import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .experiment import feature_density_histogram
from .sae import TopKSAE

PROMOTED_TOKENS = 10
CONTEXT_BEFORE = 48
CONTEXT_AFTER = 16
TOP_EXAMPLES = 12
TOP_CANDIDATES = 512
RANGE_EDGES = (1.0, 0.75, 0.5, 0.25, 0.0)
RANGE_EXAMPLES = 6
FIRES_ON_TOKENS = 8
TOKEN_SAMPLE = 4096
TOKEN_FEATURES = 20


class FeatureIndex:
    """Queries over a recorded activation index, written by record_activations.py."""

    def __init__(self, directory: Path):
        self.summary_path = directory / "summary.npz"
        self.interpretations_path = directory / "interpretations.jsonl"
        metadata = json.loads((directory / "metadata.json").read_text())
        self.run = Path(metadata["sae_dir"]).name
        self.config = metadata["sae_config"]
        self.context_size = self.config["context_size"]

        def load(name: str) -> np.ndarray:
            return np.load(directory / f"{name}.npy", mmap_mode="r")

        self.ptr = load("feature_ptr").astype(np.int64)
        self.counts = np.diff(self.ptr)
        self.max = load("feature_max")
        self.positions, self.values = load("feature_positions"), load("feature_values")
        self.token_ptr, self.token_features, self.token_values = load("token_ptr"), load("token_features"), load("token_values")
        self.token_ids = load("token_ids")

        sinks = load("sink_positions")
        self.active_tokens = len(self.token_ids) - len(sinks)
        self.density = self.counts / self.active_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_id"])
        self.vocab = [self.tokenizer.decode([token]) for token in range(len(self.tokenizer))]
        self.special_ids = set(self.tokenizer.all_special_ids)
        # Sink tokens bypass the SAE, so they don't count as occurrences.
        self.token_occurrences = np.bincount(self.token_ids, minlength=len(self.vocab)) - np.bincount(
            self.token_ids[sinks], minlength=len(self.vocab)
        )

    def summarize(self, sae: TopKSAE, model: nn.Module) -> None:
        peaks = [
            start + np.argmax(self.values[start:stop]) if stop > start else 0
            for start, stop in tqdm(zip(self.ptr[:-1], self.ptr[1:]), total=len(self.counts), desc="Peaks")
        ]
        promoted_ids, promoted_logits = promoted_tokens(sae, model, len(self.vocab))
        np.savez(self.summary_path, peak_position=self.positions[peaks], promoted_ids=promoted_ids, promoted_logits=promoted_logits)

    def load_summary(self) -> None:
        summary = np.load(self.summary_path)
        peak_tokens = self.token_ids[summary["peak_position"]]
        self.peak_tokens = [self.vocab[token] if count else None for token, count in zip(peak_tokens, self.counts)]
        self.promoted_ids, self.promoted_logits = summary["promoted_ids"], summary["promoted_logits"]

    def load_interpretations(self) -> None:
        # Written by interpret.py; features without one are shown unnamed.
        lines = self.interpretations_path.read_text().splitlines() if self.interpretations_path.exists() else []
        self.interpretations = {record["feature"]: record for record in map(json.loads, lines)}
        self.named = np.zeros(len(self.counts), dtype=bool)
        self.named[list(self.interpretations)] = True
        tests, detection = np.zeros(len(self.counts)), np.zeros(len(self.counts))
        for feature, record in self.interpretations.items():
            tests[feature] = record["generation"]["pass_rate"]
            detection[feature] = record["detection"]["balanced_accuracy"]
        # Generated tests first, detection breaking ties.
        self.check_rank = np.empty(len(self.counts), dtype=np.int64)
        self.check_rank[np.lexsort((-detection, -tests))] = np.arange(len(self.counts))

    def overview(self) -> dict:
        return {
            "run": self.run,
            "model_id": self.config["model_id"],
            "layer_index": self.config["layer_index"],
            "d_sae": len(self.counts),
            "k": self.config["k"],
            "tokens": len(self.token_ids),
            "density_histogram": feature_density_histogram(torch.from_numpy(self.counts), self.active_tokens),
        }

    def row(self, feature: int) -> dict:
        title = self.interpretations.get(int(feature), {}).get("title")
        return {"id": int(feature), "density": float(self.density[feature]), "peak_token": self.peak_tokens[feature], "title": title}

    def browse(self, sort: str, offset: int, limit: int, low: float | None, high: float | None) -> list[dict]:
        keys = {"density": -self.density, "rarity": self.density, "max": -self.max, "id": np.arange(len(self.counts)), "check": self.check_rank}
        features = np.argsort(keys[sort], kind="stable")
        if low is not None:
            # Bounds are log10 densities from the overview histogram; the tolerance keeps exact edges inside.
            with np.errstate(divide="ignore"):
                log_density = np.log10(self.density[features])
            features = features[(log_density >= low - 1e-9) & (log_density < high - 1e-9)]
        # Named features first, each group in the chosen order.
        features = features[np.argsort(~self.named[features], kind="stable")]
        return [self.row(feature) for feature in features[offset : offset + limit]]

    @lru_cache(maxsize=64)
    def feature(self, feature: int) -> dict:
        if not 0 <= feature < len(self.counts):
            raise IndexError(f"feature {feature} is outside 0-{len(self.counts) - 1}")
        positions = np.asarray(self.positions[self.ptr[feature] : self.ptr[feature + 1]], dtype=np.int64)
        values = np.asarray(self.values[self.ptr[feature] : self.ptr[feature + 1]], dtype=np.float32)
        peak = float(self.max[feature])

        rng = np.random.default_rng(feature)
        ranges = []
        for high, low in zip(RANGE_EDGES, RANGE_EDGES[1:]):
            inside = np.flatnonzero((values > low * peak) & (values <= high * peak))
            if len(inside):
                chosen = rng.choice(inside, min(RANGE_EXAMPLES, len(inside)), replace=False)
                examples = self.examples(chosen[np.argsort(-values[chosen])], positions, values)
                ranges.append({"low": low, "high": high, "share": len(inside) / len(values), "examples": examples})

        token_fires = np.bincount(self.token_ids[positions], minlength=len(self.vocab))
        fires_on = np.argsort(-token_fires, kind="stable")[:FIRES_ON_TOKENS]
        return self.row(feature) | {
            "fires": len(values),
            "max": peak,
            "top": self.examples(self.top_activations(positions, values), positions, values),
            "ranges": ranges,
            "fires_on": [
                {
                    "token": self.vocab[token],
                    "share": float(token_fires[token] / len(values)),
                    "precision": float(token_fires[token] / max(self.token_occurrences[token], 1)),
                }
                for token in fires_on
                if token_fires[token]
            ],
            "promotes": [
                {"token": self.vocab[token], "logit": float(logit)}
                for token, logit in zip(self.promoted_ids[feature], self.promoted_logits[feature])
            ],
            "interpretation": self.interpretations.get(feature),
        }

    def top_activations(self, positions: np.ndarray, values: np.ndarray) -> np.ndarray:
        # The strongest activations, at most one per context so repeated boilerplate doesn't fill the list.
        if len(values) > TOP_CANDIDATES:
            candidates = np.argpartition(-values, TOP_CANDIDATES)[:TOP_CANDIDATES]
        else:
            candidates = np.arange(len(values))
        candidates = candidates[np.argsort(-values[candidates], kind="stable")]
        _, first = np.unique(positions[candidates] // self.context_size, return_index=True)
        return candidates[np.sort(first)][:TOP_EXAMPLES]

    def examples(self, indices: np.ndarray, positions: np.ndarray, values: np.ndarray) -> list[dict]:
        return [self.example(int(positions[index]), positions, values) for index in indices]

    def example(self, position: int, positions: np.ndarray, values: np.ndarray) -> dict:
        context_start = position - position % self.context_size
        start = max(context_start, position - CONTEXT_BEFORE)
        stop = min(context_start + self.context_size, position + CONTEXT_AFTER + 1)
        ids = self.token_ids[start:stop]
        first, last = np.searchsorted(positions, (start, stop))
        activations = np.zeros(stop - start, dtype=np.float32)
        activations[positions[first:last] - start] = values[first:last]
        return {
            "tokens": [self.vocab[token] for token in ids],
            "acts": np.round(activations, 2).tolist(),
            "center": position - start,
            "special": [index for index, token in enumerate(ids) if token in self.special_ids],
        }

    def token_search(self, query: str) -> dict:
        matches = [token for token, text in enumerate(self.vocab) if text.strip() == query.strip()]
        if not matches:
            pieces = self.tokenizer.encode(" " + query.strip(), add_special_tokens=False)
            return {"matches": [], "pieces": [self.vocab[token] for token in pieces]}

        occurrences = np.flatnonzero(np.isin(self.token_ids, matches))
        sample = np.random.default_rng(0).choice(occurrences, min(TOKEN_SAMPLE, len(occurrences)), replace=False)
        # Sink tokens have no recorded activations, so they drop out of the sample.
        ranges = [np.arange(self.token_ptr[position], self.token_ptr[position + 1]) for position in sample]
        sampled = max(sum(len(activations) > 0 for activations in ranges), 1)
        flat = np.concatenate(ranges)
        mean = np.bincount(self.token_features[flat], weights=self.token_values[flat], minlength=len(self.counts)) / sampled
        rate = np.bincount(self.token_features[flat], minlength=len(self.counts)) / sampled
        total = self.token_occurrences[matches].sum()
        return {
            "matches": [{"token": self.vocab[token], "occurrences": int(self.token_occurrences[token])} for token in matches],
            "features": [
                self.row(feature) | {"mean": float(mean[feature]), "share": float(min(1.0, rate[feature] * total / self.counts[feature]))}
                for feature in np.argsort(-mean, kind="stable")[:TOKEN_FEATURES]
                if mean[feature] > 0
            ],
        }


@torch.inference_mode()
def promoted_tokens(sae: TopKSAE, model: nn.Module, vocab_size: int, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    # Each decoder direction through the final norm and unembedding, centered over the vocabulary.
    unembedding = model.get_output_embeddings().weight[:vocab_size]
    directions = torch.zeros(sae.d_sae, len(sae.keep), device=unembedding.device, dtype=unembedding.dtype)
    directions[:, sae.keep] = sae.decoder_weight.to(directions)
    ids, logits = [], []
    for chunk in tqdm(directions.split(chunk_size), desc="Logit effects"):
        chunk_logits = (model.model.norm(chunk) @ unembedding.T).float()
        values, indices = (chunk_logits - chunk_logits.mean(dim=1, keepdim=True)).topk(PROMOTED_TOKENS, dim=1)
        ids.append(indices.cpu().numpy())
        logits.append(values.cpu().numpy())
    return np.concatenate(ids), np.concatenate(logits)

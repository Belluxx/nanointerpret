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

SUMMARY_CHUNK = 1 << 25
LOGIT_TOKENS = 10
CONTEXT_BEFORE = 48
CONTEXT_AFTER = 16
TOP_EXAMPLES = 12
TOP_CANDIDATES = 512
RANGE_EDGES = (1.0, 0.75, 0.5, 0.25, 0.0)
RANGE_EXAMPLES = 6
FIRES_ON_TOKENS = 8
TOKEN_SEARCH_SAMPLE = 4096
TOKEN_SEARCH_FEATURES = 20


class FeatureIndex:
    """Queries over a recorded activation index, plus a cached per-feature summary."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.summary_path = directory / "summary.npz"
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.config = self.metadata["sae_config"]
        self.context_size = self.config["context_size"]

        def load(name: str) -> np.ndarray:
            return np.load(directory / f"{name}.npy", mmap_mode="r")

        self.ptr = np.load(directory / "feature_ptr.npy").astype(np.int64)
        self.counts = np.diff(self.ptr)
        self.max = np.load(directory / "feature_max.npy")
        self.positions, self.values = load("feature_positions"), load("feature_values")
        self.token_ptr, self.token_features, self.token_values = load("token_ptr"), load("token_features"), load("token_values")
        self.token_ids = load("token_ids")

        sinks = np.load(directory / "sink_positions.npy").astype(np.int64)
        self.active_tokens = len(self.token_ids) - len(sinks)
        self.density = self.counts / self.active_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_id"])
        self.vocab = [self.tokenizer.decode([token]) for token in range(len(self.tokenizer))]
        self.special_ids = set(self.tokenizer.all_special_ids)
        occurrences = np.bincount(self.token_ids, minlength=len(self.vocab))
        self.token_occurrences = occurrences - np.bincount(self.token_ids[sinks], minlength=len(self.vocab))

    def write_summary(self, sae: TopKSAE, model: nn.Module) -> None:
        promoted_ids, promoted_logits = promoted_tokens(sae, model, len(self.vocab))
        np.savez(
            self.summary_path, peak_position=self.peak_positions(), promoted_ids=promoted_ids, promoted_logits=promoted_logits
        )

    def peak_positions(self) -> np.ndarray:
        # One feature-major pass for the token position of each feature's strongest activation.
        d_sae = len(self.counts)
        peak_position = np.zeros(d_sae, dtype=np.int64)
        start = 0
        with tqdm(total=int(self.ptr[-1]), unit="act", unit_scale=True, desc="Summarize") as progress:
            while start < d_sae:
                stop = int(np.searchsorted(self.ptr, self.ptr[start] + SUMMARY_CHUNK, side="right")) - 1
                stop = min(max(stop, start + 1), d_sae)
                counts = self.counts[start:stop]
                values = np.asarray(self.values[self.ptr[start] : self.ptr[stop]], dtype=np.float32)
                at_peak = np.flatnonzero(values == np.repeat(self.max[start:stop], counts))
                features, first = np.unique(np.repeat(np.arange(stop - start), counts)[at_peak], return_index=True)
                peak_position[start + features] = self.positions[self.ptr[start] + at_peak[first]]
                progress.update(len(values))
                start = stop
        return peak_position

    def load_summary(self) -> None:
        summary = np.load(self.summary_path)
        self.peak_position = summary["peak_position"]
        self.promoted_ids, self.promoted_logits = summary["promoted_ids"], summary["promoted_logits"]

    def overview(self) -> dict:
        return {
            "run": Path(self.metadata["sae_dir"]).name,
            "model_id": self.config["model_id"],
            "layer_index": self.config["layer_index"],
            "d_sae": len(self.counts),
            "k": self.config["k"],
            "tokens": len(self.token_ids),
            "density_histogram": feature_density_histogram(torch.from_numpy(self.counts), self.active_tokens),
        }

    def peak_token(self, feature: int) -> str | None:
        return self.vocab[self.token_ids[self.peak_position[feature]]] if self.counts[feature] else None

    def rows(self, features: np.ndarray) -> list[dict]:
        return [
            {
                "id": int(feature),
                "density": float(self.density[feature]),
                "peak_token": self.peak_token(feature),
            }
            for feature in features
        ]

    def check(self, feature: int) -> int:
        if not 0 <= feature < len(self.counts):
            raise IndexError(f"feature {feature} is outside 0-{len(self.counts) - 1}")
        return feature

    def browse(self, sort: str, offset: int, limit: int, low: float | None, high: float | None) -> dict:
        mask = np.ones(len(self.counts), dtype=bool)
        if low is not None or high is not None:
            # Bounds are log10 densities from the overview histogram; the tolerance keeps exact edges inside.
            with np.errstate(divide="ignore"):
                log_density = np.log10(self.density)
            if low is not None:
                mask &= log_density >= low - 1e-9
            if high is not None:
                mask &= log_density < high - 1e-9
        features = np.flatnonzero(mask)
        keys = {"density": -self.density, "rarity": self.density, "max": -self.max}
        if sort != "id":
            features = features[np.argsort(keys[sort][features], kind="stable")]
        return {"total": len(features), "items": self.rows(features[offset : offset + limit])}

    def random(self) -> int:
        return int(np.random.default_rng().choice(np.flatnonzero(self.counts)))

    @lru_cache(maxsize=64)
    def feature(self, feature: int) -> dict:
        feature = self.check(feature)
        start, stop = self.ptr[feature], self.ptr[feature + 1]
        positions = np.asarray(self.positions[start:stop], dtype=np.int64)
        values = np.asarray(self.values[start:stop], dtype=np.float32)
        detail = {
            **self.rows([feature])[0],
            "fires": len(values),
            "max": float(self.max[feature]),
            "promotes": [
                {"token": self.vocab[token], "logit": float(logit)}
                for token, logit in zip(self.promoted_ids[feature], self.promoted_logits[feature])
            ],
        }
        if not len(values):
            return detail

        peak = float(self.max[feature])
        rng = np.random.default_rng(feature)
        ranges = []
        for high, low in zip(RANGE_EDGES, RANGE_EDGES[1:]):
            inside = np.flatnonzero((values > low * peak) & (values <= high * peak))
            chosen = rng.choice(inside, min(RANGE_EXAMPLES, len(inside)), replace=False)
            chosen = chosen[np.argsort(-values[chosen], kind="stable")]
            ranges.append({"low": low, "high": high, "fires": len(inside), "examples": self.examples(chosen, positions, values)})

        token_fires = np.bincount(self.token_ids[positions], minlength=len(self.vocab))
        fires_on = np.argsort(-token_fires, kind="stable")[:FIRES_ON_TOKENS]
        return detail | {
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
        }

    def top_activations(self, positions: np.ndarray, values: np.ndarray) -> np.ndarray:
        # The strongest activations, at most one per context so repeated boilerplate doesn't fill the list.
        count = min(TOP_CANDIDATES, len(values))
        candidates = np.argpartition(-values, count - 1)[:count]
        candidates = candidates[np.argsort(-values[candidates], kind="stable")]
        chosen, contexts = [], set()
        for index in candidates:
            context = positions[index] // self.context_size
            if context not in contexts:
                contexts.add(context)
                chosen.append(index)
                if len(chosen) == TOP_EXAMPLES:
                    break
        return np.asarray(chosen, dtype=np.int64)

    def examples(self, indices: np.ndarray, positions: np.ndarray, values: np.ndarray) -> list[dict]:
        return [self.example(int(positions[index]), positions, values) for index in indices]

    def example(self, position: int, positions: np.ndarray, values: np.ndarray) -> dict:
        context_start = position - position % self.context_size
        start = max(context_start, position - CONTEXT_BEFORE)
        stop = min(context_start + self.context_size, position + CONTEXT_AFTER + 1)
        ids = np.asarray(self.token_ids[start:stop])
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
        target = query.strip()
        matches = [token for token, text in enumerate(self.vocab) if text.strip() == target]
        if not matches:
            matches = [token for token, text in enumerate(self.vocab) if text.strip().casefold() == target.casefold()]
        if not target or not matches:
            pieces = self.tokenizer.encode(" " + target, add_special_tokens=False) if target else []
            return {"query": query, "matches": [], "pieces": [self.vocab[token] for token in pieces]}

        occurrences = np.flatnonzero(np.isin(self.token_ids, matches))
        rng = np.random.default_rng(0)
        sample = np.sort(rng.choice(occurrences, min(TOKEN_SEARCH_SAMPLE, len(occurrences)), replace=False))
        starts = np.asarray(self.token_ptr[sample], dtype=np.int64)
        lengths = np.asarray(self.token_ptr[sample + 1], dtype=np.int64) - starts
        # Sink tokens have no recorded activations, so they drop out of the sample.
        starts, lengths = starts[lengths > 0], lengths[lengths > 0]
        flat = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())
        features = np.asarray(self.token_features[flat], dtype=np.int64)
        values = np.asarray(self.token_values[flat], dtype=np.float32)
        sampled = len(starts)
        mean = np.bincount(features, weights=values, minlength=len(self.counts)) / max(sampled, 1)
        rate = np.bincount(features, minlength=len(self.counts)) / max(sampled, 1)
        top = np.argsort(-mean, kind="stable")[:TOKEN_SEARCH_FEATURES]
        top = top[mean[top] > 0]
        total = int(sum(self.token_occurrences[token] for token in matches))
        return {
            "query": query,
            "matches": [{"token": self.vocab[token], "occurrences": int(self.token_occurrences[token])} for token in matches],
            "features": [
                row | {
                    "mean": float(mean[row["id"]]),
                    "share": float(min(1.0, rate[row["id"]] * total / max(self.counts[row["id"]], 1))),
                }
                for row in self.rows(top)
            ],
        }


@torch.inference_mode()
def promoted_tokens(sae: TopKSAE, model: nn.Module, vocab_size: int, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    # Each decoder direction through the final norm and unembedding, centered over the vocabulary.
    unembedding = model.get_output_embeddings().weight[:vocab_size]
    directions = torch.zeros(sae.d_sae, len(sae.keep), device=unembedding.device, dtype=unembedding.dtype)
    directions[:, sae.keep] = sae.decoder_weight.to(directions)
    ids, logits = [], []
    for chunk in tqdm(directions.split(chunk_size), unit="chunk", desc="Logit effects"):
        chunk_logits = (model.model.norm(chunk) @ unembedding.T).float()
        values, indices = (chunk_logits - chunk_logits.mean(dim=1, keepdim=True)).topk(LOGIT_TOKENS, dim=1)
        ids.append(indices.cpu().numpy())
        logits.append(values.cpu().numpy())
    return np.concatenate(ids), np.concatenate(logits)

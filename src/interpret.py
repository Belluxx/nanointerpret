import json
import random
import threading
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .features import FeatureIndex
from .runtime import capture_layer_input, patch_layer_input
from .sae import FIRING_THRESHOLD, TopKSAE

WINDOW_BEFORE = 32
WINDOW_AFTER = 8
EXPLAIN_TOP = 8
# Activation-quantile bins: examples are drawn evenly from each, so weak activations are represented.
QUANTILE_EDGES = (0.0, 0.5, 0.8, 0.95, 1.0)
EXPLAIN_PER_BIN = 3
TEST_PER_BIN = 3
NEGATIVES_PER_KIND = 4
NEGATIVE_KINDS = ("random", "same_token", "neighbor")
TOKEN_STATS = 8
TOKEN_STATS_MIN_SHARE = 0.01
GENERATED_PAIRS = 6
# A generated pair passes when its contrast activates the feature under this fraction of the first text's activation.
# Irrelevant changes, such as a neutral sentence before the text, cut activation that much for under 2% of texts.
CONTRAST_RATIO = 0.5
# Rarer features barely make the top k: their activations are small and vanish with any change of context,
# so their titles come out vague and their tests fail.
MIN_FIRES = 1000
NEWLINE = "↵"
STEER_PROMPTS = (
    "In my experience,",
    "The most important thing to know is that",
    "I once heard that",
    "Here is what happened next:",
    "My favorite part of the day is",
    "According to the report,",
    "We spent the afternoon talking about",
    "The list includes",
)
# Multiples of the feature's max activation; the strongest one within STEER_KL is reported.
STEER_STRENGTHS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
STEER_KL = 1.0
STEER_TOKENS = 10
STEER_MIN_PROBABILITY = 1e-3


@dataclass
class Example:
    start: int
    stop: int
    kind: str
    activation: float = 0.0
    text: str = ""


@dataclass
class FeatureExamples:
    explain: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)


class Sampler:
    """Draws explanation examples and disjoint held-out test examples of a feature from the recorded activations."""

    def __init__(self, index: FeatureIndex, decoder: torch.Tensor, seed: int = 0):
        self.index = index
        self.decoder = decoder
        self.seed = seed
        self.eos = index.tokenizer.eos_token_id

    def activations(self, feature: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = self.index.ptr[feature], self.index.ptr[feature + 1]
        return np.asarray(self.index.positions[start:stop], dtype=np.int64), np.asarray(self.index.values[start:stop], dtype=np.float32)

    def window(self, position: int) -> tuple[int, int]:
        # Up to WINDOW_BEFORE tokens before and WINDOW_AFTER after, inside the context and the document.
        context = self.index.context_size
        context_start = position - position % context
        start = max(context_start, position - WINDOW_BEFORE)
        stop = min(context_start + context, position + WINDOW_AFTER + 1)
        ids = np.asarray(self.index.token_ids[start:stop])
        before = np.flatnonzero(ids[: position - start] == self.eos)
        after = np.flatnonzero(ids[position - start + 1 :] == self.eos)
        if len(before):
            start += before[-1] + 1
        if len(after):
            stop = position + 1 + after[0]
        return start, stop

    def render(self, start: int, stop: int, marks: list[int]) -> str:
        ids = np.asarray(self.index.token_ids[start:stop]).tolist()
        decode = self.index.tokenizer.decode
        pieces, cursor = [], start
        for mark in sorted(marks):
            pieces += [decode(ids[cursor - start : mark - start]), "<<", decode([ids[mark - start]]), ">>"]
            cursor = mark + 1
        pieces.append(decode(ids[cursor - start :]))
        return "".join(pieces).replace("\n", NEWLINE).strip()

    def example(self, position: int, kind: str, activation: float = 0.0) -> Example:
        start, stop = self.window(position)
        return Example(start, stop, kind, activation, self.render(start, stop, [position]))

    def sample(self, feature: int) -> FeatureExamples:
        positions, values = self.activations(feature)
        rng = np.random.default_rng(self.seed + feature)
        result = FeatureExamples()
        used_contexts, used_texts = set(), set()

        def take(candidates: np.ndarray, count: int, kind: str, into: list[Example]) -> None:
            # Repeated documents are common in web text, so contexts and texts are each used once.
            for candidate in candidates:
                if count == 0:
                    return
                position = int(positions[candidate])
                example = self.example(position, kind, float(values[candidate]))
                context = position // self.index.context_size
                if context in used_contexts or example.text in used_texts:
                    continue
                used_contexts.add(context)
                used_texts.add(example.text)
                into.append(example)
                count -= 1

        order = np.argsort(-values, kind="stable")
        take(order, EXPLAIN_TOP, "top", result.explain)
        ranks = np.empty(len(values), dtype=np.int64)
        ranks[order[::-1]] = np.arange(len(values))
        quantiles = ranks / max(len(values) - 1, 1)
        for bin, (low, high) in enumerate(zip(QUANTILE_EDGES, QUANTILE_EDGES[1:])):
            inside = rng.permutation(np.flatnonzero((quantiles >= low) & (quantiles <= high)))
            take(inside, EXPLAIN_PER_BIN, f"q{bin}", result.explain)
            take(inside, TEST_PER_BIN, f"q{bin}", result.test)

        firing = set(positions.tolist())
        result.test += self.same_token_negatives(positions, firing, rng)
        result.test += self.neighbor_negatives(feature, firing, rng)
        result.test += self.random_negatives(firing, rng)
        return result

    def silent(self, position: int, firing: set[int]) -> bool:
        # Not firing for this feature and not a sink token, which has no features at all.
        return position not in firing and self.index.token_ptr[position + 1] > self.index.token_ptr[position]

    def same_token_negatives(self, positions: np.ndarray, firing: set[int], rng) -> list[Example]:
        tokens, counts = np.unique(np.asarray(self.index.token_ids[positions]), return_counts=True)
        precision = counts / np.maximum(self.index.token_occurrences[tokens], 1)
        # Near-misses come from tokens it fires on at most half the time. On tokens it mostly fires on, the silent
        # occurrences are weak activations that missed the top k, which no description can predict.
        ranked = tokens[np.lexsort((-counts, (precision < 0.05) | (precision > 0.5)))]
        examples = []
        for token in ranked[:3]:
            occurrences = np.flatnonzero(self.index.token_ids == token)
            silent = [int(p) for p in rng.permutation(occurrences)[:256] if self.silent(int(p), firing)]
            for position in silent[: NEGATIVES_PER_KIND - len(examples)]:
                examples.append(self.example(position, "same_token"))
            if len(examples) == NEGATIVES_PER_KIND:
                break
        return examples

    def neighbor(self, feature: int) -> int:
        similarity = self.decoder @ self.decoder[feature]
        similarity[feature] = -1
        return int(similarity.argmax())

    def neighbor_negatives(self, feature: int, firing: set[int], rng) -> list[Example]:
        # The upper half of the closest decoder direction's activations, where this feature stays silent.
        positions, values = self.activations(self.neighbor(feature))
        strong = np.flatnonzero(values >= np.quantile(values, 0.5)) if len(values) else np.array([], dtype=np.int64)
        silent = [int(positions[i]) for i in rng.permutation(strong)[:256] if self.silent(int(positions[i]), firing)]
        return [self.example(position, "neighbor") for position in silent[:NEGATIVES_PER_KIND]]

    def random_negatives(self, firing: set[int], rng) -> list[Example]:
        examples = []
        while len(examples) < NEGATIVES_PER_KIND:
            position = int(rng.integers(len(self.index.token_ids)))
            if self.silent(position, firing):
                examples.append(self.example(position, "random"))
        return examples


def explain_prompt(sampler: Sampler, feature: int, examples: list[Example]) -> str:
    index = sampler.index
    positions, values = sampler.activations(feature)
    peak = float(values.max())
    lines = [
        f"A sparse autoencoder was trained on the residual stream of the base language model {index.config['model_id'].split('/')[-1]} "
        f"(layer {index.config['layer_index']}). Each of its features fires on some tokens. Explain feature {feature}.",
        "",
        "In each excerpt the tokens where the feature fires are wrapped in << >>, followed by their activations "
        f"on a 1-10 scale relative to the feature's maximum. {NEWLINE} marks a line break.",
    ]
    for title, top in (("Strongest activations", True), ("Activations sampled across the feature's range, weakest last", False)):
        lines += ["", f"## {title}"]
        chosen = sorted((example for example in examples if (example.kind == "top") == top), key=lambda example: -example.activation)
        for number, example in enumerate(chosen, 1):
            inside = (positions >= example.start) & (positions < example.stop)
            marks = positions[inside].tolist()
            levels = [max(1, round(10 * value / peak)) for value in values[inside]]
            text = sampler.render(example.start, example.stop, marks)
            tokens = ", ".join(f"{json.dumps(index.vocab[index.token_ids[mark]], ensure_ascii=False)} {level}" for mark, level in zip(marks, levels))
            lines.append(f"{number}. {text}\n   Activations: {tokens}")

    token_fires = np.bincount(np.asarray(index.token_ids[positions]), minlength=len(index.vocab))
    common = [token for token in np.argsort(-token_fires, kind="stable")[:TOKEN_STATS] if token_fires[token] >= TOKEN_STATS_MIN_SHARE * len(positions)]
    lines += ["", "## Tokens it fires on most"]
    for token in common:
        share = token_fires[token] / len(positions)
        precision = token_fires[token] / max(index.token_occurrences[token], 1)
        lines.append(f"- {json.dumps(index.vocab[token], ensure_ascii=False)}: {share:.0%} of its activations; fires on {precision:.0%} of this token's occurrences")
    if not common:
        lines.append(f"No token accounts for {TOKEN_STATS_MIN_SHARE:.0%} of its activations.")

    lines += [
        "",
        "Describe when the feature fires. Name the exact token(s) if it is tied to specific tokens, and the context that "
        "decides whether it fires. The description must also cover the weaker activations, not only the strongest.",
        'Return compact JSON: {"title": a specific name of at most 8 words, without the word "feature", "description": one or two precise sentences}.',
    ]
    return "\n".join(lines)


EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
    "required": ["title", "description"],
}


def detect_prompt(explanation: dict, examples: list[Example]) -> str:
    lines = [
        "A feature of a language model fires on some tokens. Its description:",
        f"Title: {explanation['title']}",
        f"Description: {explanation['description']}",
        "",
        f"In each excerpt below one token is wrapped in << >>; {NEWLINE} marks a line break. Judging only from the description, "
        "decide for each excerpt whether the feature fires on the marked token.",
        "",
    ]
    lines += [f"{number}. {example.text}" for number, example in enumerate(examples, 1)]
    lines += ["", f'Return compact JSON: {{"fires": [{len(examples)} booleans, one per excerpt in order]}}.']
    return "\n".join(lines)


def detect_schema(count: int) -> dict:
    answers = {"type": "array", "items": {"type": "boolean"}, "minItems": count, "maxItems": count}
    return {"type": "object", "properties": {"fires": answers}, "required": ["fires"]}


def generate_prompt(explanation: dict) -> str:
    return "\n".join([
        "A feature of a language model fires on some tokens. Its description:",
        f"Title: {explanation['title']}",
        f"Description: {explanation['description']}",
        "",
        f"Write {GENERATED_PAIRS} pairs of short texts (one or two sentences, like ordinary web text) to test the description. "
        "In each pair, the feature should fire on the first text and stay silent on the second, which is otherwise as similar "
        "as possible: change only what the description says the feature depends on. Make the pairs varied.",
        'Return compact JSON: {"pairs": [{"fires": text, "silent": text}, ...]}.',
    ])


GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"fires": {"type": "string"}, "silent": {"type": "string"}},
                "required": ["fires", "silent"],
            },
            "minItems": GENERATED_PAIRS,
            "maxItems": GENERATED_PAIRS,
        }
    },
    "required": ["pairs"],
}


class Judge:
    """Chat completions from an OpenAI-compatible server, such as llama-server, with JSON-schema output."""

    def __init__(self, base_url: str, model: str, timeout: float = 600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str, schema: dict, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        }
        request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])


class Probe:
    """Runs the language model with the SAE: a feature's activations on new text, and its effect when steering."""

    def __init__(self, model: nn.Module, layer: nn.Module, sae: TopKSAE, tokenizer, feature_max: np.ndarray, vocab: list[str]):
        self.model, self.layer, self.sae, self.tokenizer = model, layer, sae, tokenizer
        self.feature_max = feature_max
        self.vocab = vocab
        # Worker threads share one model, and MPS crashes on concurrent use, so all work on its device holds this lock.
        self.lock = threading.Lock()
        encoded = [tokenizer.encode(prompt) for prompt in STEER_PROMPTS]
        input_ids = torch.full((len(encoded), max(map(len, encoded))), tokenizer.pad_token_id)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids)
        # One row per prompt and strength, the unsteered strength 0 first; padding comes after each prompt's last token.
        repeats = len(STEER_STRENGTHS) + 1
        self.steer_ids = input_ids.to(model.device).repeat(repeats, 1)
        self.steer_last = torch.tensor([len(ids) - 1 for ids in encoded], device=model.device).repeat(repeats)
        self.steer_scales = torch.tensor((0.0, *STEER_STRENGTHS), device=model.device).repeat_interleave(len(encoded))

    @torch.inference_mode()
    def max_activations(self, texts: list[str], feature: int) -> list[float]:
        # The feature's strongest activation over each text, as a fraction of its recorded max.
        result = []
        with self.lock:
            for text in texts:
                input_ids = torch.tensor([self.tokenizer.encode(text)], device=self.model.device)
                indices, values, _ = self.sae.encode(self.sae.inputs(capture_layer_input(self.model, self.layer, input_ids)[0]))
                value = float((values * (indices == feature)).sum(dim=1).max()) if len(values) else 0.0
                result.append(value / float(self.feature_max[feature]) if value > FIRING_THRESHOLD else 0.0)
        return result

    @torch.inference_mode()
    def steer(self, feature: int) -> dict:
        # The next tokens made likelier by adding the decoder direction at the SAE's layer, at the strongest
        # multiple of the max activation that keeps the KL from the unsteered predictions within STEER_KL.
        # Pass-through dims are outside the SAE, so the direction leaves them unchanged.
        with self.lock:
            direction = torch.zeros(len(self.sae.keep), device=self.sae.keep.device)
            direction[self.sae.keep] = self.sae.decoder_weight[feature] / self.sae.activation_scale
            delta = self.steer_scales[:, None] * float(self.feature_max[feature]) * direction

            def add(hidden):
                steered = hidden.clone()
                # Position 0 is an attention sink and is left alone.
                steered[:, 1:] += delta[:, None].to(hidden.dtype)
                return steered

            with patch_layer_input(self.layer, add):
                logits = self.model(input_ids=self.steer_ids, use_cache=False).logits
            logprobs = torch.log_softmax(logits[torch.arange(len(logits)), self.steer_last].float(), dim=-1).cpu()
        base, *steered = logprobs.view(len(STEER_STRENGTHS) + 1, len(STEER_PROMPTS), -1)
        kls = [float((run.exp() * (run - base)).sum(dim=-1).mean()) for run in steered]
        chosen = max([index for index, kl in enumerate(kls) if kl <= STEER_KL], default=0)
        run = steered[chosen]
        shift = (run - base).mean(dim=0)
        shift[run.exp().max(dim=0).values < STEER_MIN_PROBABILITY] = -torch.inf
        return {
            "strength": STEER_STRENGTHS[chosen],
            "kl": round(kls[chosen], 3),
            "tokens": [self.vocab[token] for token in shift.topk(STEER_TOKENS).indices.tolist()],
        }


def shuffled(examples: list[Example], seed: int) -> list[Example]:
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    return examples


def detection_scores(examples: list[Example], answers: list[bool]) -> dict:
    by_kind = {}
    for example, answer in zip(examples, answers):
        by_kind.setdefault(example.kind, []).append(answer)
    recall = float(np.mean([answer for example, answer in zip(examples, answers) if example.kind not in NEGATIVE_KINDS]))
    # Each negative kind weighs the same, whatever the number of examples drawn for it.
    specificity = float(np.mean([1 - np.mean(by_kind[kind]) for kind in NEGATIVE_KINDS if kind in by_kind]))
    return {
        "balanced_accuracy": (recall + specificity) / 2,
        "recall": recall,
        "specificity": specificity,
        "says_fires": {kind: float(np.mean(answers)) for kind, answers in sorted(by_kind.items())},
    }


def measured_pairs(pairs: list[dict], fires: list[float], silent: list[float]) -> list[dict]:
    return [
        pair | {"fires_activation": round(a, 3), "silent_activation": round(b, 3), "passed": a > 0 and b < CONTRAST_RATIO * a}
        for pair, a, b in zip(pairs, fires, silent)
    ]


def generation_scores(pairs: list[dict]) -> dict:
    return {
        "pass_rate": float(np.mean([pair["passed"] for pair in pairs])),
        "recall": float(np.mean([pair["fires_activation"] > 0 for pair in pairs])),
        "specificity": float(np.mean([pair["silent_activation"] == 0 for pair in pairs])),
    }

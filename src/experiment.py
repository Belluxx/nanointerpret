from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from .data import iter_context_batches
from .sae import RunningMetrics, TopKSAE


@dataclass
class ExperimentConfig:
    model_id: str
    dataset_id: str
    dataset_config: str
    train_tokens: int
    validation_tokens: int
    context_size: int
    layer_index: int
    layer_path: str
    residual_location: str
    d_model: int
    d_sae: int
    width_multiplier: int
    k: int
    learning_rate: float
    model_batch_size: int
    sae_batch_size: int
    seed: int
    device: str
    model_dtype: str


def find_transformer_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    candidates: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and name.split(".")[-1] == "layers" and len(module) > 1:
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("could not locate the transformer's ModuleList named 'layers'")
    return max(candidates, key=lambda item: len(item[1]))


class _ActivationCaptured(Exception):
    pass


class ResidualStreamCapture:
    """Capture a layer's input residual and stop the rest of the model forward."""

    def __init__(self, layer: nn.Module):
        self.activation: Tensor | None = None

        def capture(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            self.activation = hidden.detach()
            raise _ActivationCaptured

        self.handle = layer.register_forward_pre_hook(capture, with_kwargs=True)

    @torch.no_grad()
    def __call__(
        self, model: nn.Module, input_ids: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        self.activation = None
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _ActivationCaptured:
            pass
        if self.activation is None:
            raise RuntimeError("the residual-stream hook did not run")
        return self.activation

    def close(self) -> None:
        self.handle.remove()


def save_checkpoint(
    path: Path,
    sae: TopKSAE,
    optimizer: torch.optim.Optimizer,
    processed_tokens: int,
    processed_contexts: int,
    last_fired: Tensor,
    config: ExperimentConfig,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "sae": sae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "processed_tokens": processed_tokens,
            "processed_contexts": processed_contexts,
            "last_fired": last_fired.cpu(),
            "config": asdict(config),
        },
        temporary,
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_resume_config(checkpoint_config: object, config: ExperimentConfig) -> None:
    current_config = asdict(config)
    if not isinstance(checkpoint_config, dict):
        raise ValueError("cannot resume: checkpoint has no valid experiment configuration")

    fields = sorted(checkpoint_config.keys() | current_config.keys())
    mismatches = [
        field
        for field in fields
        if checkpoint_config.get(field) != current_config.get(field)
        or (field in checkpoint_config) != (field in current_config)
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: checkpoint={checkpoint_config.get(field)!r}, "
            f"current={current_config.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"cannot resume with a different experiment configuration ({details})")


def move_and_capture_residual(
    model: nn.Module,
    capture: ResidualStreamCapture,
    input_ids: Tensor,
    attention_mask: Tensor,
    batch_tokens: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Capture packed residuals without constructing an all-true GPU mask."""
    full_batch = batch_tokens == input_ids.numel()
    input_ids = input_ids.to(device, non_blocking=True)
    if full_batch:
        residual = capture(model, input_ids, None).flatten(0, 1).float()
        return residual, input_ids, None

    attention_mask = attention_mask.to(device, non_blocking=True)
    valid_mask = attention_mask.bool()
    residual = capture(model, input_ids, attention_mask)[valid_mask].float()
    return residual, input_ids, valid_mask


def train_sae(
    model: nn.Module,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    train_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> int:
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    metrics_path = args.output_dir / "train_metrics.jsonl"
    processed_tokens = 0
    processed_contexts = 0
    last_fired = torch.full((sae.d_sae,), -1, dtype=torch.int64, device=device)

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"cannot resume: {checkpoint_path} does not exist")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        validate_resume_config(checkpoint.get("config"), config)
        sae.load_state_dict(checkpoint["sae"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        processed_tokens = int(checkpoint["processed_tokens"])
        processed_contexts = int(
            checkpoint.get("processed_contexts", math.ceil(processed_tokens / args.context_size))
        )
        last_fired.copy_(checkpoint["last_fired"].to(device))
        print(f"resumed at {processed_tokens:,} training tokens")
    elif checkpoint_path.exists():
        raise FileExistsError(
            f"{checkpoint_path} already exists; pass --resume or choose another --output-dir"
        )
    else:
        metrics_path.write_text("")

    (args.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    batches = iter_context_batches(
        train_tokens,
        args.context_size,
        args.model_batch_size,
        pad_token_id,
        shuffle=True,
        seed=args.seed,
        skip_contexts=processed_contexts,
    )
    metrics = RunningMetrics(sae.d_model, device)
    next_log = ((processed_tokens // args.log_every) + 1) * args.log_every
    next_checkpoint = ((processed_tokens // args.checkpoint_every) + 1) * args.checkpoint_every
    progress = tqdm(total=len(train_tokens), initial=processed_tokens, unit="tok", desc="train")
    start_time = time.monotonic()
    start_tokens = processed_tokens

    for input_ids, attention_mask, context_ids in batches:
        batch_tokens = int(attention_mask.sum())
        residual, _input_ids, _valid_mask = move_and_capture_residual(
            model, capture, input_ids, attention_mask, batch_tokens, device
        )

        if processed_tokens == 0:
            with torch.no_grad():
                sae.decoder_bias.copy_(residual.mean(dim=0))

        for start in range(0, len(residual), args.sae_batch_size):
            x = residual[start : start + args.sae_batch_size]
            reconstruction, indices, values = sae(x)
            loss = F.mse_loss(reconstruction, x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            sae.constrain_decoder_gradient()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), args.gradient_clip)
            optimizer.step()
            sae.normalize_decoder()

            metrics.update(x.detach(), reconstruction.detach(), values.detach())
            with torch.no_grad():
                fired = torch.unique(indices[values > 0])
                last_fired[fired] = processed_tokens + start + len(x)

        processed_tokens += batch_tokens
        processed_contexts += len(context_ids)
        progress.update(batch_tokens)

        if processed_tokens >= next_log or processed_tokens == len(train_tokens):
            record = {
                "split": "train",
                "tokens": processed_tokens,
                **metrics.compute(),
                "dead_features": int((last_fired < processed_tokens - args.dead_window).sum().item())
                if processed_tokens >= args.dead_window
                else None,
                "tokens_per_second": (processed_tokens - start_tokens)
                / max(time.monotonic() - start_time, 1e-9),
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True))
            metrics.reset()
            while next_log <= processed_tokens:
                next_log += args.log_every

        if processed_tokens >= next_checkpoint or processed_tokens == len(train_tokens):
            save_checkpoint(
                checkpoint_path,
                sae,
                optimizer,
                processed_tokens,
                processed_contexts,
                last_fired,
                config,
            )
            while next_checkpoint <= processed_tokens:
                next_checkpoint += args.checkpoint_every

    progress.close()
    save_checkpoint(
        checkpoint_path,
        sae,
        optimizer,
        processed_tokens,
        processed_contexts,
        last_fired,
        config,
    )
    torch.save({"sae": sae.state_dict(), "config": asdict(config)}, args.output_dir / "sae_final.pt")
    return processed_tokens


@torch.inference_mode()
def evaluate_sae(
    model: nn.Module,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    validation_tokens: np.memmap,
    pad_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
    excluded_token_ids: set[int],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    metrics = RunningMetrics(sae.d_model, device)
    fire_counts = np.zeros(sae.d_sae, dtype=np.int64)
    max_activation = np.zeros(sae.d_sae, dtype=np.float32)
    report_max_activation = np.zeros(sae.d_sae, dtype=np.float32)
    batches = iter_context_batches(
        validation_tokens,
        args.context_size,
        args.model_batch_size,
        pad_token_id,
        shuffle=False,
        seed=args.seed,
    )
    progress = tqdm(total=len(validation_tokens), unit="tok", desc="validate")
    for input_ids, attention_mask, _context_ids in batches:
        batch_tokens = int(attention_mask.sum())
        residual, input_ids, valid_mask = move_and_capture_residual(
            model, capture, input_ids, attention_mask, batch_tokens, device
        )
        valid_token_ids = (
            input_ids.flatten() if valid_mask is None else input_ids[valid_mask]
        ).cpu().numpy()
        for start in range(0, len(residual), args.sae_batch_size):
            x = residual[start : start + args.sae_batch_size]
            reconstruction, indices, values = sae(x)
            metrics.update(x, reconstruction, values)
            idx = indices.cpu().numpy().reshape(-1)
            val = values.cpu().numpy().reshape(-1)
            positive = val > 0
            idx, val = idx[positive], val[positive]
            fire_counts += np.bincount(idx, minlength=sae.d_sae)
            np.maximum.at(max_activation, idx, val)
            token_ids = np.repeat(valid_token_ids[start : start + len(x)], sae.k)[positive]
            reportable = ~np.isin(token_ids, tuple(excluded_token_ids))
            np.maximum.at(report_max_activation, idx[reportable], val[reportable])
        progress.update(batch_tokens)
    progress.close()

    result = {
        "split": "validation",
        "tokens": len(validation_tokens),
        **metrics.compute(),
        "dead_features": int((fire_counts == 0).sum()),
        "active_features": int((fire_counts > 0).sum()),
    }
    return result, fire_counts, max_activation, report_max_activation


@torch.inference_mode()
def collect_top_examples(
    model: nn.Module,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    validation_tokens: np.memmap,
    selected_features: np.ndarray,
    tokenizer,
    pad_token_id: int,
    device: torch.device,
    args: argparse.Namespace,
    excluded_token_ids: set[int],
) -> list[dict]:
    lookup = np.full(sae.d_sae, -1, dtype=np.int64)
    lookup[selected_features] = np.arange(len(selected_features))
    hit_values: list[list[np.ndarray]] = [[] for _ in selected_features]
    hit_positions: list[list[np.ndarray]] = [[] for _ in selected_features]
    batches = iter_context_batches(
        validation_tokens,
        args.context_size,
        args.model_batch_size,
        pad_token_id,
        shuffle=False,
        seed=args.seed,
    )
    progress = tqdm(total=len(validation_tokens), unit="tok", desc="top examples")

    for input_ids, attention_mask, context_ids in batches:
        batch_tokens = int(attention_mask.sum())
        residual, _input_ids, valid_mask = move_and_capture_residual(
            model, capture, input_ids, attention_mask, batch_tokens, device
        )
        position_grid = (
            torch.as_tensor(context_ids, device=device)[:, None] * args.context_size
            + torch.arange(args.context_size, device=device)[None, :]
        )
        absolute_positions = (
            position_grid.flatten() if valid_mask is None else position_grid[valid_mask]
        )

        for start in range(0, len(residual), args.sae_batch_size):
            x = residual[start : start + args.sae_batch_size]
            _reconstruction, indices, values = sae(x)
            idx = indices.cpu().numpy().reshape(-1)
            val = values.cpu().numpy().reshape(-1)
            positions = np.repeat(
                absolute_positions[start : start + len(x)].cpu().numpy(), sae.k
            )
            slots = lookup[idx]
            token_ids = np.asarray(validation_tokens[positions], dtype=np.int64)
            wanted = (slots >= 0) & (val > 0) & ~np.isin(token_ids, tuple(excluded_token_ids))
            slots, val, positions = slots[wanted], val[wanted], positions[wanted]
            for slot in np.unique(slots):
                mask = slots == slot
                hit_values[int(slot)].append(val[mask])
                hit_positions[int(slot)].append(positions[mask])
        progress.update(batch_tokens)
    progress.close()

    reports: list[dict] = []
    for slot, feature_id in enumerate(selected_features):
        if not hit_values[slot]:
            continue
        values = np.concatenate(hit_values[slot])
        positions = np.concatenate(hit_positions[slot])
        take = min(args.examples_per_feature, len(values))
        best = np.argpartition(-values, take - 1)[:take]
        best = best[np.argsort(-values[best])]
        examples = []
        for index in best:
            position = int(positions[index])
            context_start = (position // args.context_size) * args.context_size
            context_end = min(context_start + args.context_size, len(validation_tokens))
            left = max(context_start, position - args.example_radius)
            right = min(context_end, position + args.example_radius + 1)
            prefix = tokenizer.decode(
                validation_tokens[left:position].tolist(), skip_special_tokens=True
            )
            token_text = tokenizer.decode(
                [int(validation_tokens[position])], skip_special_tokens=False
            )
            suffix = tokenizer.decode(
                validation_tokens[position + 1 : right].tolist(), skip_special_tokens=True
            )
            examples.append(
                {
                    "activation": float(values[index]),
                    "token_position": position,
                    "text": f"{prefix}<<{token_text}>>{suffix}",
                }
            )
        reports.append({"feature": int(feature_id), "examples": examples})
    return reports


def rank_reports_by_token_coherence(
    reports: list[dict], validation_tokens: np.memmap, tokenizer
) -> list[dict]:
    """Put consistent lexical detectors first in the human-inspection report."""

    def score(report: dict) -> tuple[bool, float, float]:
        labels = [
            tokenizer.decode([int(validation_tokens[example["token_position"]])])
            .strip()
            .casefold()
            for example in report["examples"]
        ]
        labels = [label for label in labels if label]
        max_activation = max(example["activation"] for example in report["examples"])
        if not labels:
            return False, 0.0, max_activation
        dominance = max(labels.count(label) for label in set(labels)) / len(labels)
        dominant_label = max(set(labels), key=labels.count)
        lexical = any(character.isalnum() for character in dominant_label)
        return lexical, dominance, max_activation

    return sorted(reports, key=score, reverse=True)


def write_example_markdown(path: Path, reports: list[dict]) -> None:
    lines = [
        "# Top-activating validation examples",
        "",
        "Features are ordered by token-level coherence, then activation magnitude. "
        "The activating token is enclosed in `<<...>>`.",
        "",
    ]
    for report in reports:
        lines.extend([f"## Feature {report['feature']}", ""])
        for example in report["examples"]:
            text = example["text"].replace("\n", " ").strip()
            lines.append(f"- `{example['activation']:.5f}` — {text}")
        lines.append("")
    path.write_text("\n".join(lines))

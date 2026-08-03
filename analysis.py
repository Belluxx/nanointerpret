from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import (
    TokenCacheSpec,
    iter_context_batches,
    token_cache_is_valid,
    token_cache_paths,
)
from src.experiment import ResidualStreamCapture, find_transformer_layers
from src.runtime import choose_device
from src.sae import FIRING_THRESHOLD, TopKSAE


SAE_DIR = Path("artifacts/sae_gemma_3_270m")
CACHE_DIR = Path("artifacts/token_cache")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record SAE feature activations on the local evaluation set."
    )
    parser.add_argument(
        "--sae-dir",
        type=Path,
        default=SAE_DIR,
        help="Training output directory containing config.json and sae_final.pt.",
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path. Default: <sae-dir>/analysis.npz.",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=None,
        help="Process at most this many evaluation tokens.",
    )
    parser.add_argument(
        "--model-batch-size",
        type=positive_int,
        default=None,
        help="Contexts processed together. Default: the training configuration.",
    )
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    return parser.parse_args()


def load_config(sae_dir: Path) -> dict:
    config_path = sae_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"SAE configuration not found: {config_path}")
    return json.loads(config_path.read_text())


def evaluation_cache(config: dict, cache_dir: Path) -> tuple[Path, int] | None:
    spec = TokenCacheSpec(
        cache_dir=cache_dir,
        model_id=config["model_id"],
        dataset_id=config["dataset_id"],
        dataset_config=config["dataset_config"],
        train_tokens=int(config["train_tokens"]),
        validation_tokens=int(config["validation_tokens"]),
    )
    train_path, validation_path, metadata_path = token_cache_paths(spec)
    if spec.validation_tokens <= 0 or not token_cache_is_valid(
        train_path, validation_path, metadata_path, spec
    ):
        return None
    return validation_path, spec.validation_tokens


def load_sae(sae_dir: Path, config: dict, device: torch.device) -> TopKSAE:
    checkpoint_path = sae_dir / "sae_final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config:
        raise ValueError("config.json does not match the SAE checkpoint configuration")

    sae = TopKSAE(
        int(config["d_model"]),
        int(config["d_sae"]),
        int(config["k"]),
        device,
        subtract_pre_bias=bool(config["subtract_pre_bias"]),
    )
    sae.load_state_dict(checkpoint["sae"])
    return sae.eval().requires_grad_(False)


def encode_residuals(
    sae: TopKSAE, residuals: Tensor, activation_scale: float, batch_size: int
) -> tuple[Tensor, Tensor]:
    indices = []
    values = []
    for start in range(0, len(residuals), batch_size):
        x = residuals[start : start + batch_size].float()
        x.mul_(activation_scale)
        _reconstruction, batch_indices, batch_values = sae(x)
        indices.append(batch_indices.cpu())
        values.append(batch_values.cpu())
    return torch.cat(indices), torch.cat(values)


def csr_activations(
    indices: Tensor, values: Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_ids, order = indices.sort(dim=1)
    sorted_values = values.gather(1, order)
    firing = sorted_values > FIRING_THRESHOLD
    counts = firing.sum(dim=1).numpy().astype(np.uint32, copy=False)
    feature_ids = sorted_ids[firing].numpy().astype(np.uint32, copy=False)
    active_values = sorted_values[firing].numpy().astype(np.float32, copy=False)
    return counts, feature_ids, active_values


@torch.inference_mode()
def write_analysis(
    output_path: Path,
    tokenizer,
    model,
    capture: ResidualStreamCapture,
    sae: TopKSAE,
    evaluation_tokens: np.ndarray,
    config: dict,
    device: torch.device,
    model_batch_size: int,
    evaluation_path: Path,
    checkpoint_path: Path,
    max_tokens: int | None,
) -> None:
    context_size = int(config["context_size"])
    token_count = len(evaluation_tokens)
    metadata = {
        "format": "csr",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model_id"],
        "sae_checkpoint": str(checkpoint_path),
        "dataset_id": config["dataset_id"],
        "dataset_config": config["dataset_config"],
        "evaluation_split": "validation",
        "evaluation_path": str(evaluation_path),
        "available_tokens": int(config["validation_tokens"]),
        "processed_tokens": token_count,
        "max_tokens": max_tokens,
        "context_size": context_size,
        "context_count": math.ceil(token_count / context_size),
        "layer_index": int(config["layer_index"]),
        "layer_path": config["layer_path"],
        "residual_location": config["residual_location"],
        "d_model": int(config["d_model"]),
        "d_sae": int(config["d_sae"]),
        "k": int(config["k"]),
        "activation_scale": float(config["activation_scale"]),
        "firing_threshold": FIRING_THRESHOLD,
        "activation_type": "topk_post_relu",
        "device": str(device),
        "model_dtype": config["model_dtype"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    feature_temporary = temporary.with_suffix(temporary.suffix + ".feature_ids")
    values_temporary = temporary.with_suffix(temporary.suffix + ".values")
    pointer_dtype = (
        np.uint32
        if token_count * int(config["k"]) <= np.iinfo(np.uint32).max
        else np.uint64
    )
    row_ptr = np.empty(token_count + 1, dtype=pointer_dtype)
    row_ptr[0] = 0
    context_ptr = np.append(
        np.arange(0, token_count, context_size, dtype=pointer_dtype),
        np.array([token_count], dtype=pointer_dtype),
    )
    progress = tqdm(total=token_count, unit="tok", desc="Analyze", dynamic_ncols=True)
    try:
        processed_tokens = 0
        active_count = 0
        with feature_temporary.open("wb") as feature_output, values_temporary.open(
            "wb"
        ) as values_output:
            batches = iter_context_batches(
                evaluation_tokens,
                context_size,
                model_batch_size,
                tokenizer.pad_token_id,
                shuffle=False,
                seed=0,
            )
            for input_ids, attention_mask, _context_ids in batches:
                device_input_ids = input_ids.to(device, non_blocking=True)
                device_attention_mask = attention_mask.to(device, non_blocking=True)
                residuals = capture(
                    model, device_input_ids, device_attention_mask
                )[device_attention_mask.bool()]
                indices, values = encode_residuals(
                    sae,
                    residuals,
                    float(config["activation_scale"]),
                    int(config["sae_batch_size"]),
                )
                counts, feature_ids, active_values = csr_activations(indices, values)
                batch_tokens = len(counts)
                cumulative = np.cumsum(counts, dtype=pointer_dtype)
                row_ptr[
                    processed_tokens + 1 : processed_tokens + batch_tokens + 1
                ] = active_count + cumulative
                feature_ids.tofile(feature_output)
                active_values.tofile(values_output)
                processed_tokens += batch_tokens
                active_count += len(feature_ids)
                progress.update(batch_tokens)

        if processed_tokens != token_count:
            raise RuntimeError(
                f"processed {processed_tokens:,} tokens; expected {token_count:,}"
            )

        feature_ids = (
            np.memmap(
                feature_temporary,
                mode="r",
                dtype=np.uint32,
                shape=(active_count,),
            )
            if active_count
            else np.empty(0, dtype=np.uint32)
        )
        active_values = (
            np.memmap(
                values_temporary,
                mode="r",
                dtype=np.float32,
                shape=(active_count,),
            )
            if active_count
            else np.empty(0, dtype=np.float32)
        )
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                metadata=json.dumps(metadata, separators=(",", ":")),
                token_ids=np.asarray(evaluation_tokens, dtype=np.uint32),
                context_ptr=context_ptr,
                row_ptr=row_ptr,
                feature_ids=feature_ids,
                values=active_values,
            )
        os.replace(temporary, output_path)
    finally:
        progress.close()
        temporary.unlink(missing_ok=True)
        feature_temporary.unlink(missing_ok=True)
        values_temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.sae_dir)
    cache = evaluation_cache(config, args.cache_dir)
    if cache is None:
        print("No local evaluation set is available; exiting.")
        return

    evaluation_path, available_tokens = cache
    token_count = (
        available_tokens
        if args.max_tokens is None
        else min(args.max_tokens, available_tokens)
    )
    evaluation_tokens = np.memmap(
        evaluation_path, mode="r", dtype=np.uint32, shape=(available_tokens,)
    )[:token_count]

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        dtype=getattr(torch, config["model_dtype"]),
        attn_implementation="eager",
    ).to(device)
    model.eval().requires_grad_(False)
    layer_path, layers = find_transformer_layers(model)
    if layer_path != config["layer_path"]:
        raise ValueError(
            f"configured layer path {config['layer_path']!r} does not match "
            f"model layer path {layer_path!r}"
        )
    layer_index = int(config["layer_index"])
    if not 0 <= layer_index < len(layers):
        raise ValueError(
            f"configured activation layer {layer_index} is not present in the model"
        )

    sae = load_sae(args.sae_dir, config, device)
    output_path = args.output or args.sae_dir / "analysis.npz"
    model_batch_size = args.model_batch_size or int(config["model_batch_size"])
    print(
        f"Device: {device} | Evaluation: {token_count:,} tokens | "
        f"Layer: {layer_index} | Output: {output_path}"
    )
    with ResidualStreamCapture(layers[layer_index]) as capture:
        write_analysis(
            output_path,
            tokenizer,
            model,
            capture,
            sae,
            evaluation_tokens,
            config,
            device,
            model_batch_size,
            evaluation_path,
            args.sae_dir / "sae_final.pt",
            args.max_tokens,
        )


if __name__ == "__main__":
    main()

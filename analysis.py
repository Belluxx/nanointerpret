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
        help="Output JSONL path. Default: <sae-dir>/analysis.jsonl.",
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


def context_activations(indices: Tensor, values: Tensor) -> list[list[int | float]]:
    result: list[list[int | float]] = []
    for token_position, (token_indices, token_values) in enumerate(
        zip(indices, values, strict=True)
    ):
        firing = token_values > FIRING_THRESHOLD
        pairs = zip(token_indices[firing].tolist(), token_values[firing].tolist())
        result.extend(
            [token_position, int(feature_id), float(value)]
            for feature_id, value in sorted(pairs)
        )
    return result


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
        "type": "metadata",
        "schema_version": 1,
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
        "activation_columns": [
            "token_position",
            "feature_id",
            "raw_activation",
        ],
        "device": str(device),
        "model_dtype": config["model_dtype"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    progress = tqdm(total=token_count, unit="tok", desc="Analyze", dynamic_ncols=True)
    try:
        with temporary.open("w") as output:
            output.write(json.dumps(metadata, separators=(",", ":")) + "\n")
            batches = iter_context_batches(
                evaluation_tokens,
                context_size,
                model_batch_size,
                tokenizer.pad_token_id,
                shuffle=False,
                seed=0,
            )
            for input_ids, attention_mask, context_ids in batches:
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

                lengths = attention_mask.sum(dim=1).tolist()
                residual_offset = 0
                for row, context_id in enumerate(context_ids):
                    length = lengths[row]
                    token_ids = input_ids[row, :length].tolist()
                    record = {
                        "type": "context",
                        "context_id": int(context_id),
                        "text": tokenizer.decode(
                            token_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        "token_ids": token_ids,
                        "activations": context_activations(
                            indices[residual_offset : residual_offset + length],
                            values[residual_offset : residual_offset + length],
                        ),
                    }
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    progress.update(length)
                    residual_offset += length
        os.replace(temporary, output_path)
    finally:
        progress.close()


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
    output_path = args.output or args.sae_dir / "analysis.jsonl"
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

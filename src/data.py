from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm


RESIDUAL_DTYPE = np.float32


@dataclass(frozen=True)
class TokenCacheSpec:
    cache_dir: Path
    model_id: str
    dataset_id: str
    dataset_config: str
    train_tokens: int
    validation_tokens: int


@dataclass(frozen=True)
class ResidualCacheSpec:
    cache_dir: Path
    model_id: str
    dataset_id: str
    dataset_config: str
    train_tokens: int
    validation_tokens: int
    context_size: int
    activation_layer: int | None
    model_dtype: str


def residual_cache_paths(spec: ResidualCacheSpec) -> tuple[Path, Path, Path]:
    safe_model = spec.model_id.replace("/", "--")
    safe_dataset = spec.dataset_id.replace("/", "--")
    safe_config = spec.dataset_config.replace("/", "--")
    layer = "middle" if spec.activation_layer is None else str(spec.activation_layer)
    stem = (
        f"{safe_model}_{safe_dataset}_{safe_config}_"
        f"{spec.train_tokens}_{spec.validation_tokens}_ctx{spec.context_size}_"
        f"layer{layer}_{spec.model_dtype}"
    )
    dtype_name = np.dtype(RESIDUAL_DTYPE).name
    return (
        spec.cache_dir / f"{stem}_train.{dtype_name}",
        spec.cache_dir / f"{stem}_validation.{dtype_name}",
        spec.cache_dir / f"{stem}_metadata.json",
    )


def load_residual_cache_metadata(spec: ResidualCacheSpec) -> dict | None:
    train_path, validation_path, metadata_path = residual_cache_paths(spec)
    if not (train_path.exists() and validation_path.exists() and metadata_path.exists()):
        return None

    metadata = json.loads(metadata_path.read_text())
    expected = asdict(spec)
    expected.pop("cache_dir")
    if not all(metadata.get(key) == value for key, value in expected.items()):
        return None

    d_model = metadata.get("d_model")
    if not isinstance(d_model, int) or d_model <= 0:
        return None
    item_size = np.dtype(RESIDUAL_DTYPE).itemsize
    if (
        train_path.stat().st_size != spec.train_tokens * d_model * item_size
        or validation_path.stat().st_size
        != spec.validation_tokens * d_model * item_size
    ):
        return None
    return metadata


def token_cache_is_valid(
    train_path: Path,
    validation_path: Path,
    metadata_path: Path,
    spec: TokenCacheSpec,
) -> bool:
    if not (train_path.exists() and validation_path.exists() and metadata_path.exists()):
        return False
    metadata = json.loads(metadata_path.read_text())

    expected = asdict(spec)
    expected.pop("cache_dir")
    item_size = np.dtype(np.uint32).itemsize
    return (
        all(metadata.get(key) == value for key, value in expected.items())
        and train_path.stat().st_size == spec.train_tokens * item_size
        and validation_path.stat().st_size == spec.validation_tokens * item_size
    )


def build_token_cache(tokenizer, spec: TokenCacheSpec) -> tuple[Path, Path]:
    safe_model = spec.model_id.replace("/", "--")
    stem = (
        f"{safe_model}_{spec.dataset_config}_{spec.train_tokens}_"
        f"{spec.validation_tokens}"
    )
    train_path = spec.cache_dir / f"{stem}_train.uint32"
    validation_path = spec.cache_dir / f"{stem}_validation.uint32"
    metadata_path = spec.cache_dir / f"{stem}_metadata.json"
    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    if token_cache_is_valid(train_path, validation_path, metadata_path, spec):
        print(f"using token cache at {spec.cache_dir}")
        return train_path, validation_path

    from datasets import load_dataset

    train_tmp = train_path.with_suffix(train_path.suffix + ".tmp")
    validation_tmp = validation_path.with_suffix(validation_path.suffix + ".tmp")
    target_total = spec.train_tokens + spec.validation_tokens
    written = 0
    dataset = load_dataset(
        spec.dataset_id,
        name=spec.dataset_config,
        split="train",
        streaming=True,
    )
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    with train_tmp.open("wb") as train_file, validation_tmp.open("wb") as validation_file:
        progress = tqdm(total=target_total, unit="tok", desc="Token cache")
        text_batch: list[str] = []

        def write_documents(texts: list[str]) -> None:
            nonlocal written
            encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
            for document in encoded:
                if written >= target_total:
                    break
                ids = ([] if bos is None else [bos]) + document + ([] if eos is None else [eos])
                array = np.asarray(ids, dtype=np.uint32)
                take = min(len(array), target_total - written)

                split_at = max(0, min(take, spec.train_tokens - written))
                if split_at:
                    array[:split_at].tofile(train_file)
                if split_at < take:
                    array[split_at:take].tofile(validation_file)
                written += take
                progress.update(take)

        for row in dataset:
            text = row.get("text")
            if text:
                text_batch.append(text)
            if len(text_batch) >= 32:
                write_documents(text_batch)
                text_batch.clear()
            if written >= target_total:
                break
        if text_batch and written < target_total:
            write_documents(text_batch)
        progress.close()

    if written != target_total:
        raise RuntimeError(f"dataset ended after {written:,} tokens; expected {target_total:,}")

    os.replace(train_tmp, train_path)
    os.replace(validation_tmp, validation_path)
    metadata = {
        "model_id": spec.model_id,
        "dataset_id": spec.dataset_id,
        "dataset_config": spec.dataset_config,
        "train_tokens": spec.train_tokens,
        "validation_tokens": spec.validation_tokens,
        "bos_token_id": bos,
        "eos_token_id": eos,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return train_path, validation_path


def iter_context_batches(
    tokens: np.memmap,
    context_size: int,
    batch_size: int,
    pad_token_id: int,
    shuffle: bool,
    seed: int,
    skip_contexts: int = 0,
) -> Iterator[tuple[Tensor, Tensor, np.ndarray]]:
    context_count = math.ceil(len(tokens) / context_size)
    order = np.arange(context_count)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    order = order[skip_contexts:]

    for offset in range(0, len(order), batch_size):
        context_ids = order[offset : offset + batch_size]
        input_ids = np.full((len(context_ids), context_size), pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(context_ids), context_size), dtype=np.int64)
        for row, context_id in enumerate(context_ids):
            start = int(context_id) * context_size
            end = min(start + context_size, len(tokens))
            length = end - start
            input_ids[row, :length] = tokens[start:end]
            attention_mask[row, :length] = 1
        yield torch.from_numpy(input_ids), torch.from_numpy(attention_mask), context_ids


def iter_residual_batches(
    residuals: np.memmap,
    batch_size: int,
    shuffle: bool,
    seed: int,
    skip_batches: int = 0,
) -> Iterator[Tensor]:
    batch_count = math.ceil(len(residuals) / batch_size)
    order = np.arange(batch_count)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)

    for batch_id in order[skip_batches:]:
        start = int(batch_id) * batch_size
        batch = np.asarray(residuals[start : start + batch_size]).copy()
        yield torch.from_numpy(batch)

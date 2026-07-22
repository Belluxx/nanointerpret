from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm


def token_cache_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    safe_model = args.model_id.replace("/", "--")
    stem = f"{safe_model}_{args.dataset_config}_{args.train_tokens}_{args.validation_tokens}"
    return (
        args.cache_dir / f"{stem}_train.uint32",
        args.cache_dir / f"{stem}_validation.uint32",
        args.cache_dir / f"{stem}_metadata.json",
    )


def cache_is_valid(
    train_path: Path,
    validation_path: Path,
    metadata_path: Path,
    args: argparse.Namespace,
) -> bool:
    if not (train_path.exists() and validation_path.exists() and metadata_path.exists()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    expected = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "train_tokens": args.train_tokens,
        "validation_tokens": args.validation_tokens,
    }
    item_size = np.dtype(np.uint32).itemsize
    return (
        all(metadata.get(key) == value for key, value in expected.items())
        and train_path.stat().st_size == args.train_tokens * item_size
        and validation_path.stat().st_size == args.validation_tokens * item_size
    )


def build_token_cache(tokenizer, args: argparse.Namespace) -> tuple[Path, Path]:
    train_path, validation_path, metadata_path = token_cache_paths(args)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_is_valid(train_path, validation_path, metadata_path, args):
        print(f"using token cache at {args.cache_dir}")
        return train_path, validation_path

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Run: pip install -r requirements.txt"
        ) from exc

    train_tmp = train_path.with_suffix(train_path.suffix + ".tmp")
    validation_tmp = validation_path.with_suffix(validation_path.suffix + ".tmp")
    target_total = args.train_tokens + args.validation_tokens
    written = 0
    print(f"streaming {target_total:,} tokens from {args.dataset_id}/{args.dataset_config}")
    dataset = load_dataset(
        args.dataset_id,
        name=args.dataset_config,
        split="train",
        streaming=True,
    )
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    with train_tmp.open("wb") as train_file, validation_tmp.open("wb") as validation_file:
        progress = tqdm(total=target_total, unit="tok", desc="token cache")
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
                if take == 0:
                    continue

                split_at = max(0, min(take, args.train_tokens - written))
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
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "train_tokens": args.train_tokens,
        "validation_tokens": args.validation_tokens,
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

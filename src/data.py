from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

RESIDUAL_CACHE_FORMATS = ("fp16", "int8")
RESIDUAL_FP16_SCALE = 1 / 256
RESIDUAL_INT8_GROUP_SIZE = 128
ACTIVATION_VALUE_DTYPE = np.float16
TRANSPOSE_TOKENS = 1_000_000
TOKEN_CACHE_BATCH_CHARS = 32 << 20
INSUFFICIENT_TITLE = "Insufficient activation data"
UNCLEAR_TITLE = "No coherent interpretation"
FEATURE_CATEGORIES = ("token-specific", "lexical", "semantic")
INTERPRETATIONS_FILENAME = "feature_interpretations.jsonl"


@dataclass(frozen=True)
class TokenCacheSpec:
    cache_dir: Path
    model_id: str
    dataset_id: str
    dataset_config: str
    train_tokens: int
    validation_tokens: int
    recording_tokens: int


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
    cache_format: str


@dataclass(frozen=True)
class FeatureActivations:
    metadata: dict
    token_ids: np.ndarray
    feature_ptr: np.ndarray
    token_positions: np.ndarray
    values: np.ndarray
    feature_max: np.ndarray


def load_activations(path: Path) -> FeatureActivations:
    metadata = json.loads((path / "metadata.json").read_text())

    def load(name: str) -> np.ndarray:
        return np.load(path / f"{name}.npy", mmap_mode="r")

    return FeatureActivations(
        metadata=metadata,
        token_ids=load("token_ids"),
        feature_ptr=load("feature_ptr"),
        token_positions=load("token_positions"),
        values=load("values"),
        feature_max=load("feature_max"),
    )


def validate_interpretation(
    title: object,
    category: object,
) -> tuple[str, str | None]:
    if not isinstance(title, str) or not title.strip():
        raise TypeError("feature title must be a non-empty string")

    title = title.strip()
    if title in (INSUFFICIENT_TITLE, UNCLEAR_TITLE):
        if category is not None:
            raise ValueError("an uninterpretable feature must have a null category")
    elif not isinstance(category, str) or category not in FEATURE_CATEGORIES:
        raise ValueError(
            f"feature category must be one of {', '.join(FEATURE_CATEGORIES)}"
        )
    return title, category


def load_interpretations(
    path: Path | None,
) -> dict[int, dict[str, str | None]]:
    if path is None:
        return {}

    interpretations = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                feature_id = int(record["feature_id"])
                title, category = validate_interpretation(
                    record["title"], record["category"]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid feature interpretation on line {line_number} of {path}"
                ) from error
            interpretations[feature_id] = {
                "title": title,
                "category": category,
            }
    return interpretations


def save_activations(
    output_path: Path,
    metadata: dict,
    token_ids: np.ndarray,
    row_ptr: np.ndarray,
    feature_ids: np.ndarray,
    values: np.ndarray,
    feature_counts: np.ndarray,
    feature_max: np.ndarray,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"activation output already exists: {output_path}")

    token_count = len(token_ids)
    activation_count = len(feature_ids)
    d_sae = len(feature_counts)
    pointer_dtype = (
        np.uint32 if activation_count <= np.iinfo(np.uint32).max else np.uint64
    )
    position_dtype = (
        np.uint32 if token_count <= np.iinfo(np.uint32).max else np.uint64
    )
    feature_ptr = np.empty(d_sae + 1, dtype=pointer_dtype)
    feature_ptr[0] = 0
    np.cumsum(feature_counts, dtype=pointer_dtype, out=feature_ptr[1:])
    if int(feature_ptr[-1]) != activation_count:
        raise ValueError("feature counts do not match the activation count")

    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.mkdir(parents=True)
    try:
        np.save(temporary / "token_ids.npy", token_ids)
        np.save(temporary / "feature_ptr.npy", feature_ptr)
        position_output = np.lib.format.open_memmap(
            temporary / "token_positions.npy",
            mode="w+",
            dtype=position_dtype,
            shape=(activation_count,),
        )
        value_output = np.lib.format.open_memmap(
            temporary / "values.npy",
            mode="w+",
            dtype=ACTIVATION_VALUE_DTYPE,
            shape=(activation_count,),
        )
        cursors = feature_ptr[:-1].copy()

        with tqdm(
            total=token_count,
            unit="tok",
            desc="Index features",
            dynamic_ncols=True,
        ) as progress:
            for token_start in range(0, token_count, TRANSPOSE_TOKENS):
                token_stop = min(token_start + TRANSPOSE_TOKENS, token_count)
                activation_start = int(row_ptr[token_start])
                activation_stop = int(row_ptr[token_stop])
                if activation_start == activation_stop:
                    progress.update(token_stop - token_start)
                    continue
                chunk_ids = feature_ids[activation_start:activation_stop]
                order = np.argsort(chunk_ids, kind="stable")
                sorted_ids = chunk_ids[order]
                counts = np.diff(row_ptr[token_start : token_stop + 1])
                positions = np.repeat(
                    np.arange(token_start, token_stop, dtype=position_dtype),
                    counts,
                )[order]
                chunk_values = values[activation_start:activation_stop][order]
                group_starts = np.concatenate(
                    ([0], np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1)
                )
                group_stops = np.append(group_starts[1:], len(sorted_ids))

                for group_start, group_stop in zip(group_starts, group_stops):
                    feature_id = int(sorted_ids[group_start])
                    output_start = int(cursors[feature_id])
                    output_stop = output_start + int(group_stop - group_start)
                    position_output[output_start:output_stop] = positions[
                        group_start:group_stop
                    ]
                    value_output[output_start:output_stop] = chunk_values[
                        group_start:group_stop
                    ]
                    cursors[feature_id] = output_stop
                progress.update(token_stop - token_start)

        if not np.array_equal(cursors, feature_ptr[1:]):
            raise RuntimeError(
                "feature index construction did not write every activation"
            )
        position_output.flush()
        value_output.flush()
        del position_output, value_output
        np.save(temporary / "feature_max.npy", feature_max)
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        os.replace(temporary, output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def token_cache_paths(spec: TokenCacheSpec) -> tuple[Path, Path, Path, Path]:
    safe_model = spec.model_id.replace("/", "--")
    safe_dataset = spec.dataset_id.replace("/", "--")
    safe_config = spec.dataset_config.replace("/", "--")
    stem = (
        f"{safe_model}_{safe_dataset}_{safe_config}_{spec.train_tokens}_"
        f"{spec.validation_tokens}_{spec.recording_tokens}"
    )
    return (
        spec.cache_dir / f"{stem}_train.uint32",
        spec.cache_dir / f"{stem}_validation.uint32",
        spec.cache_dir / f"{stem}_recording.uint32",
        spec.cache_dir / f"{stem}_metadata.json",
    )


def residual_cache_paths(spec: ResidualCacheSpec) -> tuple[Path, Path, Path]:
    safe_model = spec.model_id.replace("/", "--")
    safe_dataset = spec.dataset_id.replace("/", "--")
    safe_config = spec.dataset_config.replace("/", "--")
    layer = "middle" if spec.activation_layer is None else str(spec.activation_layer)
    stem = (
        f"{safe_model}_{safe_dataset}_{safe_config}_"
        f"{spec.train_tokens}_{spec.validation_tokens}_ctx{spec.context_size}_"
        f"layer{layer}_{spec.model_dtype}_{spec.cache_format}"
    )
    return (
        spec.cache_dir / f"{stem}_train.npy",
        spec.cache_dir / f"{stem}_validation.npy",
        spec.cache_dir / f"{stem}_metadata.json",
    )


def residual_cache_layout(
    cache_format: str, token_count: int, d_model: int
) -> tuple[np.dtype, tuple[int, ...]]:
    if cache_format == "fp16":
        return np.dtype(np.float16), (token_count, d_model)
    if cache_format == "int8":
        if d_model % RESIDUAL_INT8_GROUP_SIZE:
            raise ValueError(
                f"INT8 residual caching requires d_model to be divisible by "
                f"{RESIDUAL_INT8_GROUP_SIZE}, got {d_model}"
            )
        dtype = np.dtype(
            [
                ("codes", np.int8, (d_model,)),
                ("scales", np.float16, (d_model // RESIDUAL_INT8_GROUP_SIZE,)),
            ]
        )
        return dtype, (token_count,)
    raise ValueError(f"unknown residual cache format: {cache_format}")


def create_residual_cache(
    path: Path, token_count: int, d_model: int, cache_format: str
) -> np.memmap:
    dtype, shape = residual_cache_layout(cache_format, token_count, d_model)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def encode_int8_residuals(residuals: Tensor) -> tuple[Tensor, Tensor]:
    groups = residuals.reshape(len(residuals), -1, RESIDUAL_INT8_GROUP_SIZE)
    scales = (groups.abs().amax(dim=2) / 127).clamp_max(
        torch.finfo(torch.float16).max
    )
    scales = scales.to(torch.float16)
    divisors = torch.where(scales == 0, 1.0, scales.float())
    codes = torch.round(groups / divisors.unsqueeze(2)).clamp_(-127, 127)
    return codes.reshape_as(residuals).to(torch.int8), scales


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
    for path, token_count in (
        (train_path, spec.train_tokens),
        (validation_path, spec.validation_tokens),
    ):
        dtype, shape = residual_cache_layout(spec.cache_format, token_count, d_model)
        cache = np.load(path, mmap_mode="r")
        if cache.dtype != dtype or cache.shape != shape:
            return None
    return metadata


def token_cache_is_valid(
    spec: TokenCacheSpec, *, recording_only: bool = False
) -> bool:
    train_path, validation_path, recording_path, metadata_path = token_cache_paths(spec)
    required_paths = (recording_path, metadata_path)
    if not recording_only:
        required_paths += (train_path, validation_path)
    if not all(path.exists() for path in required_paths):
        return False
    metadata = json.loads(metadata_path.read_text())

    expected = asdict(spec)
    expected.pop("cache_dir")
    item_size = np.dtype(np.uint32).itemsize
    return (
        all(metadata.get(key) == value for key, value in expected.items())
        and recording_path.stat().st_size == spec.recording_tokens * item_size
        and (
            recording_only
            or (
                train_path.stat().st_size == spec.train_tokens * item_size
                and validation_path.stat().st_size
                == spec.validation_tokens * item_size
            )
        )
    )


def build_token_cache(tokenizer, spec: TokenCacheSpec) -> tuple[Path, Path]:
    train_path, validation_path, recording_path, metadata_path = token_cache_paths(spec)
    spec.cache_dir.mkdir(parents=True, exist_ok=True)
    if token_cache_is_valid(spec):
        print(f"using token cache at {spec.cache_dir}")
        return train_path, validation_path

    import awkward as ak
    from datasets import load_dataset
    from gigatoken import Tokenizer as GigaTokenizer

    train_tmp = train_path.with_suffix(train_path.suffix + ".tmp")
    validation_tmp = validation_path.with_suffix(validation_path.suffix + ".tmp")
    recording_tmp = recording_path.with_suffix(recording_path.suffix + ".tmp")
    target_total = spec.train_tokens + spec.validation_tokens + spec.recording_tokens
    written = 0
    dataset = load_dataset(
        spec.dataset_id,
        name=spec.dataset_config,
        split="train",
        streaming=True,
    )
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    giga_tokenizer = GigaTokenizer(tokenizer)

    with (
        train_tmp.open("wb") as train_file,
        validation_tmp.open("wb") as validation_file,
        recording_tmp.open("wb") as recording_file,
    ):
        progress = tqdm(total=target_total, unit="tok", desc="Token cache")
        text_batch: list[str] = []
        batch_chars = 0

        def write_documents(texts: list[str]) -> None:
            nonlocal written
            rows = giga_tokenizer.encode_batch(texts)
            parts = [rows]
            if bos is not None:
                parts.insert(
                    0,
                    np.full((len(texts), 1), bos, dtype=np.uint32),
                )
            if eos is not None:
                parts.append(np.full((len(texts), 1), eos, dtype=np.uint32))
            if len(parts) > 1:
                rows = ak.concatenate(parts, axis=1)
            array = ak.to_numpy(ak.flatten(rows))
            take = min(len(array), target_total - written)

            chunk_start = written
            chunk_stop = written + take
            split_start = 0
            for split_file, split_size in (
                (train_file, spec.train_tokens),
                (validation_file, spec.validation_tokens),
                (recording_file, spec.recording_tokens),
            ):
                split_stop = split_start + split_size
                start = max(chunk_start, split_start)
                stop = min(chunk_stop, split_stop)
                if start < stop:
                    array[start - chunk_start : stop - chunk_start].tofile(split_file)
                split_start = split_stop
            written += take
            progress.update(take)

        for text in dataset["text"]:
            if text:
                text_batch.append(text)
                batch_chars += len(text)
            if batch_chars >= TOKEN_CACHE_BATCH_CHARS:
                write_documents(text_batch)
                text_batch.clear()
                batch_chars = 0
            if written >= target_total:
                break
        if text_batch and written < target_total:
            write_documents(text_batch)
        progress.close()

    if written != target_total:
        raise RuntimeError(f"dataset ended after {written:,} tokens; expected {target_total:,}")

    os.replace(train_tmp, train_path)
    os.replace(validation_tmp, validation_path)
    os.replace(recording_tmp, recording_path)
    metadata = {
        "model_id": spec.model_id,
        "dataset_id": spec.dataset_id,
        "dataset_config": spec.dataset_config,
        "train_tokens": spec.train_tokens,
        "validation_tokens": spec.validation_tokens,
        "recording_tokens": spec.recording_tokens,
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
) -> Iterator[tuple[Tensor, Tensor]]:
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
        yield torch.from_numpy(input_ids), torch.from_numpy(attention_mask)


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
        rows = residuals[start : start + batch_size]
        if rows.dtype.fields is None:
            batch = np.asarray(rows, dtype=np.float32)
            batch /= RESIDUAL_FP16_SCALE
        else:
            codes = np.asarray(rows["codes"], dtype=np.float32)
            scales = np.asarray(rows["scales"], dtype=np.float32)
            groups = codes.reshape(len(rows), -1, RESIDUAL_INT8_GROUP_SIZE)
            groups *= scales[:, :, None]
            batch = groups.reshape(len(rows), -1)
        yield torch.from_numpy(batch)

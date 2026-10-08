import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

# Residuals are cached as fp16, scaled down so outlier activations fit.
RESIDUAL_FP16_SCALE = 1 / 256
TOKENIZE_BATCH_CHARS = 32 << 20
# Residual-cache batches read in the background ahead of the one in use.
READ_AHEAD = 4


def cache_name(*parts: object) -> str:
    return "_".join(str(part).replace("/", "--") for part in parts)


def token_cache(
    cache_dir: Path, model_id: str, dataset_id: str, dataset_config: str, token_count: int
) -> np.ndarray:
    # The first `token_count` tokens of the dataset; a longer existing cache is reused.
    path = cache_dir / f"{cache_name(model_id, dataset_id, dataset_config)}.uint32"
    if not path.exists() or path.stat().st_size < token_count * 4:
        tokenize_dataset(path, model_id, dataset_id, dataset_config, token_count)
    else:
        print(f"Using token cache {path}")
    return np.memmap(path, dtype=np.uint32, mode="r")[:token_count]


def tokenize_dataset(path: Path, model_id: str, dataset_id: str, dataset_config: str, token_count: int) -> None:
    import awkward as ak
    from datasets import load_dataset
    from gigatoken import Tokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = Tokenizer(hf_tokenizer)
    bos, eos = hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id
    dataset = load_dataset(dataset_id, name=dataset_config, split="train", streaming=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    written = 0
    with temporary.open("wb") as file, tqdm(total=token_count, unit="tok", desc="Token cache") as progress:
        for texts in text_batches(filter(None, dataset["text"])):
            rows = tokenizer.encode_batch(texts)
            if bos is not None:
                rows = ak.concatenate([np.full((len(texts), 1), bos, dtype=np.uint32), rows], axis=1)
            if eos is not None:
                rows = ak.concatenate([rows, np.full((len(texts), 1), eos, dtype=np.uint32)], axis=1)
            tokens = ak.to_numpy(ak.flatten(rows))[: token_count - written]
            tokens.tofile(file)
            written += len(tokens)
            progress.update(len(tokens))
            if written == token_count:
                break
    if written != token_count:
        raise RuntimeError(f"dataset ended after {written:,} tokens; expected {token_count:,}")
    os.replace(temporary, path)


def text_batches(texts: Iterable[str]) -> Iterator[list[str]]:
    batch, chars = [], 0
    for text in texts:
        batch.append(text)
        chars += len(text)
        if chars >= TOKENIZE_BATCH_CHARS:
            yield batch
            batch, chars = [], 0
    if batch:
        yield batch


def as_contexts(tokens: np.ndarray, context_size: int) -> np.ndarray:
    # A trailing partial context is dropped.
    return tokens[: len(tokens) // context_size * context_size].reshape(-1, context_size)


def context_batches(
    context_count: int, batch_size: int, *, shuffle: bool = False, seed: int = 0, skip: int = 0
) -> Iterator[np.ndarray]:
    order = np.random.default_rng(seed).permutation(context_count) if shuffle else np.arange(context_count)
    for start in range(skip, context_count, batch_size):
        yield order[start : start + batch_size]


def residual_cache(
    path: Path, contexts: np.ndarray, capture: Callable[[np.ndarray], Tensor], batch_size: int, d_model: int
) -> np.ndarray:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(*contexts.shape, d_model))
        float16_max = torch.finfo(torch.float16).max
        with tqdm(total=contexts.size, unit="tok", desc="Residual cache",dynamic_ncols=True) as progress:
            for ids in context_batches(len(contexts), batch_size):
                residual = capture(ids).float().mul_(RESIDUAL_FP16_SCALE).clamp_(-float16_max, float16_max)
                output[ids] = residual.cpu().numpy()
                progress.update(residual.shape[:2].numel())
        output.flush()
        del output
        os.replace(temporary, path)
    return np.load(path, mmap_mode="r")


def read_residual_cache(cache: np.ndarray, batches: Iterable[np.ndarray], device: torch.device) -> Iterator[Tensor]:
    # Upcoming batches are read in background threads, so disk reads overlap GPU work.
    def to_device(residual: np.ndarray) -> Tensor:
        return torch.from_numpy(residual).to(device).flatten(0, 1).float() / RESIDUAL_FP16_SCALE

    with ThreadPoolExecutor(READ_AHEAD) as pool:
        pending = deque()
        for ids in batches:
            pending.append(pool.submit(cache.__getitem__, ids))
            if len(pending) > READ_AHEAD:
                yield to_device(pending.popleft().result())
        while pending:
            yield to_device(pending.popleft().result())

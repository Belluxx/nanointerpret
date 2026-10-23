import os
import random
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

# Residuals are cached as fp16, scaled down so outlier activations fit.
RESIDUAL_FP16_SCALE = 1 / 256
TOKENIZE_BATCH_CHARS = 8 << 20
# Residual-cache batches read in the background ahead of the one in use.
READ_AHEAD = 4
# Longer documents are cut to a random window, so a few long ones can't fill a source.
MAX_DOCUMENT_CHARS = 8192
# Parquet rows decoded at once; Stack v3 rows are whole repositories.
READ_BATCH_ROWS = 64
# Stack v3 rows are whole repositories: a few of their source files in these languages are kept.
CODE_LANGUAGES = frozenset({
    "C", "C#", "C++", "Dart", "Go", "Haskell", "Java", "JavaScript", "Julia", "Kotlin", "Lua", "MATLAB",
    "Objective-C", "PHP", "Perl", "Python", "R", "Ruby", "Rust", "Scala", "Shell", "SQL", "Swift", "TSX", "TypeScript",
})
MIN_CODE_BYTES = 256
MAX_FILES_PER_REPO = 8


@dataclass(frozen=True)
class Source:
    name: str
    # Glob of the source's parquet files, as "<dataset repo>/<path>".
    files: str
    share: float
    column: str = "text"


MIXES = {
    "fineweb": (Source("fineweb", "HuggingFaceFW/fineweb/sample/10BT/*.parquet", 1.0),),
    "broad": (
        Source("fineweb", "HuggingFaceFW/fineweb/sample/10BT/*.parquet", 0.475),
        Source("finepdfs", "HuggingFaceFW/finepdfs/data/eng_Latn/train/*.parquet", 0.1),
        Source("stack-v3", "HuggingFaceCode/stack-v3-train/data/*.parquet", 0.15, column="files"),
        Source("open-web-math", "open-web-math/open-web-math/data/*.parquet", 0.125),
        # FineWeb-2's Chinese is mostly SEO spam; the quality-filtered FineWeb2-HQ much less so.
        Source("chinese", "epfml/FineWeb2-HQ/cmn_Hani/*.parquet", 0.05),
        Source("german", "HuggingFaceFW/fineweb-2/data/deu_Latn/train/*.parquet", 0.025),
        Source("french", "HuggingFaceFW/fineweb-2/data/fra_Latn/train/*.parquet", 0.025),
        Source("spanish", "HuggingFaceFW/fineweb-2/data/spa_Latn/train/*.parquet", 0.025),
        Source("italian", "HuggingFaceFW/fineweb-2/data/ita_Latn/train/*.parquet", 0.025),
    ),
}


def cache_name(*parts: object) -> str:
    return "_".join(str(part).replace("/", "--") for part in parts)


def context_splits(train_tokens: int, validation_tokens: int, recording_tokens: int, context_size: int) -> dict[str, slice]:
    # Consecutive context ranges; each split drops its trailing partial context.
    stops = np.cumsum([count // context_size for count in (train_tokens, validation_tokens, recording_tokens)])
    starts = np.concatenate([[0], stops[:-1]])
    return {name: slice(int(start), int(stop)) for name, start, stop in zip(("train", "validation", "recording"), starts, stops)}


def token_cache(
    cache_dir: Path, model_id: str, mix: str, context_size: int, context_count: int
) -> tuple[np.ndarray, np.ndarray]:
    # The first `context_count` contexts of the mix and each one's source index. Each context comes
    # whole from one source, drawn by share, and the draws don't depend on `context_count`, so a
    # longer existing cache is reused and every split keeps the mix.
    sources = MIXES[mix]
    source_ids = np.random.default_rng(0).choice(len(sources), size=context_count, p=[s.share for s in sources])
    path = cache_dir / f"{cache_name(model_id, mix, f'ctx{context_size}')}.uint32"
    if not path.exists() or path.stat().st_size < context_count * context_size * 4:
        tokenize_mix(path, model_id, sources, source_ids, context_size)
    else:
        print(f"Using token cache {path}")
    return np.memmap(path, dtype=np.uint32, mode="r").reshape(-1, context_size)[:context_count], source_ids


def tokenize_mix(path: Path, model_id: str, sources: tuple[Source, ...], source_ids: np.ndarray, context_size: int) -> None:
    from gigatoken import Tokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = Tokenizer(hf_tokenizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    mixed = np.memmap(temporary, dtype=np.uint32, mode="w+", shape=(len(source_ids), context_size))
    # One source at a time: streams hold large buffers, so only one is open.
    for index, source in enumerate(sources):
        rows = np.flatnonzero(source_ids == index)
        streamed = source_contexts(source, tokenizer, hf_tokenizer.bos_token_id, hf_tokenizer.eos_token_id, context_size)
        with closing(streamed) as contexts:
            written = 0
            for row, context in zip(tqdm(rows, unit="ctx", desc=f"Tokenize {source.name}", dynamic_ncols=True), contexts):
                mixed[row] = context
                written += 1
        if written < len(rows):
            raise RuntimeError(f"{source.name} ended after {written:,} contexts; expected {len(rows):,}")
    mixed.flush()
    del mixed
    os.replace(temporary, path)


def source_contexts(source: Source, tokenizer, bos: int | None, eos: int | None, context_size: int) -> Iterator[np.ndarray]:
    # The source's documents, tokenized and packed into contexts.
    import awkward as ak

    tokens = np.empty(0, dtype=np.uint32)
    with closing(source_documents(source)) as documents:
        for texts in text_batches(documents):
            rows = tokenizer.encode_batch(texts)
            if bos is not None:
                rows = ak.concatenate([np.full((len(texts), 1), bos, dtype=np.uint32), rows], axis=1)
            if eos is not None:
                rows = ak.concatenate([rows, np.full((len(texts), 1), eos, dtype=np.uint32)], axis=1)
            tokens = np.concatenate([tokens, ak.to_numpy(ak.flatten(rows))])
            packed = len(tokens) // context_size * context_size
            yield from tokens[:packed].reshape(-1, context_size)
            tokens = tokens[packed:]


def source_documents(source: Source) -> Iterator[str]:
    rng = random.Random(source.name)
    for row in source_rows(source, rng):
        texts = code_files(row["files"], rng) if source.column == "files" else [row["text"]]
        for text in texts:
            if text:
                yield document_window(text, rng)


def source_rows(source: Source, rng: random.Random) -> Iterator[dict]:
    # Files in random order, one random row group of each per pass, so even a small share spans many
    # files. Only one row group is read at a time, which keeps memory low.
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.glob(f"datasets/{source.files}")
    rng.shuffle(files)
    unread_groups = {}
    while files:
        for path in list(files):
            with fs.open(path) as file:
                parquet = pq.ParquetFile(file)
                groups = unread_groups.setdefault(path, rng.sample(range(parquet.num_row_groups), parquet.num_row_groups))
                for batch in parquet.iter_batches(READ_BATCH_ROWS, row_groups=[groups.pop()], columns=[source.column]):
                    yield from batch.to_pylist()
            if not groups:
                files.remove(path)


def code_files(files: list[dict], rng: random.Random) -> list[str]:
    kept = [
        file["content"] for file in files
        if file["language"] in CODE_LANGUAGES and not file["is_vendor"] and file["size_bytes"] >= MIN_CODE_BYTES
    ]
    return rng.sample(kept, min(len(kept), MAX_FILES_PER_REPO))


def document_window(text: str, rng: random.Random) -> str:
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    start = text.find("\n", rng.randrange(len(text) - MAX_DOCUMENT_CHARS)) + 1
    return text[start : start + MAX_DOCUMENT_CHARS]


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

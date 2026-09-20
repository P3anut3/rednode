"""Chinese encoder benchmark utilities: text health, preprocessing, and encoding."""

from __future__ import annotations

import html
import json
import multiprocessing as mp
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq

from experiments.phase_02.experiment_01_zero_shot_dense.recall.dense_encoder import note_files, valid_text


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    model_name: str
    snapshot: str
    dim: int
    pooling: str
    prefix: str
    tokenizer_type: str
    language_support: str
    parameters: int


def _snapshot(model_dir: str) -> str:
    snapshots = sorted((Path.home() / ".cache/huggingface/hub" / model_dir / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshot for {model_dir}")
    return str(snapshots[-1])


def model_specs() -> dict[str, EncoderSpec]:
    return {
        "e5_base_v2": EncoderSpec(
            key="e5_base_v2", model_name="intfloat/e5-base-v2",
            snapshot=_snapshot("models--intfloat--e5-base-v2"), dim=768,
            pooling="masked_mean", prefix="passage: ", tokenizer_type="BERT WordPiece",
            language_support="English", parameters=109_482_240,
        ),
        "bge_base_zh": EncoderSpec(
            key="bge_base_zh", model_name="BAAI/bge-base-zh",
            snapshot=_snapshot("models--BAAI--bge-base-zh"), dim=768,
            pooling="cls", prefix="", tokenizer_type="Chinese BERT WordPiece",
            language_support="Chinese", parameters=102_267_648,
        ),
    }


URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
USER_RE = re.compile(r"(?<!\w)@[\w.\-\u4e00-\u9fff]+")
SPACE_RE = re.compile(r"\s+")


def is_emoji(char: str) -> bool:
    code = ord(char)
    return (
        0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF
        or 0x2300 <= code <= 0x23FF or 0xFE00 <= code <= 0xFE0F
        or 0x1F1E6 <= code <= 0x1F1FF
    )


def basic_clean(text: str, remove_emoji: bool = False) -> str:
    text = html.unescape(valid_text(text))
    text = URL_RE.sub("<URL>", text)
    text = USER_RE.sub("<USER>", text)
    kept = []
    for char in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"} and char not in "\n\t":
            continue
        if char == "\ufffd":
            continue
        if remove_emoji and is_emoji(char):
            continue
        kept.append(char)
    return SPACE_RE.sub(" ", "".join(kept)).strip()


def compose_item_text(title: object, content: object, tax1: object, tax2: object,
                      spec: EncoderSpec, preprocessing: str) -> str:
    body = (f"taxonomy1: {valid_text(tax1)} taxonomy2: {valid_text(tax2)} "
            f"title: {valid_text(title)} content: {valid_text(content)}")
    if preprocessing == "raw":
        normalized = body.strip()
    elif preprocessing == "basic_clean":
        normalized = basic_clean(body)
    elif preprocessing == "remove_emoji":
        normalized = basic_clean(body, remove_emoji=True)
    else:
        raise ValueError(preprocessing)
    return spec.prefix + normalized


def prepare_note_ids(root: Path, output_dir: Path) -> np.ndarray:
    path = output_dir / "note_ids.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for file in note_files(root):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=200_000, columns=["note_idx"]):
            parts.append(batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
    ids = np.concatenate(parts)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("note_idx is not unique")
    np.save(path, ids)
    return np.load(path, mmap_mode="r")


def load_sample_records(root: Path, note_ids: np.ndarray, sample_size: int = 10_000,
                        seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = set(map(int, rng.choice(note_ids, size=sample_size, replace=False)))
    records = []
    required: dict[str, dict] = {}
    columns = ["note_idx", "note_title", "note_content", "taxonomy1_id", "taxonomy2_id"]
    for file in note_files(root):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=100_000, columns=columns):
            values = [batch.column(i).to_pylist() for i in range(5)]
            for note, title, content, tax1, tax2 in zip(*values):
                note = int(note)
                text = f"{title or ''} {content or ''}"
                candidate = {"note_idx": note, "title": title, "content": content,
                             "taxonomy1": tax1, "taxonomy2": tax2}
                conditions = {
                    "chinese": any(0x3400 <= ord(char) <= 0x9FFF for char in text),
                    "english": any("a" <= char.lower() <= "z" for char in text),
                    "emoji": any(is_emoji(char) for char in text),
                    "special_symbol": any(char in "#@<>/&%$" for char in text),
                    "both_title_content_empty": not valid_text(title) and not valid_text(content),
                }
                for key, matches in conditions.items():
                    if matches and key not in required:
                        required[key] = candidate
                if note in selected:
                    records.append(candidate)
    records.sort(key=lambda row: row["note_idx"])
    if len(records) != sample_size:
        raise ValueError(f"Expected {sample_size} diagnostic records, got {len(records)}")
    present_ids = {row["note_idx"] for row in records}
    # Preserve the fixed random sample except for deterministic tail replacements
    # needed to guarantee every required diagnostic text class is represented.
    def represented(key: str) -> bool:
        for row in records:
            text = f"{row['title'] or ''} {row['content'] or ''}"
            if key == "chinese" and any(0x3400 <= ord(char) <= 0x9FFF for char in text): return True
            if key == "english" and any("a" <= char.lower() <= "z" for char in text): return True
            if key == "emoji" and any(is_emoji(char) for char in text): return True
            if key == "special_symbol" and any(char in "#@<>/&%$" for char in text): return True
            if key == "both_title_content_empty" and not valid_text(row["title"]) and not valid_text(row["content"]): return True
        return False
    replace_at = len(records) - 1
    for key in ("chinese", "english", "emoji", "special_symbol", "both_title_content_empty"):
        if not represented(key):
            candidate = required.get(key)
            if candidate is None:
                raise ValueError(f"No corpus record found for required diagnostic class: {key}")
            while records[replace_at]["note_idx"] in {value["note_idx"] for value in required.values()}:
                replace_at -= 1
            present_ids.discard(records[replace_at]["note_idx"])
            records[replace_at] = candidate
            present_ids.add(candidate["note_idx"])
            replace_at -= 1
    records.sort(key=lambda row: row["note_idx"])
    return records


def iter_text_range(root: Path, start: int, end: int, spec: EncoderSpec,
                    preprocessing: str, batch_size: int) -> Iterable[tuple[int, list[str]]]:
    columns = ["note_title", "note_content", "taxonomy1_id", "taxonomy2_id"]
    global_offset = 0
    for file in note_files(root):
        pf = pq.ParquetFile(file)
        file_rows = pf.metadata.num_rows
        file_start, file_end = global_offset, global_offset + file_rows
        if end <= file_start:
            break
        if start >= file_end:
            global_offset = file_end
            continue
        row_offset = file_start
        for rg in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg).num_rows
            rg_start, rg_end = row_offset, row_offset + rg_rows
            row_offset = rg_end
            if end <= rg_start:
                break
            if start >= rg_end:
                continue
            table = pf.read_row_group(rg, columns=columns)
            local_start = max(start, rg_start) - rg_start
            local_end = min(end, rg_end) - rg_start
            table = table.slice(local_start, local_end - local_start)
            values = [table.column(i).to_pylist() for i in range(4)]
            texts = [compose_item_text(*row, spec=spec, preprocessing=preprocessing) for row in zip(*values)]
            base = rg_start + local_start
            for offset in range(0, len(texts), batch_size):
                yield base + offset, texts[offset:offset + batch_size]
        global_offset = file_end


def load_model(spec: EncoderSpec, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec.snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(
        spec.snapshot, local_files_only=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).eval().to(device)
    return tokenizer, model


def encode_texts(texts: Sequence[str], tokenizer, model, spec: EncoderSpec,
                 device: str, max_length: int) -> np.ndarray:
    import torch
    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state.float()
        if spec.pooling == "cls":
            embeddings = hidden[:, 0]
        elif spec.pooling == "masked_mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            raise ValueError(spec.pooling)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float16)


def _encode_worker(root: str, embedding_path: str, progress_dir: str, spec_values: dict,
                   preprocessing: str, total: int, rank: int, world_size: int,
                   batch_size: int, max_length: int) -> None:
    import torch
    spec = EncoderSpec(**spec_values)
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    start, end = total * rank // world_size, total * (rank + 1) // world_size
    progress_path = Path(progress_dir) / f"rank_{rank}.json"
    resume, elapsed_before = start, 0.0
    if progress_path.exists():
        old = json.loads(progress_path.read_text())
        if old.get("model_name") == spec.model_name and old.get("preprocessing") == preprocessing and old.get("end") == end:
            resume = min(end, max(start, int(old.get("next_row", start))))
            elapsed_before = float(old.get("elapsed_seconds", 0.0))
    tokenizer, model = load_model(spec, device)
    mmap = np.memmap(embedding_path, mode="r+", dtype=np.float16, shape=(total, spec.dim))
    begin, next_row = time.perf_counter(), resume
    for batch_idx, (row, texts) in enumerate(iter_text_range(Path(root), resume, end, spec, preprocessing, batch_size)):
        vectors = encode_texts(texts, tokenizer, model, spec, device, max_length)
        mmap[row:row + len(vectors)] = vectors
        next_row = row + len(vectors)
        if batch_idx % 50 == 0:
            mmap.flush()
            progress_path.write_text(json.dumps({
                "rank": rank, "model_name": spec.model_name, "preprocessing": preprocessing,
                "start": start, "end": end, "next_row": next_row,
                "elapsed_seconds": elapsed_before + time.perf_counter() - begin, "complete": False,
            }, indent=2))
    mmap.flush()
    progress_path.write_text(json.dumps({
        "rank": rank, "model_name": spec.model_name, "preprocessing": preprocessing,
        "start": start, "end": end, "next_row": end,
        "elapsed_seconds": elapsed_before + time.perf_counter() - begin, "complete": True,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(rank)),
    }, indent=2))


def encode_corpus(root: Path, output_dir: Path, spec: EncoderSpec, preprocessing: str,
                  total: int, batch_size: int = 64, max_length: int = 256,
                  world_size: int = 4) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embeddings.f16"
    expected_bytes = total * spec.dim * np.dtype(np.float16).itemsize
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and embedding_path.exists() and embedding_path.stat().st_size == expected_bytes:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("complete"):
            return metadata
    if not embedding_path.exists() or embedding_path.stat().st_size != expected_bytes:
        mmap = np.memmap(embedding_path, mode="w+", dtype=np.float16, shape=(total, spec.dim))
        mmap.flush(); del mmap
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=_encode_worker, args=(
        str(root), str(embedding_path), str(progress_dir), asdict(spec), preprocessing,
        total, rank, world_size, batch_size, max_length,
    )) for rank in range(world_size)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"Encoding worker failures: {failures}")
    rank_stats = [json.loads((progress_dir / f"rank_{rank}.json").read_text()) for rank in range(world_size)]
    if not all(row["complete"] and row["next_row"] == row["end"] for row in rank_stats):
        raise RuntimeError("Incomplete embedding ranges")
    wall = time.perf_counter() - start
    metadata = {
        **asdict(spec), "preprocessing": preprocessing, "text_configuration": "taxonomy1+taxonomy2+title+content",
        "dtype": "float16", "normalization": "L2", "max_length": max_length,
        "items": total, "batch_size_per_gpu": batch_size, "num_gpus": world_size,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "encoding_wall_seconds": wall,
        "items_per_second": total / wall, "embedding_file_bytes": expected_bytes,
        "gpu_peak_bytes_max": max(int(row.get("gpu_peak_bytes", 0)) for row in rank_stats),
        "rank_stats": rank_stats, "complete": True,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata

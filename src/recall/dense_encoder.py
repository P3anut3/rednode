"""Zero-shot E5 corpus encoding with resumable multi-GPU memmap output."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq


E5_CACHE = Path.home() / ".cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
MODEL_NAME = "intfloat/e5-base-v2"


def valid_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def format_item_text(title: object, content: object, tax1: object = "", tax2: object = "",
                     text_config: str = "text_only") -> str:
    title, content = valid_text(title), valid_text(content)
    if text_config == "text_only":
        body = f"title: {title} content: {content}"
    elif text_config == "text_taxonomy":
        # Taxonomy is placed first so it is not lost when long content is truncated.
        body = (f"taxonomy1: {valid_text(tax1)} taxonomy2: {valid_text(tax2)} "
                f"title: {title} content: {content}")
    else:
        raise ValueError(text_config)
    return "passage: " + body.strip()


def note_files(root: Path) -> list[Path]:
    files = sorted((root / "data/notes").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("data/notes parquet files not found")
    return files


def prepare_catalog_metadata(root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path, stats_path = output_dir / "note_ids.npy", output_dir / "catalog_stats.json"
    if ids_path.exists() and stats_path.exists():
        return json.loads(stats_path.read_text())
    ids_parts: list[np.ndarray] = []
    stats = {"items": 0, "empty_title": 0, "empty_content": 0, "both_empty": 0,
             "taxonomy1_available": 0, "taxonomy2_available": 0, "any_taxonomy_available": 0}
    columns = ["note_idx", "note_title", "note_content", "taxonomy1_id", "taxonomy2_id"]
    for file in note_files(root):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=100_000, columns=columns):
            ids, titles, contents, tax1s, tax2s = (batch.column(i).to_pylist() for i in range(5))
            ids_parts.append(np.asarray(ids, dtype=np.int64))
            for title, content, tax1, tax2 in zip(titles, contents, tax1s, tax2s):
                title_ok, content_ok = bool(valid_text(title)), bool(valid_text(content))
                t1, t2 = bool(valid_text(tax1)), bool(valid_text(tax2))
                stats["items"] += 1
                stats["empty_title"] += not title_ok
                stats["empty_content"] += not content_ok
                stats["both_empty"] += not title_ok and not content_ok
                stats["taxonomy1_available"] += t1
                stats["taxonomy2_available"] += t2
                stats["any_taxonomy_available"] += t1 or t2
    ids = np.concatenate(ids_parts)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("note_idx is not unique")
    np.save(ids_path, ids)
    stats["note_id_min"], stats["note_id_max"] = int(ids.min()), int(ids.max())
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def iter_range(root: Path, start: int, end: int, text_config: str, batch_size: int) -> Iterable[tuple[int, list[str]]]:
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
            local_start, local_end = max(start, rg_start) - rg_start, min(end, rg_end) - rg_start
            table = table.slice(local_start, local_end - local_start)
            vals = [table.column(i).to_pylist() for i in range(4)]
            texts = [format_item_text(*x, text_config=text_config) for x in zip(*vals)]
            base = rg_start + local_start
            for offset in range(0, len(texts), batch_size):
                yield base + offset, texts[offset:offset + batch_size]
        global_offset = file_end


def load_e5(device: str = "cuda"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(E5_CACHE, local_files_only=True)
    model = AutoModel.from_pretrained(
        E5_CACHE, local_files_only=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).eval().to(device)
    return tokenizer, model


def encode_texts(texts: Sequence[str], tokenizer, model, device: str, max_length: int = 256) -> np.ndarray:
    import torch
    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state.float()
        mask = inputs["attention_mask"].unsqueeze(-1)
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy().astype(np.float16)


def _worker(root: str, embedding_path: str, progress_dir: str, text_config: str, total: int,
            dim: int, rank: int, world_size: int, batch_size: int, max_length: int) -> None:
    import torch
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    start = total * rank // world_size
    end = total * (rank + 1) // world_size
    progress_path = Path(progress_dir) / f"rank_{rank}.json"
    resume = start
    elapsed_before = 0.0
    if progress_path.exists():
        old = json.loads(progress_path.read_text())
        if old.get("text_config") == text_config and old.get("end") == end:
            resume = min(end, max(start, int(old.get("next_row", start))))
            elapsed_before = float(old.get("elapsed_seconds", 0.0))
    tokenizer, model = load_e5(device)
    mmap = np.memmap(embedding_path, mode="r+", dtype=np.float16, shape=(total, dim))
    begin = time.perf_counter()
    next_row = resume
    for batch_idx, (row, texts) in enumerate(iter_range(Path(root), resume, end, text_config, batch_size)):
        vectors = encode_texts(texts, tokenizer, model, device, max_length=max_length)
        mmap[row:row + len(vectors)] = vectors
        next_row = row + len(vectors)
        if batch_idx % 50 == 0:
            mmap.flush()
            progress_path.write_text(json.dumps({
                "rank": rank, "text_config": text_config, "start": start, "end": end,
                "next_row": next_row, "elapsed_seconds": elapsed_before + time.perf_counter() - begin,
                "complete": False,
            }, indent=2))
    mmap.flush()
    elapsed = elapsed_before + time.perf_counter() - begin
    progress_path.write_text(json.dumps({
        "rank": rank, "text_config": text_config, "start": start, "end": end,
        "next_row": end, "elapsed_seconds": elapsed, "complete": True,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(rank)),
    }, indent=2))


def encode_corpus_multi_gpu(root: Path, output_dir: Path, text_config: str, batch_size: int = 64,
                            max_length: int = 256, world_size: int = 4, dim: int = 768) -> dict:
    stats = prepare_catalog_metadata(root, output_dir.parent)
    total = int(stats["items"])
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "embeddings.f16"
    expected_bytes = total * dim * np.dtype(np.float16).itemsize
    if not embedding_path.exists() or embedding_path.stat().st_size != expected_bytes:
        mmap = np.memmap(embedding_path, mode="w+", dtype=np.float16, shape=(total, dim))
        mmap.flush(); del mmap
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(exist_ok=True)
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    processes = [ctx.Process(target=_worker, args=(
        str(root), str(embedding_path), str(progress_dir), text_config, total, dim,
        rank, world_size, batch_size, max_length,
    )) for rank in range(world_size)]
    for process in processes: process.start()
    for process in processes: process.join()
    failures = [p.exitcode for p in processes if p.exitcode != 0]
    if failures:
        raise RuntimeError(f"GPU encoding worker failures: {failures}")
    rank_stats = [json.loads((progress_dir / f"rank_{rank}.json").read_text()) for rank in range(world_size)]
    if not all(x["complete"] and x["next_row"] == x["end"] for x in rank_stats):
        raise RuntimeError("Incomplete embedding shards")
    wall = time.perf_counter() - start
    metadata = {
        "model_name": MODEL_NAME, "model_snapshot": str(E5_CACHE), "dimension": dim,
        "text_config": text_config, "item_prefix": "passage: ", "normalized": True,
        "storage_dtype": "float16", "inference_dtype": "float16", "max_length": max_length,
        "batch_size_per_gpu": batch_size, "num_gpus": world_size, "items": total,
        "encoding_wall_seconds": wall, "items_per_second": total / wall,
        "gpu_peak_bytes_max": max(int(x.get("gpu_peak_bytes", 0)) for x in rank_stats),
        "embedding_file_bytes": embedding_path.stat().st_size,
        "rank_stats": rank_stats,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


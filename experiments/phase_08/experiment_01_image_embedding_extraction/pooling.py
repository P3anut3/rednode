"""Deterministic item pooling over sharded, normalized per-image vectors."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .data import OUT, SHARD_SIZE, save_json_atomic, sha256_file
from .extractor import OUTPUT_DIM, embedding_path, valid_path, validate_extraction_chain

STRATEGIES = {
    "first": "first_image",
    "top3": "mean_top3",
    "all": "mean_all",
}


def require_pooling_source(validator=None) -> dict:
    return (validator or validate_extraction_chain)()


class ShardReader:
    def __init__(self, shard_size: int = SHARD_SIZE, embedding_fn=embedding_path, valid_fn=valid_path):
        self.shard_size = shard_size
        self.embedding_fn = embedding_fn
        self.valid_fn = valid_fn
        self._shard_id = None
        self._embedding = None
        self._valid = None

    def _load(self, shard_id: int) -> None:
        if self._shard_id != shard_id:
            self._embedding = np.load(self.embedding_fn(shard_id), mmap_mode="r")
            self._valid = np.load(self.valid_fn(shard_id), mmap_mode="r")
            self._shard_id = shard_id

    def range(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        if end <= start:
            return np.empty((0, OUTPUT_DIM), np.float32), np.empty(0, np.bool_)
        vectors, masks = [], []
        cursor = start
        while cursor < end:
            shard_id, local = divmod(cursor, self.shard_size)
            self._load(shard_id)
            take = min(end - cursor, len(self._embedding) - local)
            if take <= 0:
                raise RuntimeError(f"invalid shard boundary at global image row {cursor}")
            vectors.append(np.asarray(self._embedding[local:local + take], dtype=np.float32))
            masks.append(np.asarray(self._valid[local:local + take], dtype=np.bool_))
            cursor += take
        return np.concatenate(vectors), np.concatenate(masks)


def pool_vectors(vectors: np.ndarray, valid: np.ndarray, strategy: str) -> tuple[np.ndarray, bool, bool]:
    """Return normalized vector, availability, and first-image fallback flag."""
    vectors = np.asarray(vectors, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.bool_)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return np.zeros(vectors.shape[1] if vectors.ndim == 2 else OUTPUT_DIM, np.float32), False, False
    fallback = False
    if strategy == "first":
        chosen = indices[:1]
        fallback = int(chosen[0]) != 0
    elif strategy == "top3":
        chosen = indices[:3]
    elif strategy == "all":
        chosen = indices
    else:
        raise ValueError(f"unknown pooling strategy: {strategy}")
    pooled = vectors[chosen].mean(axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(pooled))
    if not np.isfinite(norm) or norm <= 0:
        return np.zeros(vectors.shape[1], np.float32), False, fallback
    return pooled / norm, True, fallback


def output_directory(strategy: str) -> Path:
    return OUT / "pooled_embeddings" / STRATEGIES[strategy]


def prepare_pooling_workspace(destination: Path, recover_incomplete: bool) -> Path | None:
    building = destination.with_name(destination.name + "_building")
    if destination.exists():
        if (destination / "pooling_complete.json").exists():
            return None
        if not recover_incomplete:
            raise RuntimeError(f"incomplete formal pooling directory exists; use --recover-pooling: {destination}")
        shutil.rmtree(destination)
    if building.exists():
        if not recover_incomplete:
            raise RuntimeError(f"interrupted pooling build exists; use --recover-pooling: {building}")
        shutil.rmtree(building)
    building.mkdir(parents=True, exist_ok=False)
    return building


def generate_pooling(strategy: str, item_block: int = 10_000, recover_incomplete: bool = False) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    extraction_marker = OUT / "markers/extraction_complete.json"
    # Do not trust presence of the global marker alone: re-hash every source
    # manifest, embedding, mask and decode-failure table before deriving assets.
    require_pooling_source()
    source_hash = sha256_file(extraction_marker)
    offsets = np.load(OUT / "mappings/image_offsets.npy", mmap_mode="r")
    n_items = len(offsets) - 1
    destination = output_directory(strategy)
    building = prepare_pooling_workspace(destination, recover_incomplete)
    if building is None:
        return validate_pooling(strategy)
    final_embedding = building / "embeddings.f16.npy"
    final_mask = building / "image_available.npy"
    temp_embedding = final_embedding.with_suffix(final_embedding.suffix + ".tmp")
    temp_mask = final_mask.with_suffix(final_mask.suffix + ".tmp")
    output = np.lib.format.open_memmap(temp_embedding, mode="w+", dtype=np.float16, shape=(n_items, OUTPUT_DIM))
    available = np.lib.format.open_memmap(temp_mask, mode="w+", dtype=np.bool_, shape=(n_items,))
    output[:] = 0
    available[:] = False
    reader = ShardReader()
    fallback_count = 0
    start_time = time.monotonic()
    for block_start in range(0, n_items, item_block):
        block_end = min(n_items, block_start + item_block)
        image_start, image_end = int(offsets[block_start]), int(offsets[block_end])
        vectors, masks = reader.range(image_start, image_end)
        for item_row in range(block_start, block_end):
            local_start = int(offsets[item_row] - image_start)
            local_end = int(offsets[item_row + 1] - image_start)
            vector, present, fallback = pool_vectors(vectors[local_start:local_end], masks[local_start:local_end], strategy)
            if present:
                output[item_row] = vector.astype(np.float16)
                available[item_row] = True
            fallback_count += int(fallback)
        if block_start % (item_block * 10) == 0:
            print(f"[{strategy}] pooled {block_end}/{n_items} items", flush=True)
    output.flush()
    available.flush()
    available_count = int(np.asarray(available).sum())
    del output, available
    temp_embedding.replace(final_embedding)
    temp_mask.replace(final_mask)
    sample = np.load(final_embedding, mmap_mode="r")
    mask = np.load(final_mask, mmap_mode="r")
    sample_rows = np.flatnonzero(mask)[::max(1, available_count // 10_000)][:10_000]
    norms = np.linalg.norm(np.asarray(sample[sample_rows], dtype=np.float32), axis=1) if len(sample_rows) else np.empty(0)
    metadata = {
        "strategy": STRATEGIES[strategy], "shape": [n_items, OUTPUT_DIM], "dtype": "float16",
        "available_items": available_count, "missing_items": n_items - available_count,
        "source_extraction_hash": source_hash,
        "embedding_sha256": sha256_file(final_embedding), "mask_sha256": sha256_file(final_mask),
        "norm_statistics": {
            "sample_count": len(norms), "mean": float(norms.mean()) if len(norms) else None,
            "min": float(norms.min()) if len(norms) else None,
            "max": float(norms.max()) if len(norms) else None,
        },
        "fallback_count": fallback_count if strategy == "first" else 0,
        "elapsed_seconds": time.monotonic() - start_time, "test_opened": False,
    }
    save_json_atomic(building / "metadata.json", metadata)
    checksums = (
        f"{metadata['embedding_sha256']}  {final_embedding.name}\n"
        f"{metadata['mask_sha256']}  {final_mask.name}\n"
        f"{sha256_file(building / 'metadata.json')}  metadata.json\n"
    )
    checksum_tmp = building / "SHA256SUMS.tmp"
    checksum_tmp.write_text(checksums)
    checksum_tmp.replace(building / "SHA256SUMS")
    completion = {
        "complete": True, "strategy": STRATEGIES[strategy],
        "source_extraction_hash": source_hash,
        "embedding_sha256": metadata["embedding_sha256"],
        "mask_sha256": metadata["mask_sha256"],
        "metadata_sha256": sha256_file(building / "metadata.json"),
        "sha256sums_sha256": sha256_file(building / "SHA256SUMS"),
        "test_opened": False,
    }
    save_json_atomic(building / "pooling_complete.json", completion)
    _validate_pooling_directory(strategy, building)
    building.replace(destination)
    return validate_pooling(strategy)


def _validate_pooling_directory(strategy: str, destination: Path) -> dict:
    metadata = json.loads((destination / "metadata.json").read_text())
    completion = json.loads((destination / "pooling_complete.json").read_text())
    validate_extraction_chain()
    current_source_hash = sha256_file(OUT / "markers/extraction_complete.json")
    if metadata.get("source_extraction_hash") != current_source_hash:
        raise RuntimeError(f"pooled output was derived from a different extraction marker: {strategy}")
    if not completion.get("complete") or completion.get("test_opened") is not False:
        raise RuntimeError(f"invalid pooling completion marker: {strategy}")
    completion_checks = {
        "source_extraction_hash": current_source_hash,
        "embedding_sha256": metadata["embedding_sha256"],
        "mask_sha256": metadata["mask_sha256"],
        "metadata_sha256": sha256_file(destination / "metadata.json"),
        "sha256sums_sha256": sha256_file(destination / "SHA256SUMS"),
    }
    for field, expected in completion_checks.items():
        if completion.get(field) != expected:
            raise RuntimeError(f"pooling completion mismatch for {strategy}: {field}")
    embedding = np.load(destination / "embeddings.f16.npy", mmap_mode="r")
    mask = np.load(destination / "image_available.npy", mmap_mode="r")
    if embedding.shape != tuple(metadata["shape"]) or embedding.dtype != np.float16:
        raise RuntimeError(f"invalid pooled embedding for {strategy}")
    if mask.shape != (len(embedding),) or mask.dtype != np.bool_:
        raise RuntimeError(f"invalid pooled mask for {strategy}")
    if sha256_file(destination / "embeddings.f16.npy") != metadata["embedding_sha256"]:
        raise RuntimeError(f"pooled embedding hash mismatch for {strategy}")
    if sha256_file(destination / "image_available.npy") != metadata["mask_sha256"]:
        raise RuntimeError(f"pooled mask hash mismatch for {strategy}")
    for start in range(0, len(embedding), 10_000):
        values = np.asarray(embedding[start:start + 10_000])
        present = np.asarray(mask[start:start + 10_000])
        if np.any(values[~present] != 0) or not np.isfinite(values).all():
            raise RuntimeError(f"pooled finite/missing-zero contract failed for {strategy}")
    return metadata


def validate_pooling(strategy: str) -> dict:
    return _validate_pooling_directory(strategy, output_directory(strategy))

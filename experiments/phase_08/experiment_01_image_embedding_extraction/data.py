"""Canonical note/image CSR mapping and manifest audit utilities."""

from __future__ import annotations

import html
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps, UnidentifiedImageError

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (
    NOTE_IDS_PATH,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase_08/experiment_01_image_embedding_extraction"
IMAGE_ROOT = ROOT / "data/qilin_images/mnt/ali-sh-1/usr/lihaitao/process_0106"
RESOLVED_IMAGE_ROOT = IMAGE_ROOT.resolve()
NOTES_ROOT = ROOT / "data/notes"
SHARD_SIZE = 100_000
EXPECTED_ITEMS = 1_983_938
MANIFEST_COLUMNS = (
    "global_image_row", "note_row", "note_idx", "position",
    "relative_path", "shard_id", "row_in_shard",
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def save_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)


def save_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def canonical_note_ids() -> np.ndarray:
    values = np.load(NOTE_IDS_PATH, mmap_mode="r")
    if values.shape != (EXPECTED_ITEMS,):
        raise ValueError("canonical NOTE_IDS_PATH has invalid shape")
    validate_unique_note_ids(values)
    return np.asarray(values, dtype=np.int64)


def validate_unique_note_ids(values) -> None:
    values = np.asarray(values, dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise ValueError("canonical NOTE_IDS_PATH has invalid shape or duplicate IDs")


class CanonicalLookup:
    def __init__(self, note_ids: np.ndarray):
        self.note_ids = np.asarray(note_ids, dtype=np.int64)
        self.order = np.argsort(self.note_ids)
        self.sorted_ids = self.note_ids[self.order]

    def rows(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        positions = np.searchsorted(self.sorted_ids, values)
        valid = positions < len(self.sorted_ids)
        valid &= self.sorted_ids[np.minimum(positions, len(self.sorted_ids) - 1)] == values
        if not valid.all():
            raise KeyError(f"notes parquet contains {int((~valid).sum())} non-canonical note IDs")
        return self.order[positions]


def validate_relative_path(value: str) -> tuple[str, bool, str | None]:
    text = html.unescape(str(value)).replace("\\", "/").strip()
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts:
        return text, False, "absolute_or_parent_escape"
    resolved = (RESOLVED_IMAGE_ROOT / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(RESOLVED_IMAGE_ROOT)
    except ValueError:
        return text, False, "resolved_outside_root"
    return text, True, None


def absolute_image_path(relative_path: str) -> Path:
    text, safe, reason = validate_relative_path(relative_path)
    if not safe:
        raise ValueError(f"unsafe image path ({reason}): {relative_path}")
    return IMAGE_ROOT / Path(*PurePosixPath(text).parts)


def notes_files() -> list[Path]:
    files = sorted(NOTES_ROOT.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no notes parquet under {NOTES_ROOT}")
    return files


def _link_canonical_mapping() -> Path:
    target = OUT / "mappings/note_ids.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if sha256_file(target) != sha256_file(NOTE_IDS_PATH):
            raise RuntimeError("existing Phase-8 note_ids differs from canonical mapping")
        return target
    relative = os.path.relpath(NOTE_IDS_PATH, target.parent)
    temporary = target.with_suffix(".npy.tmp")
    os.symlink(relative, temporary)
    temporary.replace(target)
    return target


def _decode_sample(rows: list[tuple[str, str]], limit_per_part: int = 10) -> dict:
    by_part: dict[str, list[str]] = {}
    for path, part in rows:
        by_part.setdefault(part, [])
        if len(by_part[part]) < limit_per_part:
            by_part[part].append(path)
    formats, sizes, failures = Counter(), [], Counter()
    for values in by_part.values():
        for relative in values:
            try:
                with Image.open(absolute_image_path(relative)) as image:
                    formats[str(image.format or "UNKNOWN")] += 1
                    transformed = ImageOps.exif_transpose(image)
                    sizes.append((int(transformed.width), int(transformed.height)))
                    transformed.convert("RGB").load()
            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
                failures[type(exc).__name__] += 1
    return {
        "sampled": sum(formats.values()) + sum(failures.values()),
        "formats": dict(formats),
        "decode_failures": dict(failures),
        "width": _describe([x[0] for x in sizes]),
        "height": _describe([x[1] for x in sizes]),
    }


def _write_manifest_shard(shard_id: int, columns: dict[str, list], directory: Path) -> None:
    expected_start = shard_id * SHARD_SIZE
    global_rows = np.asarray(columns["global_image_row"], dtype=np.int64)
    if not np.array_equal(global_rows, np.arange(expected_start, expected_start + len(global_rows))):
        raise AssertionError("manifest rows are not globally contiguous")
    target = directory / f"shard_{shard_id:05d}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    table = pa.table({
        "global_image_row": pa.array(columns["global_image_row"], type=pa.int64()),
        "note_row": pa.array(columns["note_row"], type=pa.int64()),
        "note_idx": pa.array(columns["note_idx"], type=pa.int64()),
        "position": pa.array(columns["position"], type=pa.int32()),
        "relative_path": pa.array(columns["relative_path"], type=pa.string()),
        "shard_id": pa.array(columns["shard_id"], type=pa.int32()),
        "row_in_shard": pa.array(columns["row_in_shard"], type=pa.int32()),
    })
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(target)


def _describe(values) -> dict:
    array = np.asarray(values)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)), "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)), "max": int(array.max()),
    }


def prepare_audit_workspace(output_root: Path, recover_audit: bool = False) -> Path | None:
    complete_marker = output_root / "markers/audit_complete.json"
    if complete_marker.exists():
        return None
    building = output_root / "audit_building"
    final_shards = output_root / "mappings/shards"
    partial = building.exists() or final_shards.exists() or (output_root / "mappings/image_offsets.npy").exists()
    if partial and not recover_audit:
        raise RuntimeError("incomplete audit artifacts found; inspect them, then rerun explicitly with --recover-audit")
    if partial and recover_audit:
        if (output_root / "markers/extraction_complete.json").exists() or list((output_root / "markers").glob("extract_shard_*.json")):
            raise RuntimeError("cannot recover audit after image extraction has started")
        shutil.rmtree(building, ignore_errors=True)
        shutil.rmtree(final_shards, ignore_errors=True)
        for path in (output_root / "mappings/image_offsets.npy", output_root / "mappings/note_ids.npy"):
            if path.exists() or path.is_symlink():
                path.unlink()
    building.mkdir(parents=True, exist_ok=False)
    return building


def publish_completed_audit_building(building: Path) -> dict:
    """Validate and publish a fully built audit transaction after an interrupted publish."""
    required = [
        building / "audit.json", building / "manifest_metadata.json",
        building / "smoke_sample.parquet", building / "image_offsets.npy",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError("audit_building is not complete enough for recovery publication")
    audit = json.loads((building / "audit.json").read_text())
    metadata = json.loads((building / "manifest_metadata.json").read_text())
    if int(audit["item_count"]) != EXPECTED_ITEMS or int(metadata["item_count"]) != EXPECTED_ITEMS:
        raise RuntimeError("recovered audit item count mismatch")
    if int(audit["unsafe_path_count"]) != 0:
        raise RuntimeError("unsafe paths prevent recovery publication")
    offsets = np.load(building / "image_offsets.npy", mmap_mode="r")
    if offsets.shape != (EXPECTED_ITEMS + 1,) or offsets.dtype != np.int64 or np.any(np.diff(offsets) < 0):
        raise RuntimeError("recovered audit CSR offsets are invalid")
    total = int(offsets[-1])
    if total != int(audit["total_image_paths"]) or total != int(metadata["image_count"]):
        raise RuntimeError("recovered audit image count mismatch")
    building_shards = building / "shards"
    built_shards = sorted(building_shards.glob("shard_*.parquet"))
    if len(built_shards) != int(metadata["shard_count"]):
        raise RuntimeError("recovered audit shard count mismatch")
    if sum(pq.ParquetFile(path).metadata.num_rows for path in built_shards) != total:
        raise RuntimeError("recovered audit manifest row count mismatch")
    if len(pq.read_table(building / "smoke_sample.parquet")) != 10_000:
        raise RuntimeError("recovered audit smoke sample is not exactly 10,000 rows")
    final_shards = OUT / "mappings/shards"
    if final_shards.exists() or (OUT / "markers/audit_complete.json").exists():
        raise RuntimeError("formal audit artifacts already exist; recovery publication refused")
    (OUT / "mappings").mkdir(parents=True, exist_ok=True)
    building_shards.replace(final_shards)
    (building / "image_offsets.npy").replace(OUT / "mappings/image_offsets.npy")
    _link_canonical_mapping()
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name in ("audit.json", "manifest_metadata.json", "smoke_sample.parquet"):
        (building / name).replace(audit_dir / name)
    building.rmdir()
    manifest_hashes = [sha256_file(path) for path in sorted(final_shards.glob("shard_*.parquet"))]
    complete = {
        "complete": True, "item_count": EXPECTED_ITEMS, "image_count": total,
        "shard_count": len(built_shards), "canonical_note_ids_sha256": sha256_file(NOTE_IDS_PATH),
        "image_offsets_sha256": sha256_file(OUT / "mappings/image_offsets.npy"),
        "manifest_digest": _hash_strings(manifest_hashes),
        "audit_sha256": sha256_file(audit_dir / "audit.json"),
        "manifest_metadata_sha256": sha256_file(audit_dir / "manifest_metadata.json"),
        "smoke_sample_sha256": sha256_file(audit_dir / "smoke_sample.parquet"),
        "test_opened": False,
    }
    save_json_atomic(OUT / "markers/audit_complete.json", complete)
    (OUT / "markers/audit_blocked_unsafe.json").unlink(missing_ok=True)
    validate_audit_complete()
    return audit


def build_audit_and_manifests(seed: int = 42, recover_audit: bool = False) -> dict:
    """One bounded parquet pass in memory, then transactional publication."""
    complete_marker = OUT / "markers/audit_complete.json"
    existing_building = OUT / "audit_building"
    if recover_audit and existing_building.exists():
        required = ("audit.json", "manifest_metadata.json", "smoke_sample.parquet", "image_offsets.npy")
        if all((existing_building / name).exists() for name in required):
            return publish_completed_audit_building(existing_building)
    building = prepare_audit_workspace(OUT, recover_audit)
    if building is None:
        validate_audit_complete()
        return json.loads((OUT / "audit/audit.json").read_text())
    building_shards = building / "shards"
    final_shards = OUT / "mappings/shards"
    building_shards.mkdir()
    note_ids = canonical_note_ids()
    lookup = CanonicalLookup(note_ids)
    paths_by_row: list[list[str] | None] = [None] * len(note_ids)
    image_num_by_row = np.full(len(note_ids), -1, dtype=np.int64)
    seen = np.zeros(len(note_ids), dtype=np.bool_)
    audited_notes = 0
    audit_started = time.monotonic()
    unsafe: list[dict] = []
    path_counter: Counter[str] = Counter()
    part_counter: Counter[str] = Counter()
    for file in notes_files():
        schema = pq.ParquetFile(file).schema_arrow
        columns = ["note_idx", "image_path"]
        if "image_num" in schema.names:
            columns.append("image_num")
        for batch in pq.ParquetFile(file).iter_batches(batch_size=10_000, columns=columns):
            values = batch.to_pydict()
            rows = lookup.rows(values["note_idx"])
            if len(np.unique(rows)) != len(rows):
                raise ValueError("duplicate note IDs occur within one notes parquet batch")
            if seen[rows].any():
                duplicates = np.asarray(values["note_idx"])[seen[rows]][:10].tolist()
                raise ValueError(f"duplicate note IDs in notes parquet: {duplicates}")
            seen[rows] = True
            for local, row in enumerate(rows):
                paths = []
                for position, raw in enumerate(values["image_path"][local] or []):
                    text, safe, reason = validate_relative_path(raw)
                    if not safe:
                        unsafe.append({"note_idx": int(note_ids[row]), "position": position, "path": text, "reason": reason})
                    paths.append(text)
                    path_counter[text] += 1
                    match = re.search(r"(?:^|/)part_(\d+)(?:/|$)", text)
                    part_counter[f"part_{match.group(1)}" if match else "<UNKNOWN>"] += 1
                paths_by_row[row] = paths
                if "image_num" in values and values["image_num"][local] is not None:
                    raw_image_num = values["image_num"][local]
                    try:
                        if np.isfinite(float(raw_image_num)):
                            image_num_by_row[row] = int(raw_image_num)
                    except (TypeError, ValueError):
                        pass
            audited_notes += len(rows)
            if audited_notes % 100_000 < len(rows):
                elapsed = time.monotonic() - audit_started
                print(
                    f"[audit] notes={audited_notes:,}/{len(note_ids):,} "
                    f"image_paths={sum(part_counter.values()):,} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    if not seen.all():
        raise ValueError(f"canonical IDs missing from notes parquet: {int((~seen).sum())}")
    counts = np.fromiter((len(value or ()) for value in paths_by_row), dtype=np.int64, count=len(note_ids))
    offsets = np.empty(len(note_ids) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    save_npy_atomic(building / "image_offsets.npy", offsets)
    total = int(offsets[-1])
    shard_count = (total + SHARD_SIZE - 1) // SHARD_SIZE
    current_shard = 0
    buffer = {column: [] for column in MANIFEST_COLUMNS}
    existence_candidates: list[tuple[int, str]] = []
    decode_candidates: list[tuple[str, str]] = []
    decode_part_counts: Counter[str] = Counter()
    smoke_candidates: list[dict] = []
    smoke_bin_counts: Counter[str] = Counter()
    rng = random.Random(seed)
    reservoir_seen = 0
    for note_row, paths in enumerate(paths_by_row):
        for position, relative in enumerate(paths or ()):  # canonical order
            global_row = int(offsets[note_row] + position)
            shard_id, row_in_shard = divmod(global_row, SHARD_SIZE)
            if shard_id != current_shard:
                _write_manifest_shard(current_shard, buffer, building_shards)
                current_shard = shard_id
                buffer = {column: [] for column in MANIFEST_COLUMNS}
            row = (global_row, note_row, int(note_ids[note_row]), position, relative, shard_id, row_in_shard)
            for column, value in zip(MANIFEST_COLUMNS, row):
                buffer[column].append(value)
            reservoir_seen += 1
            if len(existence_candidates) < 10_000:
                existence_candidates.append((global_row, relative))
            else:
                replacement = rng.randrange(reservoir_seen)
                if replacement < 10_000:
                    existence_candidates[replacement] = (global_row, relative)
            part = next((part for part in PurePosixPath(relative).parts if part.startswith("part_")), "<UNKNOWN>")
            if decode_part_counts[part] < 10:
                decode_candidates.append((relative, part))
                decode_part_counts[part] += 1
            image_count = int(counts[note_row])
            image_bin = "1" if image_count == 1 else "2_3" if image_count <= 3 else "4_10" if image_count <= 10 else "11_plus"
            # Fixed deterministic coverage first, then reservoir-fill below.
            key = f"{part}:{image_bin}"
            if smoke_bin_counts[key] < 2:
                smoke_candidates.append({
                    "global_image_row": global_row, "note_row": note_row,
                    "note_idx": int(note_ids[note_row]), "position": position,
                    "relative_path": relative, "part": part, "image_count_bin": image_bin,
                })
                smoke_bin_counts[key] += 1
    if total:
        _write_manifest_shard(current_shard, buffer, building_shards)
    # Add the global reservoir while retaining deterministic part/count coverage.
    selected_rows = {int(row["global_image_row"]) for row in smoke_candidates}
    for global_row, relative in existence_candidates:
        if len(smoke_candidates) >= 10_000:
            break
        if global_row in selected_rows:
            continue
        note_row = int(np.searchsorted(offsets, global_row, side="right") - 1)
        position = int(global_row - offsets[note_row])
        part = next((p for p in PurePosixPath(relative).parts if p.startswith("part_")), "<UNKNOWN>")
        image_count = int(counts[note_row])
        image_bin = "1" if image_count == 1 else "2_3" if image_count <= 3 else "4_10" if image_count <= 10 else "11_plus"
        smoke_candidates.append({
            "global_image_row": global_row, "note_row": note_row,
            "note_idx": int(note_ids[note_row]), "position": position,
            "relative_path": relative, "part": part, "image_count_bin": image_bin,
        })
        selected_rows.add(global_row)
    smoke_candidates.sort(key=lambda row: row["global_image_row"])
    smoke_path = building / "smoke_sample.parquet"
    smoke_tmp = smoke_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(smoke_candidates[:10_000]), smoke_tmp, compression="zstd")
    smoke_tmp.replace(smoke_path)
    exists = [absolute_image_path(path).is_file() for _, path in existence_candidates if validate_relative_path(path)[1]]
    valid_image_num = image_num_by_row >= 0
    agreement = (image_num_by_row[valid_image_num] == counts[valid_image_num])
    bins = {
        "0": int((counts == 0).sum()), "1": int((counts == 1).sum()),
        "2_3": int(((counts >= 2) & (counts <= 3)).sum()),
        "4_10": int(((counts >= 4) & (counts <= 10)).sum()),
        "11_plus": int((counts >= 11).sum()),
    }
    audit = {
        "item_count": len(note_ids), "total_image_paths": total,
        "items_with_images": int((counts > 0).sum()),
        "items_without_images": int((counts == 0).sum()),
        "path_count_distribution": bins, "max_images_per_item": int(counts.max()),
        "duplicate_relative_path_occurrences": int(sum(value - 1 for value in path_counter.values() if value > 1)),
        "duplicate_relative_path_unique": int(sum(value > 1 for value in path_counter.values())),
        "unsafe_path_count": len(unsafe), "unsafe_path_examples": unsafe[:100],
        "part_directories": dict(sorted(part_counter.items())),
        "part_directory_count": len(part_counter),
        "image_num_audited_items": int(valid_image_num.sum()),
        "image_num_path_count_agreement_rate": float(agreement.mean()) if len(agreement) else None,
        "random_path_existence_sample": len(exists),
        "random_path_existence_rate": float(np.mean(exists)) if exists else None,
        "decode_sample": _decode_sample(decode_candidates),
        "manifest_shards": shard_count, "shard_size": SHARD_SIZE,
        "canonical_note_ids_source": str(NOTE_IDS_PATH.relative_to(ROOT)),
        "canonical_note_ids_sha256": sha256_file(NOTE_IDS_PATH),
        "test_opened": False,
    }
    save_json_atomic(building / "audit.json", audit)
    save_json_atomic(building / "manifest_metadata.json", {
        "item_count": len(note_ids), "image_count": total,
        "shard_count": shard_count, "shard_size": SHARD_SIZE,
        "manifest_columns": MANIFEST_COLUMNS,
    })
    built_shards = sorted(building_shards.glob("shard_*.parquet"))
    if len(built_shards) != shard_count:
        raise RuntimeError("transactional audit produced an incomplete manifest shard set")
    manifest_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in built_shards)
    if manifest_rows != total:
        raise RuntimeError("transactional audit manifest row count mismatch")
    if unsafe:
        # Publish only the audit finding; unsafe paths never enter formal mappings.
        audit_dir = OUT / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (building / "audit.json").replace(audit_dir / "audit.json")
        save_json_atomic(OUT / "markers/audit_blocked_unsafe.json", {
            "complete": False, "unsafe_path_count": len(unsafe), "test_opened": False,
        })
        shutil.rmtree(building, ignore_errors=True)
        raise RuntimeError(f"audit found {len(unsafe)} unsafe paths; formal mappings were not published")
    # Publish immutable mappings first and write audit_complete last. A crash in
    # this small window is recoverable only through explicit --recover-audit.
    (OUT / "mappings").mkdir(parents=True, exist_ok=True)
    building_shards.replace(final_shards)
    (building / "image_offsets.npy").replace(OUT / "mappings/image_offsets.npy")
    _link_canonical_mapping()
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name in ("audit.json", "manifest_metadata.json", "smoke_sample.parquet"):
        (building / name).replace(audit_dir / name)
    building.rmdir()
    manifest_hashes = [sha256_file(path) for path in sorted(final_shards.glob("shard_*.parquet"))]
    complete = {
        "complete": True, "item_count": len(note_ids), "image_count": total,
        "shard_count": shard_count, "canonical_note_ids_sha256": sha256_file(NOTE_IDS_PATH),
        "image_offsets_sha256": sha256_file(OUT / "mappings/image_offsets.npy"),
        "manifest_digest": _hash_strings(manifest_hashes),
        "audit_sha256": sha256_file(audit_dir / "audit.json"), "test_opened": False,
        "manifest_metadata_sha256": sha256_file(audit_dir / "manifest_metadata.json"),
        "smoke_sample_sha256": sha256_file(audit_dir / "smoke_sample.parquet"),
    }
    save_json_atomic(complete_marker, complete)
    (OUT / "markers/audit_blocked_unsafe.json").unlink(missing_ok=True)
    validate_audit_complete()
    return audit


def _hash_strings(values: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def validate_audit_complete() -> dict:
    marker_path = OUT / "markers/audit_complete.json"
    marker = json.loads(marker_path.read_text())
    metadata = json.loads((OUT / "audit/manifest_metadata.json").read_text())
    shards = sorted((OUT / "mappings/shards").glob("shard_*.parquet"))
    if not marker.get("complete") or marker.get("test_opened") is not False:
        raise RuntimeError("invalid audit completion marker")
    if len(shards) != int(marker["shard_count"]):
        raise RuntimeError("audit manifest shard count mismatch")
    if sum(pq.ParquetFile(path).metadata.num_rows for path in shards) != int(marker["image_count"]):
        raise RuntimeError("audit manifest row count mismatch")
    checks = {
        "canonical_note_ids_sha256": sha256_file(OUT / "mappings/note_ids.npy"),
        "image_offsets_sha256": sha256_file(OUT / "mappings/image_offsets.npy"),
        "manifest_digest": _hash_strings([sha256_file(path) for path in shards]),
        "audit_sha256": sha256_file(OUT / "audit/audit.json"),
        "manifest_metadata_sha256": sha256_file(OUT / "audit/manifest_metadata.json"),
        "smoke_sample_sha256": sha256_file(OUT / "audit/smoke_sample.parquet"),
    }
    for field, actual in checks.items():
        if marker.get(field) != actual:
            raise RuntimeError(f"audit completion hash mismatch: {field}")
    if int(metadata["image_count"]) != int(marker["image_count"]):
        raise RuntimeError("audit metadata image count mismatch")
    return marker


def manifest_path(shard_id: int) -> Path:
    return OUT / f"mappings/shards/shard_{shard_id:05d}.parquet"


def load_manifest(shard_id: int) -> pa.Table:
    return pq.read_table(manifest_path(shard_id))


def manifest_shard_count() -> int:
    metadata = json.loads((OUT / "audit/manifest_metadata.json").read_text())
    return int(metadata["shard_count"])

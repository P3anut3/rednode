#!/usr/bin/env python3
"""Phase 8 frozen image feature extraction orchestration."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.phase_08.experiment_01_image_embedding_extraction.data import (  # noqa: E402
    EXPECTED_ITEMS, NOTE_IDS_PATH, OUT, SHARD_SIZE, build_audit_and_manifests,
    canonical_note_ids, manifest_path, save_json_atomic,
    sha256_file, validate_audit_complete,
)
from experiments.phase_08.experiment_01_image_embedding_extraction.extractor import (  # noqa: E402
    FORMAL_INFERENCE_PROTOCOL, OUTPUT_DIM, embedding_path, extract_shard, hash_string_sequence,
    load_siglip, marker_path, run_smoke, valid_path, validate_completed_shard,
    validate_extraction_chain, validate_shard_against_smoke,
)
from experiments.phase_08.experiment_01_image_embedding_extraction.pooling import (  # noqa: E402
    STRATEGIES, ShardReader, generate_pooling, output_directory, pool_vectors,
    validate_pooling,
)

STAGES = (
    "plan", "audit", "smoke", "extract", "finalize-extraction",
    "pool-first", "pool-top3", "pool-all", "validate", "report",
)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_shard_lock(
    lock_path: Path,
    worker_rank: int,
    recover_stale: bool = False,
    recovery_root: Path | None = None,
) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        if not recover_stale:
            raise RuntimeError(f"shard lock exists; explicit --recover-stale-locks is required: {lock_path}")
        try:
            owner = json.loads(lock_path.read_text())
        except Exception as exc:
            raise RuntimeError(f"cannot audit malformed stale lock: {lock_path}") from exc
        if owner.get("hostname") != socket.gethostname():
            raise RuntimeError("stale lock belongs to another hostname and cannot be safely recovered")
        if _pid_is_alive(int(owner["pid"])):
            raise RuntimeError(f"refusing to remove active shard lock owned by PID {owner['pid']}")
        recovery_base = recovery_root or (OUT / "diagnostics/stale_locks")
        recovery = recovery_base / f"{lock_path.name}.{int(time.time())}.json"
        recovery.parent.mkdir(parents=True, exist_ok=True)
        lock_path.replace(recovery)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"shard is already owned by another worker: {lock_path}") from exc
    payload = {
        "pid": os.getpid(), "hostname": socket.gethostname(),
        "worker_rank": worker_rank, "created_unix": time.time(),
    }
    os.write(descriptor, json.dumps(payload).encode())
    os.fsync(descriptor)
    return descriptor


def require_confirm(args) -> None:
    if args.stage != "plan" and not args.confirm_run:
        raise SystemExit(f"--stage {args.stage} writes/validates formal artifacts; pass --confirm-run")


def plan() -> None:
    print(json.dumps({
        "experiment": "phase_08/experiment_01_image_embedding_extraction",
        "stages": list(STAGES),
        "default_action": "plan only; no image/model scan and no artifact writes",
        "canonical_note_ids": str(NOTE_IDS_PATH),
        "manifest_shard_images": SHARD_SIZE,
        "model": "google/siglip-base-patch16-224 (local_files_only=True)",
        "formal_inference": "CUDA only",
        "recommendation_test_opened": False,
    }, indent=2, ensure_ascii=False))


def disk_preflight(image_count: int) -> dict:
    image_bytes = image_count * OUTPUT_DIM * 2
    valid_bytes = image_count
    pooled_bytes = EXPECTED_ITEMS * OUTPUT_DIM * 2 * 3
    pooled_masks = EXPECTED_ITEMS * 3
    manifest_bytes = sum(path.stat().st_size for path in (OUT / "mappings/shards").glob("shard_*.parquet"))
    existing_image_bytes = sum(path.stat().st_size for path in (OUT / "image_embeddings/siglip_base_patch16_224/shards").glob("*.npy"))
    existing_valid_bytes = sum(path.stat().st_size for path in (OUT / "diagnostics/decode_status").glob("*.npy"))
    existing_pooled_bytes = sum(path.stat().st_size for path in (OUT / "pooled_embeddings").glob("*/embeddings.f16.npy"))
    existing_pooled_masks = sum(path.stat().st_size for path in (OUT / "pooled_embeddings").glob("*/image_available.npy"))
    remaining = (
        max(0, image_bytes - existing_image_bytes)
        + max(0, valid_bytes - existing_valid_bytes)
        + max(0, pooled_bytes - existing_pooled_bytes)
        + max(0, pooled_masks - existing_pooled_masks)
    )
    minimum = int(remaining * 1.25 + SHARD_SIZE * OUTPUT_DIM * 2)
    usage = shutil.disk_usage(OUT.parent if OUT.parent.exists() else OUT.parents[2])
    result = {
        "image_embedding_bytes": image_bytes, "valid_mask_bytes": valid_bytes,
        "three_pooled_embedding_bytes": pooled_bytes, "three_pooled_mask_bytes": pooled_masks,
        "manifest_bytes": manifest_bytes,
        "existing_formal_bytes": existing_image_bytes + existing_valid_bytes + existing_pooled_bytes + existing_pooled_masks,
        "remaining_estimated_bytes": remaining,
        "minimum_free_bytes_with_safety": minimum, "free_bytes": usage.free,
    }
    if usage.free < minimum:
        raise RuntimeError(f"insufficient disk space: {result}")
    return result


def smoke(args) -> dict:
    validate_audit_complete()
    audit = json.loads((OUT / "audit/audit.json").read_text())
    if audit["unsafe_path_count"]:
        raise RuntimeError("unsafe image paths found; formal extraction is blocked")
    sample = pq.read_table(OUT / "audit/smoke_sample.parquet")
    if len(sample) != 10_000:
        raise RuntimeError(f"smoke sample must contain 10,000 images, got {len(sample)}")
    batch_size = args.batch_size
    while batch_size >= 1:
        try:
            before_fds = len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else None
            autocast_result, autocast_vectors, autocast_valid = run_smoke(
                sample, args.device, batch_size, args.num_workers,
                precision="autocast", return_vectors=True,
                prefetch_factor=args.prefetch_factor,
                min_host_available_gb=args.min_host_available_gb,
                max_process_tree_rss_gb=args.max_process_tree_rss_gb,
            )
            fds_after_autocast = len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else None
            result = autocast_result
            if args.smoke_autocast_only:
                result["precision_comparison"] = {
                    "fp32": None,
                    "autocast": {key: autocast_result[key] for key in (
                        "elapsed_seconds", "images_per_second", "gpu_peak_memory_bytes",
                        "valid_count", "invalid_count",
                    )},
                    "not_run_reason": "resource-safety benchmark; FP32 agreement was established by an earlier 10k smoke",
                }
            else:
                fp32_result, fp32_vectors, fp32_valid = run_smoke(
                    sample, args.device, batch_size, args.num_workers,
                    precision="fp32", return_vectors=True,
                    prefetch_factor=args.prefetch_factor,
                    min_host_available_gb=args.min_host_available_gb,
                    max_process_tree_rss_gb=args.max_process_tree_rss_gb,
                )
                if not np.array_equal(autocast_valid, fp32_valid):
                    raise RuntimeError("FP32/autocast smoke produced different valid-image masks")
                gc.collect()
                common = autocast_valid & fp32_valid
                cosine = np.sum(
                    autocast_vectors[common].astype(np.float32) * fp32_vectors[common].astype(np.float32),
                    axis=1,
                )
                result["precision_comparison"] = {
                    "fp32": fp32_result,
                    "autocast": {key: autocast_result[key] for key in (
                        "elapsed_seconds", "images_per_second", "gpu_peak_memory_bytes",
                        "valid_count", "invalid_count",
                    )},
                    "common_valid_count": int(common.sum()),
                    "embedding_cosine_mean": float(cosine.mean()) if len(cosine) else None,
                    "embedding_cosine_p01": float(np.percentile(cosine, 1)) if len(cosine) else None,
                    "embedding_cosine_min": float(cosine.min()) if len(cosine) else None,
                }
            after_fds = len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else None
            result["file_descriptors_before"] = before_fds
            result["file_descriptors_after_autocast"] = fds_after_autocast
            result["file_descriptors_after"] = after_fds
            result["file_descriptor_delta"] = None if before_fds is None else after_fds - before_fds
            result["steady_state_file_descriptor_delta"] = (
                None if fds_after_autocast is None else after_fds - fds_after_autocast
            )
            if result["steady_state_file_descriptor_delta"] is not None and abs(result["steady_state_file_descriptor_delta"]) > 4:
                raise RuntimeError(
                    "DataLoader file descriptors continue to grow after CUDA/IPC initialization: "
                    f"{result['steady_state_file_descriptor_delta']}"
                )
            result["disk_preflight"] = disk_preflight(int(audit["total_image_paths"]))
            throughput = result["images_per_second"]
            result["estimated_first_image_seconds"] = int(audit["items_with_images"]) / max(throughput, 1e-9)
            result["estimated_all_image_seconds"] = int(audit["total_image_paths"]) / max(throughput, 1e-9)
            result["batch_size_reductions"] = int(np.log2(args.batch_size // batch_size)) if batch_size else 0
            result["selected_batch_size"] = batch_size
            result["audit_complete_sha256"] = sha256_file(OUT / "markers/audit_complete.json")
            result["smoke_sample_sha256"] = sha256_file(OUT / "audit/smoke_sample.parquet")
            result["model_fingerprint"] = result["model"]["model_fingerprint"]
            result["inference_protocol"] = FORMAL_INFERENCE_PROTOCOL
            save_json_atomic(OUT / "smoke/smoke_result.json", result)
            save_json_atomic(
                OUT / f"smoke/benchmark_bs{batch_size}_w{args.num_workers}_pf{args.prefetch_factor}.json",
                result,
            )
            return result
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size //= 2
            print(f"CUDA OOM; retrying smoke with batch_size={batch_size}", flush=True)
    raise RuntimeError("smoke could not fit on GPU even with batch_size=1")


def validate_smoke_contract(
    result: dict,
    audit_marker_path: Path,
    smoke_sample_path: Path,
    expected_model_fingerprint: str | None = None,
) -> dict:
    if not result.get("complete") or result.get("sample_count") != 10_000:
        raise RuntimeError("smoke completion contract failed")
    expected = {
        "audit_complete_sha256": sha256_file(audit_marker_path),
        "smoke_sample_sha256": sha256_file(smoke_sample_path),
        "inference_protocol": FORMAL_INFERENCE_PROTOCOL,
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise RuntimeError(f"smoke is stale or incompatible with current formal inputs: {field}")
    if result.get("model_fingerprint") != result.get("model", {}).get("model_fingerprint"):
        raise RuntimeError("smoke model fingerprint contract failed")
    if expected_model_fingerprint is not None and result["model_fingerprint"] != expected_model_fingerprint:
        raise RuntimeError("formal extraction model fingerprint differs from smoke")
    if int(result.get("selected_batch_size", 0)) < 1 or int(result["selected_batch_size"]) != int(result["batch_size"]):
        raise RuntimeError("smoke selected batch-size contract failed")
    return result


def require_smoke() -> dict:
    path = OUT / "smoke/smoke_result.json"
    if not path.exists():
        raise RuntimeError("successful 10k smoke is required before formal extraction")
    return validate_smoke_contract(
        json.loads(path.read_text()),
        OUT / "markers/audit_complete.json",
        OUT / "audit/smoke_sample.parquet",
    )


def extract(args) -> None:
    validate_audit_complete()
    smoke_result = require_smoke()
    effective_batch_size = min(args.batch_size, int(smoke_result["batch_size"]))
    metadata = json.loads((OUT / "audit/manifest_metadata.json").read_text())
    disk_preflight(int(metadata["image_count"]))
    if args.world_size < 1 or not (0 <= args.worker_rank < args.world_size):
        raise ValueError("worker-rank must satisfy 0 <= rank < world-size")
    log_path = OUT / f"diagnostics/worker_logs/worker_{args.worker_rank:02d}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    assigned = [shard_id for shard_id in range(int(metadata["shard_count"])) if shard_id % args.world_size == args.worker_rank]
    if args.max_shards is not None:
        if args.max_shards < 1:
            raise ValueError("--max-shards must be >= 1")
        assigned = assigned[:args.max_shards]
    pending = [shard_id for shard_id in assigned if args.overwrite_debug or not marker_path(shard_id).exists()]
    runtime = load_siglip(args.device) if pending else None
    if runtime is not None:
        validate_smoke_contract(
            smoke_result,
            OUT / "markers/audit_complete.json",
            OUT / "audit/smoke_sample.parquet",
            expected_model_fingerprint=runtime[2]["model_fingerprint"],
        )
    for shard_id in assigned:
        lock_path = OUT / f"markers/.extract_shard_{shard_id:05d}.lock"
        descriptor = acquire_shard_lock(lock_path, args.worker_rank, args.recover_stale_locks)
        try:
            result = extract_shard(
                shard_id, args.device, effective_batch_size, args.num_workers,
                args.overwrite_debug, runtime=runtime,
                prefetch_factor=args.prefetch_factor,
                min_host_available_gb=args.min_host_available_gb,
                max_process_tree_rss_gb=args.max_process_tree_rss_gb,
            )
            validate_shard_against_smoke(result, smoke_result)
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)
        with log_path.open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"worker {args.worker_rank}: completed/verified shard {shard_id}", flush=True)


def finalize_extraction() -> dict:
    validate_audit_complete()
    smoke_result = require_smoke()
    metadata = json.loads((OUT / "audit/manifest_metadata.json").read_text())
    total_rows = valid_count = invalid_count = 0
    global_cursor = 0
    shard_hashes, valid_hashes, manifest_hashes, failure_hashes, marker_hashes = [], [], [], [], []
    model_snapshots = set()
    model_fingerprints = set()
    norm_samples = []
    offsets = np.load(OUT / "mappings/image_offsets.npy", mmap_mode="r")
    note_ids = np.load(OUT / "mappings/note_ids.npy", mmap_mode="r")
    for shard_id in range(int(metadata["shard_count"])):
        marker = validate_completed_shard(shard_id)
        validate_shard_against_smoke(marker, smoke_result)
        model_snapshots.add(marker["model_snapshot"])
        model_fingerprints.add(marker["model_fingerprint"])
        if marker["global_start"] != global_cursor or marker["global_end"] - marker["global_start"] != marker["row_count"]:
            raise RuntimeError(f"non-contiguous extraction shard {shard_id}")
        embedding = np.load(embedding_path(shard_id), mmap_mode="r")
        valid = np.load(valid_path(shard_id), mmap_mode="r")
        manifest = pq.read_table(manifest_path(shard_id), columns=["global_image_row", "note_row", "note_idx", "position"])
        columns = manifest.to_pydict()
        note_rows = np.asarray(columns["note_row"], dtype=np.int64)
        positions = np.asarray(columns["position"], dtype=np.int64)
        global_rows = np.asarray(columns["global_image_row"], dtype=np.int64)
        manifest_note_ids = np.asarray(columns["note_idx"], dtype=np.int64)
        if (
            np.any(note_rows < 0) or np.any(note_rows >= len(note_ids))
            or not np.array_equal(offsets[note_rows] + positions, global_rows)
            or not np.array_equal(note_ids[note_rows], manifest_note_ids)
        ):
            raise RuntimeError(f"manifest/CSR/note mapping mismatch in shard {shard_id}")
        for start in range(0, len(embedding), 10_000):
            values = np.asarray(embedding[start:start + 10_000], dtype=np.float32)
            masks = np.asarray(valid[start:start + 10_000])
            if not np.isfinite(values).all() or np.any(values[~masks] != 0):
                raise RuntimeError(f"finite/invalid-zero contract failed in shard {shard_id}")
        rows = np.flatnonzero(valid)[::max(1, int(valid.sum()) // 200)][:200]
        if len(rows):
            norm_samples.extend(np.linalg.norm(np.asarray(embedding[rows], dtype=np.float32), axis=1).tolist())
        total_rows += marker["row_count"]
        valid_count += marker["valid_count"]
        invalid_count += marker["invalid_count"]
        global_cursor = marker["global_end"]
        shard_hashes.append(marker["embedding_sha256"])
        valid_hashes.append(marker["valid_mask_sha256"])
        manifest_hashes.append(marker["manifest_sha256"])
        failure_hashes.append(marker["decode_failure_sha256"])
        marker_hashes.append(sha256_file(marker_path(shard_id)))
    if total_rows != int(offsets[-1]) or total_rows != int(metadata["image_count"]):
        raise RuntimeError("total extraction rows do not match CSR offsets/audit")
    if len(model_snapshots) != 1:
        raise RuntimeError(f"multiple model snapshots found across shards: {sorted(model_snapshots)}")
    if len(model_fingerprints) != 1:
        raise RuntimeError("multiple model fingerprints found across shards")
    if not np.allclose(norm_samples, 1.0, atol=2e-3):
        raise RuntimeError("valid extraction norm sample is not near one")
    summary = {
        "complete": True, "shard_count": int(metadata["shard_count"]),
        "image_count": total_rows, "valid_count": valid_count, "invalid_count": invalid_count,
        "global_start": 0, "global_end": global_cursor,
        "norm_sample_count": len(norm_samples), "norm_mean": float(np.mean(norm_samples)),
        "canonical_note_ids_sha256": sha256_file(NOTE_IDS_PATH),
        "image_offsets_sha256": sha256_file(OUT / "mappings/image_offsets.npy"),
        "audit_complete_sha256": sha256_file(OUT / "markers/audit_complete.json"),
        "shard_marker_digest": hash_string_sequence(marker_hashes),
        "shard_embedding_hashes_sha256": hash_string_sequence(shard_hashes),
        "shard_valid_hashes_sha256": hash_string_sequence(valid_hashes),
        "shard_manifest_hashes_sha256": hash_string_sequence(manifest_hashes),
        "shard_decode_failure_hashes_sha256": hash_string_sequence(failure_hashes),
        "model_snapshot": next(iter(model_snapshots)),
        "model_fingerprint": next(iter(model_fingerprints)),
        "smoke_result_sha256": sha256_file(OUT / "smoke/smoke_result.json"),
        "smoke_model_fingerprint": smoke_result["model_fingerprint"],
        "smoke_inference_protocol": smoke_result["inference_protocol"],
        "test_opened": False,
    }
    save_json_atomic(OUT / "diagnostics/extraction_summary.json", summary)
    save_json_atomic(OUT / "markers/extraction_complete.json", summary)
    validate_extraction_chain()
    return summary


def validate_all(seed: int = 42) -> dict:
    extraction = validate_extraction_chain()
    note_ids = np.load(OUT / "mappings/note_ids.npy", mmap_mode="r")
    canonical = canonical_note_ids()
    offsets = np.load(OUT / "mappings/image_offsets.npy", mmap_mode="r")
    if note_ids.shape != (EXPECTED_ITEMS,) or not np.array_equal(note_ids, canonical):
        raise RuntimeError("Phase-8 note IDs differ from canonical mapping")
    if offsets.shape != (EXPECTED_ITEMS + 1,) or offsets.dtype != np.int64 or np.any(np.diff(offsets) < 0):
        raise RuntimeError("invalid CSR image offsets")
    if int(offsets[-1]) != extraction["image_count"]:
        raise RuntimeError("CSR/extraction image count mismatch")
    pool_metadata = {strategy: validate_pooling(strategy) for strategy in STRATEGIES}
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(np.diff(offsets) > 0)
    sample_items = rng.choice(candidates, size=min(1_000, len(candidates)), replace=False)
    reader = ShardReader()
    for strategy in STRATEGIES:
        pooled = np.load(output_directory(strategy) / "embeddings.f16.npy", mmap_mode="r")
        mask = np.load(output_directory(strategy) / "image_available.npy", mmap_mode="r")
        for row in sample_items:
            vectors, valid = reader.range(int(offsets[row]), int(offsets[row + 1]))
            expected, present, _ = pool_vectors(vectors, valid, strategy)
            if bool(mask[row]) != present or not np.allclose(np.asarray(pooled[row], np.float32), expected, atol=2e-3):
                raise RuntimeError(f"pooled recomputation mismatch strategy={strategy}, note_row={row}")
    result = {
        "complete": True, "item_count": len(note_ids), "image_count": int(offsets[-1]),
        "canonical_mapping_sha256": sha256_file(OUT / "mappings/note_ids.npy"),
        "pooling": pool_metadata, "recomputed_items_per_strategy": len(sample_items),
        "test_opened": False,
    }
    save_json_atomic(OUT / "diagnostics/validation.json", result)
    save_json_atomic(OUT / "markers/validation_complete.json", result)
    return result


def report() -> str:
    if not (OUT / "markers/validation_complete.json").exists():
        raise RuntimeError("validate must complete before report")
    audit = json.loads((OUT / "audit/audit.json").read_text())
    smoke_result = json.loads((OUT / "smoke/smoke_result.json").read_text())
    extraction = json.loads((OUT / "diagnostics/extraction_summary.json").read_text())
    pools = {key: json.loads((output_directory(key) / "metadata.json").read_text()) for key in STRATEGIES}
    markers = [json.loads(path.read_text()) for path in sorted((OUT / "markers").glob("extract_shard_*.json"))]
    embedding_size = sum(embedding_path(marker["shard_id"]).stat().st_size for marker in markers)
    worker_elapsed = []
    for path in sorted((OUT / "diagnostics/worker_logs").glob("worker_*.jsonl")):
        worker_elapsed.append(sum(json.loads(line)["elapsed_seconds"] for line in path.read_text().splitlines() if line.strip()))
    extraction_seconds = max(worker_elapsed, default=max((marker["elapsed_seconds"] for marker in markers), default=0.0))
    sum_gpu_seconds = sum(marker["elapsed_seconds"] for marker in markers)
    precision = smoke_result.get("precision_comparison", {})
    if precision.get("fp32"):
        precision_line = (
            f"- FP32 / autocast 吞吐：{precision['fp32']['images_per_second']:.2f} / "
            f"{precision['autocast']['images_per_second']:.2f} images/s；"
            f"embedding cosine mean={precision['embedding_cosine_mean']:.8f}。"
        )
    else:
        precision_line = "- 本次安全 smoke 仅运行正式 autocast 路径；FP32 一致性见既有 benchmark。"
    lines = [
        "# Phase 8 Experiment 01：冻结图片 Embedding 基础设施", "",
        "> 本实验未读取 recommendation test 标签；`test_opened=false`。", "",
        "## 数据审计", "",
        f"- Item：{audit['item_count']:,}",
        f"- 有图 / 无图：{audit['items_with_images']:,} / {audit['items_without_images']:,}",
        f"- 图片路径：{audit['total_image_paths']:,}",
        f"- 图片数分布：`{json.dumps(audit['path_count_distribution'], ensure_ascii=False)}`",
        f"- part 目录：{audit['part_directory_count']} 个",
        f"- duplicate occurrence / unsafe：{audit['duplicate_relative_path_occurrences']:,} / {audit['unsafe_path_count']:,}",
        f"- 10k 路径存在率：{audit['random_path_existence_rate']:.6%}", "",
        "## 模型与 Smoke", "",
        f"- 模型：`{smoke_result['model']['model_id']}`，snapshot `{smoke_result['model']['snapshot_commit']}`",
        f"- 输出：{OUTPUT_DIM}d，逐图 float16；归一化在 float32 完成。",
        f"- Smoke：{smoke_result['images_per_second']:.2f} images/s，失败 {smoke_result['invalid_count']:,}。",
        precision_line,
        f"- 抽样实际格式：`{json.dumps(smoke_result['formats'], ensure_ascii=False)}`", "",
        "## 正式提取", "",
        f"- 成功 / 失败：{extraction['valid_count']:,} / {extraction['invalid_count']:,}",
        f"- Shard：{extraction['shard_count']}，逐图 embedding 总大小：{embedding_size / 2**30:.3f} GiB。",
        f"- 四 worker 墙钟近似 / 累计 GPU worker 时间：{extraction_seconds:.1f}s / {sum_gpu_seconds:.1f}s。", "",
        "## Pooling", "",
        "| Strategy | Available | Missing | Fallback | Size (GiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy, metadata in pools.items():
        size = (output_directory(strategy) / "embeddings.f16.npy").stat().st_size / 2**30
        lines.append(f"| {STRATEGIES[strategy]} | {metadata['available_items']:,} | {metadata['missing_items']:,} | {metadata['fallback_count']:,} | {size:.3f} |")
    lines += [
        "", "## 完整性与产物", "",
        "- 每个逐图 shard 均由 marker 校验 shape、dtype、manifest/embedding/mask SHA-256。",
        "- CSR、canonical mapping、逐图 embedding 与三种 pooling 已完成抽样双向复算。",
        f"- 结果目录：`{OUT.relative_to(OUT.parents[2])}`",
        "- `test_opened=false`。", "",
        "## 下一步建议", "",
        "在独立后续实验中将 first/top3/all 作为冻结 item-side 图片分支接入 H2，先做缺图 mask 与小门控消融；本实验不自动执行图文融合训练。", "",
    ]
    text = "\n".join(lines)
    path = OUT / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="plan")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--min-host-available-gb", type=float, default=32.0)
    parser.add_argument("--max-process-tree-rss-gb", type=float, default=32.0)
    parser.add_argument(
        "--smoke-autocast-only", action="store_true",
        help="Run only the formal autocast path for resource/concurrency benchmarking",
    )
    parser.add_argument(
        "--max-shards", type=int,
        help="Safety/canary limit: process at most this many assigned shards",
    )
    parser.add_argument(
        "--recover-audit", action="store_true",
        help="Explicitly discard an incomplete pre-extraction audit build and rebuild it transactionally",
    )
    parser.add_argument(
        "--recover-stale-locks", action="store_true",
        help="Explicitly recover same-host shard locks only after verifying their owner PID is dead",
    )
    parser.add_argument(
        "--recover-pooling", action="store_true",
        help="Explicitly discard an incomplete pooling build/directory; completed pooling remains immutable",
    )
    parser.add_argument(
        "--overwrite-debug", action="store_true",
        help="Explicitly regenerate an assigned shard while preserving its old marker in diagnostics/overwritten_shards",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_confirm(args)
    if args.stage == "plan":
        plan()
    elif args.stage == "audit":
        print(json.dumps(build_audit_and_manifests(recover_audit=args.recover_audit), indent=2, ensure_ascii=False))
    elif args.stage == "smoke":
        print(json.dumps(smoke(args), indent=2, ensure_ascii=False))
    elif args.stage == "extract":
        extract(args)
    elif args.stage == "finalize-extraction":
        print(json.dumps(finalize_extraction(), indent=2, ensure_ascii=False))
    elif args.stage.startswith("pool-"):
        strategy = {"pool-first": "first", "pool-top3": "top3", "pool-all": "all"}[args.stage]
        print(json.dumps(generate_pooling(strategy, recover_incomplete=args.recover_pooling), indent=2, ensure_ascii=False))
    elif args.stage == "validate":
        print(json.dumps(validate_all(), indent=2, ensure_ascii=False))
    elif args.stage == "report":
        print(report())


if __name__ == "__main__":
    main()

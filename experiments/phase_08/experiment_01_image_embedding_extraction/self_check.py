#!/usr/bin/env python3
"""Synthetic-only checks for CSR, ragged shards, pooling and integrity guards."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.phase_08.experiment_01_image_embedding_extraction.data import (  # noqa: E402
    save_json_atomic, save_npy_atomic, sha256_file, validate_relative_path,
    validate_unique_note_ids, prepare_audit_workspace,
)
from experiments.phase_08.experiment_01_image_embedding_extraction.extractor import (  # noqa: E402
    DecodeRecord, FORMAL_INFERENCE_PROTOCOL, ProcessorCollator,
    validate_artifact_contract, validate_shard_against_smoke,
)
from experiments.phase_08.experiment_01_image_embedding_extraction.pooling import (  # noqa: E402
    ShardReader, pool_vectors, prepare_pooling_workspace, require_pooling_source,
)
from experiments.phase_08.experiment_01_image_embedding_extraction.run import (  # noqa: E402
    acquire_shard_lock, validate_smoke_contract,
)


def expect_raises(function, exception=Exception):
    try:
        function()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8_self_check_") as directory:
        root = Path(directory)
        # The multiprocessing collate payload contains processed tensors and
        # scalar metadata only; decoded PIL images must never cross the queue.
        class FakeProcessor:
            def __call__(self, images, return_tensors):
                count = len(images) if isinstance(images, list) else 1
                return {"pixel_values": torch.zeros((count, 3, 4, 4))}

        decoded = Image.new("RGB", (4, 4), "white")
        collated = ProcessorCollator(FakeProcessor())([{
            "local_row": 0, "global_image_row": 10, "note_idx": 20,
            "position": 0, "relative_path": "image/part_0/a.webp",
            "record": DecodeRecord(decoded, None, "WEBP", 4, 4),
        }])
        assert "examples" not in collated and "valid_examples" not in collated
        assert collated["valid_local_rows"] == [0]
        assert tuple(collated["pixel_values"].shape) == (1, 3, 4, 4)
        assert not any(isinstance(value, Image.Image) for value in collated.values())
        # Items: empty, one valid, three with position-0 invalid, and an all-invalid cross-shard item.
        offsets = np.asarray([0, 0, 1, 4, 6], dtype=np.int64)
        assert np.array_equal(np.diff(offsets), [0, 1, 3, 2]) and offsets[-1] == 6
        vectors = np.asarray([
            [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0],
        ], dtype=np.float16)
        valid = np.asarray([1, 0, 1, 1, 0, 0], dtype=np.bool_)
        shard_size = 2
        for shard in range(3):
            save_npy_atomic(root / f"emb_{shard}.npy", vectors[shard * 2:(shard + 1) * 2])
            save_npy_atomic(root / f"valid_{shard}.npy", valid[shard * 2:(shard + 1) * 2])
        reader = ShardReader(
            shard_size=shard_size,
            embedding_fn=lambda shard: root / f"emb_{shard}.npy",
            valid_fn=lambda shard: root / f"valid_{shard}.npy",
        )
        cross_vectors, cross_valid = reader.range(1, 5)
        assert cross_vectors.shape == (4, 3) and cross_valid.tolist() == [False, True, True, False]
        empty, present, _ = pool_vectors(np.empty((0, 3), np.float32), np.empty(0, bool), "all")
        assert not present and np.all(empty == 0)
        one, present, fallback = pool_vectors(vectors[:1], valid[:1], "first")
        assert present and not fallback and np.allclose(one, [1, 0, 0])
        first, present, fallback = pool_vectors(vectors[1:4], valid[1:4], "first")
        assert present and fallback and np.allclose(first, [0, 1, 0])
        top3, present, _ = pool_vectors(vectors[1:4], valid[1:4], "top3")
        expected = np.asarray([0, 1, 1], np.float32) / np.sqrt(2)
        assert present and np.allclose(top3, expected, atol=1e-6)
        all_vector, present, _ = pool_vectors(vectors[1:4], valid[1:4], "all")
        assert present and np.allclose(all_vector, expected, atol=1e-6)
        all_invalid, present, _ = pool_vectors(vectors[4:6], valid[4:6], "all")
        assert not present and np.all(all_invalid == 0)
        # FP32 accumulation must survive values that would overflow float16 summation.
        repeated = np.tile(np.asarray([[0.7, 0.7, 0.0]], np.float16), (100_000, 1))
        pooled, present, _ = pool_vectors(repeated, np.ones(len(repeated), bool), "all")
        assert present and np.isfinite(pooled).all() and np.isclose(np.linalg.norm(pooled), 1.0)
        # Atomic file and marker/hash integrity.
        artifact = root / "artifact.npy"
        save_npy_atomic(artifact, np.arange(10, dtype=np.int64))
        digest = sha256_file(artifact)
        marker = root / "marker.json"
        save_json_atomic(marker, {"complete": True, "sha256": digest})
        assert json.loads(marker.read_text())["sha256"] == sha256_file(artifact)
        with artifact.open("ab") as stream:
            stream.write(b"changed")
        assert sha256_file(artifact) != digest
        # Formal shard contract: every source file is hash-bound, and the model
        # fingerprint is part of the contract.
        contract_dir = root / "contract"
        contract_dir.mkdir()
        contract_embedding = contract_dir / "embedding.npy"
        contract_valid = contract_dir / "valid.npy"
        contract_manifest = contract_dir / "manifest.parquet"
        contract_failure = contract_dir / "failure.parquet"
        save_npy_atomic(contract_embedding, np.ones((2, 3), dtype=np.float16))
        save_npy_atomic(contract_valid, np.asarray([True, True], dtype=np.bool_))
        pq.write_table(pa.table({"row": [0, 1]}), contract_manifest)
        pq.write_table(pa.table({"error": pa.array([], type=pa.string())}), contract_failure)
        paths = {
            "embedding": contract_embedding, "valid": contract_valid,
            "manifest": contract_manifest, "failure": contract_failure,
        }
        contract_marker = {
            "complete": True, "test_opened": False, "model_fingerprint": "model-a",
            "inference_protocol": FORMAL_INFERENCE_PROTOCOL,
            "embedding_shape": [2, 3], "embedding_sha256": sha256_file(contract_embedding),
            "valid_mask_sha256": sha256_file(contract_valid),
            "manifest_sha256": sha256_file(contract_manifest),
            "decode_failure_sha256": sha256_file(contract_failure),
        }
        validate_artifact_contract(contract_marker, paths, expected_model_fingerprint="model-a")
        matching_smoke = {
            "model_fingerprint": "model-a", "inference_protocol": FORMAL_INFERENCE_PROTOCOL,
        }
        validate_shard_against_smoke(contract_marker, matching_smoke)
        expect_raises(
            lambda: validate_shard_against_smoke(contract_marker, {**matching_smoke, "model_fingerprint": "model-b"}),
            RuntimeError,
        )
        expect_raises(
            lambda: validate_shard_against_smoke(contract_marker, {**matching_smoke, "inference_protocol": "other"}),
            RuntimeError,
        )
        expect_raises(
            lambda: validate_artifact_contract(contract_marker, paths, expected_model_fingerprint="model-b"),
            RuntimeError,
        )
        for key in ("embedding", "valid", "manifest", "failure"):
            original = paths[key].read_bytes()
            paths[key].write_bytes(original + b"corrupt")
            expect_raises(lambda: validate_artifact_contract(contract_marker, paths), Exception)
            paths[key].write_bytes(original)
        # Pooling must execute source validation before touching derived output.
        called = []
        require_pooling_source(lambda: called.append(True) or {"complete": True})
        assert called == [True]
        expect_raises(lambda: require_pooling_source(lambda: (_ for _ in ()).throw(RuntimeError("bad source"))), RuntimeError)
        pooling_destination = root / "pooled/mean_top3"
        interrupted = pooling_destination.with_name("mean_top3_building")
        interrupted.mkdir(parents=True)
        (interrupted / "partial").write_text("partial")
        expect_raises(lambda: prepare_pooling_workspace(pooling_destination, False), RuntimeError)
        recovered_build = prepare_pooling_workspace(pooling_destination, True)
        assert recovered_build == interrupted and not (interrupted / "partial").exists()
        # Simulate a crash after a partial formal directory was published.
        import shutil
        shutil.rmtree(interrupted)
        pooling_destination.mkdir(parents=True)
        (pooling_destination / "embeddings.f16.npy").write_bytes(b"partial")
        expect_raises(lambda: prepare_pooling_workspace(pooling_destination, False), RuntimeError)
        recovered_build = prepare_pooling_workspace(pooling_destination, True)
        assert recovered_build == interrupted and not pooling_destination.exists()
        # Interrupted audit builds require explicit recovery and cannot be
        # recovered after extraction has begun.
        audit_root = root / "audit_case"
        (audit_root / "audit_building").mkdir(parents=True)
        (audit_root / "audit_building/partial").write_text("partial")
        expect_raises(lambda: prepare_audit_workspace(audit_root, False), RuntimeError)
        rebuilt = prepare_audit_workspace(audit_root, True)
        assert rebuilt == audit_root / "audit_building" and not (rebuilt / "partial").exists()
        (rebuilt / "partial").write_text("again")
        (audit_root / "markers").mkdir(parents=True)
        (audit_root / "markers/extract_shard_00000.json").write_text("{}")
        expect_raises(lambda: prepare_audit_workspace(audit_root, True), RuntimeError)
        # Smoke is bound to both the current audit/sample bytes and model.
        smoke_audit = root / "smoke_audit.json"
        smoke_sample = root / "smoke_sample.parquet"
        smoke_audit.write_text("audit-v1")
        smoke_sample.write_text("sample-v1")
        smoke_contract = {
            "complete": True, "sample_count": 10_000, "batch_size": 64,
            "selected_batch_size": 64, "inference_protocol": FORMAL_INFERENCE_PROTOCOL,
            "audit_complete_sha256": sha256_file(smoke_audit),
            "smoke_sample_sha256": sha256_file(smoke_sample),
            "model_fingerprint": "model-a", "model": {"model_fingerprint": "model-a"},
        }
        validate_smoke_contract(smoke_contract, smoke_audit, smoke_sample, "model-a")
        smoke_sample.write_text("sample-v2")
        expect_raises(lambda: validate_smoke_contract(smoke_contract, smoke_audit, smoke_sample), RuntimeError)
        smoke_sample.write_text("sample-v1")
        expect_raises(lambda: validate_smoke_contract(smoke_contract, smoke_audit, smoke_sample, "model-b"), RuntimeError)
        # Stale lock recovery is explicit, same-host and dead-PID only.
        lock = root / "lock_case/.extract.lock"
        lock.parent.mkdir()
        lock.write_text(json.dumps({"pid": 999_999_999, "hostname": socket.gethostname(), "worker_rank": 0, "created_unix": 0}))
        descriptor = acquire_shard_lock(lock, worker_rank=1, recover_stale=True, recovery_root=root / "recovered_locks")
        os.close(descriptor)
        lock.unlink()
        lock.write_text(json.dumps({"pid": os.getpid(), "hostname": socket.gethostname(), "worker_rank": 0, "created_unix": 0}))
        expect_raises(lambda: acquire_shard_lock(lock, worker_rank=1, recover_stale=True), RuntimeError)
        # A failed replacement preparation never touches the old formal file.
        old_asset = root / "old_asset.npy"
        save_npy_atomic(old_asset, np.asarray([7], dtype=np.int64))
        old_hash = sha256_file(old_asset)
        bad_candidate = root / "bad_candidate.npy"
        bad_candidate.write_bytes(b"not-a-npy")
        expect_raises(lambda: np.load(bad_candidate, allow_pickle=False), Exception)
        assert sha256_file(old_asset) == old_hash
        # Path safety and duplicate-ID rejection.
        assert validate_relative_path("image/part_1/a.webp")[1]
        assert not validate_relative_path("../escape.jpg")[1]
        assert not validate_relative_path("/absolute.jpg")[1]
        expect_raises(lambda: validate_unique_note_ids([1, 2, 2]), ValueError)
        validate_unique_note_ids([1, 2, 3])
    print("Phase 8 synthetic self-check: PASS (no full images or recommendation labels opened)")


if __name__ == "__main__":
    main()

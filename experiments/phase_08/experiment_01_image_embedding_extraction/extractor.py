"""Frozen SigLIP image extraction with resumable, hash-verified shards."""

from __future__ import annotations

import json
import hashlib
import gc
import os
import platform
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import psutil
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from .data import (
    OUT, absolute_image_path, load_manifest, manifest_path,
    save_json_atomic, sha256_file, validate_audit_complete,
)

MODEL_ID = "google/siglip-base-patch16-224"
OUTPUT_DIM = 768
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 0
DEFAULT_PREFETCH_FACTOR = 1
FORMAL_INFERENCE_PROTOCOL = "siglip_vision_cuda_autocast_fp16_pooler_fp32_l2_f16_storage_v1"


@dataclass
class DecodeRecord:
    image: Image.Image | None
    error_type: str | None
    actual_format: str | None
    width: int | None
    height: int | None


class ImageManifestDataset(Dataset):
    def __init__(self, manifest: pa.Table):
        self.rows = manifest.to_pydict()

    def __len__(self) -> int:
        return len(self.rows["global_image_row"])

    def __getitem__(self, index: int) -> dict:
        relative = self.rows["relative_path"][index]
        try:
            with Image.open(absolute_image_path(relative)) as source:
                actual_format = str(source.format or "UNKNOWN")
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
            record = DecodeRecord(image, None, actual_format, image.width, image.height)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            record = DecodeRecord(None, type(exc).__name__, None, None, None)
        return {
            "local_row": index,
            "global_image_row": int(self.rows["global_image_row"][index]),
            "note_idx": int(self.rows["note_idx"][index]),
            "position": int(self.rows["position"][index]),
            "relative_path": relative,
            "record": record,
        }


class ProcessorCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: list[dict]) -> dict:
        # The collate function runs inside the DataLoader worker.  Never return
        # `examples` or DecodeRecord: doing so serializes the decoded PIL images
        # back to the parent in addition to pixel_values and can exhaust host
        # memory/shared memory at large batch/worker counts.
        valid_indices = [index for index, example in enumerate(examples) if example["record"].image is not None]
        valid_examples = [examples[index] for index in valid_indices]
        pixel_values = None
        errors = [example["record"].error_type for example in examples]
        if valid_examples:
            try:
                pixel_values = self.processor(
                    images=[example["record"].image for example in valid_examples],
                    return_tensors="pt",
                )["pixel_values"]
            except Exception:
                tensors, retained, retained_indices = [], [], []
                for example_index, example in zip(valid_indices, valid_examples):
                    try:
                        tensors.append(self.processor(images=example["record"].image, return_tensors="pt")["pixel_values"][0])
                        retained.append(example)
                        retained_indices.append(example_index)
                    except Exception as exc:  # processor exception types are model/version specific
                        errors[example_index] = type(exc).__name__
                valid_examples = retained
                valid_indices = retained_indices
                pixel_values = torch.stack(tensors) if tensors else None
        batch = {
            "local_rows": [int(example["local_row"]) for example in examples],
            "global_image_rows": [int(example["global_image_row"]) for example in examples],
            "actual_formats": [example["record"].actual_format for example in examples],
            "errors": errors,
            "valid_local_rows": [int(example["local_row"]) for example in valid_examples],
            "pixel_values": pixel_values,
        }
        # Ensure decoded images are not retained by the object returned through
        # the multiprocessing queue.
        del valid_examples, examples
        return batch


class ResourceMonitor:
    """Track parent/worker RSS and host pressure without a monitoring thread."""

    def __init__(self, min_available_gb: float, max_process_tree_rss_gb: float):
        self.process = psutil.Process(os.getpid())
        self.min_available_bytes = int(min_available_gb * 2**30)
        self.max_process_tree_rss_bytes = int(max_process_tree_rss_gb * 2**30)
        self.peak_process_tree_rss_bytes = 0
        self.minimum_host_available_bytes = 2**63 - 1
        self.peak_swap_used_bytes = 0
        self.maximum_child_processes = 0
        self.last = {}

    def sample(self, enforce: bool = True) -> dict:
        children = [child for child in self.process.children(recursive=True) if child.is_running()]
        processes = [self.process, *children]
        rss = 0
        for process in processes:
            try:
                rss += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        self.peak_process_tree_rss_bytes = max(self.peak_process_tree_rss_bytes, rss)
        self.minimum_host_available_bytes = min(self.minimum_host_available_bytes, int(virtual.available))
        self.peak_swap_used_bytes = max(self.peak_swap_used_bytes, int(swap.used))
        self.maximum_child_processes = max(self.maximum_child_processes, len(children))
        self.last = {
            "process_tree_rss_bytes": rss,
            "host_available_bytes": int(virtual.available),
            "swap_used_bytes": int(swap.used),
            "child_processes": len(children),
        }
        if enforce and int(virtual.available) < self.min_available_bytes:
            raise MemoryError(
                f"host available memory below safety floor: {virtual.available / 2**30:.2f} GiB "
                f"< {self.min_available_bytes / 2**30:.2f} GiB"
            )
        if enforce and rss > self.max_process_tree_rss_bytes:
            raise MemoryError(
                f"extractor process-tree RSS exceeded safety ceiling: {rss / 2**30:.2f} GiB "
                f"> {self.max_process_tree_rss_bytes / 2**30:.2f} GiB"
            )
        return self.last

    def summary(self) -> dict:
        return {
            "peak_process_tree_rss_bytes": self.peak_process_tree_rss_bytes,
            "minimum_host_available_bytes": self.minimum_host_available_bytes,
            "peak_swap_used_bytes": self.peak_swap_used_bytes,
            "maximum_child_processes": self.maximum_child_processes,
            "final": self.last,
        }


def shutdown_loader(loader: DataLoader, iterator=None) -> None:
    """Close worker queues deterministically, including exceptional exits."""
    candidates = [iterator, getattr(loader, "_iterator", None)]
    seen = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        shutdown = getattr(candidate, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    if hasattr(loader, "_iterator"):
        loader._iterator = None
    gc.collect()


def configure_worker_sharing() -> None:
    # The default file_descriptor strategy retains many shared-memory FDs after
    # each short-lived shard loader. file_system avoids per-shard FD growth.
    torch.multiprocessing.set_sharing_strategy("file_system")


def _local_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, local_files_only=True))


def _model_metadata(snapshot: Path, processor, model) -> dict:
    import PIL
    import transformers

    files = {}
    for pattern in ("config.json", "preprocessor_config.json", "model.safetensors", "model-*.safetensors", "model.safetensors.index.json"):
        for path in sorted(snapshot.glob(pattern)):
            files[path.name] = sha256_file(path)
    if not any(name.endswith(".safetensors") for name in files):
        raise FileNotFoundError("local SigLIP snapshot has no model.safetensors weights")
    image_size = getattr(getattr(model.config, "vision_config", model.config), "image_size", None)
    metadata = {
        "model_id": MODEL_ID,
        "snapshot_path": str(snapshot),
        "snapshot_commit": snapshot.name,
        "file_sha256": files,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "pillow_version": PIL.__version__,
        "python_version": platform.python_version(),
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "input_resolution": image_size,
        "output_dim": OUTPUT_DIM,
        "normalization": "float32 L2 before float16 storage",
        "local_files_only": True,
    }
    fingerprint_payload = {
        "model_id": metadata["model_id"],
        "file_sha256": metadata["file_sha256"],
        "transformers_version": metadata["transformers_version"],
        "processor_class": metadata["processor_class"],
        "model_class": metadata["model_class"],
        "input_resolution": metadata["input_resolution"],
        "output_dim": metadata["output_dim"],
        "normalization": metadata["normalization"],
    }
    metadata["model_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return metadata


def load_siglip(device: str):
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("formal SigLIP extraction requires CUDA; CPU fallback is forbidden")
    from transformers import SiglipImageProcessor, SiglipVisionModel

    snapshot = _local_snapshot()
    # Image-only processor: do not instantiate SigLIP's tokenizer/text tower.
    processor = SiglipImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model, loading = SiglipVisionModel.from_pretrained(
        snapshot, local_files_only=True, output_loading_info=True,
    )
    missing = loading.get("missing_keys", [])
    unexpected = loading.get("unexpected_keys", [])
    mismatched = loading.get("mismatched_keys", [])
    if missing or mismatched:
        raise RuntimeError(f"SigLIP vision weights incomplete: missing={missing[:10]}, mismatched={mismatched[:10]}")
    model.requires_grad_(False).eval().to(device)
    metadata = _model_metadata(snapshot, processor, model)
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    metadata["gpu_name"] = properties.name
    metadata["gpu_total_memory_bytes"] = int(properties.total_memory)
    metadata["cuda_device"] = str(device)
    metadata["missing_keys"] = missing
    metadata["unexpected_keys"] = unexpected
    metadata["mismatched_keys"] = mismatched
    return processor, model, metadata


def embedding_path(shard_id: int) -> Path:
    return OUT / f"image_embeddings/siglip_base_patch16_224/shards/shard_{shard_id:05d}.f16.npy"


def valid_path(shard_id: int) -> Path:
    return OUT / f"diagnostics/decode_status/shard_{shard_id:05d}.valid.npy"


def marker_path(shard_id: int) -> Path:
    return OUT / f"markers/extract_shard_{shard_id:05d}.json"


def failure_path(shard_id: int) -> Path:
    return OUT / f"diagnostics/decode_failures/shard_{shard_id:05d}.parquet"


def hash_string_sequence(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def validate_shard_against_smoke(marker: dict, smoke_result: dict) -> None:
    if marker.get("model_fingerprint") != smoke_result.get("model_fingerprint"):
        raise RuntimeError("completed shard model fingerprint differs from current smoke")
    if marker.get("inference_protocol") != smoke_result.get("inference_protocol"):
        raise RuntimeError("completed shard inference protocol differs from current smoke")


def validate_artifact_contract(
    marker: dict,
    paths: dict[str, Path],
    strict_hash: bool = True,
    expected_model_fingerprint: str | None = None,
) -> None:
    embedding = np.load(paths["embedding"], mmap_mode="r")
    valid = np.load(paths["valid"], mmap_mode="r")
    if not marker.get("complete") or marker.get("test_opened") is not False:
        raise RuntimeError("invalid shard marker completion contract")
    if not marker.get("model_fingerprint"):
        raise RuntimeError("shard marker has no model fingerprint")
    if marker.get("inference_protocol") != FORMAL_INFERENCE_PROTOCOL:
        raise RuntimeError("shard inference protocol mismatch")
    if expected_model_fingerprint is not None and marker["model_fingerprint"] != expected_model_fingerprint:
        raise RuntimeError("shard model fingerprint mismatch")
    if embedding.shape != tuple(marker["embedding_shape"]) or embedding.dtype != np.float16:
        raise RuntimeError("embedding shape/dtype contract mismatch")
    if valid.shape != (len(embedding),) or valid.dtype != np.bool_:
        raise RuntimeError("valid-mask shape/dtype contract mismatch")
    if strict_hash:
        hash_fields = {
            "embedding": "embedding_sha256", "valid": "valid_mask_sha256",
            "manifest": "manifest_sha256", "failure": "decode_failure_sha256",
        }
        for key, field in hash_fields.items():
            if sha256_file(paths[key]) != marker[field]:
                raise RuntimeError(f"hash mismatch for shard artifact: {paths[key]}")


def validate_completed_shard(shard_id: int, strict_hash: bool = True) -> dict:
    marker_file = marker_path(shard_id)
    if not marker_file.exists():
        raise FileNotFoundError(marker_file)
    marker = json.loads(marker_file.read_text())
    validate_artifact_contract(marker, {
        "embedding": embedding_path(shard_id), "valid": valid_path(shard_id),
        "manifest": manifest_path(shard_id), "failure": failure_path(shard_id),
    }, strict_hash=strict_hash)
    return marker


def validate_extraction_chain() -> dict:
    """Revalidate every immutable source artifact against the global marker."""
    validate_audit_complete()
    complete_path = OUT / "markers/extraction_complete.json"
    if not complete_path.exists():
        raise RuntimeError("extraction_complete.json is missing")
    complete = json.loads(complete_path.read_text())
    if not complete.get("complete") or complete.get("test_opened") is not False:
        raise RuntimeError("invalid global extraction completion marker")
    marker_hashes, embedding_hashes, valid_hashes = [], [], []
    manifest_hashes, failure_hashes, fingerprints = [], [], set()
    total_rows = 0
    smoke_path = OUT / "smoke/smoke_result.json"
    if not smoke_path.exists() or sha256_file(smoke_path) != complete.get("smoke_result_sha256"):
        raise RuntimeError("global extraction marker is not bound to the current smoke result")
    smoke_result = json.loads(smoke_path.read_text())
    if smoke_result.get("audit_complete_sha256") != sha256_file(OUT / "markers/audit_complete.json"):
        raise RuntimeError("smoke result is not bound to the current audit marker")
    if smoke_result.get("smoke_sample_sha256") != sha256_file(OUT / "audit/smoke_sample.parquet"):
        raise RuntimeError("smoke result is not bound to the current smoke sample")
    if complete.get("smoke_model_fingerprint") != smoke_result.get("model_fingerprint"):
        raise RuntimeError("global extraction smoke model fingerprint mismatch")
    if complete.get("smoke_inference_protocol") != smoke_result.get("inference_protocol"):
        raise RuntimeError("global extraction smoke inference protocol mismatch")
    for shard_id in range(int(complete["shard_count"])):
        marker = validate_completed_shard(shard_id)
        validate_shard_against_smoke(marker, smoke_result)
        marker_hashes.append(sha256_file(marker_path(shard_id)))
        embedding_hashes.append(marker["embedding_sha256"])
        valid_hashes.append(marker["valid_mask_sha256"])
        manifest_hashes.append(marker["manifest_sha256"])
        failure_hashes.append(marker["decode_failure_sha256"])
        fingerprints.add(marker["model_fingerprint"])
        total_rows += int(marker["row_count"])
    checks = {
        "shard_marker_digest": hash_string_sequence(marker_hashes),
        "shard_embedding_hashes_sha256": hash_string_sequence(embedding_hashes),
        "shard_valid_hashes_sha256": hash_string_sequence(valid_hashes),
        "shard_manifest_hashes_sha256": hash_string_sequence(manifest_hashes),
        "shard_decode_failure_hashes_sha256": hash_string_sequence(failure_hashes),
    }
    for key, actual in checks.items():
        if complete.get(key) != actual:
            raise RuntimeError(f"global extraction marker mismatch: {key}")
    if len(fingerprints) != 1 or complete.get("model_fingerprint") != next(iter(fingerprints)):
        raise RuntimeError("model fingerprint differs across extraction chain")
    if total_rows != int(complete["image_count"]):
        raise RuntimeError("global extraction row count mismatch")
    mapping_checks = {
        "canonical_note_ids_sha256": OUT / "mappings/note_ids.npy",
        "image_offsets_sha256": OUT / "mappings/image_offsets.npy",
        "audit_complete_sha256": OUT / "markers/audit_complete.json",
    }
    for field, path in mapping_checks.items():
        if sha256_file(path) != complete.get(field):
            raise RuntimeError(f"global extraction mapping/audit mismatch: {field}")
    return complete


def extract_shard(
    shard_id: int,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_WORKERS,
    overwrite_debug: bool = False,
    progress_seconds: float = 60.0,
    runtime=None,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    min_host_available_gb: float = 32.0,
    max_process_tree_rss_gb: float = 32.0,
) -> dict:
    marker_file = marker_path(shard_id)
    if marker_file.exists() and not overwrite_debug:
        return validate_completed_shard(shard_id)
    replacing = marker_file.exists() and overwrite_debug
    manifest = load_manifest(shard_id)
    row_count = len(manifest)
    if row_count == 0:
        raise ValueError(f"empty manifest shard {shard_id}")
    processor, model, model_metadata = runtime if runtime is not None else load_siglip(device)
    # Shard-specific metadata avoids four workers racing on one writable path.
    save_json_atomic(OUT / f"configs/model_metadata_shard_{shard_id:05d}.json", model_metadata)
    dataset = ImageManifestDataset(manifest)
    configure_worker_sharing()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=ProcessorCollator(processor),
    )
    token = uuid.uuid4().hex
    embedding_tmp = embedding_path(shard_id).with_name(embedding_path(shard_id).name + f".{token}.tmp")
    valid_tmp = valid_path(shard_id).with_name(valid_path(shard_id).name + f".{token}.tmp")
    embedding_tmp.parent.mkdir(parents=True, exist_ok=True)
    valid_tmp.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.lib.format.open_memmap(embedding_tmp, mode="w+", dtype=np.float16, shape=(row_count, OUTPUT_DIM))
    valid_mask = np.lib.format.open_memmap(valid_tmp, mode="w+", dtype=np.bool_, shape=(row_count,))
    embeddings[:] = 0
    valid_mask[:] = False
    failures, formats = [], {}
    start = last_log = time.monotonic()
    data_wait = inference_time = 0.0
    last_batch_end = start
    torch.cuda.reset_peak_memory_stats(device)
    stop_requested = False
    resources = ResourceMonitor(min_host_available_gb, max_process_tree_rss_gb)
    resources.sample()

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGTERM, request_stop)
    processed = 0
    iterator = None
    try:
        iterator = iter(loader)
        with torch.inference_mode():
            for batch_index, batch in enumerate(iterator):
                now = time.monotonic()
                data_wait += now - last_batch_end
                resource_state = resources.sample()
                if stop_requested:
                    raise InterruptedError("SIGTERM received; incomplete temporary shard retained but is not formal")
                for batch_row, local_row in enumerate(batch["local_rows"]):
                    actual_format = batch["actual_formats"][batch_row]
                    if actual_format:
                        formats[actual_format] = formats.get(actual_format, 0) + 1
                    error = batch["errors"][batch_row]
                    if error:
                        failures.append({
                            "global_image_row": int(dataset.rows["global_image_row"][local_row]),
                            "note_idx": int(dataset.rows["note_idx"][local_row]),
                            "position": int(dataset.rows["position"][local_row]),
                            "relative_path": dataset.rows["relative_path"][local_row],
                            "error_type": error,
                        })
                pixels = batch["pixel_values"]
                if pixels is not None and len(batch["valid_local_rows"]):
                    infer_start = time.monotonic()
                    pixels = pixels.to(device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(pixel_values=pixels).pooler_output
                    output = output.float()
                    if output.shape[1] != OUTPUT_DIM:
                        raise RuntimeError(f"unexpected SigLIP output {tuple(output.shape)}")
                    output = F.normalize(output, p=2, dim=-1)
                    torch.cuda.synchronize(device)
                    inference_time += time.monotonic() - infer_start
                    values = output.cpu().numpy().astype(np.float16)
                    rows = np.asarray(batch["valid_local_rows"], dtype=np.int64)
                    embeddings[rows] = values
                    valid_mask[rows] = True
                processed += len(batch["local_rows"])
                last_batch_end = time.monotonic()
                if batch_index % 100 == 0 or last_batch_end - last_log >= progress_seconds:
                    elapsed = last_batch_end - start
                    print(
                        f"[shard {shard_id:05d}] {processed}/{row_count} "
                        f"({processed/max(elapsed,1e-9):.1f} img/s) "
                        f"rss={resource_state['process_tree_rss_bytes']/2**30:.2f}GiB "
                        f"host_avail={resource_state['host_available_bytes']/2**30:.1f}GiB "
                        f"swap={resource_state['swap_used_bytes']/2**30:.2f}GiB "
                        f"children={resource_state['child_processes']}",
                        flush=True,
                    )
                    last_log = last_batch_end
    finally:
        shutdown_loader(loader, iterator)
        signal.signal(signal.SIGTERM, old_handler)
    embeddings.flush()
    valid_mask.flush()
    del embeddings, valid_mask
    failure_tmp = failure_path(shard_id).with_name(failure_path(shard_id).name + f".{token}.tmp")
    failure_tmp.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("global_image_row", pa.int64()), ("note_idx", pa.int64()),
        ("position", pa.int32()), ("relative_path", pa.string()), ("error_type", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(failures, schema=schema), failure_tmp, compression="zstd")
    elapsed = time.monotonic() - start
    valid_count = row_count - len(failures)
    marker = {
        "complete": True, "shard_id": shard_id,
        "global_start": int(manifest["global_image_row"][0].as_py()),
        "global_end": int(manifest["global_image_row"][-1].as_py()) + 1,
        "row_count": row_count, "valid_count": valid_count, "invalid_count": len(failures),
        "embedding_shape": [row_count, OUTPUT_DIM], "embedding_dtype": "float16",
        "embedding_sha256": sha256_file(embedding_tmp),
        "valid_mask_sha256": sha256_file(valid_tmp),
        "manifest_sha256": sha256_file(manifest_path(shard_id)),
        "decode_failure_sha256": sha256_file(failure_tmp),
        "model_snapshot": model_metadata["snapshot_commit"],
        "model_fingerprint": model_metadata["model_fingerprint"],
        "inference_protocol": FORMAL_INFERENCE_PROTOCOL,
        "elapsed_seconds": elapsed, "images_per_second": row_count / max(elapsed, 1e-9),
        "data_wait_seconds": data_wait, "gpu_inference_seconds": inference_time,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "actual_formats": formats, "batch_size": batch_size, "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "resource_usage": resources.summary(),
        "test_opened": False,
    }
    # New files are fully materialized and hashed before any old completed asset
    # is touched. Explicit replacement also keeps hard-link backups for recovery.
    backups = {}
    if replacing:
        recovery = OUT / "diagnostics/overwritten_shards" / f"shard_{shard_id:05d}_{int(time.time())}"
        recovery.mkdir(parents=True, exist_ok=False)
        for name, path in {
            "embedding": embedding_path(shard_id), "valid": valid_path(shard_id),
            "failure": failure_path(shard_id), "marker": marker_file,
        }.items():
            if path.exists():
                backup = recovery / path.name
                os.link(path, backup)
                backups[name] = str(backup)
    embedding_tmp.replace(embedding_path(shard_id))
    valid_tmp.replace(valid_path(shard_id))
    failure_tmp.replace(failure_path(shard_id))
    save_json_atomic(marker_file, marker)
    validate_completed_shard(shard_id)
    if backups:
        marker["recovery_backups"] = backups
        save_json_atomic(marker_file, marker)
    return marker


def run_smoke(
    sample: pa.Table,
    device: str,
    batch_size: int,
    num_workers: int,
    precision: str = "autocast",
    return_vectors: bool = False,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    min_host_available_gb: float = 32.0,
    max_process_tree_rss_gb: float = 32.0,
):
    """Run the same extraction path in-memory without creating formal shards."""
    processor, model, model_metadata = load_siglip(device)
    resolution = int(model_metadata["input_resolution"] or 224)
    warmup = torch.zeros((2, 3, resolution, resolution), device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(2):
            if precision == "autocast":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(pixel_values=warmup).pooler_output
            elif precision == "fp32":
                model(pixel_values=warmup).pooler_output
            else:
                raise ValueError(f"unknown smoke precision: {precision}")
    torch.cuda.synchronize(device)
    del warmup
    configure_worker_sharing()
    loader = DataLoader(
        ImageManifestDataset(sample), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=ProcessorCollator(processor),
    )
    output = np.zeros((len(sample), OUTPUT_DIM), dtype=np.float16)
    valid = np.zeros(len(sample), dtype=np.bool_)
    failures, formats = [], {}
    start = time.monotonic()
    data_wait = inference_time = 0.0
    previous = start
    torch.cuda.reset_peak_memory_stats(device)
    resources = ResourceMonitor(min_host_available_gb, max_process_tree_rss_gb)
    resources.sample()
    iterator = None
    processed = 0
    last_log = start
    try:
        iterator = iter(loader)
        with torch.inference_mode():
            for batch_index, batch in enumerate(iterator):
                now = time.monotonic()
                data_wait += now - previous
                resource_state = resources.sample()
                for batch_row, global_row in enumerate(batch["global_image_rows"]):
                    actual_format = batch["actual_formats"][batch_row]
                    if actual_format:
                        formats[actual_format] = formats.get(actual_format, 0) + 1
                    error = batch["errors"][batch_row]
                    if error:
                        failures.append({"global_image_row": global_row, "error_type": error})
                if batch["pixel_values"] is not None:
                    infer_start = time.monotonic()
                    pixels = batch["pixel_values"].to(device, non_blocking=True)
                    if precision == "autocast":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            raw = model(pixel_values=pixels).pooler_output
                    elif precision == "fp32":
                        raw = model(pixel_values=pixels).pooler_output
                    else:
                        raise ValueError(f"unknown smoke precision: {precision}")
                    vector = F.normalize(raw.float(), p=2, dim=-1)
                    torch.cuda.synchronize(device)
                    inference_time += time.monotonic() - infer_start
                    rows = np.asarray(batch["valid_local_rows"])
                    output[rows] = vector.cpu().numpy().astype(np.float16)
                    valid[rows] = True
                previous = time.monotonic()
                processed += len(batch["local_rows"])
                if batch_index % 100 == 0 or previous - last_log >= 60.0:
                    print(
                        f"[smoke {precision}] {processed}/{len(sample)} "
                        f"({processed/max(previous-start,1e-9):.1f} img/s) "
                        f"rss={resource_state['process_tree_rss_bytes']/2**30:.2f}GiB "
                        f"host_avail={resource_state['host_available_bytes']/2**30:.1f}GiB "
                        f"swap={resource_state['swap_used_bytes']/2**30:.2f}GiB "
                        f"children={resource_state['child_processes']}",
                        flush=True,
                    )
                    last_log = previous
    finally:
        shutdown_loader(loader, iterator)
    elapsed = time.monotonic() - start
    norms = np.linalg.norm(output[valid].astype(np.float32), axis=1)
    if output.shape != (len(sample), OUTPUT_DIM) or not np.isfinite(output).all():
        raise RuntimeError("smoke embedding shape/finite check failed")
    if not valid.any():
        raise RuntimeError("smoke decoded no valid images")
    if valid.any() and not np.allclose(norms, 1.0, atol=2e-3):
        raise RuntimeError("smoke L2 norm check failed")
    if np.any(output[~valid] != 0) or np.any(np.linalg.norm(output[valid].astype(np.float32), axis=1) == 0):
        raise RuntimeError("smoke zero-vector/valid-mask contract failed")
    result = {
        "complete": True, "sample_count": len(sample), "valid_count": int(valid.sum()),
        "invalid_count": int((~valid).sum()), "formats": formats,
        "embedding_shape": list(output.shape), "embedding_dtype": str(output.dtype),
        "norm_mean": float(norms.mean()) if len(norms) else None,
        "norm_min": float(norms.min()) if len(norms) else None,
        "norm_max": float(norms.max()) if len(norms) else None,
        "batch_size": batch_size, "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "elapsed_seconds": elapsed, "images_per_second": len(sample) / max(elapsed, 1e-9),
        "data_wait_seconds": data_wait, "gpu_inference_seconds": inference_time,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "inference_precision": precision,
        "warmup_forwards": 2,
        "speed_comparison_note": "Approximate: both modes are GPU-warmed, but autocast runs first so filesystem cache order is not counterbalanced.",
        "model": model_metadata, "test_opened": False,
        "resource_usage": resources.summary(),
    }
    return (result, output, valid) if return_vectors else result

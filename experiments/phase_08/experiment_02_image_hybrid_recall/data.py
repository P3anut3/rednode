"""Read-only Phase-8 image assets aligned to the Phase-7 canonical item rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import NOTE_IDS_PATH, PROTOCOL
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import Phase7Collator
from experiments.phase_08.experiment_01_image_embedding_extraction.data import sha256_file


ROOT = Path(__file__).resolve().parents[3]
IMAGE_OUT = ROOT / "results/phase_08/experiment_01_image_embedding_extraction"
STRATEGIES = {"i1_first": "first_image", "i2_top3": "mean_top3", "i3_all": "mean_all"}


def image_paths(strategy: str) -> tuple[Path, Path, Path]:
    folder = IMAGE_OUT / "pooled_embeddings" / STRATEGIES[strategy]
    return folder / "embeddings.f16.npy", folder / "image_available.npy", folder / "metadata.json"


def verify_image_assets(strategy: str, full_hash: bool = True) -> dict:
    if not (IMAGE_OUT / "markers/validation_complete.json").exists():
        raise RuntimeError("Phase 8-01 validation marker is missing")
    mapping = IMAGE_OUT / "mappings/note_ids.npy"
    canonical = np.load(NOTE_IDS_PATH, mmap_mode="r")
    image_ids = np.load(mapping, mmap_mode="r")
    if canonical.shape != image_ids.shape or not np.array_equal(canonical, image_ids):
        raise RuntimeError("image/Phase-7 canonical note mapping mismatch")
    embeddings_path, mask_path, metadata_path = image_paths(strategy)
    metadata = json.loads(metadata_path.read_text())
    completion = json.loads((metadata_path.parent / "pooling_complete.json").read_text())
    images = np.load(embeddings_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    if not completion.get("complete") or metadata["strategy"] != STRATEGIES[strategy]:
        raise RuntimeError("image pooling is incomplete or mismatched")
    if (sha256_file(metadata_path) != completion["metadata_sha256"]
            or completion["embedding_sha256"] != metadata["embedding_sha256"]
            or completion["mask_sha256"] != metadata["mask_sha256"]):
        raise RuntimeError("image pooling completion metadata mismatch")
    if images.shape != (PROTOCOL.corpus_items, 768) or images.dtype != np.float16:
        raise RuntimeError("image embedding shape/dtype mismatch")
    if mask.shape != (PROTOCOL.corpus_items,) or mask.dtype != np.bool_:
        raise RuntimeError("image mask shape/dtype mismatch")
    if int(mask.sum()) != int(metadata["available_items"]):
        raise RuntimeError("image availability count mismatch")
    if full_hash and (sha256_file(embeddings_path) != metadata["embedding_sha256"] or sha256_file(mask_path) != metadata["mask_sha256"]):
        raise RuntimeError("image asset SHA-256 mismatch")
    return {
        "strategy": strategy,
        "image_path": str(embeddings_path.relative_to(ROOT)),
        "mask_path": str(mask_path.relative_to(ROOT)),
        "image_sha256": metadata["embedding_sha256"],
        "mask_sha256": metadata["mask_sha256"],
        "mapping_sha256": sha256_file(mapping),
        "canonical_sha256": sha256_file(NOTE_IDS_PATH),
        "available_items": int(metadata["available_items"]),
    }


class ImageCollator(Phase7Collator):
    """One mmap per worker; transfer only the rows in this batch to the GPU."""

    def __init__(self, strategy: str):
        super().__init__(hybrid=True)
        self.strategy = strategy
        self._images = None
        self._mask = None

    def __call__(self, rows: list[dict]) -> dict:
        corpus_rows = {
            prefix: np.stack([np.asarray(row[f"{prefix}_corpus_row"]) for row in rows])
            for prefix in ("history", "target", "negative")
        }
        result = super().__call__(rows)
        if self._images is None:
            image_path, mask_path, _ = image_paths(self.strategy)
            self._images = np.load(image_path, mmap_mode="r")
            self._mask = np.load(mask_path, mmap_mode="r")
        for prefix, indices in corpus_rows.items():
            valid = indices >= 0
            available = np.zeros(indices.shape, dtype=np.bool_)
            available[valid] = self._mask[indices[valid]]
            image = np.zeros(indices.shape + (768,), dtype=np.float32)
            image[available] = self._images[indices[available]].astype(np.float32)
            result[f"{prefix}_image"] = torch.from_numpy(image)
            result[f"{prefix}_image_available"] = torch.from_numpy(available)
        return result

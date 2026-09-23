"""Immutable protocol and paths for Phase 7 Experiment 02."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (
    DeadlineExceeded,
    check_deadline,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phase_07/experiment_02_concat_fusion_ablation"
EXP01_OUT = ROOT / "results/phase_07/experiment_01_feature_hybrid_two_tower"

MODEL_NAMES = ("m1_direct_concat_id64", "m2_concat_residual_id64")
CONDITIONAL_MODEL = "add_id64_control"


@dataclass(frozen=True)
class Protocol:
    train_samples: int = 254_583
    valid_samples: int = 47_733
    corpus_items: int = 1_983_938
    history_n: int = 20
    content_dim: int = 768
    id_dim: int = 64
    item_id_rows: int = 412_467
    user_id_rows: int = 9_533
    output_dim: int = 128
    fusion_hidden_dim: int = 512
    dropout: float = 0.1
    beta_init: float = 0.05
    temperature: float = 0.05
    pair_lambda: float = 0.5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 6
    patience: int = 2
    seed: int = 42
    proxy_requests: int = 5_000
    proxy_candidates: int = 100_000
    smoke_train_samples: int = 10_000
    smoke_valid_requests: int = 1_000
    overfetch: int = 600
    topk: int = 500
    cold_noninferiority_margin: float = -0.001

    def to_dict(self) -> dict:
        return asdict(self)


PROTOCOL = Protocol()


def ensure_output_layout() -> None:
    for relative in (
        "smoke",
        "configs",
        "checkpoints",
        "training_curves",
        "proxy",
        "validation",
        "validation_rankings",
        "embeddings",
        "diagnostics",
        "timing",
        "locks",
        "terminal_test",
    ):
        (OUT / relative).mkdir(parents=True, exist_ok=True)

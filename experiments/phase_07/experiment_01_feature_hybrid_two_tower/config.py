"""Immutable Phase 7 protocol and path contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/phase_07/experiment_01_feature_hybrid_two_tower"
OUT = ROOT / "results/phase_07/experiment_01_feature_hybrid_two_tower"
PHASE5_MINING = ROOT / "results/phase_05/experiment_01_request_hard_negative_mining"
PHASE5_TRAINING = (
    ROOT / "results/phase_05/experiment_02_bias_aware_hard_negative_training"
)
PHASE6 = ROOT / "results/phase_06/experiment_01_id_two_tower_retrieval"
FROZEN_BGE = (
    ROOT / "results/phase_04/experiment_02_history_n20_multi_interest/"
    "baseline_releases/v1_frozen_bge_n20_single_attention"
)
LEGACY_EVALUATOR_CACHE = (
    ROOT / "results/phase_04/experiment_02_history_n20_multi_interest/cache/evaluator"
)

TRAIN_PATH = PHASE5_MINING / "train_samples_base.parquet"
VALID_PATH = PHASE5_MINING / "valid_samples_base.parquet"
HARD_NEGATIVE_PATH = PHASE5_MINING / "train_hard_negatives.parquet"
NOTE_IDS_PATH = FROZEN_BGE / "mapping/note_ids.npy"
BGE_PATH = FROZEN_BGE / "embeddings/embeddings.f16"


@dataclass(frozen=True)
class Protocol:
    train_samples: int = 254_583
    valid_samples: int = 47_733
    corpus_items: int = 1_983_938
    history_n: int = 20
    content_dim: int = 768
    output_dim: int = 128
    temperature: float = 0.05
    seed: int = 42
    epochs: int = 6
    patience: int = 2
    proxy_requests: int = 5_000
    proxy_candidates: int = 100_000
    smoke_train_samples: int = 10_000
    smoke_valid_requests: int = 1_000
    smoke_candidates: int = 100_000
    overfetch: int = 600
    topk: int = 500
    alpha_init: float = 0.05
    cold_noninferiority_margin: float = -0.001

    def to_dict(self) -> dict:
        return asdict(self)


PROTOCOL = Protocol()


class DeadlineExceeded(RuntimeError):
    """Raised at a safe batch/chunk boundary when a stage exceeds its budget."""


def check_deadline(deadline: float | None, stage: str) -> None:
    import time

    if deadline is not None and time.monotonic() > deadline:
        raise DeadlineExceeded(f"{stage} exceeded its protected deadline")


USER_CATEGORICAL = ("gender", "platform", "age")
USER_NUMERIC = ("fans_num", "follows_num")
ITEM_CATEGORICAL = ("note_type", "taxonomy1_id", "taxonomy2_id", "commercial_flag")
ITEM_NUMERIC = ("video_duration", "aspect_ratio", "image_num", "content_length")
FORBIDDEN_ITEM_COLUMNS = (
    "imp_num",
    "imp_rec_num",
    "imp_search_num",
    "click_num",
    "click_rec_num",
    "click_search_num",
    "like_num",
    "collect_num",
    "comment_num",
    "share_num",
    "screenshot_num",
    "hide_num",
    "rec_like_num",
    "rec_collect_num",
    "rec_comment_num",
    "rec_share_num",
    "rec_follow_num",
    "search_like_num",
    "search_collect_num",
    "search_comment_num",
    "search_share_num",
    "search_follow_num",
    "accum_like_num",
    "accum_collect_num",
    "accum_comment_num",
    "view_time",
    "rec_view_time",
    "search_view_time",
    "valid_view_times",
    "full_view_times",
)

MODEL_NAMES = (
    "id_b0_user_history_control",
    "id_b1a_stable_profile",
    "id_b1b_full_profile",
    "id_b2_item_features",
    "id_b3_all",
    "h0_content_control",
    "h1_side_features",
    "h2_side_features_id",
    "h3_anonymous_dense",
)
ID_MODELS = frozenset(MODEL_NAMES[:5])
HYBRID_MODELS = frozenset(MODEL_NAMES[5:])


def ensure_output_layout() -> None:
    for relative in (
        "cache",
        "audit",
        "smoke",
        "checkpoints",
        "configs",
        "training_curves",
        "proxy",
        "validation",
        "validation_rankings",
        "locks",
        "terminal_test",
        "timing",
    ):
        (OUT / relative).mkdir(parents=True, exist_ok=True)

"""Fixed Phase 7 objectives: Phase-6 logQ and Phase-5 TF-IDF hard negatives."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from experiments.phase_06.experiment_01_id_two_tower_retrieval.losses import (
    invalid_inbatch_mask,
)
from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.loss import (
    phase4_inbatch_loss,
)


def logq_inbatch_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    batch: dict,
    log_q: torch.Tensor,
    temperature: float = 0.05,
):
    logits = query @ target.T / temperature - log_q[None, :]
    invalid, stats = invalid_inbatch_mask(batch)
    count = len(batch["user_id"])
    diagonal = torch.eye(count, dtype=torch.bool, device=query.device)
    same_user = batch["user_id"][:, None].eq(batch["user_id"][None, :]) & ~diagonal
    invalid |= same_user
    stats["masked_same_user"] = int(same_user.sum())
    logits = logits.masked_fill(invalid, -1e4)
    return F.cross_entropy(logits, torch.arange(len(query), device=query.device)), stats


def tfidf_hard_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    negative: torch.Tensor,
    negative_mask: torch.Tensor,
    batch: dict,
    temperature: float = 0.05,
    pair_lambda: float = 0.5,
):
    # Match Phase 5 exactly: duplicate target, same user, and in-history masks.
    inbatch, stats = phase4_inbatch_loss(
        query[:, None, :],
        target,
        batch["target_note_id"],
        batch["user_id"],
        batch["history_note_ids"],
        temperature,
    )
    positive_score = (query * target).sum(-1, keepdim=True)
    negative_score = torch.einsum("bd,bnd->bn", query, negative)
    pair = F.softplus(negative_score - positive_score)
    pair_value = pair[negative_mask].mean() if negative_mask.any() else pair.sum() * 0.0
    return inbatch + pair_lambda * pair_value, {
        **stats,
        "inbatch_loss": float(inbatch.detach()),
        "tfidf_pair_loss": float(pair_value.detach()),
    }

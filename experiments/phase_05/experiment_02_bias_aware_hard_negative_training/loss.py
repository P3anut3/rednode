"""In-batch InfoNCE plus source-specific pairwise hard-negative losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.loss import (
    phase4_inbatch_loss,
)
from experiments.phase_05.experiment_02_bias_aware_hard_negative_training.dataset import (
    SOURCE_DENSE,
    SOURCE_IMPRESSION,
    SOURCE_TFIDF,
)


def phase5_loss(interests: torch.Tensor, targets: torch.Tensor, negatives: torch.Tensor,
                negative_mask: torch.Tensor, negative_sources: torch.Tensor,
                negative_weights: torch.Tensor, target_ids: torch.Tensor,
                user_ids: torch.Tensor, history_ids: torch.Tensor,
                temperature: float = 0.05, lambdas=(0.5, 0.5, 0.5)):
    inbatch, info = phase4_inbatch_loss(interests, targets, target_ids, user_ids,
                                        history_ids, temperature)
    user = interests[:, 0]
    positive_score = (user * targets).sum(-1, keepdim=True)
    negative_score = torch.einsum("bd,bnd->bn", user, negatives)
    pair = F.softplus(negative_score - positive_score)
    source_losses = []
    for source, coefficient in zip((SOURCE_DENSE, SOURCE_TFIDF, SOURCE_IMPRESSION), lambdas):
        mask = negative_mask & negative_sources.eq(source)
        if mask.any():
            weights = negative_weights if source == SOURCE_IMPRESSION else torch.ones_like(pair)
            value = (pair * weights)[mask].mean()
        else:
            value = pair.sum() * 0.0
        source_losses.append(value)
    total = inbatch + sum(float(coef) * value for coef, value in zip(lambdas, source_losses))
    details = {**info, "inbatch_loss": float(inbatch.detach()),
               "dense_pair_loss": float(source_losses[0].detach()),
               "tfidf_pair_loss": float(source_losses[1].detach()),
               "impression_pair_loss": float(source_losses[2].detach())}
    return total, details

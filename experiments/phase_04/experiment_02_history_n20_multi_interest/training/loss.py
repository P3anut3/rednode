"""Phase 4.1 InfoNCE plus an explicit anti-collapse regularizer."""

from __future__ import annotations

import torch

from experiments.phase_04.experiment_01_learnable_user_tower_n10.training.loss import phase4_inbatch_loss


def phase4_1_loss(interests: torch.Tensor, targets: torch.Tensor,
                  target_ids: torch.Tensor, user_ids: torch.Tensor,
                  history_ids: torch.Tensor, temperature: float = 0.05,
                  diversity_lambda: float = 0.0) -> tuple[torch.Tensor, dict]:
    contrastive, info = phase4_inbatch_loss(
        interests, targets, target_ids, user_ids, history_ids, temperature)
    if interests.shape[1] < 2:
        diversity = contrastive.new_zeros(())
    else:
        similarity = interests @ interests.transpose(1, 2)
        indices = torch.triu_indices(interests.shape[1], interests.shape[1], 1,
                                     device=interests.device)
        # Squared cosine penalizes both identical and antipodal duplicate slots.
        diversity = similarity[:, indices[0], indices[1]].square().mean()
    total = contrastive + diversity_lambda * diversity
    return total, {**info, "contrastive_loss": float(contrastive.detach()),
                   "diversity_loss": float(diversity.detach()),
                   "diversity_lambda": diversity_lambda}

"""Phase 4 in-batch contrastive loss with false-negative masking."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def phase4_inbatch_loss(interests: torch.Tensor, targets: torch.Tensor,
                        target_ids: torch.Tensor, user_ids: torch.Tensor,
                        history_ids: torch.Tensor, temperature: float = 0.05
                        ) -> tuple[torch.Tensor, dict]:
    # [B,K,D] @ [B,D] -> [B,K,B], then target matches its best interest.
    scores_by_interest = torch.einsum("bkd,jd->bkj", interests, targets)
    scores, winner = scores_by_interest.max(dim=1)
    count = len(target_ids)
    diagonal = torch.eye(count, device=target_ids.device, dtype=torch.bool)
    duplicate = target_ids[:, None].eq(target_ids[None, :]) & ~diagonal
    same_user = user_ids[:, None].eq(user_ids[None, :]) & ~diagonal
    in_history = history_ids[:, :, None].eq(target_ids[None, None, :]).any(dim=1) & ~diagonal
    masked = duplicate | same_user | in_history
    logits = (scores / temperature).masked_fill(masked, -1e4)
    loss = F.cross_entropy(logits, torch.arange(count, device=logits.device))
    positive_winner = winner.diagonal()
    return loss, {"masked_duplicate": int(duplicate.sum().item()),
                  "masked_same_user": int(same_user.sum().item()),
                  "masked_in_history": int(in_history.sum().item()),
                  "positive_winner": positive_winner.detach()}

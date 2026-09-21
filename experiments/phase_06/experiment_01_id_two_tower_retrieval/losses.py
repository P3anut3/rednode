"""Phase 6 contrastive objectives and false-negative masks."""

from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def invalid_inbatch_mask(batch: dict) -> tuple[torch.Tensor, dict]:
    targets = batch["target_note_id"]
    count = len(targets)
    diagonal = torch.eye(count, dtype=torch.bool, device=targets.device)
    duplicate = targets[:, None].eq(targets[None, :]) & ~diagonal
    history = batch["history_note_ids"][:, :, None].eq(targets[None, None, :]).any(1) & ~diagonal
    same_request = torch.zeros_like(duplicate)
    positives = batch["same_request_positive_ids"]
    target_list = targets.detach().cpu().tolist()
    for row, allowed in enumerate(positives):
        allowed_set = set(map(int, allowed))
        same_request[row] = torch.as_tensor(
            [column != row and note in allowed_set for column, note in enumerate(target_list)],
            device=targets.device)
    return duplicate | history | same_request, {
        "masked_duplicate": int(duplicate.sum()), "masked_history": int(history.sum()),
        "masked_same_request": int(same_request.sum())}


def phase6_inbatch_loss(query: torch.Tensor, target: torch.Tensor, batch: dict,
                        temperature: float = 0.05, log_q: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, dict]:
    logits = query @ target.T / temperature
    if log_q is not None:
        logits = logits - log_q[None, :]
    invalid, stats = invalid_inbatch_mask(batch)
    logits = logits.masked_fill(invalid, -1e4)
    loss = F.cross_entropy(logits, torch.arange(len(query), device=query.device))
    return loss, stats


def uniform_pair_loss(query: torch.Tensor, positive: torch.Tensor,
                      negatives: torch.Tensor) -> torch.Tensor:
    pos = (query * positive).sum(-1, keepdim=True)
    neg = torch.einsum("bd,bnd->bn", query, negatives)
    return F.softplus(neg - pos).mean()

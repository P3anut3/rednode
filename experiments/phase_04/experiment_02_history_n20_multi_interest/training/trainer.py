"""GPU-capable Phase 4.1 trainer using frozen item vectors."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .loss import phase4_1_loss


def phase4_1_train_epoch(model: torch.nn.Module, dataset, optimizer: torch.optim.Optimizer,
                         device: str, batch_size: int, seed: int, epoch: int,
                         diversity_lambda: float, max_batches: int | None = None,
                         temperature: float = 0.05) -> dict:
    generator = torch.Generator().manual_seed(seed + epoch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=generator, drop_last=True)
    model.train()
    losses, contrastive, diversity = [], [], []
    masked = [0, 0, 0]
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        history = batch["history"].to(device)
        mask = batch["mask"].to(device)
        target = F.normalize(batch["target"].to(device), dim=-1).detach()
        interests = model(history, mask)
        loss, info = phase4_1_loss(
            interests, target, batch["target_id"].to(device), batch["user_id"].to(device),
            batch["history_ids"].to(device), temperature, diversity_lambda)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Phase 4.1 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        contrastive.append(info["contrastive_loss"])
        diversity.append(info["diversity_loss"])
        masked[0] += info["masked_duplicate"]
        masked[1] += info["masked_same_user"]
        masked[2] += info["masked_in_history"]
    return {"train_loss": float(np.mean(losses)),
            "contrastive_loss": float(np.mean(contrastive)),
            "diversity_loss": float(np.mean(diversity)), "batches": len(losses),
            "masked_duplicate": masked[0], "masked_same_user": masked[1],
            "masked_in_history": masked[2]}

"""Phase 5B trainer for frozen-item single-attention user towers."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.phase_05.experiment_02_bias_aware_hard_negative_training.loss import phase5_loss


def train_epoch(model, dataset, optimizer, device: str, batch_size: int, epoch: int,
                seed: int = 42, max_batches: int | None = None) -> dict:
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(seed + epoch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=generator, drop_last=True, pin_memory=device.startswith("cuda"))
    model.train(); totals = []
    detail_totals = {key: [] for key in ("inbatch_loss", "dense_pair_loss",
                                          "tfidf_pair_loss", "impression_pair_loss")}
    masked = [0, 0, 0]
    sampled = [0, 0, 0]
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches: break
        history = batch["history"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = F.normalize(batch["target"].to(device, non_blocking=True), dim=-1)
        negatives = F.normalize(batch["negatives"].to(device, non_blocking=True), dim=-1)
        interests = model(history, mask)
        loss, details = phase5_loss(
            interests, target, negatives, batch["negative_mask"].to(device),
            batch["negative_sources"].to(device), batch["negative_weights"].to(device),
            batch["target_id"].to(device), batch["user_id"].to(device),
            batch["history_ids"].to(device),
        )
        if not torch.isfinite(loss): raise FloatingPointError("non-finite Phase 5 loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        totals.append(float(loss.detach()))
        for key in detail_totals: detail_totals[key].append(details[key])
        masked[0] += details["masked_duplicate"]; masked[1] += details["masked_same_user"]
        masked[2] += details["masked_in_history"]
        source_values = batch["negative_sources"]
        valid_values = batch["negative_mask"]
        for source in range(3):
            sampled[source] += int((valid_values & source_values.eq(source)).sum().item())
    return {"train_loss": float(np.mean(totals)), "batches": len(totals),
            **{key: float(np.mean(values)) for key, values in detail_totals.items()},
            "masked_duplicate": masked[0], "masked_same_user": masked[1],
            "masked_in_history": masked[2], "sampled_dense": sampled[0],
            "sampled_tfidf": sampled[1], "sampled_impression": sampled[2]}

"""Phase 4 CPU/GPU trainer; frozen item vectors are read-only data."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .loss import phase4_inbatch_loss


def phase4_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def phase4_train_epoch(model: torch.nn.Module, dataset, optimizer: torch.optim.Optimizer,
                       device: str, batch_size: int, seed: int, epoch: int,
                       max_batches: int | None = None, temperature: float = 0.05) -> dict:
    generator = torch.Generator().manual_seed(seed + epoch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=generator, drop_last=True)
    model.train(); losses, masked = [], [0, 0, 0]
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches: break
        history = batch["history"].to(device)
        mask = batch["mask"].to(device)
        target = F.normalize(batch["target"].to(device), dim=-1)
        target.requires_grad_(False)
        interests = model(history, mask)
        loss, info = phase4_inbatch_loss(
            interests, target, batch["target_id"].to(device), batch["user_id"].to(device),
            batch["history_ids"].to(device), temperature,
        )
        if not torch.isfinite(loss): raise FloatingPointError("non-finite Phase 4 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        masked[0] += info["masked_duplicate"]
        masked[1] += info["masked_same_user"]
        masked[2] += info["masked_in_history"]
    return {"train_loss": float(np.mean(losses)), "batches": len(losses),
            "masked_duplicate": masked[0], "masked_same_user": masked[1],
            "masked_in_history": masked[2]}


@torch.inference_mode()
def phase4_encode_requests(model: torch.nn.Module, requests, embeddings: np.ndarray,
                           lookup: np.ndarray, device: str, history_n: int = 10,
                           batch_size: int = 512, return_attention: bool = False):
    model.eval(); vectors, attention = [], []
    dim = embeddings.shape[1]
    for offset in range(0, len(requests), batch_size):
        part = requests[offset:offset + batch_size]
        history = np.zeros((len(part), history_n, dim), dtype=np.float32)
        mask = np.zeros((len(part), history_n), dtype=np.bool_)
        for i, request in enumerate(part):
            valid = [int(n) for n in request.history[-history_n:]
                     if 0 <= int(n) < len(lookup) and lookup[int(n)] >= 0]
            if valid:
                rows = lookup[np.asarray(valid, dtype=np.int64)]
                history[i, :len(rows)] = np.asarray(embeddings[rows], dtype=np.float32)
                mask[i, :len(rows)] = True
        values = model(torch.from_numpy(history).to(device), torch.from_numpy(mask).to(device),
                       return_attention=return_attention)
        if return_attention:
            value, weights = values
            attention.append(weights.cpu().numpy())
        else:
            value = values
        vectors.append(value.cpu().numpy().astype(np.float32))
    result = np.concatenate(vectors)
    return (result, np.concatenate(attention)) if return_attention else result


def phase4_save_checkpoint(path: Path, model: torch.nn.Module, meta: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": meta}, path)
    return path.stat().st_size

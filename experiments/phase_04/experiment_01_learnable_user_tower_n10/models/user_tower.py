"""Phase 4 frozen-item learnable user towers."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1)


class Phase4MeanProjection(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, dim))

    def forward(self, history: torch.Tensor, mask: torch.Tensor,
                return_attention: bool = False):
        base = masked_mean(history, mask)
        output = F.normalize(base + self.project(base), dim=-1)
        return (output[:, None, :], None) if return_attention else output[:, None, :]


class Phase4SingleAttention(nn.Module):
    def __init__(self, dim: int = 768, attention_dim: int = 128,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, dim))

    def forward(self, history: torch.Tensor, mask: torch.Tensor,
                return_attention: bool = False):
        scores = self.score(torch.tanh(self.key(history))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=1) * mask.to(scores.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        base = (weights.unsqueeze(-1) * history).sum(1)
        output = F.normalize(base + self.project(base), dim=-1)[:, None, :]
        return (output, weights[:, None, :]) if return_attention else output


class Phase4MultiInterest(nn.Module):
    def __init__(self, dim: int = 768, k: int = 4, attention_dim: int = 128,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.key = nn.Linear(dim, attention_dim)
        self.queries = nn.Parameter(torch.randn(k, attention_dim) * 0.1)
        self.project = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, dim))

    def forward(self, history: torch.Tensor, mask: torch.Tensor,
                return_attention: bool = False):
        keys = torch.tanh(self.key(history))
        scores = torch.einsum("bnd,kd->bkn", keys, self.queries) / math.sqrt(keys.shape[-1])
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        weights = F.softmax(scores, dim=-1) * mask[:, None, :].to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        base = torch.einsum("bkn,bnd->bkd", weights, history)
        output = F.normalize(base + self.project(base), dim=-1)
        return (output, weights) if return_attention else output


def phase4_make_model(name: str, dim: int, k: int = 4) -> nn.Module:
    if name == "mean_mlp": return Phase4MeanProjection(dim)
    if name == "single_attention": return Phase4SingleAttention(dim)
    if name == "multi_interest": return Phase4MultiInterest(dim, k=k)
    raise ValueError(name)

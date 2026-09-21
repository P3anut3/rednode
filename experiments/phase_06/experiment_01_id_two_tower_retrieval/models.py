"""Pure-ID and content-ID residual towers for Phase 6."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TargetIndependentAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1, bias=False))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(values).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1) * mask.float()
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        return (values * weights.unsqueeze(-1)).sum(1)


class PureIdTower(nn.Module):
    def __init__(self, kind: str, vocab_size: int, user_size: int, dim: int = 128):
        super().__init__()
        if kind not in {"m0_user_mf", "m1_history_mean", "m1a_history_attention",
                        "m1b_user_history_attention"}:
            raise ValueError(kind)
        self.kind, self.dim = kind, dim
        self.item = nn.Embedding(vocab_size + 1, dim, padding_idx=0)
        self.user = nn.Embedding(user_size + 1, dim, padding_idx=0)
        self.attention = TargetIndependentAttention(dim)
        self.combine = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.item.weight[1:], std=0.02)
        nn.init.normal_(self.user.weight[1:], std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_(); self.user.weight[0].zero_()

    def query(self, history_vocab: torch.Tensor, mask: torch.Tensor,
              user_row: torch.Tensor) -> torch.Tensor:
        values = self.item(history_vocab)
        if self.kind == "m0_user_mf":
            query = self.user(user_row)
        elif self.kind == "m1_history_mean":
            query = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        else:
            history = self.attention(values, mask)
            query = history if self.kind == "m1a_history_attention" else self.combine(
                torch.cat((history, self.user(user_row)), dim=-1))
        return F.normalize(query, dim=-1)

    def target(self, target_vocab: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item(target_vocab), dim=-1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query(batch["history_vocab"], batch["mask"], batch["user_row"]), \
               self.target(batch["target_vocab"])


class ContentIdResidualTower(nn.Module):
    def __init__(self, vocab_size: int, user_size: int, item_to_vocab: torch.Tensor,
                 dim: int = 128, content_dim: int = 768, residual: bool = False,
                 frequency_gate: torch.Tensor | None = None, id_dropout: float = 0.0):
        super().__init__()
        self.dim, self.residual_enabled, self.id_dropout = dim, residual, id_dropout
        self.content_projection = nn.Linear(content_dim, dim, bias=False)
        self.attention = TargetIndependentAttention(dim)
        self.item_residual = nn.Embedding(vocab_size + 1, dim, padding_idx=0)
        self.user = nn.Embedding(user_size + 1, dim, padding_idx=0)
        self.user_projection = nn.Linear(dim, dim, bias=False)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.9444))  # softplus ~= 0.051
        self.register_buffer("item_to_vocab", item_to_vocab.long())
        gate = torch.ones(vocab_size + 1) if frequency_gate is None else frequency_gate.float()
        self.register_buffer("frequency_gate", gate)
        nn.init.normal_(self.item_residual.weight[1:], std=0.02)
        nn.init.normal_(self.user.weight[1:], std=0.02)
        with torch.no_grad():
            self.item_residual.weight[0].zero_(); self.user.weight[0].zero_()

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    def _drop(self, value: torch.Tensor) -> torch.Tensor:
        if self.training and self.id_dropout > 0:
            shape = value.shape[:-1] + (1,)
            keep = (torch.rand(shape, device=value.device) >= self.id_dropout).float()
            return value * keep / (1.0 - self.id_dropout)
        return value

    def represent(self, content: torch.Tensor, note_ids: torch.Tensor) -> torch.Tensor:
        projected = self.content_projection(content)
        if self.residual_enabled:
            safe = note_ids.clamp(min=0, max=len(self.item_to_vocab) - 1)
            vocab = self.item_to_vocab[safe]
            vocab = torch.where(note_ids >= 0, vocab, torch.zeros_like(vocab))
            residual = self._drop(self.item_residual(vocab))
            projected = projected + self.alpha * self.frequency_gate[vocab].unsqueeze(-1) * residual
        return F.normalize(projected, dim=-1)

    def query(self, history_content: torch.Tensor, history_note_ids: torch.Tensor,
              mask: torch.Tensor, user_row: torch.Tensor) -> torch.Tensor:
        values = self.represent(history_content, history_note_ids)
        history = self.attention(values, mask)
        if self.residual_enabled:
            history = history + self._drop(self.user_projection(self.user(user_row)))
        return F.normalize(history, dim=-1)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(batch["history_content"], batch["history_note_ids"],
                           batch["content_mask"], batch["user_row"])
        target = self.represent(batch["target_content"], batch["target_note_id"])
        return query, target

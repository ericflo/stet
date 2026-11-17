"""The student: a small, ordinary GPT. Fixed for a season; nobody gets to tune it.

Deliberately plain (pre-norm, rotary positions, tied embeddings, no dropout) so that the
only thing that can explain a difference in score is the data.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _rotary(seq_len: int, head_dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.norm1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.out = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.n_head, c // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = _apply_rotary(q, cos[:, :, :t], sin[:, :, :t]), _apply_rotary(k, cos[:, :, :t], sin[:, :, :t])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.out(F.gelu(self.fc(self.norm2(x))))


class GPT(nn.Module):
    def __init__(self, n_layer: int, n_head: int, d_model: int, ctx: int, vocab: int):
        super().__init__()
        assert d_model % n_head == 0 and (d_model // n_head) % 2 == 0
        self.ctx = ctx
        self.head_dim = d_model // n_head
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied
        self.apply(self._init)
        for blk in self.blocks:  # scale the residual writes so depth does not blow up the stream
            for lin in (blk.proj, blk.out):
                nn.init.normal_(lin.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        self._rot: tuple | None = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self._rot is None or self._rot[0].device != idx.device:
            self._rot = _rotary(self.ctx, self.head_dim, idx.device)
        cos, sin = self._rot
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.head(self.norm(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

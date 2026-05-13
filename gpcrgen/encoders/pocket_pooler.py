"""Pool variable-length residue embeddings into a fixed-size target conditioning vector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPocketPooler(nn.Module):
    """
    Single-query attention pool: one learned query attends over residue embeddings.

    Input ``[N, d]`` (one pocket) or batched ``[B, N, d]`` with ``mask`` ``[B, N]``.
    """

    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            h: ``[N, d]`` or ``[B, N, d]``.
            mask: Optional bool ``[B, N]`` or ``[N]``; True means **valid** residue.

        Returns:
            ``[d]`` or ``[B, d]`` pooled embedding.
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze = True
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
        else:
            squeeze = False

        b, n, d = h.shape
        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=h.device)

        q = self.w_q(self.q.expand(b, -1, -1))
        k = self.w_k(h)
        v = self.w_v(h)

        qh = q.view(b, 1, self.num_heads, self.head_dim)
        kh = k.view(b, n, self.num_heads, self.head_dim)
        vh = v.view(b, n, self.num_heads, self.head_dim)

        scores = (qh * kh).sum(-1) / (self.head_dim**0.5)
        scores = scores.masked_fill(~mask.unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=1)
        ctx = (attn.unsqueeze(-1) * vh).sum(dim=1)
        ctx = ctx.reshape(b, d)
        out = self.out(F.gelu(ctx))
        return out.squeeze(0) if squeeze else out


class MeanPocketPooler(nn.Module):
    """Mean pool over residues (respects mask)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze = True
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
        else:
            squeeze = False
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            w = mask.to(h.dtype).unsqueeze(-1)
            pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        out = self.norm(pooled)
        return out.squeeze(0) if squeeze else out

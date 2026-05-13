"""Fuse sequence and structure pocket channels into a single conditioning vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class FuseCondition(nn.Module):
    """
    Combine sequence-derived and structure-derived pocket summaries for diffusion / AR models.

    Modes:
        - ``concat_mlp``: concatenate then MLP.
        - ``gated``: sigmoid gate on the structure branch scaled by sequence context.
    """

    def __init__(
        self,
        seq_dim: int,
        struct_dim: int,
        out_dim: int,
        mode: str = "concat_mlp",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in {"concat_mlp", "gated"}:
            raise ValueError("mode must be 'concat_mlp' or 'gated'")
        self.mode = mode
        if mode == "concat_mlp":
            self.net = nn.Sequential(
                nn.Linear(seq_dim + struct_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.seq_to_gate = nn.Linear(seq_dim, struct_dim)
            self.net = nn.Sequential(
                nn.Linear(seq_dim + struct_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )

    def forward(self, seq_vec: torch.Tensor, struct_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_vec: ``[B, seq_dim]`` or ``[seq_dim]``.
            struct_vec: ``[B, struct_dim]`` or ``[struct_dim]``.
        """
        squeeze = seq_vec.dim() == 1 and struct_vec.dim() == 1
        if seq_vec.dim() == 1:
            seq_vec = seq_vec.unsqueeze(0)
        if struct_vec.dim() == 1:
            struct_vec = struct_vec.unsqueeze(0)

        if self.mode == "gated":
            g = torch.sigmoid(self.seq_to_gate(seq_vec))
            struct_vec = struct_vec * g

        z = torch.cat([seq_vec, struct_vec], dim=-1)
        out = self.net(z)
        return out.squeeze(0) if squeeze else out

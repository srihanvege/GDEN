"""Graph-style encoder over pocket residue features (message passing)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_neighbor_agg(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean of neighbor embeddings for each node (``h`` is ``[N, D]``)."""
    if edge_index.numel() == 0:
        return torch.zeros_like(h)
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(h)
    out.index_add_(0, dst, h[src])
    deg = torch.bincount(dst, minlength=h.size(0)).to(h.dtype).clamp(min=1.0).unsqueeze(-1)
    return out / deg


class StructEncoder(nn.Module):
    """
    Encode each pocket residue given node features and sparse edges (``edge_index``).

    Uses mean-neighbor message passing (no PyG dependency); suitable for 30–150 residues.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[N, in_dim]`` node features for one graph.
            edge_index: ``[2, E]`` directed edges (both directions expected).

        Returns:
            ``[N, out_dim]`` node embeddings.
        """
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            agg = _mean_neighbor_agg(h, edge_index)
            h = norm(h + F.gelu(layer(agg)))
        return self.out_proj(h)

"""Sequence encoders: lightweight transformer over amino-acid tokens (ESM-like interface)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    import esm  # type: ignore[import-not-found]

    _HAS_ESM = True
except Exception:
    esm = None  # type: ignore[assignment]
    _HAS_ESM = False

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i + 1 for i, a in enumerate(AA20)}  # 0 = pad
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}


def tokenize_sequence(seq: str, device: torch.device | None = None) -> torch.Tensor:
    """Map one-letter sequence to long indices ``[1, L]`` (batch dim omitted)."""
    idx = [AA_TO_IDX.get(c.upper(), 1) for c in seq if c.upper() in AA_TO_IDX]
    return torch.tensor([idx], dtype=torch.long, device=device)


class LocalSequenceEncoder(nn.Module):
    """
    Small transformer encoder over residue embeddings (train from scratch or fine-tune).

    This is the default when ``fair-esm`` is not installed; it matches the expected
    role of "sequence → residue embeddings" in GDEN.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 21,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: ``[B, L]`` long indices (0 pad).
            key_padding_mask: ``[B, L]`` bool, True marks padding positions.

        Returns:
            Residue embeddings ``[B, L, d_model]``.
        """
        x = self.embed(tokens) * math.sqrt(self.d_model)
        if key_padding_mask is None:
            key_padding_mask = tokens == 0
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class ESM2EncoderWrapper(nn.Module):
    """Thin wrapper around ``facebook/esm2_*`` when ``fair-esm`` is available."""

    def __init__(self, model_name: str = "esm2_t6_8M_UR50D", device: str | None = None) -> None:
        if not _HAS_ESM:
            raise RuntimeError("Install fair-esm to use ESM2EncoderWrapper.")
        super().__init__()
        self.model, _alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        if device:
            self.model = self.model.to(device)
        self.model.eval()
        self.d_model = self.model.embed_dim

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns sequence representations from ESM2 (last hidden layer)."""
        out = self.model(tokens, repr_layers=[self.model.num_layers])["representations"][self.model.num_layers]
        return out


def build_sequence_encoder(
    *,
    kind: str = "local",
    d_model: int = 256,
    esm_model_name: str = "esm2_t6_8M_UR50D",
    device: str | None = None,
) -> nn.Module:
    """
    Factory: ``kind="local"`` (default) or ``kind="esm2"`` if fair-esm is installed.
    """
    if kind == "esm2":
        return ESM2EncoderWrapper(model_name=esm_model_name, device=device)
    if kind != "local":
        raise ValueError(f"Unknown sequence encoder kind: {kind}")
    return LocalSequenceEncoder(d_model=d_model)

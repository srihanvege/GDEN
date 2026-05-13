"""Build residue-level pocket definitions for encoders (sequence + optional 3D graph)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpcrgen.utils.io import ensure_dir

AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {a: i for i, a in enumerate(AA20)}


@dataclass
class PocketDefinition:
    """Residues selected as the binding pocket (1D sequence + optional Cα coordinates)."""

    protein_id: str
    residue_index: list[int]
    residue_one_letter: list[str]
    ca_coords: np.ndarray | None
    sequence_full: str | None = None

    def to_tensor_features(self, device: torch.device | None = None) -> torch.Tensor:
        """One-hot amino acids for each pocket residue, shape ``[N, 20]``."""
        idx = [_AA_TO_IDX.get(a.upper(), 0) for a in self.residue_one_letter]
        t = torch.zeros(len(idx), 20, device=device)
        t[torch.arange(len(idx)), torch.tensor(idx, dtype=torch.long, device=device)] = 1.0
        return t


def pseudo_helical_coords(n_residues: int, *, pitch: float = 1.5, radius: float = 4.5) -> np.ndarray:
    """Helix-shaped Cα proxy when no structure is available (demo / ablation)."""
    t = np.linspace(0, 3.2 * np.pi, n_residues, dtype=np.float32)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * np.arange(n_residues, dtype=np.float32)
    return np.stack([x, y, z], axis=-1)


def build_residue_graph(
    ca_coords: np.ndarray,
    cutoff_angstrom: float = 8.0,
) -> torch.Tensor:
    """
    Undirected edges between residues with Cα distance below ``cutoff_angstrom``.

    Returns ``edge_index`` of shape ``[2, E]`` (unique unordered pairs, both directions).
    """
    d = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
    i, j = np.where((d < cutoff_angstrom) & (d > 1e-3))
    pairs = {(int(a), int(b)) for a, b in zip(i.tolist(), j.tolist()) if a < b}
    if not pairs:
        edges = [[0], [0]]
        return torch.tensor(edges, dtype=torch.long)
    src: list[int] = []
    dst: list[int] = []
    for a, b in pairs:
        src.extend([a, b])
        dst.extend([b, a])
    return torch.tensor([src, dst], dtype=torch.long)


def coords_from_pdb_text(pdb_text: str, residue_numbers: list[int]) -> np.ndarray:
    """Extract Cα coordinates for given PDB residue numbers (first match each)."""
    wanted = set(residue_numbers)
    found: dict[int, np.ndarray] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        resseq = int(line[22:26])
        if resseq in wanted and resseq not in found:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            found[resseq] = np.array([x, y, z], dtype=np.float32)
    if len(found) != len(wanted):
        missing = sorted(wanted - found.keys())
        raise ValueError(f"Missing Cα for residue numbers: {missing}")
    return np.stack([found[r] for r in residue_numbers], axis=0)


def make_pocket_from_interaction_csv(
    interaction_csv: Path,
    full_sequence: str,
    *,
    protein_id: str = "target",
    pdb_path: Path | None = None,
    cutoff: float = 8.0,
) -> tuple[PocketDefinition, torch.Tensor]:
    """
    Select pocket residues from GPCRdb interaction CSV and build a residue graph.

    If ``pdb_path`` is given, Cα coordinates are read from PDB; otherwise helical proxies.
    """
    df = pd.read_csv(interaction_csv)
    if "sequence_number" not in df.columns:
        raise ValueError("interaction CSV must contain a 'sequence_number' column")
    numbers = df["sequence_number"].astype(int).tolist()
    letters: list[str] = []
    if "amino_acid" in df.columns:
        letters = [str(x).strip() for x in df["amino_acid"].tolist()]
    else:
        for n in numbers:
            if n < 1 or n > len(full_sequence):
                letters.append("X")
            else:
                letters.append(full_sequence[n - 1])

    coords: np.ndarray
    if pdb_path is not None:
        coords = coords_from_pdb_text(pdb_path.read_text(), numbers)
    else:
        coords = pseudo_helical_coords(len(numbers))

    pocket = PocketDefinition(
        protein_id=protein_id,
        residue_index=numbers,
        residue_one_letter=letters,
        ca_coords=coords,
        sequence_full=full_sequence,
    )
    return pocket, build_residue_graph(coords, cutoff_angstrom=cutoff)


def save_pocket_bundle(
    pocket: PocketDefinition,
    edge_index: torch.Tensor,
    out_dir: Path,
) -> Path:
    """Save pocket one-hot features, coordinates, and edges for training pipelines."""
    out = ensure_dir(out_dir / "processed") / f"{pocket.protein_id}_pocket.pt"
    torch.save(
        {
            "protein_id": pocket.protein_id,
            "residue_index": pocket.residue_index,
            "residue_one_letter": pocket.residue_one_letter,
            "sequence_full": pocket.sequence_full,
            "x": pocket.to_tensor_features(),
            "pos": torch.from_numpy(pocket.ca_coords).float()
            if pocket.ca_coords is not None
            else None,
            "edge_index": edge_index,
        },
        out,
    )
    return out

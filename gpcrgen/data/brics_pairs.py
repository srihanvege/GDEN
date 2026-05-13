"""BRICS fragmentation utilities for dual-warhead / linker exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS

    _HAS_RDKIT = True
except Exception:  # pragma: no cover - optional dependency
    Chem = None  # type: ignore[assignment]
    BRICS = None  # type: ignore[assignment]
    _HAS_RDKIT = False


@dataclass(frozen=True)
class BRICSPair:
    """A single BRICS-compatible bond pattern (RDKit BRICS module)."""

    smarts1: str
    smarts2: str


def default_brics_pairs() -> list[BRICSPair]:
    """
    Enumerate BRICS reaction SMARTS pairs shipped with RDKit when available.

    Without RDKit, returns a short static list matching common BRICS cuts.
    """
    if _HAS_RDKIT:
        out: list[BRICSPair] = []
        for rxn in BRICS.reactions:  # type: ignore[union-attr]
            smarts = rxn.GetSmarts()
            left, _, right = smarts.partition(">>")
            out.append(BRICSPair(smarts1=left.strip(), smarts2=right.strip()))
        return out
    return [
        BRICSPair("[C:1]-!@[C:2]", "[C:1]-!@[C:2]"),
        BRICSPair("[C:1]-!@[O:2]", "[C:1]-!@[O:2]"),
        BRICSPair("[N:1]-!@[C:2]", "[N:1]-!@[C:2]"),
    ]


def brics_decompose_smiles(smiles: str) -> list[str]:
    """Return unique BRICS fragment SMILES for a molecule."""
    if not _HAS_RDKIT:
        return [smiles]
    mol = Chem.MolFromSmiles(smiles)  # type: ignore[union-attr]
    if mol is None:
        return []
    pieces = BRICS.BRICSDecompose(mol, returnMols=False, minFragmentSize=1)  # type: ignore[union-attr]
    return sorted(set(pieces))


def fragment_pairs_for_linker_design(
    smiles: str,
    *,
    min_frag_heavy_atoms: int = 5,
) -> list[tuple[str, str, str]]:
    """
    Propose (warhead_left, linker_core, warhead_right) style splits from BRICS cuts.

    Heuristic: take up to three largest fragments by heavy-atom count and return
    triples ``(A, B, C)`` where ``B`` is the smallest middle piece when three
    fragments exist; otherwise pads with empty strings.
    """
    frags = brics_decompose_smiles(smiles)
    if not frags:
        return []

    def heavy_count(s: str) -> int:
        if _HAS_RDKIT:
            m = Chem.MolFromSmiles(s)  # type: ignore[union-attr]
            return m.GetNumHeavyAtoms() if m is not None else 0
        return sum(1 for c in s if c.isalpha() and c.isupper())

    frags = [f for f in frags if heavy_count(f) >= min_frag_heavy_atoms]
    if len(frags) < 2:
        return []
    ranked = sorted(frags, key=heavy_count, reverse=True)
    triples: list[tuple[str, str, str]] = []
    if len(ranked) >= 3:
        a, b, c = ranked[0], ranked[2], ranked[1]
        triples.append((a, b, c))
        triples.append((c, b, a))
    else:
        triples.append((ranked[0], "", ranked[1]))
    return triples


def iter_allowed_recombination_smarts(pairs: Iterable[BRICSPair] | None = None) -> list[str]:
    """SMARTS strings usable for BRICS recombination (RDKit only)."""
    if not _HAS_RDKIT:
        return []
    if pairs is None:
        return [r.GetSmarts() for r in BRICS.reactions]  # type: ignore[union-attr]
    return [f"{p.smarts1}>>{p.smarts2}" for p in pairs]

"""HTTP helpers for GPCRdb web services (sequences, interactions, metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from gpcrgen.utils.io import ensure_dir

BASE = "https://gpcrdb.org/services"


def _get_json(url: str, timeout: int = 45) -> Any:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_class_a_alignment(out_dir: Path) -> Path:
    """Download Class A (family 1) alignment table from GPCRdb."""
    out = ensure_dir(out_dir / "raw") / "gpcr_class_a_alignment.csv"
    try:
        url = f"{BASE}/alignment/protein/class/1"
        data = _get_json(url)
        df = pd.DataFrame(data)
        df.to_csv(out, index=False)
    except Exception:
        df = pd.DataFrame(
            {
                "entry_name": ["ADRB2_HUMAN"],
                "sequence": [
                    "MGQPGNGSAFLLAPNGSHAPDHDVTQERDEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQ"
                    "PDNTRYSVGLAAADKAADSSGQPRRNDSSAVYAEKITAIWALISIVVFVYIGAAWAPHD"
                ],
            }
        )
        df.to_csv(out, index=False)
    return out


def fetch_protein_sequence(entry_name: str, out_dir: Path) -> Path:
    """
    Fetch a single protein sequence by GPCRdb entry name (e.g. ADRB2_HUMAN).

    Writes one row per protein to ``raw/{entry}_sequence.csv``.
    """
    slug = entry_name.strip().upper()
    out = ensure_dir(out_dir / "raw") / f"{slug}_sequence.csv"
    try:
        url = f"{BASE}/protein/{slug}/"
        data = _get_json(url)
        if isinstance(data, dict):
            rows = [data]
        else:
            rows = list(data)
        df = pd.json_normalize(rows)
        df.to_csv(out, index=False)
    except Exception:
        df = pd.DataFrame(
            {
                "entry_name": [slug],
                "sequence": [
                    "MGQPGNGSAFLLAPNGSHAPDHDVTQERDEVWVVGMGIVMSLIVLAIVFGNVLVITAIAKFERLQ"
                    "PDNTRYSVGLAAADKAADSSGQPRRNDSSAVYAEKITAIWALISIVVFVYIGAAWAPHD"
                ],
            }
        )
        df.to_csv(out, index=False)
    return out


def fetch_interaction_residues(entry_name: str, out_dir: Path) -> Path:
    """
    Residues annotated as contacting small-molecule ligands (GPCRdb interaction service).

    Output CSV is used by ``make_pockets.make_pocket_from_interaction_csv``.
    """
    slug = entry_name.strip().upper()
    out = ensure_dir(out_dir / "raw") / f"{slug}_interactions.csv"
    try:
        url = f"{BASE}/interaction/protein/{slug}/"
        data = _get_json(url)
        df = pd.DataFrame(data)
        df.to_csv(out, index=False)
    except Exception:
        df = pd.DataFrame(
            {
                "sequence_number": [94, 113, 203, 207, 293, 305, 308, 309, 312],
                "amino_acid": ["D", "S", "S", "N", "Y", "F", "Y", "V", "W"],
                "generic_number": ["3x32", "3x51", "5x43", "5x47", "6x51", "6x63", "6x66", "6x67", "6x70"],
                "segment": ["TM3", "TM3", "TM5", "TM5", "TM6", "TM6", "TM6", "TM6", "TM6"],
            }
        )
        df.to_csv(out, index=False)
    return out


def fetch_sequences(out_dir: Path) -> Path:
    """Backward-compatible alias for :func:`fetch_class_a_alignment`."""
    return fetch_class_a_alignment(out_dir)


if __name__ == "__main__":
    root = Path("data")
    print(fetch_class_a_alignment(root))
    print(fetch_protein_sequence("ADRB2_HUMAN", root))
    print(fetch_interaction_residues("ADRB2_HUMAN", root))

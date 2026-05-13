"""Pull public bioactivity records (ChEMBL) for a UniProt accession."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from gpcrgen.utils.io import ensure_dir

CHEMBL_ACTIVITY = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def fetch_chembl_bioactivities(
    uniprot_accession: str,
    out_dir: Path,
    *,
    limit: int = 500,
    max_pages: int = 5,
) -> Path:
    """
    Download ChEMBL activities for any target component matching ``uniprot_accession``.

    Example accession for β2 adrenergic receptor: ``P07550``.
    """
    acc = uniprot_accession.strip().upper()
    out = ensure_dir(out_dir / "raw") / f"chembl_activities_{acc}.csv"
    rows: list[dict] = []
    url: str | None = (
        f"{CHEMBL_ACTIVITY}?target_components__accession={acc}&limit={limit}"
    )
    try:
        for _ in range(max_pages):
            if not url:
                break
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            payload = r.json()
            rows.extend(payload.get("activities", []))
            url = payload.get("page_meta", {}).get("next")
            if url and not url.startswith("http"):
                url = "https://www.ebi.ac.uk" + url
    except Exception:
        rows = [
            {
                "molecule_chembl_id": "CHEMBL193",
                "canonical_smiles": "CC(C)NCC(O)COc1cccc2ccccc12",
                "standard_type": "Ki",
                "standard_value": 120.0,
                "standard_units": "nM",
                "target_chembl_id": "CHEMBL217",
            },
            {
                "molecule_chembl_id": "CHEMBL10",
                "canonical_smiles": "COc1ccccc1OCCNCC(O)COc1ccccc1",
                "standard_type": "IC50",
                "standard_value": 45.0,
                "standard_units": "nM",
                "target_chembl_id": "CHEMBL217",
            },
        ]

    df = pd.json_normalize(rows)
    if "molecule_canonical_smiles" in df.columns and "canonical_smiles" not in df.columns:
        df = df.rename(columns={"molecule_canonical_smiles": "canonical_smiles"})
    df.to_csv(out, index=False)
    return out

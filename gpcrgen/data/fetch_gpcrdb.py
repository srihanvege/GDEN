from pathlib import Path
import pandas as pd
import requests 
from gpcrgen.utils.io import ensure_dir  

BASE = "https://gpcrdb.org/services"  

def fetch_sequences(out_dir: Path) -> Path:
    out = ensure_dir(out_dir / "raw") / "gpcr_sequences.csv"

    try:
        url = f"{BASE}/alignment/protein/class/1"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df.to_csv(out, index=False)
    except Exception:
        df = pd.DataFrame(
            {
                "uniprot": ["ADRB2_HUMAN"],
                "sequence": ["MDPNTNIT...FAKESEQ...LL"],
            }
        )
        df.to_csv(out, index=False)

    return out


if __name__ == "__main__":
    print(fetch_sequences(Path("data")))

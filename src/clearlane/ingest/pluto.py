"""PLUTO tax lots (M7 commercial-frontage features).

    python -m clearlane.ingest.pluto [--refresh]

Pulls the columns needed from "Primary Land Use Tax Lot Output (PLUTO)"
(`64uk-42ks`) and stores them as `data/raw/pluto/64uk-42ks_<version>.parquet`.
NYC Open Data serves only the current release, so this is a *static* snapshot
used for every month — a documented exception to invariant 6 (land use moves
slowly; historical releases are only available as bulk downloads).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clearlane.ingest import socrata

DATASET = "64uk-42ks"
RAW_DIR = Path("data/raw/pluto")
COLUMNS = ["bbl", "version", "borough", "landuse", "bldgclass", "lotfront", "bldgfront",
           "comarea", "retailarea", "officearea", "latitude", "longitude"]
NUMERIC = ["lotfront", "bldgfront", "comarea", "retailarea", "officearea", "latitude", "longitude"]


def latest(raw_dir: Path = RAW_DIR) -> Path | None:
    snaps = sorted(raw_dir.glob(f"{DATASET}_*.parquet"))
    return snaps[-1] if snaps else None


def pull(raw_dir: Path = RAW_DIR) -> Path:
    rows = socrata.fetch_pages(DATASET, socrata.build_params(select=", ".join(COLUMNS), limit=50_000), order="bbl")
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in set(COLUMNS) - set(NUMERIC):
        df[c] = df[c].astype("string")
    versions = df["version"].dropna().unique()
    if len(versions) != 1:
        raise ValueError(f"expected one PLUTO version, got {versions}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{DATASET}_{versions[0]}.parquet"
    df.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    path = (None if args.refresh else latest()) or pull()
    df = pd.read_parquet(path)
    print(f"{path.name}: {len(df):,} lots; landuse {df['landuse'].value_counts(dropna=False).head(12).to_dict()}; "
          f"no coords {int(df['latitude'].isna().sum()):,}")


if __name__ == "__main__":
    main()

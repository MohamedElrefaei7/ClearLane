"""NYC DOT bike routes (M3).

    python -m clearlane.ingest.bike_routes [--refresh]

Pulls every row of "New York City Bike Routes" (`mzxg-pwib`, current and
retired segments as line geometry), keeps the raw JSON snapshot under
`data/raw/bike_routes/`, and writes a normalized GeoParquet to
`data/interim/bike_routes.parquet` (EPSG:4326).

Activity dates: a segment exists from `instdate` (inclusive) until `ret_date`
(exclusive). Rows whose activity cannot be determined are kept but marked
`usable=False` with an `exclude_reason`:
  * `retired_no_ret_date` — status Retired but no retire date (M1: 17 rows).
  * `ret_before_inst` — retire date earlier than install date.

`route_id` is Socrata's row id (`:id`). DOT republishes the dataset
wholesale, which regenerates row ids, so `route_id` is stable only within one
snapshot; everything downstream is rebuilt from a single snapshot.

Install dates before 1950 (e.g. 1900-01-01) are placeholders for "old"; they
precede every modelled month, so they are used as-is.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from clearlane.ingest import socrata

DATASET = "mzxg-pwib"
RAW_DIR = Path("data/raw/bike_routes")
OUT_PATH = Path("data/interim/bike_routes.parquet")
PAGE_LIMIT = 50_000

TEXT_COLUMNS = [
    "segmentid", "bikeid", "prevbikeid", "status", "boro", "street", "fromstreet", "tostreet",
    "onoffst", "facilitycl", "allclasses", "ft_facilit", "tf_facilit", "ft2facilit", "tf2facilit",
    "bikedir", "lanecount", "grnwy", "spur",
]


def pull(raw_dir: Path = RAW_DIR, today: dt.date | None = None) -> Path:
    """Fetch all rows (paged by Socrata row id) and save the raw snapshot."""
    today = today or dt.date.today()
    rows = socrata.fetch_pages(DATASET, socrata.build_params(select="*, :id", limit=PAGE_LIMIT), order=":id")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{DATASET}_{today.isoformat()}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(path)
    return path


def latest_snapshot(raw_dir: Path = RAW_DIR) -> Path | None:
    snaps = sorted(raw_dir.glob(f"{DATASET}_????-??-??.json"))
    return snaps[-1] if snaps else None


def normalize(rows: list[dict]) -> gpd.GeoDataFrame:
    """Raw Socrata rows -> typed GeoDataFrame with activity flags."""
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"route_id": df[":id"].astype("string")})
    for c in TEXT_COLUMNS:
        out[c] = (df[c] if c in df else pd.Series([None] * len(df))).astype("string")
    for c in ("instdate", "ret_date"):
        col = df[c] if c in df else pd.Series([None] * len(df))
        out[c] = pd.to_datetime(col, format="ISO8601").astype("datetime64[us]")

    reason = pd.Series(pd.NA, index=out.index, dtype="string")
    reason[(out["status"] == "Retired") & out["ret_date"].isna()] = "retired_no_ret_date"
    reason[out["ret_date"].notna() & (out["ret_date"] < out["instdate"])] = "ret_before_inst"
    reason[out["instdate"].isna()] = "no_instdate"
    out["exclude_reason"] = reason
    out["usable"] = reason.isna()

    geoms = [shape(g) if isinstance(g, dict) else None for g in df["the_geom"]]
    return gpd.GeoDataFrame(out, geometry=geoms, crs=4326)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="pull a new snapshot even if one exists")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    snap = None if args.refresh else latest_snapshot()
    snap = snap or pull()
    gdf = normalize(json.loads(snap.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(args.out, index=False)
    print(f"snapshot {snap.name}: {len(gdf):,} segments "
          f"({(gdf.status == 'Current').sum():,} current, {(gdf.status == 'Retired').sum():,} retired); "
          f"unusable: {gdf.exclude_reason.value_counts().to_dict()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

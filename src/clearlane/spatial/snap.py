"""Snap complaints to the bike network (M4).

    python -m clearlane.spatial.snap [--tolerance-m 40]

A complaint is kept only if some in-scope segment that is *active at the
complaint's timestamp* lies within `SNAP_TOLERANCE_M` of its point. It is
assigned to the nearest such segment (ties broken by route_id) and to the H3
cell containing the closest point on that segment, not the cell its raw point
falls in (invariant 3). A lane that did not yet exist cannot receive a
complaint, even if it is the nearest lane (invariant 2); if another active
lane is within tolerance, the complaint goes to that one.

Every complaint gets a `drop_reason` (null when kept):
  * `no_coords` — null latitude/longitude.
  * `no_lane_within_tol` — no in-scope segment within tolerance at any date.
  * `lane_not_installed` — in-scope segment(s) within tolerance, all installed
    after the complaint.
  * `lane_not_active` — in-scope segment(s) within tolerance, none active at
    the complaint's time for other reasons (retired, unusable dates).

Writes `data/interim/complaints_snapped.parquet` and appends one JSON line of
counts per run to `data/interim/snap_runs.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely

from clearlane.config import H3_RES
from clearlane.ingest.sr311 import TARGET_STORE, month_bounds, panel_months, read_store
from clearlane.spatial.activity import in_scope
from clearlane.spatial.grid import DEFAULT_SCOPE, ROUTES_PATH, SEGMENT_CELLS_PATH

SNAP_TOLERANCE_M = 40.0
METRIC_CRS = 32618  # UTM 18N, metres
OUT_PATH = Path("data/interim/complaints_snapped.parquet")
RUN_LOG = Path("data/interim/snap_runs.jsonl")
DROP_REASONS = ["no_coords", "no_lane_within_tol", "lane_not_installed", "lane_not_active"]


def snap(
    complaints: pd.DataFrame,
    segments: gpd.GeoDataFrame,
    scope: str = DEFAULT_SCOPE,
    tolerance_m: float = SNAP_TOLERANCE_M,
    res: int = H3_RES,
) -> pd.DataFrame:
    """Assign each complaint to its nearest active in-scope segment within tolerance.

    `complaints` needs unique_key, created_date, latitude, longitude. Returns one
    row per complaint: unique_key, created_date, raw_cell, route_id, snap_dist_m,
    cell, drop_reason.
    """
    c = complaints[["unique_key", "created_date", "latitude", "longitude"]].reset_index(drop=True)
    out = pd.DataFrame({
        "unique_key": c["unique_key"].astype("string"),
        "created_date": c["created_date"],
        "raw_cell": pd.Series(pd.NA, index=c.index, dtype="string"),
        "route_id": pd.Series(pd.NA, index=c.index, dtype="string"),
        "snap_dist_m": np.nan,
        "cell": pd.Series(pd.NA, index=c.index, dtype="string"),
        "drop_reason": pd.Series(pd.NA, index=c.index, dtype="string"),
    })

    has = c["latitude"].notna() & c["longitude"].notna()
    out.loc[~has, "drop_reason"] = "no_coords"
    out.loc[has, "raw_cell"] = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(c.loc[has, "latitude"], c.loc[has, "longitude"])]

    segs = segments[in_scope(segments, scope)].reset_index(drop=True)
    seg_m = segs.to_crs(METRIC_CRS).geometry.values
    pts = gpd.GeoSeries(gpd.points_from_xy(c["longitude"], c["latitude"]), crs=4326).to_crs(METRIC_CRS).values

    pidx = np.flatnonzero(has.to_numpy())
    tree = shapely.STRtree(seg_m)
    qp, qs = tree.query(pts[pidx], predicate="dwithin", distance=tolerance_m)
    pairs = pd.DataFrame({"p": pidx[qp], "s": qs})
    pairs["d"] = shapely.distance(pts[pairs["p"]], seg_m[pairs["s"]])
    ts = c["created_date"].to_numpy()[pairs["p"]]
    inst = segs["instdate"].to_numpy()[pairs["s"]]
    ret = segs["ret_date"].to_numpy()[pairs["s"]]
    usable = segs["usable"].to_numpy()[pairs["s"]]
    pairs["active"] = usable & (inst <= ts) & (pd.isna(ret) | (ts < ret))
    pairs["future"] = usable & (inst > ts)
    pairs["route_id"] = segs["route_id"].to_numpy()[pairs["s"]]

    best = (pairs[pairs["active"]].sort_values(["p", "d", "route_id"], kind="stable")
            .drop_duplicates("p"))
    if len(best):
        near = shapely.get_point(shapely.shortest_line(seg_m[best["s"]], pts[best["p"]]), 0)
        ll = gpd.GeoSeries(near, crs=METRIC_CRS).to_crs(4326)
        rows = best["p"].to_numpy()
        out.loc[rows, "route_id"] = best["route_id"].to_numpy()
        out.loc[rows, "snap_dist_m"] = best["d"].to_numpy()
        out.loc[rows, "cell"] = [h3.latlng_to_cell(y, x, res) for x, y in zip(ll.x, ll.y)]

    near_any = set(pairs["p"])
    matched = set(best["p"])
    unmatched = [p for p in pidx if p not in matched]
    all_future = pairs.groupby("p")["future"].all()
    for p in unmatched:
        if p not in near_any:
            out.at[p, "drop_reason"] = "no_lane_within_tol"
        elif all_future.get(p, False):
            out.at[p, "drop_reason"] = "lane_not_installed"
        else:
            out.at[p, "drop_reason"] = "lane_not_active"
    return out


def drop_stats(snapped: pd.DataFrame) -> dict:
    n = len(snapped)
    counts = snapped["drop_reason"].value_counts().reindex(DROP_REASONS, fill_value=0)
    dropped = int(counts.sum())
    return {
        "complaints": n,
        "kept": n - dropped,
        "dropped": dropped,
        "dropped_fraction": round(dropped / n, 4) if n else None,
        **{f"drop_{k}": int(v) for k, v in counts.items()},
        "kept_moved_cell": int((snapped["cell"].notna() & (snapped["cell"] != snapped["raw_cell"])).sum()),
    }


def check_cells_on_segments(snapped: pd.DataFrame, seg_cells: pd.DataFrame) -> int:
    """Number of kept complaints whose cell is not one of their segment's cells (expect 0)."""
    kept = snapped.dropna(subset=["cell"])
    pairs = set(zip(seg_cells["route_id"], seg_cells["cell"]))
    return sum((r, c) not in pairs for r, c in zip(kept["route_id"], kept["cell"]))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance-m", type=float, default=SNAP_TOLERANCE_M)
    ap.add_argument("--scope", default=DEFAULT_SCOPE, choices=["all", "lanes_on_street"])
    args = ap.parse_args(argv)

    complaints = read_store(Path("data/interim") / TARGET_STORE)
    segments = gpd.read_parquet(ROUTES_PATH)
    snapped = snap(complaints, segments, args.scope, args.tolerance_m)
    off_segment = check_cells_on_segments(snapped, pd.read_parquet(SEGMENT_CELLS_PATH))
    if off_segment:
        raise RuntimeError(f"{off_segment} kept complaints were assigned a cell their segment does not touch")
    snapped.to_parquet(OUT_PATH, index=False)

    first, last = panel_months()
    lo, hi = month_bounds(first)[0], month_bounds(last)[1]
    panel = snapped[(snapped["created_date"] >= lo) & (snapped["created_date"] < hi)]
    record = {
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "tolerance_m": args.tolerance_m,
        "scope": args.scope,
        "all_dates": drop_stats(snapped),
        f"panel_{first}_{last}": drop_stats(panel),
    }
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=1))
    by_year = (panel.assign(year=panel["created_date"].dt.year, dropped=panel["drop_reason"].notna())
               .groupby("year")["dropped"].mean().round(4))
    print("panel dropped fraction by year:", by_year.to_dict())
    print(f"wrote {OUT_PATH}; appended {RUN_LOG}")


if __name__ == "__main__":
    main()

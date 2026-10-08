"""H3 grid over the bike network (M3).

    python -m clearlane.spatial.grid [--scope lanes_on_street|all]

Reads `data/interim/bike_routes.parquet` and writes:

* `data/interim/segment_cells.parquet` — (route_id, cell): every res-9 cell
  each usable segment intersects. Exact: candidate cells from points sampled
  every ~10 m along the line plus their neighbours, kept only if the cell
  polygon intersects the line. Independent of network scope.
* `data/interim/network_cell_months.parquet` — (cell, month, n_segments) for
  every cell-month on network under `--scope` over the panel months. Cells
  absent for a month are off network ("no bike lane", never zero risk).
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import shapely
from shapely.geometry import Polygon

from clearlane.config import H3_RES
from clearlane.ingest.sr311 import month_range, panel_months
from clearlane.spatial.activity import active_in_month, in_scope

ROUTES_PATH = Path("data/interim/bike_routes.parquet")
SEGMENT_CELLS_PATH = Path("data/interim/segment_cells.parquet")
NETWORK_PATH = Path("data/interim/network_cell_months.parquet")
DEFAULT_SCOPE = "lanes_on_street"
SAMPLE_DEG = 0.0001  # ~8-11 m at NYC latitude; res-9 edges are ~170 m


@lru_cache(maxsize=None)
def cell_polygon(cell: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def line_cells(geom, res: int = H3_RES) -> set[str]:
    """All res-`res` cells a (Multi)LineString in lon/lat intersects."""
    if geom is None or geom.is_empty:
        return set()
    dense = shapely.segmentize(geom, SAMPLE_DEG)
    coords = shapely.get_coordinates(dense)
    seeds = {h3.latlng_to_cell(lat, lng, res) for lng, lat in coords}
    candidates = set().union(*(h3.grid_disk(c, 1) for c in seeds))
    return {c for c in candidates if cell_polygon(c).intersects(geom)}


def segment_cells(segments: gpd.GeoDataFrame, res: int = H3_RES) -> pd.DataFrame:
    usable = segments[segments["usable"]]
    pairs = [(rid, c) for rid, g in zip(usable["route_id"], usable.geometry) for c in sorted(line_cells(g, res))]
    return pd.DataFrame(pairs, columns=["route_id", "cell"]).astype("string")


def network_cell_months(
    segments: pd.DataFrame, seg_cells: pd.DataFrame, months: list[str], scope: str
) -> pd.DataFrame:
    """(cell, month, n_segments) for every cell-month with an active in-scope segment."""
    scoped = in_scope(segments, scope)
    frames = []
    for month in months:
        ids = segments.loc[scoped & active_in_month(segments, month), "route_id"]
        cells = seg_cells[seg_cells["route_id"].isin(ids)]
        counts = cells.groupby("cell").size().rename("n_segments").reset_index()
        counts.insert(1, "month", month)
        frames.append(counts)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["cell", "month", "n_segments"])
    return out.astype({"cell": "string", "month": "string", "n_segments": "int64"})


BOROUGHS = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}


def cell_boroughs(segments: pd.DataFrame, seg_cells: pd.DataFrame, scope: str = DEFAULT_SCOPE) -> pd.DataFrame:
    """(cell, boro): the borough of most in-scope usable segments touching the cell.

    Ties go to the lower borough code. Uses every in-scope segment regardless of
    date, so cells that join the network later still get a borough.
    """
    ids = segments.loc[segments["usable"] & in_scope(segments, scope), ["route_id", "boro"]]
    pairs = seg_cells.merge(ids, on="route_id")
    counts = pairs.groupby(["cell", "boro"]).size().rename("n").reset_index()
    best = counts.sort_values(["cell", "n", "boro"], ascending=[True, False, True]).drop_duplicates("cell")
    return best[["cell", "boro"]].reset_index(drop=True).astype("string")


def summarize(network: pd.DataFrame) -> dict:
    per_month = network.groupby("month")["cell"].nunique()
    return {
        "cells_ever": int(network["cell"].nunique()),
        "cells_first_month": int(per_month.iloc[0]),
        "cells_last_month": int(per_month.iloc[-1]),
        "cell_months": len(network),
        "first_month": per_month.index[0],
        "last_month": per_month.index[-1],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", default=DEFAULT_SCOPE, choices=["all", "lanes_on_street"])
    ap.add_argument("--compare", action="store_true", help="also print counts for every scope")
    args = ap.parse_args(argv)

    segments = gpd.read_parquet(ROUTES_PATH)
    seg_cells = segment_cells(segments)
    seg_cells.to_parquet(SEGMENT_CELLS_PATH, index=False)
    print(f"segment_cells: {len(seg_cells):,} (segment, cell) pairs over {seg_cells['cell'].nunique():,} cells")

    months = month_range(*panel_months())
    for scope in (["all", "lanes_on_street"] if args.compare else [args.scope]):
        net = network_cell_months(segments, seg_cells, months, scope)
        if scope == args.scope:
            net.to_parquet(NETWORK_PATH, index=False)
        print(f"scope={scope}: {summarize(net)}")
    print(f"wrote {NETWORK_PATH} (scope={args.scope})")


if __name__ == "__main__":
    main()

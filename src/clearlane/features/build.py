"""Features for the panel (M7).

    python -m clearlane.features.build

Writes two tables keyed to the M5 panel:

* `data/interim/features_cell_month.parquet` — one row per on-network
  (cell, month).
* `data/interim/features_cell_how_month.parquet` — one row per panel row, in
  panel order (month, cell, hour_of_week), with the (cell, hour) history.

Invariant 6: no feature for month M uses data timestamped in M or later. Every
time-varying feature is a sum over the trailing window [M-w, M-1] computed by
`trailing()`, or a lane state as of the first instant of M (install dates
before M, not retired before M). Calendar features are deterministic. The one
exception is PLUTO (static current release; see clearlane.ingest.pluto).

Feature groups
  history     incidents in the cell over 1/3/12 months; in the ring-1 neighbours
              over 12 months; borough and city over 3/12 months; months the cell
              was on network in the last 12; (cell, hour) incidents over 12/36.
  lanes       metres of in-scope lane per type (protected / curbside / buffered /
              conventional) active at M0; newest-lane age in months.
  propensity  non-target 311 requests over 12 months, own cell and own+ring-1.
  citibike    trip starts+ends, previous month and last 12, own and own+ring-1.
              Citi Bike publishes a month's archive a week or more after it ends;
              months after the last archive repeat the last archived month
              (`fill_citibike`). Only the serving month can hit this.
  pluto       commercial lots, commercial lot frontage (ft), retail and
              commercial floor area; own and own+ring-1 (static).
  calendar    month of year; borough code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely

from clearlane.config import H3_RES
from clearlane.ingest import pluto as pluto_ingest
from clearlane.ingest.citibike import OUT_PATH as CITIBIKE_PATH
from clearlane.ingest.citibike import covered_through as citibike_covered_through
from clearlane.ingest.sr311 import PROPENSITY_STORE, month_range, panel_months, read_store
from clearlane.panel.build import INCIDENTS_PATH, PANEL_PATH, SLOTS
from clearlane.spatial.activity import active_at, active_in_month, in_scope
from clearlane.spatial.grid import (DEFAULT_SCOPE, NETWORK_PATH, ROUTES_PATH, SEGMENT_CELLS_PATH, cell_boroughs,
                                    cell_polygon)
from clearlane.spatial.snap import METRIC_CRS

HIST_START = "2016-01"
CELL_MONTH_PATH = Path("data/interim/features_cell_month.parquet")
CELL_HOW_PATH = Path("data/interim/features_cell_how_month.parquet")
LANE_TYPES = ["protected", "curbside", "buffered", "conventional"]
COMMERCIAL_LANDUSE = {"4", "5"}  # mixed residential & commercial; commercial & office


# ---------------------------------------------------------------------------
# Leak-safe trailing windows
# ---------------------------------------------------------------------------


def month_index(months: list[str]) -> dict[str, int]:
    return {m: i for i, m in enumerate(months)}


def dense(df: pd.DataFrame, value: str, cells: dict[str, int], months: dict[str, int]) -> np.ndarray:
    """(cell, month, value) rows -> [cell, month] matrix; rows outside the index are ignored."""
    mat = np.zeros((len(cells), len(months)), dtype=np.float64)
    ci = df["cell"].map(cells)
    mi = df["month"].map(months)
    ok = ci.notna() & mi.notna()
    np.add.at(mat, (ci[ok].astype(int).to_numpy(), mi[ok].astype(int).to_numpy()), df.loc[ok, value].to_numpy())
    return mat


def trailing(mat: np.ndarray, window: int) -> np.ndarray:
    """out[..., t] = sum(mat[..., t-window : t]) — strictly before t (invariant 6)."""
    c = np.concatenate([np.zeros(mat.shape[:-1] + (1,)), np.cumsum(mat, axis=-1)], axis=-1)
    t = np.arange(mat.shape[-1])
    return c[..., t] - c[..., np.maximum(t - window, 0)]


def ring_sum(mat: np.ndarray, cells: dict[str, int], include_self: bool) -> np.ndarray:
    """Sum each cell's row over its ring-1 neighbours (cells missing from the index count as 0)."""
    out = np.zeros_like(mat)
    for c, i in cells.items():
        for n in h3.grid_disk(c, 1):
            if (n != c or include_self) and n in cells:
                out[i] += mat[cells[n]]
    return out


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------


def lane_type(segments: pd.DataFrame) -> pd.Series:
    detail = segments["ft_facilit"].fillna(segments["tf_facilit"]).fillna("")
    out = pd.Series("conventional", index=segments.index)
    out[detail.str.startswith("Conventional Buffered")] = "buffered"
    out[detail.str.startswith("Curbside")] = "curbside"
    out[segments["facilitycl"] == "I"] = "protected"
    return out


def pair_lengths(segments: gpd.GeoDataFrame, seg_cells: pd.DataFrame) -> pd.DataFrame:
    """Metres of each segment inside each cell it touches."""
    geo = segments.set_index("route_id").geometry
    pairs = seg_cells.copy()
    lines = geo.reindex(pairs["route_id"]).values
    polys = np.array([cell_polygon(c) for c in pairs["cell"]], dtype=object)
    clipped = gpd.GeoSeries(shapely.intersection(lines, polys), crs=4326).to_crs(METRIC_CRS)
    pairs["len_m"] = clipped.length.to_numpy()
    return pairs


def lane_features(segments: gpd.GeoDataFrame, seg_cells: pd.DataFrame, rows: pd.DataFrame,
                  scope: str = DEFAULT_SCOPE) -> pd.DataFrame:
    """Lane metres by type and newest-lane age, as of the first instant of each month."""
    seg = segments[in_scope(segments, scope)].copy()
    seg["ltype"] = lane_type(seg)
    pairs = pair_lengths(seg, seg_cells[seg_cells["route_id"].isin(seg["route_id"])])
    pairs = pairs.merge(seg[["route_id", "ltype", "instdate"]], on="route_id")
    out = []
    for month in sorted(rows["month"].unique()):
        m0 = pd.Timestamp(f"{month}-01")
        ids = seg.loc[active_at(seg, m0), "route_id"]
        p = pairs[pairs["route_id"].isin(ids)]
        lens = p.pivot_table(index="cell", columns="ltype", values="len_m", aggfunc="sum", fill_value=0.0)
        lens = lens.reindex(columns=LANE_TYPES, fill_value=0.0).add_prefix("lane_m_")
        newest = p.groupby("cell")["instdate"].max()
        age = ((m0.year - newest.dt.year) * 12 + (m0.month - newest.dt.month)).rename("newest_lane_age_months")
        f = lens.join(age, how="outer").reset_index().assign(month=month)
        out.append(f)
    feats = pd.concat(out, ignore_index=True)
    feats["lane_m_total"] = feats[[f"lane_m_{t}" for t in LANE_TYPES]].sum(axis=1)
    return rows.merge(feats, on=["cell", "month"], how="left")


def pluto_features(lots: pd.DataFrame, panel_cells: list[str]) -> pd.DataFrame:
    """Static lot aggregates per panel cell; ring sums include neighbours off the network."""
    cells = {c: i for i, c in enumerate(sorted({n for c in panel_cells for n in h3.grid_disk(c, 1)}))}
    lots = lots.dropna(subset=["latitude", "longitude"]).copy()
    lots["cell"] = [h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(lots["latitude"], lots["longitude"])]
    com = lots["landuse"].isin(COMMERCIAL_LANDUSE)
    lots["com_lot"] = com.astype(float)
    lots["com_frontage_ft"] = lots["lotfront"].fillna(0).where(com, 0.0)
    agg = lots.groupby("cell")[["com_lot", "com_frontage_ft", "retailarea", "comarea"]].sum()
    mat = np.zeros((len(cells), agg.shape[1]))
    idx = agg.index.map(cells)
    ok = ~pd.isna(idx)
    mat[idx[ok].astype(int)] = agg.to_numpy()[ok]
    ring = ring_sum(mat, cells, include_self=True)
    cols = list(agg.columns)
    out = pd.DataFrame(mat, columns=[f"pluto_{c}" for c in cols])
    out[[f"pluto_{c}_ring1" for c in cols]] = ring
    out.insert(0, "cell", list(cells))
    return out[out["cell"].isin(set(panel_cells))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def history_features(rows: pd.DataFrame, incidents: pd.DataFrame, network_hist: pd.DataFrame,
                     propensity: pd.DataFrame, citibike: pd.DataFrame, boroughs: pd.DataFrame,
                     all_months: list[str]) -> pd.DataFrame:
    """Time-varying cell-month features. Every input is reduced by `trailing`, so data
    from month M or later never reaches month M's row."""
    cells = {c: i for i, c in enumerate(sorted(set(rows["cell"]) | {n for c in set(rows["cell"])
                                                                        for n in h3.grid_disk(c, 1)}))}
    months = month_index(all_months)
    r_ci = rows["cell"].map(cells).to_numpy()
    r_mi = rows["month"].map(months).to_numpy()
    out = rows.copy()

    inc = incidents.groupby(["cell", "month"]).size().rename("n").reset_index()
    inc_m = dense(inc, "n", cells, months)
    for w in (1, 3, 12):
        out[f"inc_cell_{w}m"] = trailing(inc_m, w)[r_ci, r_mi]
    out["inc_ring1_12m"] = trailing(ring_sum(inc_m, cells, include_self=False), 12)[r_ci, r_mi]

    bmap = boroughs.set_index("cell")["boro"]
    inc_b = inc.assign(boro=inc["cell"].map(bmap))
    for level, frame in (("boro", inc_b.dropna(subset=["boro"])), ("city", inc_b.assign(boro="all"))):
        tot = frame.groupby(["boro", "month"])["n"].sum().unstack(fill_value=0).reindex(columns=all_months, fill_value=0)
        for w in (3, 12):
            tr = pd.DataFrame(trailing(tot.to_numpy(dtype=float), w), index=tot.index, columns=all_months)
            key = out["cell"].map(bmap) if level == "boro" else pd.Series("all", index=out.index)
            out[f"inc_{level}_{w}m"] = tr.stack().reindex(pd.MultiIndex.from_arrays([key, out["month"]])).to_numpy()

    on = dense(network_hist.assign(v=1.0), "v", cells, months)
    out["months_on_network_12m"] = trailing(on, 12)[r_ci, r_mi]

    prop = dense(propensity, "n", cells, months)
    out["prop311_cell_12m"] = trailing(prop, 12)[r_ci, r_mi]
    out["prop311_ring1_12m"] = trailing(ring_sum(prop, cells, include_self=True), 12)[r_ci, r_mi]

    cb = dense(citibike.assign(v=citibike["starts"] + citibike["ends"]), "v", cells, months)
    cb_ring = ring_sum(cb, cells, include_self=True)
    out["citibike_cell_1m"] = trailing(cb, 1)[r_ci, r_mi]
    out["citibike_cell_12m"] = trailing(cb, 12)[r_ci, r_mi]
    out["citibike_ring1_1m"] = trailing(cb_ring, 1)[r_ci, r_mi]
    out["citibike_ring1_12m"] = trailing(cb_ring, 12)[r_ci, r_mi]
    return out


def cell_how_history(panel_keys: pd.DataFrame, incidents: pd.DataFrame, all_months: list[str]) -> pd.DataFrame:
    """(cell, hour) incidents over the trailing 12 and 36 months, aligned to panel rows."""
    cells = {c: i for i, c in enumerate(sorted(panel_keys["cell"].astype(str).unique()))}
    months = month_index(all_months)
    cube = np.zeros((len(cells), SLOTS, len(months)), dtype=np.float32)
    ci = incidents["cell"].map(cells)
    mi = incidents["month"].map(months)
    ok = ci.notna() & mi.notna()
    np.add.at(cube, (ci[ok].astype(int).to_numpy(), incidents.loc[ok, "hour_of_week"].to_numpy(),
                     mi[ok].astype(int).to_numpy()), 1)
    p_ci = panel_keys["cell"].astype(str).map(cells).to_numpy()
    p_mi = panel_keys["month"].astype(str).map(months).to_numpy()
    p_h = panel_keys["hour_of_week"].to_numpy()
    out = pd.DataFrame({"cell": panel_keys["cell"], "month": panel_keys["month"], "hour_of_week": p_h})
    for w in (12, 36):
        out[f"inc_cellhow_{w}m"] = trailing(cube, w)[p_ci, p_h, p_mi].astype(np.float32)
    return out


def network_history(segments: pd.DataFrame, seg_cells: pd.DataFrame, months: list[str],
                    scope: str = DEFAULT_SCOPE) -> pd.DataFrame:
    """On-network (cell, month) for months before the panel too (same whole-month rule)."""
    scoped = in_scope(segments, scope)
    frames = []
    for m in months:
        ids = segments.loc[scoped & active_in_month(segments, m), "route_id"]
        frames.append(pd.DataFrame({"cell": seg_cells.loc[seg_cells["route_id"].isin(ids), "cell"].unique(), "month": m}))
    return pd.concat(frames, ignore_index=True)


def propensity_cells(prop: pd.DataFrame) -> pd.DataFrame:
    """M2 grid points (cell centres of a 0.002° grid) -> H3 cells."""
    located = prop.dropna(subset=["grid_lat", "grid_lon"])
    pts = located[["grid_lat", "grid_lon"]].drop_duplicates()
    pts["cell"] = [h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(pts["grid_lat"], pts["grid_lon"])]
    return (located.merge(pts, on=["grid_lat", "grid_lon"])
            .groupby(["cell", "month"])["n"].sum().reset_index())


@dataclass
class Inputs:
    """Everything the feature builder reads, loaded once."""
    segments: gpd.GeoDataFrame
    seg_cells: pd.DataFrame
    boroughs: pd.DataFrame
    incidents: pd.DataFrame
    propensity: pd.DataFrame
    citibike: pd.DataFrame
    lots: pd.DataFrame
    citibike_through: str  # last month with its own Citi Bike archive


def load_inputs() -> Inputs:
    segments = gpd.read_parquet(ROUTES_PATH)
    seg_cells = pd.read_parquet(SEGMENT_CELLS_PATH)
    incidents = pd.read_parquet(INCIDENTS_PATH)
    incidents["cell"] = incidents["cell"].astype(str)
    incidents["month"] = incidents["month"].astype(str)
    return Inputs(
        segments=segments,
        seg_cells=seg_cells,
        boroughs=cell_boroughs(segments, seg_cells).astype(str),
        incidents=incidents,
        propensity=propensity_cells(read_store(Path("data/interim") / PROPENSITY_STORE)),
        citibike=pd.read_parquet(CITIBIKE_PATH).astype({"cell": str, "month": str}),
        lots=pd.read_parquet(pluto_ingest.latest()),
        citibike_through=citibike_covered_through(),
    )


def citibike_gap(covered: str, last_month: str) -> list[str]:
    """Months a row up to `last_month` can read that have no Citi Bike archive yet."""
    if covered is None:
        raise ValueError("no Citi Bike archives processed")
    return month_range(covered, last_month)[1:-1]


def fill_citibike(citibike: pd.DataFrame, covered: str, last_month: str) -> pd.DataFrame:
    """Drop rows after the last archived month, then repeat that month for each uncovered month
    before `last_month` (stale but closer than zero, which the model reads as no stations)."""
    cb = citibike[citibike["month"] <= covered]
    last = cb[cb["month"] == covered]
    return pd.concat([cb] + [last.assign(month=m) for m in citibike_gap(covered, last_month)], ignore_index=True)


def cell_month_features(rows: pd.DataFrame, inp: Inputs, last_month: str) -> pd.DataFrame:
    """Features for (cell, month) `rows` (sorted month, cell), any months up to `last_month`."""
    all_months = month_range(HIST_START, last_month)
    net_hist = network_history(inp.segments, inp.seg_cells, all_months)
    citibike = fill_citibike(inp.citibike, inp.citibike_through, last_month)
    feats = history_features(rows, inp.incidents, net_hist, inp.propensity, citibike, inp.boroughs, all_months)
    feats = lane_features(inp.segments, inp.seg_cells, feats)
    feats = feats.merge(pluto_features(inp.lots, sorted(rows["cell"].unique())), on="cell", how="left")
    feats["month_of_year"] = feats["month"].str.slice(5, 7).astype("int8")
    feats["boro"] = feats["cell"].map(inp.boroughs.set_index("cell")["boro"]).astype("int8")
    return feats


def main() -> None:
    network = pd.read_parquet(NETWORK_PATH)
    rows = network[["cell", "month"]].astype(str).sort_values(["month", "cell"]).reset_index(drop=True)
    last = panel_months()[1]
    if sorted(rows["month"].unique())[-1] != last:
        raise RuntimeError("network_cell_months does not end at the last complete 311 month; rerun grid / snap / panel")
    inp = load_inputs()

    feats = cell_month_features(rows, inp, last)
    feats.to_parquet(CELL_MONTH_PATH, index=False)
    print(f"wrote {CELL_MONTH_PATH}: {len(feats):,} rows x {feats.shape[1]} columns")

    keys = pd.read_parquet(PANEL_PATH, columns=["cell", "month", "hour_of_week"])
    ch = cell_how_history(keys, inp.incidents, month_range(HIST_START, last))
    ch.to_parquet(CELL_HOW_PATH, index=False)
    print(f"wrote {CELL_HOW_PATH}: {len(ch):,} rows")


if __name__ == "__main__":
    main()

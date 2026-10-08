"""311 ingestion (M2).

    python -m clearlane.ingest.sr311 {target,propensity,all} [--start YYYY-MM] [--end YYYY-MM]

Two stores under `data/interim/`, one Parquet file per calendar month of
`created_date` (America/New_York wall-clock, stored naive as Socrata serves it):

* `sr311_target/YYYY-MM.parquet` — raw rows of the target pair. Each pull
  upserts on `unique_key` (the latest pull of a row wins), so re-running any
  overlapping range never changes row counts (invariant 1). Rows that vanish
  from the source are reported, not deleted.
* `sr311_propensity/YYYY-MM.parquet` — counts of every *other* 311 request per
  (month, 0.002° grid cell), aggregated server-side with SoQL `snap_to_grid`.
  Each pull replaces the whole month. Requests with no location get a null
  cell. `month_complete` is False when the month had not ended at least
  `PUBLISH_GRACE_DAYS` before pull time.

Each store has a `_manifest.json` recording, per month, when it was last
pulled and whether the month was complete then. Without `--start`, a run
re-pulls the last `OVERLAP_MONTHS` stored months (to pick up late rows and
status updates) through the current month.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from clearlane.ingest import socrata
from clearlane.ingest.socrata import build_params

log = logging.getLogger(__name__)

# NYC Open Data split the 311 dataset: erm2-nwe9 now covers 2020 onward and
# 2010-2019 lives in 76ig-c548. Queries spanning both are routed to each and
# combined. Bounds are [start, end) on created_date; None = open-ended.
SR_DATASETS = [
    ("76ig-c548", "311 Service Requests from 2010 to 2019", "2010-01-01", "2020-01-01"),
    ("erm2-nwe9", "311 Service Requests from 2020 to Present", "2020-01-01", None),
]

# Confirmed in M1 (reports/m1_audit.md §2); first row 2016-10-19.
TARGET = ("Illegal Parking", "Blocked Bike Lane")
DEFAULT_START = "2016-01"
OVERLAP_MONTHS = 2
PUBLISH_GRACE_DAYS = 3  # a month counts as complete only this many days after it ends (see is_complete)
GRID_DEG = 0.002  # snap_to_grid rounds to the nearest multiple: points are cell centres
PAGE_LIMIT = 50_000
WORKERS = 4
DATA_DIR = Path("data/interim")
TARGET_STORE = "sr311_target"
PROPENSITY_STORE = "sr311_propensity"
MANIFEST = "_manifest.json"

# Both datasets share one schema (checked in M2 against the catalog).
TARGET_COLUMNS = [
    "unique_key", "created_date", "closed_date", "resolution_action_updated_date",
    "complaint_type", "descriptor", "status", "resolution_description",
    "open_data_channel_type", "location_type", "incident_address", "street_name",
    "cross_street_1", "cross_street_2", "intersection_street_1", "intersection_street_2",
    "address_type", "incident_zip", "borough", "community_board", "police_precinct",
    "latitude", "longitude",
]
DATE_COLUMNS = ["created_date", "closed_date", "resolution_action_updated_date"]
FLOAT_COLUMNS = ["latitude", "longitude"]
META_COLUMNS = ["source_dataset", "ingested_at"]
TS = "datetime64[us]"  # naive NYC wall clock; fixed unit so every month file has one schema
TS_UTC = "datetime64[us, UTC]"
PROPENSITY_COLUMNS = ["month", "grid_lat", "grid_lon", "n", "month_complete", "source_dataset", "ingested_at"]


# ---------------------------------------------------------------------------
# Query routing (shared with the M1 audit)
# ---------------------------------------------------------------------------


def soql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def target_where(pair: tuple[str, str]) -> str:
    return f"complaint_type={soql_str(pair[0])} AND descriptor={soql_str(pair[1])}"


def not_target_where(pair: tuple[str, str]) -> str:
    """Complement of `target_where`, keeping rows with a null type or descriptor."""
    return f"complaint_type IS NULL OR descriptor IS NULL OR NOT ({target_where(pair)})"


def date_where(start: str | None, end: str | None) -> list[str]:
    out = []
    if start:
        out.append(f"created_date >= '{start}T00:00:00'")
    if end:
        out.append(f"created_date < '{end}T00:00:00'")
    return out


def datasets_for(start: str | None, end: str | None) -> list[tuple[str, str | None, str | None]]:
    """311 datasets overlapping [start, end), with the range clipped to each."""
    out = []
    for ds, _name, ds_start, ds_end in SR_DATASETS:
        lo = max(filter(None, [start, ds_start]), default=None)
        hi = min(filter(None, [end, ds_end]), default=None)
        if lo and hi and lo >= hi:
            continue
        out.append((ds, lo, hi))
    return out


def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end) (ISO dates) at calendar-month boundaries.

    Socrata times out or drops the connection on multi-month aggregates over
    the 311 dataset (a full-year `upper(...) like` scan can exceed 10 minutes),
    while single-month ones return in about a second.
    """
    out = []
    lo = start
    while lo < end:
        y, m = int(lo[:4]), int(lo[5:7])
        nxt = f"{y + (m == 12)}-{m % 12 + 1:02d}-01"
        hi = min(nxt, end)
        out.append((lo, hi))
        lo = hi
    return out


def where_clause(clauses: list[str]) -> str:
    return " AND ".join(f"({c})" for c in clauses)


# ---------------------------------------------------------------------------
# Months
# ---------------------------------------------------------------------------


def month_bounds(month: str) -> tuple[str, str]:
    """'2023-05' -> ('2023-05-01', '2023-06-01')."""
    y, m = int(month[:4]), int(month[5:7])
    return f"{month}-01", f"{y + (m == 12)}-{m % 12 + 1:02d}-01"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' months from start to end."""
    if start > end:
        return []
    return [lo[:7] for lo, _ in month_chunks(f"{start}-01", month_bounds(end)[1])]


def default_start(stored_months: list[str], overlap: int = OVERLAP_MONTHS) -> str:
    if not stored_months:
        return DEFAULT_START
    latest = max(stored_months)
    y, m = int(latest[:4]), int(latest[5:7]) - (overlap - 1)
    while m < 1:
        y, m = y - 1, m + 12
    return f"{y}-{m:02d}"


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class Source(Protocol):
    def target_rows(self, dataset: str, lo: str, hi: str) -> list[dict]: ...

    def propensity_rows(self, dataset: str, lo: str, hi: str) -> list[dict]: ...


class SocrataSource:
    """Live NYC Open Data source. Raw rows are paged; propensity is a server-side group-by."""

    def target_rows(self, dataset: str, lo: str, hi: str) -> list[dict]:
        params = build_params(
            select=", ".join(TARGET_COLUMNS),
            where=where_clause([target_where(TARGET)] + date_where(lo, hi)),
            limit=PAGE_LIMIT,
        )
        return socrata.fetch_pages(dataset, params, order="unique_key")

    def propensity_rows(self, dataset: str, lo: str, hi: str) -> list[dict]:
        params = build_params(
            select=f"snap_to_grid(location, {GRID_DEG}) as g, count(*) as n",
            where=where_clause([not_target_where(TARGET)] + date_where(lo, hi)),
            group="g",
            limit=PAGE_LIMIT,
        )
        return socrata.fetch_live(dataset, params)


# ---------------------------------------------------------------------------
# Frames (pure)
# ---------------------------------------------------------------------------


def target_frame(rows: list[dict], source_dataset: str, ingested_at: pd.Timestamp) -> pd.DataFrame:
    """Normalize raw Socrata rows to the target-store schema."""
    df = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    for c in TARGET_COLUMNS:
        if c in DATE_COLUMNS:
            df[c] = pd.to_datetime(df[c], format="ISO8601").astype(TS)
        elif c in FLOAT_COLUMNS:
            df[c] = pd.to_numeric(df[c]).astype("float64")
        else:
            df[c] = df[c].astype("string")
    df["source_dataset"] = pd.Series([source_dataset] * len(df), dtype="string")
    df["ingested_at"] = pd.Series([ingested_at] * len(df), dtype=TS_UTC)
    return df


def empty_target_frame() -> pd.DataFrame:
    return target_frame([], "", pd.Timestamp.now(tz="UTC"))


def merge_target(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Upsert `new` into `existing` on unique_key; the row from `new` wins."""
    old_keys = set(existing["unique_key"])
    new_keys = set(new["unique_key"])
    if len(new_keys) != len(new):
        raise ValueError("duplicate unique_key within a single pull")
    content = TARGET_COLUMNS[1:]

    def as_text(df: pd.DataFrame) -> pd.DataFrame:
        # NA-safe comparison: a value turning null must count as a change.
        return df.astype("string").fillna("\x00NA")

    both = as_text(existing[existing["unique_key"].isin(new_keys)].set_index("unique_key")[content])
    fresh = as_text(new[new["unique_key"].isin(old_keys)].set_index("unique_key")[content]).reindex(both.index)
    changed = (both != fresh).any(axis=1)
    merged = (
        pd.concat([existing, new], ignore_index=True)
        .drop_duplicates("unique_key", keep="last")
        .sort_values(["created_date", "unique_key"], kind="stable")
        .reset_index(drop=True)
    )
    stats = {
        "fetched": len(new),
        "new": len(new_keys - old_keys),
        "updated": int(changed.sum()),
        "missing_from_source": len(old_keys - new_keys),
        "rows": len(merged),
    }
    return merged, stats


def propensity_frame(
    rows: list[dict], month: str, complete: bool, source_dataset: str, ingested_at: pd.Timestamp
) -> pd.DataFrame:
    recs = []
    for r in rows:
        g = r.get("g")
        lon, lat = (round(float(g["coordinates"][0]), 3), round(float(g["coordinates"][1]), 3)) if g else (None, None)
        recs.append({"grid_lat": lat, "grid_lon": lon, "n": int(r["n"])})
    df = pd.DataFrame(recs, columns=["grid_lat", "grid_lon", "n"])
    df = df.astype({"grid_lat": "float64", "grid_lon": "float64", "n": "int64"})
    df.insert(0, "month", pd.Series([month] * len(df), dtype="string"))
    df["month_complete"] = complete
    df["source_dataset"] = pd.Series([source_dataset] * len(df), dtype="string")
    df["ingested_at"] = pd.Series([ingested_at] * len(df), dtype=TS_UTC)
    return df.sort_values(["grid_lat", "grid_lon"], na_position="first", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Store I/O
# ---------------------------------------------------------------------------


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_manifest(store: Path) -> dict[str, dict]:
    p = store / MANIFEST
    return json.loads(p.read_text()) if p.exists() else {}


def write_manifest(store: Path, manifest: dict[str, dict]) -> None:
    store.mkdir(parents=True, exist_ok=True)
    tmp = store / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    tmp.replace(store / MANIFEST)


def read_store(store: Path) -> pd.DataFrame:
    files = sorted(store.glob("????-??.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def is_complete(month: str, now: pd.Timestamp, grace_days: int = PUBLISH_GRACE_DAYS) -> bool:
    """Whether `month` had ended (NYC wall clock) at least `grace_days` full days before `now`.

    Open Data publishes each day's requests about a day later, so a pull made just after
    month end lacks the last day (the 2026-10-01 09:39 ET pull had none of 2026-09-30).
    """
    nyc_today = now.tz_convert("America/New_York").date()
    return dt.date.fromisoformat(month_bounds(month)[1]) + dt.timedelta(days=grace_days) <= nyc_today


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _pull_month(source: Source, kind: str, month: str) -> list[tuple[str, list[dict]]]:
    lo, hi = month_bounds(month)
    fn = source.target_rows if kind == "target" else source.propensity_rows
    return [(ds, fn(ds, dlo, dhi)) for ds, dlo, dhi in datasets_for(lo, hi)]


def ingest_target(
    months: list[str], data_dir: Path, source: Source, now: pd.Timestamp | None = None, workers: int = WORKERS
) -> list[dict]:
    """Upsert the target pair's raw rows for each month. Returns per-month stats."""
    now = now or pd.Timestamp.now(tz="UTC")
    store = data_dir / TARGET_STORE

    def one(month: str) -> dict:
        pulls = _pull_month(source, "target", month)
        new = pd.concat([target_frame(rows, ds, now) for ds, rows in pulls] or [empty_target_frame()],
                        ignore_index=True)
        stray = new[new["created_date"].dt.strftime("%Y-%m") != month]
        if len(stray):
            raise ValueError(f"{len(stray)} rows returned for {month} have created_date outside it")
        path = store / f"{month}.parquet"
        existing = pd.read_parquet(path) if path.exists() else empty_target_frame()
        merged, stats = merge_target(existing, new)
        _atomic_parquet(merged, path)
        return {"month": month, **stats}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, months))
    manifest = read_manifest(store)
    for r in results:
        manifest[r["month"]] = {"pulled_at": now.isoformat(), "complete": is_complete(r["month"], now),
                                "rows": r["rows"]}
    write_manifest(store, manifest)
    return results


def ingest_propensity(
    months: list[str], data_dir: Path, source: Source, now: pd.Timestamp | None = None, workers: int = WORKERS
) -> list[dict]:
    """Replace each month's non-target grid counts. Returns per-month stats."""
    now = now or pd.Timestamp.now(tz="UTC")
    store = data_dir / PROPENSITY_STORE

    def one(month: str) -> dict:
        complete = is_complete(month, now)
        frames = [propensity_frame(rows, month, complete, ds, now) for ds, rows in _pull_month(source, "propensity", month)]
        df = pd.concat(frames, ignore_index=True) if frames else propensity_frame([], month, complete, "", now)
        _atomic_parquet(df, store / f"{month}.parquet")
        return {"month": month, "cells": int(df["grid_lat"].notna().sum()), "requests": int(df["n"].sum()),
                "no_location": int(df.loc[df["grid_lat"].isna(), "n"].sum()), "complete": complete}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, months))
    manifest = read_manifest(store)
    for r in results:
        manifest[r["month"]] = {"pulled_at": now.isoformat(), "complete": r["complete"],
                                "requests": r["requests"]}
    write_manifest(store, manifest)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summary(kind: str, results: list[dict]) -> str:
    df = pd.DataFrame(results)
    if df.empty:
        return f"{kind}: no months"
    if kind == "target":
        tot = df[["fetched", "new", "updated", "missing_from_source"]].sum()
        return (f"target: {len(df)} months {df['month'].min()}..{df['month'].max()}; fetched={tot['fetched']:,} "
                f"new={tot['new']:,} updated={tot['updated']:,} missing_from_source={tot['missing_from_source']:,}")
    return (f"propensity: {len(df)} months {df['month'].min()}..{df['month'].max()}; "
            f"requests={df['requests'].sum():,} no_location={df['no_location'].sum():,} "
            f"incomplete_months={int((~df['complete']).sum())}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["target", "propensity", "all"])
    ap.add_argument("--start", help="first month YYYY-MM (default: incremental from the store)")
    ap.add_argument("--end", help="last month YYYY-MM, inclusive (default: current month)")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    now = pd.Timestamp.now(tz="UTC")
    end = args.end or now.tz_convert("America/New_York").strftime("%Y-%m")
    source = SocrataSource()
    kinds = ["target", "propensity"] if args.what == "all" else [args.what]
    for kind in kinds:
        store = args.data_dir / (TARGET_STORE if kind == "target" else PROPENSITY_STORE)
        start = args.start or default_start(list(read_manifest(store)))
        months = month_range(start, end)
        fn = ingest_target if kind == "target" else ingest_propensity
        print(_summary(kind, fn(months, args.data_dir, source, now)))
        if kind == "target":
            print(f"target store: {sum(m['rows'] for m in read_manifest(store).values()):,} rows")


if __name__ == "__main__":
    main()

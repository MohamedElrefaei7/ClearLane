"""Citi Bike trips -> starts/ends per (H3 cell, month) (M7 exposure proxy).

    python -m clearlane.ingest.citibike [--from-year 2020]

Streams the public NYC trip archives from s3://tripdata one at a time
(download -> aggregate -> delete; recent monthly zips are up to ~1 GB), so
peak disk use is one archive. Yearly archives nest monthly zips/CSVs; both CSV
schemas (pre-2021 `starttime`/`start station latitude`..., later
`started_at`/`start_lat`...) are handled.

Per archive, trips are counted by (month, rounded coordinate) for starts
(month of the start time) and ends (month of the end time), then coordinates
are mapped to H3 res-9 cells. Per-archive results go to
`data/interim/citibike_parts/` (resume-safe); the combined table is
`data/interim/citibike_cell_month.parquet` (cell, month, starts, ends).
Coordinates outside the NYC bounding box (same box as the M1 audit) or missing
are dropped and counted.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h3
import pandas as pd
import requests

from clearlane.config import H3_RES
from clearlane.ingest.socrata import NYC_LAT_MAX, NYC_LAT_MIN, NYC_LON_MAX, NYC_LON_MIN

BUCKET = "https://s3.amazonaws.com/tripdata"
RAW_DIR = Path("data/raw/citibike")
PARTS_DIR = Path("data/interim/citibike_parts")
OUT_PATH = Path("data/interim/citibike_cell_month.parquet")
CHUNK = 2_000_000

SCHEMAS = [  # (start time, end time, start lat, start lng, end lat, end lng)
    ("started_at", "ended_at", "start_lat", "start_lng", "end_lat", "end_lng"),
    ("starttime", "stoptime", "start station latitude", "start station longitude",
     "end station latitude", "end station longitude"),
]


def list_archives(from_year: int) -> list[str]:
    xml = requests.get(f"{BUCKET}/?list-type=2", timeout=60).text
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    out = []
    for k in keys:
        m = re.match(r"(\d{4})(\d{2})?-citibike-tripdata(\.csv)?\.zip$", k)
        if m and int(m.group(1)) >= from_year:
            out.append(k)
    return sorted(out)


def covered_through(parts_dir: Path = PARTS_DIR) -> str | None:
    """Last month with its own processed monthly archive ('YYYY-MM'). Months after it are not
    covered, even if trips spanning an archive's end put a few rows there."""
    months = [f"{m.group(1)}-{m.group(2)}" for p in parts_dir.glob("*.parquet")
              if (m := re.match(r"(\d{4})(\d{2})-citibike-tripdata", p.name))]
    return max(months) if months else None


def aggregate_csv(f) -> pd.DataFrame:
    """(kind, month, lat, lng, n) counts from one trip CSV stream."""
    header = pd.read_csv(f, nrows=0).columns.str.strip().str.lower()
    f.seek(0)
    schema = next((s for s in SCHEMAS if set(s) <= set(header)), None)
    if schema is None:
        raise ValueError(f"unrecognised Citi Bike schema: {list(header)}")
    t0, t1, la0, lo0, la1, lo1 = schema
    parts = []
    for chunk in pd.read_csv(f, usecols=lambda c: c.strip().lower() in schema, chunksize=CHUNK,
                             dtype=str, low_memory=False):
        chunk.columns = chunk.columns.str.strip().str.lower()
        for kind, t, la, lo in (("start", t0, la0, lo0), ("end", t1, la1, lo1)):
            d = pd.DataFrame({
                "month": chunk[t].str.slice(0, 7),
                "lat": pd.to_numeric(chunk[la], errors="coerce").round(5),
                "lng": pd.to_numeric(chunk[lo], errors="coerce").round(5),
            })
            g = d.groupby(["month", "lat", "lng"], dropna=False).size().rename("n").reset_index()
            g.insert(0, "kind", kind)
            parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out.groupby(["kind", "month", "lat", "lng"], dropna=False)["n"].sum().reset_index()


def iter_csvs(zf: zipfile.ZipFile):
    for name in zf.namelist():
        base = name.rsplit("/", 1)[-1]
        if base.startswith(("._", ".")) or "__MACOSX" in name:
            continue
        if name.lower().endswith(".csv"):
            with zf.open(name) as raw:
                yield name, io.BytesIO(raw.read())
        elif name.lower().endswith(".zip"):
            with zf.open(name) as raw, zipfile.ZipFile(io.BytesIO(raw.read())) as inner:
                yield from iter_csvs(inner)


def to_cells(coords: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(kind, month, lat, lng, n) -> (kind, month, cell, n); drops missing / out-of-bbox."""
    ok = coords["lat"].between(NYC_LAT_MIN, NYC_LAT_MAX) & coords["lng"].between(NYC_LON_MIN, NYC_LON_MAX)
    good = coords[ok].copy()
    uniq = good[["lat", "lng"]].drop_duplicates()
    uniq["cell"] = [h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(uniq["lat"], uniq["lng"])]
    good = good.merge(uniq, on=["lat", "lng"])
    return good.groupby(["kind", "month", "cell"])["n"].sum().reset_index(), int(coords.loc[~ok, "n"].sum())


def download(key: str, path: Path, attempts: int = 4) -> None:
    """Stream one archive to disk, restarting from scratch on network errors."""
    for attempt in range(1, attempts + 1):
        path.unlink(missing_ok=True)  # partial download from an interrupted attempt or run
        try:
            with requests.get(f"{BUCKET}/{key}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for block in r.iter_content(8 << 20):
                        f.write(block)
            return
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
            time.sleep(30 * attempt)


def process_archive(key: str) -> dict:
    part = PARTS_DIR / (key.replace(".zip", "") + ".parquet")
    if part.exists():
        return json.loads(part.with_suffix(".json").read_text())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / key
    download(key, path)
    try:
        frames, files = [], []
        with zipfile.ZipFile(path) as zf:
            for name, f in iter_csvs(zf):
                frames.append(aggregate_csv(f))
                files.append(name)
        coords = pd.concat(frames, ignore_index=True).groupby(["kind", "month", "lat", "lng"], dropna=False)["n"].sum().reset_index()
        cells, dropped = to_cells(coords)
        PARTS_DIR.mkdir(parents=True, exist_ok=True)
        cells.to_parquet(part, index=False)
        meta = {"archive": key, "csv_files": len(files), "trip_ends_counted": int(cells["n"].sum()),
                "trip_ends_dropped": dropped,
                "months": sorted(cells["month"].unique().tolist())}
        part.with_suffix(".json").write_text(json.dumps(meta))
        return meta
    finally:
        path.unlink(missing_ok=True)


def combine() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(PARTS_DIR.glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True).groupby(["kind", "month", "cell"])["n"].sum().reset_index()
    wide = df.pivot_table(index=["cell", "month"], columns="kind", values="n", fill_value=0).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"start": "starts", "end": "ends"})
    return wide.astype({"cell": "string", "month": "string", "starts": "int64", "ends": "int64"})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-year", type=int, default=2020)
    ap.add_argument("--workers", type=int, default=3, help="archives processed in parallel (~3 GB RAM each)")
    args = ap.parse_args(argv)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        metas = list(pool.map(process_archive, list_archives(args.from_year)))
    for meta in metas:
        key = meta["archive"]
        print(f"{key}: {meta['trip_ends_counted']:,} ends counted, {meta['trip_ends_dropped']:,} dropped, "
              f"months {meta['months'][0]}..{meta['months'][-1]}", flush=True)
    out = combine()
    out.to_parquet(OUT_PATH, index=False)
    by_month = out.groupby("month")["starts"].sum()
    print(f"wrote {OUT_PATH}: {len(out):,} cell-months, {out['cell'].nunique():,} cells, "
          f"months {by_month.index.min()}..{by_month.index.max()}")


if __name__ == "__main__":
    main()

"""M1 data audit: writes reports/m1_audit.md and reports/m1_monthly.png.

    python -m clearlane.audit.m1_audit [--refresh] [--as-of YYYY-MM-DD]

Two stages:
  * `collect` runs every Socrata query (through the on-disk cache) and returns a
    dict of named raw responses. All 311 queries are server-side aggregates
    except one `$limit=5` sample.
  * `render_report` turns that dict into markdown + a chart. It is pure apart
    from writing the two output files; a section whose inputs are missing is
    rendered as "MISSING: <reason>" rather than skipped.

The report states facts and numbers. It does not make modelling decisions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from clearlane.ingest import socrata
from clearlane.ingest.socrata import (
    DISCOVERY_URL,
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LON_MAX,
    NYC_LON_MIN,
    build_params,
    to_hour_of_week,
)
from clearlane.ingest.sr311 import SR_DATASETS, date_where, datasets_for, month_chunks, target_where

WORKERS = 4  # concurrent month-chunk requests per query

DOW_CHECK_DATE = dt.date(2025, 6, 2)  # a Monday
COVID_LINE = "2020-03"
SPARSITY_CELLS = (2_000, 4_000, 6_000)
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
POINT_TYPES = {"point", "multipoint", "location", "line", "multiline", "polygon", "multipolygon"}
COORD_NAME_RE = re.compile(r"(^|_)(lat|latitude|lon|lng|longitude|the_geom|geom|point|georeference|x_coord|y_coord)($|_)", re.I)

SECTIONS = [
    "1. Run metadata",
    "2. 311 taxonomy",
    "3. Volume by year",
    "4. Volume by month",
    "5. Coordinate quality",
    "6. Day-of-week convention check",
    "7. Hour-of-week distribution",
    "8. Sparsity estimate",
    "9. Resolution descriptions",
    "10. DOT bike-route dataset",
    "11. DOF Parking Violations",
    "12. Open questions",
]


# ---------------------------------------------------------------------------
# Pure helpers (tested offline)
# ---------------------------------------------------------------------------


def sparsity(annual_count: float, n_cells: int, n_slots: int = 168) -> float:
    """Expected events per (cell, slot, month) if events were spread evenly."""
    return annual_count / (n_slots * 12) / n_cells


def mom_changes(series: pd.Series, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Largest month-over-month drops and rises.

    `series` is indexed by month label in chronological order. Each row reports
    the month the change lands in, the previous and current value, the absolute
    change, and the percent change.
    """
    s = series.astype(float)
    df = pd.DataFrame(
        {
            "month": s.index[1:],
            "prev": s.values[:-1],
            "value": s.values[1:],
        }
    )
    df["change"] = df["value"] - df["prev"]
    df["pct"] = (df["change"] / df["prev"].where(df["prev"] != 0)) * 100
    drops = df[df["change"] < 0].nsmallest(k, "change").reset_index(drop=True)
    rises = df[df["change"] > 0].nlargest(k, "change").reset_index(drop=True)
    return drops, rises


def plausible_target_pairs(rows: list[dict]) -> list[dict]:
    """Pairs that look like 'vehicle blocking a bike lane'."""
    out = []
    for r in rows:
        text = f"{r.get('complaint_type', '')} {r.get('descriptor', '')}".upper()
        if "BIKE" in text and "BLOCK" in text:
            out.append(r)
    return sorted(out, key=lambda r: -int(r["total"]))


def choose_target(rows: list[dict]) -> tuple[tuple[str, str] | None, list[dict]]:
    plausible = plausible_target_pairs(rows)
    if not plausible:
        return None, plausible
    top = plausible[0]
    return (top["complaint_type"], top["descriptor"]), plausible


def infer_sunday_is_zero(rows: list[dict]) -> bool:
    """Given grouped dow counts for a single known Monday, infer the convention."""
    values = {int(r["dow"]) for r in rows if int(r.get("n", 0)) > 0}
    if values == {1}:
        return True
    if values == {0}:
        return False
    raise ValueError(f"ambiguous dow values for a single Monday: {sorted(values)}")


def filter_pairs(by_dataset: dict[str, list[dict]], pattern: str) -> dict[str, list[dict]]:
    """Client-side equivalent of `upper(complaint_type) like '%P%' OR upper(descriptor) like '%P%'`."""
    pattern = pattern.upper()
    return {
        ds: [r for r in rows
             if pattern in (r.get("complaint_type") or "").upper() or pattern in (r.get("descriptor") or "").upper()]
        for ds, rows in by_dataset.items()
    }


def merge_pair_counts(by_dataset: dict[str, list[dict]]) -> list[dict]:
    """Combine grouped (complaint_type, descriptor) counts across 311 datasets."""
    merged: dict[tuple[str, str], dict] = {}
    for ds, rows in by_dataset.items():
        for r in rows:
            key = (r.get("complaint_type", ""), r.get("descriptor", ""))
            m = merged.setdefault(
                key, {"complaint_type": key[0], "descriptor": key[1], "total": 0, "first": None, "last": None}
            )
            n = int(r["n"])
            m[ds] = m.get(ds, 0) + n
            m["total"] += n
            if r.get("first") and (m["first"] is None or r["first"] < m["first"]):
                m["first"] = r["first"]
            if r.get("last") and (m["last"] is None or r["last"] > m["last"]):
                m["last"] = r["last"]
    return sorted(merged.values(), key=lambda m: -m["total"])


def sum_grouped(rows_lists: list[list[dict]], keys: list[str], value: str = "n") -> pd.DataFrame:
    """Concatenate grouped responses and sum `value` by `keys`."""
    rows = [r for rows in rows_lists for r in rows]
    if not rows:
        return pd.DataFrame(columns=keys + [value])
    df = pd.DataFrame(rows)
    for k in keys:
        if k not in df:
            df[k] = None
    df[value] = df[value].astype(int)
    return df.groupby(keys, dropna=False, as_index=False)[value].sum()


def find_coord_columns(names: list[str], types: list[str]) -> list[tuple[str, str]]:
    out = []
    for name, typ in zip(names, types):
        if (typ or "").lower() in POINT_TYPES or COORD_NAME_RE.search(name):
            out.append((name, typ))
    return out


def identify_bike_columns(names: list[str]) -> dict[str, list[str]]:
    lower = [n.lower() for n in names]
    pick = lambda pat: [n for n, l in zip(names, lower) if re.search(pat, l)]  # noqa: E731
    return {
        "facility type": pick(r"facilit|allclasses"),
        "install date": pick(r"inst|install"),
        "retire date": pick(r"ret_?date|retire"),
        "status": pick(r"^status$"),
    }


def how_label(how: int) -> str:
    return f"{DAY_NAMES[how // 24]} {how % 24:02d}:00"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for col, v in zip(df.columns, row.values):
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:,.4g}" if abs(v) < 1000 else f"{v:,.0f}")
            elif isinstance(v, int) and not isinstance(v, bool) and str(col) != "year":
                cells.append(f"{v:,}")
            else:
                cells.append("" if v is None else str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Collection (network, via cache)
# ---------------------------------------------------------------------------


@dataclass
class Missing:
    reason: str


class Collector:
    def __init__(self, cache_dir: Path, refresh: bool, as_of: dt.date):
        self.cache_dir = cache_dir
        self.open_end = (as_of.replace(day=1) + dt.timedelta(days=32)).replace(day=1).isoformat()
        self.refresh = refresh
        self.R: dict[str, Any] = {}

    def get(self, name: str, fn: Callable[[], Any]) -> Any:
        try:
            self.R[name] = fn()
        except Exception as e:  # recorded as MISSING in the report
            self.R[name] = Missing(f"{type(e).__name__}: {e}")
        return self.R[name]

    def ds(self, dataset_id: str, **kw) -> Any:
        return socrata.fetch(dataset_id, build_params(**kw), self.cache_dir, self.refresh)

    def url(self, url: str, params: dict) -> Any:
        return socrata.fetch_url(url, params, self.cache_dir, self.refresh)

    def sr(self, *, select: str, group: str, where: list[str], start: str | None, end: str | None, **kw) -> dict:
        """Grouped 311 query routed to every dataset overlapping [start, end).

        Each dataset's range is queried one calendar month at a time; the rows
        from all chunks are concatenated (callers sum them by group key).
        """
        out = {}
        for ds, lo, hi in datasets_for(start, end):
            wheres = [" AND ".join(f"({c})" for c in list(where) + date_where(clo, chi))
                      for clo, chi in month_chunks(lo, hi or self.open_end)]
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                chunks = pool.map(lambda w: self.ds(ds, select=select, group=group, where=w, **kw), wheres)
                out[ds] = [r for rows in chunks for r in rows]
        return out


def collect(cache_dir: Path, refresh: bool, as_of: dt.date) -> dict[str, Any]:
    c = Collector(cache_dir, refresh, as_of)
    R = c.R
    R["as_of"] = as_of.isoformat()
    year = as_of.year
    last3 = (f"{year - 3}-01-01", f"{year}-01-01")
    last2 = (f"{year - 2}-01-01", f"{year}-01-01")

    pair_sel = "complaint_type, descriptor, count(*) as n, min(created_date) as first, max(created_date) as last"
    pair_grp = "complaint_type, descriptor"
    # One unfiltered grouped count of every (complaint_type, descriptor) pair,
    # matched to the patterns client-side. Server-side `upper(...) like` scans
    # take minutes per month on 76ig-c548; the plain group-by takes ~1 s.
    c.get("taxonomy_all", lambda: c.sr(select=pair_sel, group=pair_grp, where=[], start=None, end=None))
    for key, pat in (("taxonomy_bike", "BIKE"), ("taxonomy_double_park", "DOUBLE PARK")):
        if isinstance(R["taxonomy_all"], Missing):
            R[key] = R["taxonomy_all"]
        else:
            R[key] = filter_pairs(R["taxonomy_all"], pat)

    c.get("sample_311", lambda: c.ds("erm2-nwe9", limit=5))

    target = None
    if not isinstance(R["taxonomy_bike"], Missing):
        target, _ = choose_target(merge_pair_counts(R["taxonomy_bike"]))
    if target is None:
        reason = "no target pair (taxonomy missing or no plausible blocked-bike-lane pair)"
        for k in ("yearly", "monthly", "coord_total", "coord_null", "coord_outside", "how", "resolution"):
            R[k] = Missing(reason)
    else:
        tw = target_where(target)
        c.get("yearly", lambda: c.sr(
            select="date_extract_y(created_date) as year, count(*) as n", group="year",
            where=[tw], start="2016-01-01", end=None))
        c.get("monthly", lambda: c.sr(
            select="date_trunc_ym(created_date) as month, count(*) as n", group="month",
            where=[tw], start="2016-01-01", end=None))
        ysel = "date_extract_y(created_date) as year, count(*) as n"
        c.get("coord_total", lambda: c.sr(select=ysel, group="year", where=[tw], start=last3[0], end=last3[1]))
        c.get("coord_null", lambda: c.sr(
            select=ysel, group="year", where=[tw, "latitude IS NULL OR longitude IS NULL"],
            start=last3[0], end=last3[1]))
        inside = (
            f"latitude between {NYC_LAT_MIN} and {NYC_LAT_MAX} "
            f"AND longitude between {NYC_LON_MIN} and {NYC_LON_MAX}"
        )
        c.get("coord_outside", lambda: c.sr(
            select=ysel, group="year",
            where=[tw, "latitude IS NOT NULL AND longitude IS NOT NULL", f"NOT ({inside})"],
            start=last3[0], end=last3[1]))
        c.get("how", lambda: c.sr(
            select="date_extract_dow(created_date) as dow, date_extract_hh(created_date) as hh, count(*) as n",
            group="dow, hh", where=[tw], start=last2[0], end=last2[1]))
        c.get("resolution", lambda: c.sr(
            select="resolution_description, count(*) as n", group="resolution_description",
            where=[tw], start=last2[0], end=last2[1]))

    # Day-of-week convention on a known Monday.
    d0, d1 = DOW_CHECK_DATE.isoformat(), (DOW_CHECK_DATE + dt.timedelta(days=1)).isoformat()
    dsel = "date_extract_dow(created_date) as dow, count(*) as n"

    def dow_check():
        if target is not None:
            rows = c.sr(select=dsel, group="dow", where=[target_where(target)], start=d0, end=d1)
            if sum(int(r["n"]) for rs in rows.values() for r in rs) > 0:
                return {"scope": "target pair", "rows": rows}
        rows = c.sr(select=dsel, group="dow", where=["1=1"], start=d0, end=d1)
        return {"scope": "all 311 complaint types (fallback)", "rows": rows}

    c.get("dow_check", dow_check)

    # DOT bike routes.
    c.get("bike_catalog", lambda: c.url(DISCOVERY_URL, {
        "domains": "data.cityofnewyork.us", "q": "bicycle routes", "limit": "20"}))
    bike_id = None
    if not isinstance(R["bike_catalog"], Missing):
        bike_id = pick_bike_dataset(R["bike_catalog"])
    if bike_id is None:
        for k in ("bike_meta", "bike_sample", "bike_status"):
            R[k] = Missing("no bike-route dataset identified via discovery")
    else:
        c.get("bike_meta", lambda: c.url(DISCOVERY_URL, {"ids": bike_id}))
        c.get("bike_sample", lambda: c.ds(bike_id, limit=5))
        c.get("bike_status", lambda: c.ds(
            bike_id,
            select="status, count(*) as n, count(ret_date) as n_with_ret_date, "
                   "min(instdate) as min_instdate, max(instdate) as max_instdate",
            group="status"))
    R["bike_id"] = bike_id

    # DOF parking violations.
    c.get("pv_catalog", lambda: c.url(DISCOVERY_URL, {
        "domains": "data.cityofnewyork.us", "q": "Parking Violations Issued", "limit": "40"}))
    pv_id = None
    if not isinstance(R["pv_catalog"], Missing):
        pv_id = pick_parking_dataset(R["pv_catalog"])
    if pv_id is None:
        R["pv_meta"] = R["pv_sample"] = Missing("no Parking Violations Issued dataset identified via discovery")
    else:
        c.get("pv_meta", lambda: c.url(DISCOVERY_URL, {"ids": pv_id}))
        c.get("pv_sample", lambda: c.ds(pv_id, limit=5))
    R["pv_id"] = pv_id
    return R


def _catalog_rows(catalog: dict) -> list[dict]:
    rows = []
    for r in catalog.get("results", []):
        res = r["resource"]
        rows.append({
            "id": res["id"],
            "name": res["name"].strip(),
            "type": res.get("type"),
            "updated": (res.get("data_updated_at") or "")[:10],
            "description": res.get("description") or "",
        })
    return rows


def pick_bike_dataset(catalog: dict) -> str | None:
    rows = [r for r in _catalog_rows(catalog) if r["type"] == "dataset"]
    for r in rows:
        desc = r["description"].lower()
        if "historic" in desc and "line segment" in desc and "bicycle" in desc:
            return r["id"]
    for r in rows:
        if re.search(r"bike routes|bicycle routes", r["name"], re.I):
            return r["id"]
    return None


def pick_parking_dataset(catalog: dict) -> str | None:
    best = None
    for r in _catalog_rows(catalog):
        m = re.fullmatch(r"Parking Violations Issued - Fiscal Year (\d{4})", r["name"])
        if r["type"] == "dataset" and m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), r["id"])
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Rendering (pure apart from writing the outputs)
# ---------------------------------------------------------------------------


class MissingData(Exception):
    pass


def need(R: dict, key: str) -> Any:
    if key not in R:
        raise MissingData(f"response '{key}' not available")
    if isinstance(R[key], Missing):
        raise MissingData(f"response '{key}': {R[key].reason}")
    return R[key]


@dataclass
class Ctx:
    as_of: dt.date
    out_dir: Path
    notes: list[str]  # open questions accumulated while rendering
    target: tuple[str, str] | None = None
    sunday_is_zero: bool | None = None
    yearly: pd.DataFrame | None = None


def sec_metadata(R: dict, ctx: Ctx, meta: dict) -> str:
    hits, misses = meta.get("cache_hits", 0), meta.get("cache_misses", 0)
    source = "cache" if misses == 0 else ("live" if hits == 0 else "mixed (cache + live)")
    return "\n".join([
        f"- Generated: {meta.get('generated', '?')}",
        f"- As-of date (defines 'current year' / 'last full years'): {ctx.as_of.isoformat()}",
        f"- Git commit: {meta.get('git_commit', '?')}",
        f"- Data source: {source} — cache hits={hits}, misses={misses}",
        f"- Cache dir: `{meta.get('cache_dir', '?')}`",
        "- 311 datasets queried: " + "; ".join(
            f"`{ds}` ({name})" for ds, name, _, _ in SR_DATASETS),
    ])


def _pair_table(merged: list[dict]) -> pd.DataFrame:
    rows = []
    for m in merged:
        row = {"complaint_type": m["complaint_type"], "descriptor": m["descriptor"]}
        for ds, *_ in SR_DATASETS:
            row[ds] = m.get(ds, 0)
        row.update(total=m["total"], first_seen=(m["first"] or "")[:10], last_seen=(m["last"] or "")[:10])
        rows.append(row)
    return pd.DataFrame(rows)


def sec_taxonomy(R: dict, ctx: Ctx) -> str:
    bike = merge_pair_counts(need(R, "taxonomy_bike"))
    out = ["Grouped `(complaint_type, descriptor)` counts, all dates, across both 311 datasets. Counts come from "
           "an unfiltered server-side group-by of every pair (one query per month), matched client-side with the "
           "equivalent of `upper(field) like '%PATTERN%'` on either field.", "",
           "### Pairs matching `%BIKE%`", "", md_table(_pair_table(bike)), ""]
    target, plausible = choose_target(bike)
    ctx.target = target
    if target is None:
        out.append("**No pair mentions both BIKE and BLOCK. No target chosen.**")
    else:
        out.append(f"**Target pair used in the rest of this report:** `complaint_type = {target[0]!r}`, "
                   f"`descriptor = {target[1]!r}` ({plausible[0]['total']:,} rows).")
        if len(plausible) > 1:
            out.append("")
            out.append("**WARNING: more than one plausible blocked-bike-lane pair:** " + "; ".join(
                f"`{p['complaint_type']} / {p['descriptor']}` ({p['total']:,})" for p in plausible))
        others = [m for m in bike if "BIKE LANE" in f"{m['complaint_type']} {m['descriptor']}".upper()
                  and (m["complaint_type"], m["descriptor"]) != target]
        if others:
            out.append("")
            out.append("Other pairs mentioning 'bike lane' (not treated as target): " + "; ".join(
                f"`{m['complaint_type']} / {m['descriptor']}` ({m['total']:,})" for m in others))
    out += ["", "### Pairs matching `%DOUBLE PARK%`", ""]
    try:
        dp = merge_pair_counts(need(R, "taxonomy_double_park"))
        out.append(md_table(_pair_table(dp)))
        out.append("")
        out.append("Double-parking descriptors found: " + (", ".join(
            f"`{m['complaint_type']} / {m['descriptor']}`" for m in dp) or "none"))
    except MissingData as e:
        out.append(f"MISSING: {e}")
    return "\n".join(out)


def sec_yearly(R: dict, ctx: Ctx) -> str:
    df = sum_grouped(list(need(R, "yearly").values()), ["year"])
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")
    full = pd.DataFrame({"year": range(2016, ctx.as_of.year + 1)})
    df = full.merge(df, on="year", how="left").fillna({"n": 0})
    df["n"] = df["n"].astype(int)
    df["note"] = ["partial (to as-of date)" if y == ctx.as_of.year else
                  ("category starts mid-year" if y == 2016 else "") for y in df["year"]]
    lines = [f"Target pair, by `date_extract_y(created_date)`, combined across 311 datasets.", "",
             md_table(df)]
    try:
        if ctx.target is None:
            raise MissingData("no target pair")
        cov_rows = []
        for ds, rows in need(R, "taxonomy_bike").items():
            mine = [r for r in rows if (r.get("complaint_type"), r.get("descriptor")) == ctx.target]
            cov_rows.append({
                "dataset": ds,
                "rows (all dates)": sum(int(r["n"]) for r in mine),
                "first": min((r.get("first") or "" for r in mine), default="")[:19],
                "last": max((r.get("last") or "" for r in mine), default="")[:19],
            })
        lines += ["", "Per-dataset coverage for the target pair (from the section 2 taxonomy query):", "",
                  md_table(pd.DataFrame(cov_rows))]
        tax_total = sum(r["rows (all dates)"] for r in cov_rows)
        yearly_total = int(df["n"].sum())
        if tax_total != yearly_total:
            ctx.notes.append(
                f"The target pair's total differs between the taxonomy query ({tax_total:,}, all dates) and the "
                f"yearly query ({yearly_total:,}, 2016 onward). The queries ran minutes apart against a live "
                "dataset, and the taxonomy also includes any rows before 2016.")
        first = min((r["first"] for r in cov_rows if r["first"]), default="")
        if first and first < "2016-11-01":
            ctx.notes.append(
                f"The earliest target row is {first[:10]}, before the Nov 2016 start date in CLAUDE.md.")
    except MissingData as e:
        lines += ["", f"Coverage: MISSING: {e}"]
    ctx.yearly = df
    return "\n".join(lines)


def monthly_series(R: dict) -> pd.Series:
    df = sum_grouped(list(need(R, "monthly").values()), ["month"])
    df["month"] = df["month"].str[:7]
    s = df.set_index("month")["n"].sort_index()
    idx = pd.period_range(s.index.min(), s.index.max(), freq="M").strftime("%Y-%m")
    return s.reindex(idx, fill_value=0)


def render_chart(s: pd.Series, path: Path, partial_month: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, line = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    x = pd.PeriodIndex(s.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(x, s.values, color=line, linewidth=2)
    if partial_month and partial_month in s.index:
        i = list(s.index).index(partial_month)
        ax.plot(x[i], s.values[i], marker="o", markersize=8, markerfacecolor="#fcfcfb",
                markeredgecolor=line, markeredgewidth=2)
        ax.annotate("partial month", (x[i], s.values[i]), xytext=(0, -18), textcoords="offset points",
                    ha="center", color=ink2, fontsize=8)
    covid = pd.Timestamp(COVID_LINE + "-01")
    ax.axvline(covid, color=ink2, linewidth=1, linestyle="--")
    ax.annotate("2020-03", (covid, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                color=ink2, fontsize=8)
    ax.set_title("311 'Blocked Bike Lane' requests per month (reported, raw requests)",
                 loc="left", color=ink, fontsize=11)
    ax.set_ylabel("requests / month", color=ink2, fontsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def sec_monthly(R: dict, ctx: Ctx) -> str:
    s = monthly_series(R)
    cur = ctx.as_of.strftime("%Y-%m")
    render_chart(s, ctx.out_dir / "m1_monthly.png", cur)
    complete = s[s.index < cur]
    drops, rises = mom_changes(complete)
    for df in (drops, rises):
        df["pct"] = df["pct"].map(lambda v: "" if pd.isna(v) else f"{v:+.0f}%")
        for col in ("prev", "value", "change"):
            df[col] = df[col].astype(int)
    lines = [
        "![Monthly volume](m1_monthly.png)", "",
        f"Months covered: {s.index[0]} → {s.index[-1]}. "
        f"Month-over-month changes below exclude the current month ({cur}, partial).", "",
        "**Five largest month-over-month drops**", "", md_table(drops), "",
        "**Five largest month-over-month rises**", "", md_table(rises), "",
        "<details><summary>Full monthly series</summary>", "",
        md_table(pd.DataFrame({"month": s.index, "n": s.values.astype(int)})), "", "</details>",
    ]
    return "\n".join(lines)


def sec_coords(R: dict, ctx: Ctx) -> str:
    tot = sum_grouped(list(need(R, "coord_total").values()), ["year"]).rename(columns={"n": "rows"})
    nul = sum_grouped(list(need(R, "coord_null").values()), ["year"]).rename(columns={"n": "null_latlon"})
    out = sum_grouped(list(need(R, "coord_outside").values()), ["year"]).rename(columns={"n": "outside_bbox"})
    df = tot.merge(nul, on="year", how="left").merge(out, on="year", how="left").fillna(0)
    df["year"] = df["year"].astype(int)
    for c in ("rows", "null_latlon", "outside_bbox"):
        df[c] = df[c].astype(int)
    df = df.sort_values("year")
    total = df[["rows", "null_latlon", "outside_bbox"]].sum()
    df = pd.concat([df, pd.DataFrame([{"year": "all", **total.to_dict()}])], ignore_index=True)
    df["null_share"] = (df["null_latlon"] / df["rows"]).map(lambda v: f"{v:.2%}")
    df["outside_share"] = (df["outside_bbox"] / df["rows"]).map(lambda v: f"{v:.2%}")
    return "\n".join([
        f"Target pair, last 3 full years ({ctx.as_of.year - 3}–{ctx.as_of.year - 1}). Bbox check is server-side: "
        f"lat {NYC_LAT_MIN}–{NYC_LAT_MAX}, lon {NYC_LON_MIN} to {NYC_LON_MAX}; "
        "'outside' counts rows with non-null coordinates outside that box.", "", md_table(df)])


def sec_dow(R: dict, ctx: Ctx) -> str:
    chk = need(R, "dow_check")
    rows = [r for rs in chk["rows"].values() for r in rs]
    lines = [f"Query: `date_extract_dow(created_date)` grouped, for created_date on {DOW_CHECK_DATE} "
             f"(a {DOW_CHECK_DATE.strftime('%A')}). Scope: {chk['scope']}.", "",
             md_table(pd.DataFrame(rows))]
    try:
        ctx.sunday_is_zero = infer_sunday_is_zero(rows)
        conv = "0 = Sunday … 6 = Saturday" if ctx.sunday_is_zero else "0 = Monday … 6 = Sunday"
        lines += ["", f"**Verified convention:** Monday returned dow={1 if ctx.sunday_is_zero else 0}, so "
                  f"Socrata `date_extract_dow` uses {conv}. Section 7 uses `sunday_is_zero={ctx.sunday_is_zero}`."]
    except ValueError as e:
        lines += ["", f"**Convention could not be verified:** {e}"]
    return "\n".join(lines)


def sec_how(R: dict, ctx: Ctx) -> str:
    if ctx.sunday_is_zero is None:
        raise MissingData("day-of-week convention not verified in section 6")
    df = sum_grouped(list(need(R, "how").values()), ["dow", "hh"])
    counts = [0] * 168
    for _, r in df.iterrows():
        counts[to_hour_of_week(int(r["dow"]), int(r["hh"]), ctx.sunday_is_zero)] += int(r["n"])
    s = pd.Series(counts, index=range(168))
    tbl = lambda ss: md_table(pd.DataFrame(  # noqa: E731
        {"hour_of_week": ss.index, "slot": [how_label(i) for i in ss.index], "events": ss.values}))
    top = s.sort_values(ascending=False, kind="stable").head(10)
    bottom = s.sort_values(ascending=True, kind="stable").head(10)
    grid = pd.DataFrame([[counts[d * 24 + h] for h in range(24)] for d in range(7)],
                        index=DAY_NAMES, columns=[f"{h:02d}" for h in range(24)]).reset_index(names="day")
    return "\n".join([
        f"Target pair, {ctx.as_of.year - 2}–{ctx.as_of.year - 1}, 168 bins (0 = Mon 00:00), raw requests.", "",
        f"- min per bin: {s.min():,} ({how_label(int(s.idxmin()))})",
        f"- median per bin: {s.median():,.1f}",
        f"- max per bin: {s.max():,} ({how_label(int(s.idxmax()))})",
        f"- total: {s.sum():,}", "",
        "**Top 10 bins**", "", tbl(top), "", "**Bottom 10 bins**", "", tbl(bottom), "",
        "<details><summary>Full 7×24 grid</summary>", "", md_table(grid), "", "</details>"])


def sec_sparsity(R: dict, ctx: Ctx) -> str:
    yearly = ctx.yearly
    if yearly is None:
        raise MissingData("yearly volume (section 3) not available")
    yrs = [ctx.as_of.year - 2, ctx.as_of.year - 1]
    vals = yearly.set_index("year").loc[yrs, "n"]
    annual = float(vals.mean())
    rows = []
    for n in SPARSITY_CELLS:
        rows.append({"N cells": n,
                     "per cell-slot-month (168 hourly slots)": sparsity(annual, n, 168),
                     "per cell-slot-month (56 three-hour slots)": sparsity(annual, n, 56)})
    return "\n".join([
        f"Annual target count = mean of {yrs[0]} ({int(vals.iloc[0]):,}) and {yrs[1]} ({int(vals.iloc[1]):,}) "
        f"= {annual:,.0f} raw requests (before dedup to incidents and before snapping).",
        "Expected events per cell-slot-month = annual ÷ (slots × 12) ÷ N, assuming events spread evenly.", "",
        md_table(pd.DataFrame(rows))])


def sec_resolution(R: dict, ctx: Ctx) -> str:
    df = sum_grouped(list(need(R, "resolution").values()), ["resolution_description"])
    df["resolution_description"] = df["resolution_description"].fillna("(null)")
    total = df["n"].sum()
    top = df.sort_values("n", ascending=False).head(15).copy()
    top["share"] = (top["n"] / total).map(lambda v: f"{v:.1%}")
    return "\n".join([
        f"Target pair, {ctx.as_of.year - 2}–{ctx.as_of.year - 1}. {len(df):,} distinct values; "
        f"total {total:,}. Context only.", "", md_table(top.reset_index(drop=True))])


def _catalog_table(catalog: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: r[k] for k in ("id", "name", "type", "updated")} for r in _catalog_rows(catalog)])


def _meta_columns(meta: dict) -> tuple[list[str], list[str]]:
    res = meta["results"][0]["resource"]
    return res.get("columns_field_name", []), res.get("columns_datatype", [])


def sec_bike(R: dict, ctx: Ctx) -> str:
    catalog = need(R, "bike_catalog")
    lines = ["Discovery query: `q=bicycle routes` on data.cityofnewyork.us.", "",
             md_table(_catalog_table(catalog)), ""]
    bike_id = R.get("bike_id")
    if not bike_id:
        raise MissingData("no candidate describes bicycle routes as line segments")
    name = next(r["name"] for r in _catalog_rows(catalog) if r["id"] == bike_id)
    lines.append(f"**Picked:** `{bike_id}` — {name}.")
    meta = need(R, "bike_meta")
    desc = meta["results"][0]["resource"].get("description", "")
    lines += ["", f"> {desc.strip()}", ""]
    names, types = _meta_columns(meta)
    sample = need(R, "bike_sample")
    sample_cols = sorted(set().union(*(r.keys() for r in sample))) if sample else []
    lines += [f"Columns from a `$limit=5` sample ({len(sample_cols)}): " + ", ".join(f"`{c}`" for c in sample_cols), "",
              f"Columns from catalog schema ({len(names)}): " + ", ".join(
                  f"`{n}` ({t})" for n, t in zip(names, types)), ""]
    absent = sorted(set(names) - set(sample_cols))
    if absent:
        lines.append("In the schema but absent from the 5-row sample (Socrata omits nulls): "
                     + ", ".join(f"`{c}`" for c in absent))
        lines.append("")
    ident = identify_bike_columns(sorted(set(names) | set(sample_cols)))
    lines.append("**Column identification:**")
    lines.append("")
    for role, cols in ident.items():
        lines.append(f"- {role}: " + (", ".join(f"`{c}`" for c in cols) if cols else "**NOT IDENTIFIED**"))
        if not cols:
            ctx.notes.append(f"No {role} column could be identified in the bike-route dataset `{bike_id}`.")
    if len(ident["facility type"]) > 1:
        ctx.notes.append(
            "The bike-route dataset has several facility-type columns: "
            + ", ".join(f"`{c}`" for c in ident["facility type"])
            + ". Which one (or which combination) defines lane type is not settled by this audit.")
    lines += ["", "**Rows by status:**", ""]
    st = pd.DataFrame(need(R, "bike_status"))
    for c in ("n", "n_with_ret_date"):
        if c in st:
            st[c] = st[c].astype(int)
    for c in ("min_instdate", "max_instdate"):
        if c in st:
            st[c] = st[c].fillna("").str[:10]
    lines.append(md_table(st.sort_values("n", ascending=False).reset_index(drop=True)))
    if {"status", "n", "n_with_ret_date"} <= set(st.columns):
        for _, r in st.iterrows():
            if str(r["status"]).lower() == "retired" and r["n"] != r["n_with_ret_date"]:
                ctx.notes.append(
                    f"{r['n'] - r['n_with_ret_date']:,} of {r['n']:,} 'Retired' bike-route rows have no `ret_date`.")
            if str(r["status"]).lower() != "retired" and r["n_with_ret_date"] > 0:
                ctx.notes.append(
                    f"{r['n_with_ret_date']:,} '{r['status']}' bike-route rows have a `ret_date` set.")
    if "min_instdate" in st:
        early = sorted(d for d in st["min_instdate"] if d and d < "1950")
        if early:
            ctx.notes.append(
                "The earliest bike-route `instdate` values are " + ", ".join(early) + ". These may be placeholder "
                "dates; this audit does not count how many rows carry them.")
    return "\n".join(lines)


def sec_parking(R: dict, ctx: Ctx) -> str:
    catalog = need(R, "pv_catalog")
    pv_id = R.get("pv_id")
    lines = ["Discovery query: `q=Parking Violations Issued` on data.cityofnewyork.us.", "",
             md_table(_catalog_table(catalog)), ""]
    if not pv_id:
        raise MissingData("no 'Parking Violations Issued - Fiscal Year NNNN' dataset in discovery results")
    name = next(r["name"] for r in _catalog_rows(catalog) if r["id"] == pv_id)
    lines.append(f"**Picked (most recent fiscal year):** `{pv_id}` — {name}.")
    names, types = _meta_columns(need(R, "pv_meta"))
    sample = need(R, "pv_sample")
    sample_cols = sorted(set().union(*(r.keys() for r in sample))) if sample else []
    lines += ["", f"Columns from catalog schema ({len(names)}): " + ", ".join(
        f"`{n}` ({t})" for n, t in zip(names, types)), "",
        f"Columns seen in a `$limit=5` sample ({len(sample_cols)}): " + ", ".join(f"`{c}`" for c in sample_cols), ""]
    extra = sorted(set(sample_cols) - set(names))
    all_names = names + extra
    all_types = types + ["(sample only)"] * len(extra)
    coords = find_coord_columns(all_names, all_types)
    if coords:
        lines.append("**Coordinate-like columns found:** " + ", ".join(f"`{n}` ({t})" for n, t in coords))
    else:
        lines.append("**No latitude/longitude or point/location column exists.** Location fields are "
                     "address- or code-based (e.g. house number, street name, street codes, precinct).")
    return "\n".join(lines)


def sec_open_questions(R: dict, ctx: Ctx, missing: list[str]) -> str:
    notes = list(ctx.notes)
    if any(isinstance(R.get(k), dict) and len(R[k]) > 1 for k in ("yearly", "monthly")):
        notes.insert(0, "The 311 data now lives in two datasets: `erm2-nwe9` covers 2020 onward only, and 2010–2019 "
                     "is in `76ig-c548`. CLAUDE.md lists only `erm2-nwe9`. This audit queries both and combines them.")
    for m in missing:
        notes.append(f"Section '{m}' could not be produced (see MISSING note).")
    if not notes:
        return "_None._"
    return "\n".join(f"- {n}" for n in notes)


def render_report(R: dict, meta: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = dt.date.fromisoformat(R.get("as_of") or dt.date.today().isoformat())
    ctx = Ctx(as_of=as_of, out_dir=out_dir, notes=[])
    bodies: dict[str, str] = {}
    missing: list[str] = []
    funcs = {
        SECTIONS[1]: sec_taxonomy, SECTIONS[2]: sec_yearly, SECTIONS[3]: sec_monthly,
        SECTIONS[4]: sec_coords, SECTIONS[5]: sec_dow, SECTIONS[6]: sec_how, SECTIONS[7]: sec_sparsity,
        SECTIONS[8]: sec_resolution, SECTIONS[9]: sec_bike, SECTIONS[10]: sec_parking,
    }
    for title, fn in funcs.items():
        try:
            bodies[title] = fn(R, ctx)
        except MissingData as e:
            bodies[title] = f"MISSING: {e}"
            missing.append(title)
        except Exception as e:  # surface unexpected failures in the report instead of crashing
            bodies[title] = f"MISSING: unexpected {type(e).__name__}: {e}"
            missing.append(title)
    bodies[SECTIONS[0]] = sec_metadata(R, ctx, meta)
    bodies[SECTIONS[11]] = sec_open_questions(R, ctx, missing)

    header = [
        "# M1 data audit",
        "",
        "_Generated by `python -m clearlane.audit.m1_audit` from cached Socrata responses. Do not hand-edit._",
        "",
        "All counts are **reported** obstruction (311 service requests), not observed obstruction.",
    ]
    if "taxonomy_bike" in R and not isinstance(R["taxonomy_bike"], Missing):
        _, plausible = choose_target(merge_pair_counts(R["taxonomy_bike"]))
        if len(plausible) > 1:
            header += ["", "> **⚠ MORE THAN ONE PLAUSIBLE BLOCKED-BIKE-LANE PAIR — see section 2.**"]
        if not plausible:
            header += ["", "> **⚠ NO BLOCKED-BIKE-LANE PAIR FOUND — see section 2.**"]
    parts = header
    for title in SECTIONS:
        parts += ["", f"## {title}", "", bodies[title]]
    md = "\n".join(parts) + "\n"
    (out_dir / "m1_audit.md").write_text(md)
    return md


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + (" (working tree dirty)" if dirty else "")
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-query Socrata")
    ap.add_argument("--as-of", type=dt.date.fromisoformat, default=dt.date.today(),
                    help="date defining 'current year' (default: today)")
    ap.add_argument("--cache-dir", type=Path, default=socrata.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    socrata.reset_stats()
    R = collect(args.cache_dir, args.refresh, args.as_of)
    meta = {
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "cache_hits": socrata.STATS["hits"],
        "cache_misses": socrata.STATS["misses"],
        "cache_dir": str(args.cache_dir),
    }
    render_report(R, meta, args.out_dir)
    print(f"cache: hits={meta['cache_hits']} misses={meta['cache_misses']}")
    print(f"wrote {args.out_dir / 'm1_audit.md'} and {args.out_dir / 'm1_monthly.png'}")


if __name__ == "__main__":
    main()

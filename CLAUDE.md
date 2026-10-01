# NYC Blocked Bike Lane Risk Map — Spec

Working name: `clearlane`. This file is the stable contract. Read it as ground truth at the start of every session. Running log, decisions, and "Up Next" live in CONTEXT.md.

## What this is

An interactive map of all five boroughs showing, for any day of the week and hour of the day, how likely each part of the bike network is to be reported as blocked (double parking, standing vehicles, etc.). Underneath it is a pipeline that ingests NYC 311 complaints, joins them to the DOT bike network on an H3 hex grid, and fits a count model over cell × hour-of-week.

## The one claim we are allowed to make

The target is **reported** obstruction (311 "Blocked Bike Lane" requests), not observed obstruction. Complaint volume depends on who rides, who reports, and who knows 311 exists, and that varies sharply by neighborhood. Every user-facing surface (map legend, tooltip, README, eval writeup) says "reported." The reporting-adjusted layer is a heuristic and is labeled as one. If the optional ground-truth sample (M11) is done, its numbers are reported as-is, including if they make the model look worse.

## Data sources

| Source | Use | Notes |
|---|---|---|
| 311 Service Requests, 2010–present (NYC Open Data `erm2-nwe9`, Socrata API) | Target events | Filter to the blocked-bike-lane category. Exact `complaint_type`/`descriptor` values to be confirmed in M1. Category exists from Nov 2016. |
| Same 311 dataset, all other complaint types | Reporting-propensity feature | General "how much does this cell call 311" signal. |
| NYC DOT Bicycle Routes (NYC Open Data, line segments) | Spatial frame + lane-type features | Uses facility type, `instdate`, `ret_date`, `status`. Field names to be confirmed in M1. |
| Citi Bike trip data (public S3 tripdata) | Cyclist-exposure proxy | Station-level trip starts/ends, aggregated to cells. Coverage is uneven outside Manhattan/Brooklyn/western Queens — treat as a feature, not ground truth. |
| MapPLUTO land use | Commercial-frontage feature | Delivery double-parking should track commercial density. |
| NOAA daily weather (Central Park) | v2 feature only | Not in v1. |
| DOF Parking Violations Issued | **Deferred** | Believed to carry street address, not coordinates; geocoding millions of rows is out of v1 scope. Confirm in M1. |

## Core design decisions

**Spatial unit.** H3 resolution 9 (~0.1 km² hexes). The modeled grid is every cell that intersects at least one active bike-lane segment. Cells with no bike lane render grey ("no bike lane"), never as zero risk. Rough sizing: NYC is ~780 km², so ~7–8k res-9 cells total and likely a few thousand on the network; confirm in M3.

**Lane activity over time.** A cell is "on network" for a given month only if some segment in it was installed before and not retired during that month. Complaints in a cell before its lane existed are dropped and counted.

**Snapping.** A complaint is kept only if it lies within `SNAP_TOLERANCE_M` (start at 40 m) of an active lane segment; it is then assigned to that segment's cell, not the cell its raw point falls in. Dropped fraction is logged per run.

**Deduplication.** Repeat reports of the same vehicle inflate counts. An "incident" is all kept complaints in the same cell within a 15-minute window; the target counts incidents, not raw requests. Raw counts are kept alongside for comparison.

**Time.** `created_date` is interpreted as America/New_York wall-clock time. Hour-of-week is 0–167 with 0 = Monday 00:00. DST transitions must not create or drop an hour-of-week bin (tested).

**Panel grain.** One row per (cell, hour_of_week, calendar month). `y` = incident count. `exposure` = number of times that hour-of-week occurred in that month while the cell was on network. Zero rows are explicit.

**Splits.** Strictly temporal, never random. Initial proposal: train through 2024-12, validate 2025-01 → 2025-06, test 2025-07 → 2026-06. Adjust once M1 shows regime breaks (2020 COVID dip, any 311 app changes). Test period is touched once per model version.

**Models.**
Baseline: empirical-Bayes rate per (cell, hour_of_week) — Gamma-Poisson shrinkage toward the borough × hour-of-week rate.
Main: LightGBM, `objective="poisson"`, `init_score = log(exposure)`.
The main model ships only if it beats the baseline on validation Poisson deviance. If it doesn't, the baseline ships and the writeup says so.

**Metrics.** Mean Poisson deviance on test; top-decile capture (share of test incidents falling in the model's top 10% of cell-hour slots); calibration by predicted-rate decile (predicted vs observed incidents).

**Map layers.**
1. Predicted reported rate (default).
2. Reporting-adjusted: predicted rate divided by the cell's normalized 311 propensity index. Labeled "heuristic."
3. Historical observed rate for the selected slot (no model), for sanity-checking layer 1.

**Serving.** Predictions are precomputed for every (cell, hour_of_week) using the most recent month's features and written as a static artifact. FastAPI serves it; nothing is computed at request time.

## Invariants (each guarded by a test that goes red when violated)

1. Re-running ingestion over an overlapping date range does not change row counts (idempotent on `unique_key`).
2. A complaint dated before its nearest lane's install date is excluded.
3. A complaint 100 m from every lane is excluded; one 10 m from a lane in an adjacent hex is assigned to the lane's hex.
4. Two complaints 5 minutes apart in one cell are one incident; 20 minutes apart are two.
5. The panel for any (cell, month) has exactly 168 rows; DST months included.
6. No feature for month M uses any data timestamped in month M or later (constructed leakage case: inject a future spike and assert features are unchanged).
7. Baseline beats a constant-rate model on validation deviance (if this fails, the pipeline is broken, not the model).
8. The served artifact covers every on-network cell × 168 slots with no NaN and no negative rate.

## Stack

Python 3.12, DuckDB (with the spatial extension) over Parquet in `data/`, `h3`, `geopandas`/`shapely`, LightGBM, FastAPI, pytest. Frontend: a single static page with MapLibre GL + deck.gl `H3HexagonLayer`.

## Repo layout

```
clearlane/
  CLAUDE.md  CONTEXT.md  README.md
  src/clearlane/
    ingest/      # 311, bike network, citibike, pluto pulls
    spatial/     # grid build, snapping, lane-activity
    panel/       # incident dedup, panel construction
    features/
    models/      # baseline, lgbm, eval harness
    serve/       # artifact export, FastAPI app
  web/           # index.html, map.js
  tests/
  data/          # gitignored; raw/, interim/, artifacts/
  reports/       # M1 audit, eval writeups
```

## UI wireframe

```
+----------------------------------------------------------------------+
| ClearLane — reported blocked-bike-lane risk          [ About / caveats ]
+----------------------------------------------------------------------+
| Day:  [Mon][Tue][Wed][Thu][Fri][Sat][Sun]                            |
| Hour: |----o-----------------------------| 8:00 AM   [> play week]   |
| Layer: (o) Predicted  ( ) Reporting-adjusted*  ( ) Historical         |
+-------------------------------------------------+--------------------+
|                                                 | Cell detail        |
|                                                 | -----------        |
|        [ full-NYC map, H3 hexes on the          | Lanes: painted 60% |
|          bike network shaded by rate;           |   protected 40%    |
|          non-network areas grey ]               | Pred. rate: 0.8/wk |
|                                                 | Observed (hist):   |
|                                                 |   0.7/wk           |
|                                                 | [168-slot sparkline|
|                                                 |  for this cell]    |
+-------------------------------------------------+--------------------+
| Legend: low ███████ high   grey = no bike lane                        |
| * Heuristic. All layers show REPORTED obstruction via 311.            |
+----------------------------------------------------------------------+
```

The "play week" control animates the hour slider across all 168 slots so the weekly cycle (weekday delivery peaks vs. evening/weekend patterns) is visible at a glance.
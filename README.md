# ClearLane

An interactive map of New York City's bike network that shows, for any day of the week and hour of the day, how often each part of the network is **reported** as blocked by a parked or standing vehicle.

Everything here is **reported** obstruction: NYC 311 "Illegal Parking / Blocked Bike Lane" service requests. It is not observed obstruction. How often a blocked lane gets reported depends on who rides there, who reports, and who knows 311 exists, and that varies sharply by neighbourhood. A dark hexagon can mean "rarely blocked" or "rarely reported"; this project cannot tell those apart. No ground-truth sample has been collected (milestone M11 was not done).

The spec is [CLAUDE.md](CLAUDE.md); the running log of decisions and numbers is [CONTEXT.md](CONTEXT.md); the full evaluation write-up is [reports/writeup.md](reports/writeup.md).

## What the map shows

Each H3 resolution-9 hexagon (~0.1 km²) that touches an on-street protected or painted bike lane is coloured for the selected hour of the week. Values are **expected reports per week at that hour**. Grey hexagons have no such lane; they are "no bike lane", never "zero risk".

| Layer | What it is |
|---|---|
| **Predicted** (default) | A LightGBM Poisson model of recent per-cell report history, lane type and length, Citi Bike activity, land use and calendar, with its overall level recalibrated each week to the last eight weeks of reports. |
| **Reporting-adjusted*** | Predicted ÷ the cell's index of how much its area calls 311 about anything else. **A heuristic**, not a measurement: a rough way to ask "how high is this given how much people here report?" |
| **Historical** | What was actually reported at that hour over the last 12 months. No model; for sanity-checking the predicted layer. |

Click a hexagon for its lane mix, weekly totals and a 168-hour sparkline of predicted vs observed reports.

## Results

Temporal splits: train 2021-01 → 2024-09, validate 2024-10 → 2025-09, test 2025-10 → 2026-09. Metrics are on (cell, hour-of-week, month) rows. Top-decile capture is the share of incidents that fall in the 10% of (cell, hour-of-week) slots the model rates riskiest.

**Test split, evaluated once per model version** (13,961 incidents; ledger: [reports/test_evaluations.json](reports/test_evaluations.json)):

| Model | Mean Poisson deviance | Top-decile capture | Observed ÷ predicted |
|---|---|---|---|
| Empirical-Bayes baseline | 0.027163 | 58.5% | 0.910 |
| LightGBM | 0.021564 | 84.8% | 0.871 |
| LightGBM + 3-month recalibration (shipped until 2026-10-07) | 0.021525 | 84.8% | 0.995 |
| **LightGBM + weekly recalibration (shipped)** | **0.021506** | **84.8%** | **0.999** |

Validation (14,595 incidents): baseline 0.026467 / 66.5%; LightGBM 0.021355 / 86.8%; with 3-month recalibration 0.021293; with weekly recalibration 0.021275. LightGBM ships because it beats the baseline on validation deviance, the rule set in CLAUDE.md; the weekly recalibration replaced the 3-month one on validation deviance and month-level error (it lags turning points less). Details, calibration tables and caveats: [reports/m6_baseline.md](reports/m6_baseline.md), [reports/m8_lgbm.md](reports/m8_lgbm.md), [reports/m8_test.md](reports/m8_test.md), [reports/writeup.md](reports/writeup.md).

## Reproducing

Requirements: Python 3.12; about 5 GB of free disk during the Citi Bike stage (three ~1–1.6 GB archives stream in parallel and are deleted after aggregation; 127 MB is kept under `data/`); 17 GB of RAM was enough (peak ~6.3 GB while training); on macOS the OpenMP runtime for LightGBM (`brew install libomp`).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # offline tests (network tests: pytest -m live)
python -m clearlane.pipeline             # everything, from the public APIs
python -m clearlane.pipeline --list      # stages; --from / --to run a range
```

A full first run downloads the 311 aggregates (~6 min), the DOT bike routes, PLUTO and ~28 GB of Citi Bike trip archives (streamed and discarded, 30–45 min depending on bandwidth), then trains (~10 min). Re-running from cached data, e.g. `python -m clearlane.pipeline --from features`, takes about 11 minutes (measured: 638 s, most of it training).

How reproduction works:

- **Same data, same numbers.** Training is deterministic: retraining on the same data reproduces the byte-identical LightGBM model. The `test-eval` stage then *verifies* that the recomputed test metrics match the recorded ones instead of recording a new evaluation (`python -m clearlane.models.test_eval --verify` does this alone).
- **Fresh data, new version.** 311 and the DOT bike-route dataset are live and get revised, so a pull on a later date produces slightly different data and therefore a new model version. That version is evaluated on the test split once and appended to the ledger; it will not match the table above exactly.
- The M1 data audit regenerates offline from its cache: `python -m clearlane.audit.m1_audit --as-of 2026-09-30`.

## Running the map

```bash
python -m clearlane.serve.export         # precompute the served artifact (if not already done by the pipeline)
uvicorn clearlane.serve.app:app --port 8000
# open http://localhost:8000   (add ?theme=light or ?theme=dark to force a theme)
```

### Keeping it current

```bash
python -m clearlane.pipeline --refresh   # new 311 and Citi Bike data -> served artifact; no training, no test eval
scripts/weekly_refresh.sh                # the same, logged to data/logs/ (what the schedule runs)
```

- **Level factor.** Recomputed on every refresh from the 56 days of reports ending two days before the latest 311 pull (Open Data publishes about a day late). `python -m clearlane.serve.export --as-of YYYY-MM-DD` pins the date.
- **Monthly rollover is automatic.** The panel runs to the last month that both 311 stores hold as complete (pulled at least 3 days after it ended), and the served month is the one after. Building refuses to run if any month in range is missing or incomplete.
- **Citi Bike lag.** Citi Bike publishes a month a week or more after it ends. Until then the served month repeats the last published month for the Citi Bike features; `meta.json` → `citibike` says when.
- **The running API picks up a refresh** without a restart (it reloads when `meta.json` changes).
- **Schedule (macOS).** `scripts/com.clearlane.refresh.plist` runs the script every Monday at 07:00 (on wake if the Mac was asleep). Install and remove commands are in the file. macOS does not let background jobs read `~/Desktop`, `~/Documents` or `~/Downloads`; with the repo in one of those, the job fails with "Operation not permitted" (exit 126), so keep the checkout elsewhere or grant access in System Settings → Privacy & Security.

The API is precomputed and read-only: `/api/meta`, `/api/slot?day=Mon&hour=8&layer=predicted` (or `how=0..167`), `/api/cell/{h3}`, `/api/grid`. The page loads MapLibre, deck.gl and h3-js from unpkg and Carto basemaps, so it needs a network connection; Carto's free basemaps carry attribution and usage terms.

## Data sources

| Source | NYC Open Data / location | Used for |
|---|---|---|
| 311 Service Requests from 2010 to 2019 | `76ig-c548` | Target rows before 2020; non-target reporting propensity |
| 311 Service Requests from 2020 to Present | `erm2-nwe9` | Target rows from 2020; non-target reporting propensity |
| New York City Bike Routes (DOT) | `mzxg-pwib` | Network, lane type, install/retire dates |
| PLUTO (release 26v2) | `64uk-42ks` | Commercial frontage (static snapshot) |
| Citi Bike trip data | `s3://tripdata` | Cyclist-exposure proxy, 2020-01 onward (published with a lag; see the artifact's `citibike` metadata) |

DOF parking tickets were left out: the dataset has no coordinates (checked in the [M1 audit](reports/m1_audit.md)).

## What this cannot claim

- **Reported, not observed.** The target is 311 reports. Neighbourhoods that report less look safer than they are. The reporting-adjusted layer divides by a crude 311 propensity index; it is labelled a heuristic and should be read as one.
- **No ground truth.** M11, comparing predictions with street-level observation, was not done. There is no evidence here on how well reports track actual blocking.
- **Only dedicated on-street lanes.** The network is protected and painted lanes (DOT class I/II on-street). Shared routes, links and park paths are excluded, which drops about 2% of complaints; 9.3% of 2021+ complaints are not within 40 m of an in-network lane that existed at the time.
- **Level drift.** The city-wide report rate moves (2023 peak, lower since). The shipped model rescales each month by the last three months' observed ÷ predicted ratio, which lags turning points (up to 1.35× off in single months on the test split).
- **Static and patchy features.** PLUTO is a 2026 snapshot used for every month, a documented exception to the no-future-data rule, kept because it slightly but consistently improved validation deviance (0.27%; 0.14–0.50% across five paired seeds). Citi Bike covers only part of the city: 39% of network cells have no station nearby.
- **Recent months are provisional.** September 2026 was pulled on 2026-10-01; late-arriving reports for it may be missing.

## Repository layout

```
src/clearlane/
  audit/      M1 data audit (reports/m1_audit.md)
  ingest/     Socrata client, 311, bike routes, Citi Bike, PLUTO
  spatial/    H3 grid, lane activity, snapping
  panel/      incident dedup, zero-filled panel with DST-correct exposure
  features/   leak-safe trailing-window features
  models/     eval harness, empirical-Bayes baseline, LightGBM, recalibration, one-time test eval
  serve/      artifact export, FastAPI app
  pipeline.py one command for the whole pipeline
web/          the map (index.html, map.js)
tests/        pytest suite; the eight CLAUDE.md invariants each have a test
reports/      generated reports, test ledger, write-up
data/         gitignored: raw/, interim/, artifacts/
```

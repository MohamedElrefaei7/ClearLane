# CONTEXT — ClearLane

Running log, decisions, and what's next. The stable spec is CLAUDE.md.

## Milestones

Each milestone is one or a small number of commits with a single done-condition. Nothing downstream starts until the milestone above it is green.

**M0 — Scaffold.** Repo layout from CLAUDE.md, `pyproject.toml`, pytest wired, `data/` gitignored, empty package imports. Done when `pytest` runs and passes with one trivial test.

**M1 — Data audit (go/no-go).** No pipeline code yet; a script plus `reports/m1_audit.md`. Confirm the exact 311 `complaint_type`/`descriptor` values for blocked bike lanes and the double-parking descriptors, yearly counts 2016→present, share of rows with null or out-of-NYC coordinates, monthly volume chart to spot regime breaks, and the actual column names in the DOT bike-route dataset (facility type, install/retire dates). Also confirm whether DOF Parking Violations carries coordinates. Done when the report exists with real numbers and the split dates in CLAUDE.md are either confirmed or revised. If annual volume is too thin to support hour-of-week modeling, this is where we find out.

**M2 — 311 ingestion.** Incremental Socrata pull for the target category and (separately) all-category monthly counts by location for propensity, written to Parquet. Done when invariant 1 is tested and a full historical pull completes.

**M3 — Bike network + grid.** Load DOT routes, build the res-9 cell set, implement month-level lane activity. Done when invariants 2 is tested and the on-network cell count is recorded below.

**M4 — Snapping.** Assign complaints to cells via nearest active segment within tolerance. Done when invariant 3 is tested and the dropped fraction is logged and recorded below.

**M5 — Incidents + panel.** Dedup to incidents, build the zero-filled (cell, hour_of_week, month) panel with exposure. Done when invariants 4 and 5 are tested (including a DST month) and panel row count matches cells × 168 × months.

**M6 — Eval harness + baseline.** Temporal split, Poisson deviance, top-decile capture, calibration table; empirical-Bayes baseline. Done when invariant 7 is tested and baseline numbers are recorded below.

**M7 — Features.** Lane-type mix and lane length per cell, commercial frontage from PLUTO, reporting-propensity index, Citi Bike exposure, calendar features. Done when invariant 6 (the injected-future-spike test) passes.

**M8 — LightGBM.** Poisson with log-exposure offset. Done when validation metrics vs. baseline are recorded below, whichever way they come out, and the ship decision is logged.

**M9 — Artifact + API.** Precompute cell × 168 predictions for all three layers; FastAPI endpoint returns one slot's values. Done when invariant 8 is tested and a live `curl` for Monday 8 AM returns plausible data.

**M10 — Map frontend.** Wireframe from CLAUDE.md: day/hour controls, play-week, layer toggle, cell detail panel with sparkline, grey non-network areas, caveat footer. Done when the page loads locally against the API and a manual check of three known hotspot corridors from M1 shows them lit at weekday-morning slots.

**M11 — Ground-truth sample (stretch).** Manually label a stratified sample of cell-slots (e.g. street-level imagery or in-person counts on a few corridors) and compare against predicted rates across high- and low-reporting neighborhoods. Done when the comparison is written up honestly, including if it shows the model mostly tracks reporting behavior.

**M12 — README + writeup.** What the map shows, what it can't claim, data sources, eval numbers reproduced from a single command. Done when someone cloning the repo can regenerate the reported metrics.

## Decisions log

- 2026-09-30 — Target is reported obstruction (311), framed as such everywhere; true-obstruction claims require M11.
- 2026-09-30 — H3 res 9, hour-of-week × month panel, Poisson LightGBM with exposure offset vs. empirical-Bayes baseline.
- 2026-09-30 — DOF ticket data deferred from v1 pending M1 check on coordinates.
- 2026-10-01 — M1: Socrata `date_extract_dow` verified as 0 = Sunday (2025-06-02, a Monday, returned 1); target is `complaint_type='Illegal Parking'`, `descriptor='Blocked Bike Lane'`.

## Numbers to fill in

- 311 blocked-bike-lane rows per year (M1): 2016: 916 (from 2016-10-19) · 2017: 3,605 · 2018: 5,700 · 2019: 17,699 · 2020: 8,255 · 2021: 13,362 · 2022: 20,642 · 2023: 28,127 · 2024: 23,548 · 2025: 18,887 · 2026: 14,613 (partial, to 2026-09-29). Raw requests, `Illegal Parking / Blocked Bike Lane`, combined across `76ig-c548` (2010–2019) and `erm2-nwe9` (2020–present). Source: reports/m1_audit.md §3.
- On-network cell count (M3):
- Snap dropped fraction (M4):
- Baseline val deviance / top-decile capture (M6):
- LightGBM val deviance / top-decile capture (M8):

## Up Next

Review M1 audit; decide split dates and hourly vs 3-hour bins.

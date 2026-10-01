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
- 2026-10-01 — Hourly hour-of-week bins (168 slots), not 3-hour bins. Accepted with M1's sparsity numbers in view (~0.002–0.005 expected raw requests per cell-slot-month); signal will come from pooling across months and baseline shrinkage.
- 2026-10-01 — Splits: train 2021-01 → 2024-09, validate 2024-10 → 2025-09, test 2025-10 → 2026-09. Replaces the initial proposal (train ≤ 2024-12 / val 2025-01→06 / test 2025-07→2026-06), which predated data through 2026-09 and had a spring/summer-only validation window. Train start skips pre-2019-07 low volume and the 2020 COVID dip. Raw requests in the target store (pull of 2026-10-01): train 80,409, val 19,656, test 19,114.
- 2026-10-01 — Propensity stays "all non-target 311 complaint types" (CLAUDE.md as written). Double-parking complaints are not used as the propensity denominator: they track the same street behaviour as the target and would cancel real hotspots in the reporting-adjusted layer.
- 2026-10-01 — M4: snapping measures distance in UTM 18N (EPSG:32618, metres); candidates are in-scope segments within 40 m that are active at the complaint's timestamp; the nearest wins (ties by route_id) and the complaint goes to the H3 cell of the closest point on that segment. Invariant 2 is read as "a lane that did not yet exist cannot receive a complaint": if the nearest lane is not yet installed but another active lane is within tolerance, the complaint goes to the active one. Drop reasons: no_coords / no_lane_within_tol / lane_not_installed / lane_not_active. Each run appends counts to `data/interim/snap_runs.jsonl`.
- 2026-10-01 — M4: why 40 m (CLAUDE.md's starting value, checked against data, kept). Panel complaints by distance to the nearest active in-scope lane: 69% within 5 m (geocoded onto the centerline), a second bump at 15–20 m (curbside address points, ~half a street width), a sharp fall after ~25 m, then a flat tail of ~1k per 5 m band. Share of complaints whose street / cross / intersection street names the snapped lane's street, by band: 0–5 m 92%, 5–10 m 50% (likely intersection points snapped to the crossing lane; same hex either way), 10–20 m 92–95%, 20–30 m 84–85%, 30–40 m 86%, 40–50 m 77%, 50–60 m 75%, 60–80 m 43%, 80–100 m 39%. Kept share at tolerance: 20 m 86.3%, 30 m 88.6%, 40 m 90.7%, 50 m 92.2%, 60 m 93.1%, 100 m 95.8%. 40 m is the last band that is still mostly the right street; 50 m is defensible (+1.5 pts at ~77%); beyond 60 m most matches are another street. The name matcher is crude normalization, so the shares are relative, not exact precision.
- 2026-10-01 — M4 observation (no rule change): the 2,253 panel complaints dropped as lane_not_installed are mostly well before the install date (median 151 days; 13% within 30 days), so this is not mainly an `instdate` lag. 2021 has the highest drop rate (14.3%).
- 2026-10-01 — M3: the bike network = on-street class I (protected) and II (painted) segments of `mzxg-pwib` (`facilitycl` in I/II, `onoffst = ON`), scope `lanes_on_street`. Shared/signed routes (III), links and off-street paths are excluded: no lane to block / no car access. Costs ~2% of 2021+ complaints (90.0% vs 92.2% within 40 m of an in-scope active lane) and removes ~1,000 near-structural-zero cells (2,530 vs 3,566 ever on network).
- 2026-10-01 — M3: segment activity = [instdate, ret_date); a cell is on network in month M only if an in-scope segment touching it is active for all of M. 18 segments with undeterminable dates (17 Retired without `ret_date`, 1 retired before installed) are never active. `route_id` (Socrata `:id`) is stable only within one bike-routes snapshot.
- 2026-10-01 — M2: 311 is pulled from both `76ig-c548` (2010–2019) and `erm2-nwe9` (2020–present), one created-month per query. Target store = raw rows of the target pair, upserted on `unique_key` (rows that vanish upstream are reported, not deleted). Propensity store = all *non-target* 311 requests counted per month × 0.002° grid cell via server-side `snap_to_grid` (rounds to nearest; points are cell centres, ~0.04 km² cells, finer than H3 res 9); mapping to H3 happens in M3/M7. `created_date` stored naive (NYC wall clock), never tz-localized.

## Numbers to fill in

- 311 blocked-bike-lane rows per year (M1): 2016: 916 (from 2016-10-19) · 2017: 3,605 · 2018: 5,700 · 2019: 17,699 · 2020: 8,255 · 2021: 13,362 · 2022: 20,642 · 2023: 28,127 · 2024: 23,548 · 2025: 18,887 · 2026: 14,613 (partial, to 2026-09-29). Raw requests, `Illegal Parking / Blocked Bike Lane`, combined across `76ig-c548` (2010–2019) and `erm2-nwe9` (2020–present). Source: reports/m1_audit.md §3.
- 311 rows ingested (M2, pull of 2026-10-01): target 155,354 (2016-10-19 → 2026-09-30; 494 with null coords); propensity 32,801,521 non-target requests 2016-01 → 2026-09, 1,190,057 without location.
- On-network cell count (M3, snapshot 2026-10-01, scope lanes_on_street): 2,530 cells ever on network 2021-01 → 2026-09; 2,147 in 2021-01, 2,520 in 2026-09; 162,257 cell-months.
- Snap dropped fraction (M4, 40 m, lanes_on_street): panel 2021-01 → 2026-09: 9.34% (11,137 of 119,179) — no_lane_within_tol 8,422; lane_not_installed 2,253; lane_not_active 285; no_coords 177. By year: 2021 14.3%, 2022 9.6%, 2023 8.2%, 2024 8.6%, 2025 7.9%, 2026 9.7%. All dates: 10.89%. 1,307 kept panel complaints moved from their raw hex to the lane's hex.
- Baseline val deviance / top-decile capture (M6):
- LightGBM val deviance / top-decile capture (M8):

## Up Next

M5 — incidents + panel: dedup kept complaints to incidents (same cell, 15-minute window), build the zero-filled (cell, hour_of_week, month) panel with exposure over on-network cell-months; test invariants 4 and 5 (incl. a DST month). Complaints whose cell-month is not on network (e.g. lane opened mid-month) are dropped and counted there.

# ClearLane — evaluation write-up

This is the human-written account of what was built, how it was evaluated, and what the results do and do not support. The numbers come from the generated reports in this folder ([m1_audit.md](m1_audit.md), [m6_baseline.md](m6_baseline.md), [m8_lgbm.md](m8_lgbm.md), [m8_test.md](m8_test.md)), the test ledger ([test_evaluations.json](test_evaluations.json)) and the decisions log in [CONTEXT.md](../CONTEXT.md). Data were pulled on 2026-10-01.

## 1. The question and the one claim

For each part of NYC's bike network and each of the 168 hours of the week, how often is a blocked bike lane **reported** to 311?

The target is reports, not obstruction. A lane that is blocked every morning but where nobody files a 311 request looks clean here. Every surface of the project says "reported", and the reporting-adjusted layer is labelled a heuristic. No ground-truth comparison was made (M11 not done), so nothing here measures how well reports track actual blocking.

## 2. Data

**Target.** 311 requests with `complaint_type = 'Illegal Parking'` and `descriptor = 'Blocked Bike Lane'`, the only pair matching both "bike" and "block" (155,354 rows from 2016-10-19 to 2026-09-30). NYC Open Data now splits 311 across two datasets, `76ig-c548` (2010–2019) and `erm2-nwe9` (2020 onward); both are used.

Volume is not stable over time:

| Year | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| Requests | 3,605 | 5,700 | 17,699 | 8,255 | 13,362 | 20,642 | 28,127 | 23,548 | 18,887 |

The jump in mid-2019, the April 2020 collapse (−90% month on month) and the 2023 peak are why training starts in 2021. Coordinates are reliable: 0.12% of 2023–2025 target rows have none, and none fall outside the NYC bounding box.

**Bike network.** DOT "New York City Bike Routes" (`mzxg-pwib`), 29,695 current and retired segments with install and retire dates. The network is restricted to on-street class I (protected) and II (painted) lanes. Within 40 m of an active segment, 96% of complaints are nearest a class I/II on-street lane; shared routes and park paths add about 1,000 cells where a report is near-impossible. That leaves 2,530 H3 res-9 cells ever on the network between 2021-01 and 2026-09 (2,520 in 2026-09).

**Features.** Non-target 311 counts per small grid cell (reporting propensity), Citi Bike trip starts and ends from the public `s3://tripdata` archives (2020-01 → 2026-08), PLUTO release 26v2 (commercial lots, frontage and floor area), and lane length by type.

## 3. From complaints to a panel

- **Snapping.** A complaint is kept if an in-network lane that existed at the complaint's timestamp lies within 40 m; it is assigned to the H3 cell where that lane passes closest. 40 m came from the spec and was checked against the data: in every distance band up to 40 m except 5–10 m, 84–95% of complaints name the snapped lane's street (or a cross or intersection street); the 5–10 m band is 50%, mostly intersection points snapped to the crossing lane, which lands in the same cell either way. At 40–60 m agreement falls to about 76%, beyond 60 m to about 40%. In 2021-01 → 2026-09, 9.34% of complaints are dropped: 8,422 with no in-network lane within 40 m, 2,253 near a lane not yet installed (median 151 days before its recorded install date, so mostly not an install-date lag), 285 near retired lanes, 177 without coordinates.
- **Incidents.** Repeat reports of the same blockage are collapsed: complaints in one cell within 15 minutes of the incident's first report are one incident. 107,872 kept complaints become 82,937 incidents. Chaining reports instead (each within 15 minutes of the previous) changes the count by only 0.8%.
- **Panel.** One row per on-network (cell, hour-of-week, month), 27,259,176 rows, 99.72% zeros. Exposure is elapsed hours on the New York clock, so DST months are exact (March 2024 has 743 hours, and its Sunday 02:00 slot occurs 4 times rather than 5); every slot keeps its row.

## 4. Models

- **Baseline.** Two-level empirical-Bayes Gamma-Poisson: each (cell, hour-of-week) rate shrinks toward its borough × hour rate, which shrinks toward the borough rate. Shrinkage strengths are fitted by marginal likelihood on the training data (100 pseudo-hours at cell level).
- **LightGBM.** Poisson objective with `init_score = log(exposure × base rate)`. 36 features, all built from data strictly before the month being predicted (the leakage invariant is tested by injecting a future spike), except the static PLUTO snapshot. Trained on train, early-stopped on validation (578 rounds). Retraining on the same data reproduces the byte-identical model.
- **Recalibration.** The model ranks cells well but its overall level follows the training years. Each month's predictions are multiplied by observed ÷ predicted reports over the previous three months, using only past data.

## 5. Results

**Validation** (2024-10 → 2025-09, 14,595 incidents):

| Model | Deviance | Top-decile capture | Observed ÷ predicted |
|---|---|---|---|
| Constant rate | 0.035087 | — | — |
| Empirical-Bayes baseline | 0.026467 | 66.5% | 0.964 |
| LightGBM, no PLUTO | 0.021413 | 86.8% | 0.843 |
| LightGBM | 0.021355 | 86.8% | 0.854 |
| LightGBM + recalibration | 0.021293 | 86.7% | 0.998 |

**Test** (2025-10 → 2026-09, 13,842 incidents; each version evaluated once and recorded in the ledger):

| Model | Deviance | Top-decile capture | Observed ÷ predicted |
|---|---|---|---|
| Empirical-Bayes baseline | 0.026923 | 58.6% | 0.902 |
| LightGBM | 0.021368 | 85.2% | 0.864 |
| LightGBM + recalibration (shipped) | 0.021326 | 85.2% | 0.987 |

What these say:

- **LightGBM clearly beats the baseline**, by 21% in deviance on test, and its top 10% of slots hold 85% of test incidents against 59%. The gap is as large on test as on validation, so early stopping on validation did not flatter it much.
- **Most of the signal is recent history.** The cell's own incident counts over recent months, overall (1, 3, 12 months) and for the same hour of the week (12, 36 months), carry about 72% of the model's split gain, and hour of day another 10%. Lanes, Citi Bike, 311 propensity and PLUTO each add little.
- **The baseline's problem is drift, not shrinkage.** On validation its lowest predicted-rate decile saw 8.7× the incidents predicted and its highest 0.86×, while borough totals moved in different directions (Manhattan and Queens down, the Bronx and Brooklyn up). Stronger shrinkage made validation deviance worse, not better; recency features fixed most of it.
- **Raw LightGBM predicts 16–17% too many reports** (observed ÷ predicted 0.854 on validation, 0.864 on test) because the city-wide report rate fell after 2023. The error is close to a uniform level shift: on validation, predicted-rate deciles 3–10 have observed ÷ predicted between 0.78 and 0.98. The three-month recalibration removes it on both splits but lags turning points: individual test months range from 0.81 to 1.35.
- **PLUTO barely matters.** Keeping it improves validation deviance by 0.27%, which is why it stays by the agreed rule. The margin was not checked against seed variation.

## 6. Limitations and threats to validity

1. **Reporting bias is the main one.** Reports depend on who rides, who reports and who knows 311. The model learns where people report. The reporting-adjusted layer divides by a crude index (non-target 311 volume in the cell and its neighbours, relative to the median) and is only a heuristic. It assumes reporting about blocked lanes scales with reporting about everything else, which is untested.
2. **No ground truth.** Without M11 there is no evidence on how predicted reports relate to actual blocking, in high- or low-reporting areas.
3. **Network definition.** Only dedicated on-street lanes count. Complaints on shared routes, about lanes not yet built, or more than 40 m from a lane are excluded (9.3% of 2021+ complaints, higher in 2021).
4. **Drift.** Report volume shifts year to year. Recalibration fixes the level only, with a lag. Spatial drift (which cells rise or fall) is handled only through recency features.
5. **Feature caveats.** PLUTO is a 2026 snapshot used for every month, a documented exception to the no-future-data invariant. Citi Bike only covers part of the city: 39% of network cells have no station nearby, so zero there means "no data". Lane install dates include placeholders (1,030 segments dated before 1950), and 17 retired segments lack a retire date and are excluded.
6. **Data freshness.** 311 and DOT data are live. September 2026 was pulled on 2026-10-01, so late reports for it may be missing; a later pull produces a new model version with slightly different numbers.
7. **Evaluation choices.** Single train/validation/test split, a single LightGBM configuration with no hyperparameter search, early stopping on validation, metrics on raw (cell, hour, month) rows dominated by zeros.

## 7. Reproducing these numbers

```bash
pip install -e ".[dev]"
python -m clearlane.pipeline                  # full run from the public APIs
python -m clearlane.pipeline --from features  # rebuild from cached data
python -m clearlane.models.test_eval --verify # recompute test metrics and compare with the ledger
```

With the cached 2026-10-01 data, the pipeline regenerates the reports in this folder, retrains the identical LightGBM model, and verifies the test numbers against the ledger without recording a new evaluation. A fresh clone pulling live data gets a new model version, which is evaluated once and appended to the ledger.

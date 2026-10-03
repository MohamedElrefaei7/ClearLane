"""Precompute the served artifact (M9).

    python -m clearlane.serve.export [--month YYYY-MM]

Serving month M defaults to the month after the last panel month (the latest
complete month). Features for M use only data through M-1. Writes to
`data/artifacts/serving/`:

* `predictions.parquet` — every on-network cell x 168 hour-of-week slots:
    predicted   expected reported incidents per occurrence of that hour
                (= per week for that slot): shipped LightGBM x the level factor
                (observed / raw predicted over M-3 .. M-1).
    adjusted    predicted / the cell's reporting-propensity index. HEURISTIC.
    historical  observed incidents / exposure hours over M-12 .. M-1 (no model);
                `historical_hours` = the exposure behind it (0 = no history).
* `cells.parquet` — per cell: centre, borough, lane metres by type, propensity
  index, weekly totals per layer.
* `meta.json` — month, model version, factor, units, caveats.

Invariant 8 is checked before anything is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from clearlane.config import PANEL_MONTHS
from clearlane.features.build import LANE_TYPES, cell_how_history, cell_month_features, load_inputs, month_range
from clearlane.ingest.sr311 import month_bounds
from clearlane.models.lgbm import ARTIFACT_DIR, assemble, load_model, predict_mu
from clearlane.models.recalibrate import WINDOW, monthly_totals, raw_predictions, trailing_factor
from clearlane.panel.build import PANEL_PATH, SLOTS
from clearlane.spatial.grid import BOROUGHS, DEFAULT_SCOPE, network_cell_months

SERVING_DIR = ARTIFACT_DIR / "serving"
LAYERS = ["predicted", "adjusted", "historical"]
PROPENSITY_FLOOR_QUANTILE = 0.05
CAVEAT = ("All layers show REPORTED obstruction: 311 'Blocked Bike Lane' requests, not observed obstruction. "
          "Reporting varies sharply by neighborhood. The reporting-adjusted layer is a heuristic.")


def next_month(month: str) -> str:
    return month_bounds(month)[1][:7]


def propensity_index(raw: pd.Series) -> pd.Series:
    """Normalize to median 1 across on-network cells, floored at the 5th percentile."""
    if not raw.median() > 0:
        raise ValueError("median propensity is not positive; cannot normalize")
    idx = raw / raw.median()
    floor = idx.quantile(PROPENSITY_FLOOR_QUANTILE)
    if not floor > 0:
        raise ValueError("propensity floor is not positive; adjusted layer would be unbounded")
    return idx.clip(lower=floor)


def validate_artifact(preds: pd.DataFrame, cells: pd.Series) -> None:
    """Invariant 8: every on-network cell x 168 slots, no NaN (or infinity), no negative rate."""
    expected = set(cells)
    got = preds.groupby("cell")["hour_of_week"].agg(["count", "nunique", "min", "max"])
    problems = []
    if set(got.index) != expected:
        problems.append(f"cells missing {len(expected - set(got.index))}, unexpected {len(set(got.index) - expected)}")
    bad = got[(got["count"] != SLOTS) | (got["nunique"] != SLOTS) | (got["min"] != 0) | (got["max"] != SLOTS - 1)]
    if len(bad):
        problems.append(f"{len(bad)} cells without exactly 168 distinct slots")
    for col in LAYERS + ["historical_hours"]:
        if not np.isfinite(preds[col].to_numpy(dtype=float)).all():
            problems.append(f"NaN or infinite values in {col}")
        if (preds[col] < 0).any():
            problems.append(f"negative {col}")
    if problems:
        raise ValueError("invariant 8 violated: " + "; ".join(problems))


def historical_rates(cells: list[str], month: str) -> pd.DataFrame:
    """Observed incidents per exposure hour over the 12 months before `month`."""
    months = month_range(month_range("2000-01", month)[-13], month_range("2000-01", month)[-2])
    p = pd.read_parquet(PANEL_PATH, columns=["cell", "month", "hour_of_week", "y", "exposure"],
                        filters=[("month", "in", months)])
    agg = (p.assign(cell=p["cell"].astype(str)).groupby(["cell", "hour_of_week"], observed=True)[["y", "exposure"]]
           .sum().reset_index())
    grid = pd.DataFrame({"cell": np.repeat(cells, SLOTS), "hour_of_week": np.tile(np.arange(SLOTS), len(cells))})
    out = grid.merge(agg, on=["cell", "hour_of_week"], how="left").fillna({"y": 0, "exposure": 0})
    out["historical"] = np.where(out["exposure"] > 0, out["y"] / out["exposure"].where(out["exposure"] > 0, 1), 0.0)
    return out.rename(columns={"exposure": "historical_hours"})[["cell", "hour_of_week", "historical", "historical_hours"]]


def build(month: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    decision = json.loads((ARTIFACT_DIR / "m8_decision.json").read_text())
    variant = decision["chosen_variant"]
    inp = load_inputs()

    network = network_cell_months(inp.segments, inp.seg_cells, [month], DEFAULT_SCOPE)
    rows = network[["cell", "month"]].astype(str).sort_values("cell").reset_index(drop=True)
    cells = rows["cell"].tolist()
    cm = cell_month_features(rows, inp, month)
    keys = pd.DataFrame({"cell": np.repeat(cells, SLOTS), "month": month,
                         "hour_of_week": np.tile(np.arange(SLOTS), len(cells)), "y": 0, "exposure": 1})
    ch = cell_how_history(keys, inp.incidents, month_range("2016-01", month))
    drop = () if variant == "full" else ("pluto_",)
    X, _, e = assemble(keys, cm, ch, drop)

    booster, meta = load_model(variant)
    raw = predict_mu(booster, X[meta["features"]], e, meta["base_rate"])
    prior = month_range(month_range("2000-01", month)[-1 - WINDOW], month_range("2000-01", month)[-2])
    pk, pmu = raw_predictions(prior, variant)
    totals = monthly_totals(pk, pmu)
    totals.loc[month] = [np.nan, np.nan]
    factor = float(trailing_factor(totals)[month])

    preds = keys[["cell", "hour_of_week"]].copy()
    preds["predicted"] = raw * factor
    pidx = propensity_index(cm.set_index("cell")["prop311_ring1_12m"])
    preds["adjusted"] = preds["predicted"] / preds["cell"].map(pidx).to_numpy()
    preds = preds.merge(historical_rates(cells, month), on=["cell", "hour_of_week"], how="left")
    validate_artifact(preds, rows["cell"])

    lanes = cm.set_index("cell")[[f"lane_m_{t}" for t in LANE_TYPES] + ["lane_m_total"]]
    weekly = preds.groupby("cell")[LAYERS].sum().add_suffix("_weekly")
    centre = pd.DataFrame([h3.cell_to_latlng(c) for c in cells], index=cells, columns=["lat", "lng"])
    cell_tab = (centre.join(cm.set_index("cell")["boro"].map(lambda b: BOROUGHS[str(b)]).rename("borough"))
                .join(lanes).join(pidx.rename("propensity_index")).join(weekly).rename_axis("cell").reset_index())

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta_out = {
        "month": month,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": sha,
        "model": {"variant": variant, "base_rate": meta["base_rate"], "best_iteration": meta["best_iteration"]},
        "recalibration": {"window_months": WINDOW, "months": prior, "factor": factor},
        "cells": len(cells),
        "slots": SLOTS,
        "hour_of_week": "0 = Monday 00:00 ... 167 = Sunday 23:00, NYC wall clock",
        "units": "expected reported incidents per occurrence of the hour (i.e. per week for one slot)",
        "layers": {
            "predicted": "LightGBM x level factor (reported obstruction)",
            "adjusted": "predicted / reporting-propensity index (non-target 311 over the last 12 months, own + ring-1 "
                        "cells, divided by the median on-network cell, floored at the 5th percentile). HEURISTIC.",
            "historical": "observed incidents / exposure hours over the previous 12 months; no model",
        },
        "caveat": CAVEAT,
    }
    return preds, cell_tab, meta_out


def write(preds: pd.DataFrame, cells: pd.DataFrame, meta: dict, out_dir: Path = SERVING_DIR) -> None:
    validate_artifact(preds, cells["cell"])
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.astype({"hour_of_week": "int16"}).to_parquet(out_dir / "predictions.parquet", index=False)
    cells.to_parquet(out_dir / "cells.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", default=next_month(PANEL_MONTHS[1]))
    args = ap.parse_args(argv)
    preds, cells, meta = build(args.month)
    write(preds, cells, meta)
    print(json.dumps({k: meta[k] for k in ("month", "cells", "recalibration", "model")}, indent=1))
    print(cells[[f"{l}_weekly" for l in LAYERS]].describe().round(3).to_string())
    print(f"wrote {SERVING_DIR}")


if __name__ == "__main__":
    main()

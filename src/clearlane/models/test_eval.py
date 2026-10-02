"""One-time test-split evaluation (M8).

    python -m clearlane.models.test_eval [--force]

Evaluates on the test split (CLAUDE.md: touched once per model version):
  * baseline — M6 empirical Bayes fitted on train
  * lightgbm_raw — the shipped M8 LightGBM (`full`)
  * lightgbm_recal — the same, times the trailing 3-month level factor
    (clearlane.models.recalibrate); the first test months use validation months,
    all out-of-sample.

Each version is identified by a hash of its fitted parameters. Results are
appended to `reports/test_evaluations.json` (tracked); a version already present
is refused unless `--force` (which should be recorded as a decision). Writes
`reports/m8_test.md`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from clearlane.config import TEST, TRAIN
from clearlane.ingest.sr311 import month_range
from clearlane.models.baseline import EBBaseline
from clearlane.models.eval import calibration_table, evaluate
from clearlane.models.lgbm import ARTIFACT_DIR, _table
from clearlane.models.recalibrate import WINDOW, apply_factor, bias, monthly_totals, raw_predictions, trailing_factor
from clearlane.panel.build import PANEL_PATH
from clearlane.spatial.grid import ROUTES_PATH, SEGMENT_CELLS_PATH, cell_boroughs

LEDGER = Path("reports/test_evaluations.json")  # tracked in git so a fresh clone knows what has been tested
REPORT_PATH = Path("reports/m8_test.md")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-evaluate versions already in the ledger")
    args = ap.parse_args(argv)

    lgbm_hash = file_hash(ARTIFACT_DIR / "lgbm_full.txt") + file_hash(ARTIFACT_DIR / "lgbm_full.json")
    segments = gpd.read_parquet(ROUTES_PATH)
    boroughs = cell_boroughs(segments, pd.read_parquet(SEGMENT_CELLS_PATH))
    train = pd.read_parquet(PANEL_PATH, columns=["cell", "month", "hour_of_week", "y", "exposure"],
                            filters=[("month", "in", month_range(*TRAIN))])
    base = EBBaseline().fit(train, boroughs)
    del train
    base_hash = hashlib.sha256(pd.util.hash_pandas_object(base.slot_rate[["cell", "hour_of_week", "rate"]],
                                                          index=False).values.tobytes()).hexdigest()[:12]
    versions = {"baseline": f"baseline:{base_hash}", "lightgbm_raw": f"lgbm_full:{lgbm_hash}",
                "lightgbm_recal": f"lgbm_full+recal{WINDOW}m:{lgbm_hash}"}

    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    seen = {e["version"] for e in ledger}
    already = [v for v in versions.values() if v in seen]
    if already and not args.force:
        raise SystemExit(f"test split already evaluated for {already}; pass --force only if that is a recorded decision")

    test_months = month_range(*TEST)
    lead = month_range(month_range("2000-01", test_months[0])[-WINDOW - 1], test_months[-1])
    keys, mu = raw_predictions(lead)
    factor = trailing_factor(monthly_totals(keys, mu))
    is_test = keys["month"].isin(test_months).to_numpy()
    tk, traw = keys[is_test].reset_index(drop=True), mu[is_test]
    preds = {
        "baseline": base.predict_rate(tk, boroughs) * tk["exposure"].to_numpy(),
        "lightgbm_raw": traw,
        "lightgbm_recal": apply_factor(tk, traw, factor),
    }
    results = {}
    for name, p in preds.items():
        r = evaluate(tk, p)
        results[name] = {**r, "obs_over_pred": bias(tk, p), "version": versions[name]}

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    ledger += [{"version": v["version"], "evaluated_at": now, "commit": sha,
                "test_months": [test_months[0], test_months[-1]],
                **{k: v[k] for k in ("mean_poisson_deviance", "top_decile_capture", "obs_over_pred", "incidents",
                                     "predicted_incidents")}} for v in results.values()]
    LEDGER.write_text(json.dumps(ledger, indent=1, default=float))

    metrics = pd.DataFrame([{"model": k, "mean_poisson_deviance": v["mean_poisson_deviance"],
                             "top_decile_capture": v["top_decile_capture"], "obs_over_pred": v["obs_over_pred"],
                             "predicted_incidents": v["predicted_incidents"]} for k, v in results.items()])
    months = (monthly_totals(tk, preds["lightgbm_raw"]).rename(columns={"predicted": "raw"})
              .join(factor).join(monthly_totals(tk, preds["lightgbm_recal"])["predicted"].rename("recal"))
              .join(monthly_totals(tk, preds["baseline"])["predicted"].rename("baseline")))
    months["obs/raw"] = months["observed"] / months["raw"]
    months["obs/recal"] = months["observed"] / months["recal"]
    months = months.reset_index()
    md = "\n".join([
        "# M8 — test-split evaluation (one-time)",
        "",
        "_Generated by `python -m clearlane.models.test_eval`. Do not hand-edit._ All counts are **reported** "
        "obstruction (311 incidents), not observed obstruction.",
        "",
        f"- Generated: {now} at commit `{sha}`; ledger: `{LEDGER}`",
        f"- Test {TEST[0]} → {TEST[1]}: {results['baseline']['rows']:,} rows, {results['baseline']['incidents']:,} incidents",
        f"- Models fitted on train ({TRAIN[0]} → {TRAIN[1]}); LightGBM early-stopped on validation. "
        f"Recalibration factor = observed / raw predicted over the previous {WINDOW} months.",
        "- Versions: " + "; ".join(f"`{v}`" for v in versions.values()),
        "",
        "## Test metrics",
        "",
        _table(metrics),
        "",
        "## By month",
        "",
        _table(months),
        "",
        "## Calibration by predicted-rate decile — LightGBM recalibrated",
        "",
        _table(calibration_table(tk, preds["lightgbm_recal"])),
        "",
        "## Calibration by predicted-rate decile — LightGBM raw",
        "",
        _table(calibration_table(tk, preds["lightgbm_raw"])),
        "",
        "## Calibration by predicted-rate decile — baseline",
        "",
        _table(calibration_table(tk, preds["baseline"])),
        "",
    ])
    REPORT_PATH.write_text(md)
    print(metrics.to_string(index=False))
    print(months.round(3).to_string(index=False))
    print(f"wrote {REPORT_PATH}; appended {LEDGER}")


if __name__ == "__main__":
    main()

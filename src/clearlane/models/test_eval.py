"""One-time test-split evaluation (M8).

    python -m clearlane.models.test_eval [--force]

Evaluates on the test split (CLAUDE.md: touched once per model version):
  * baseline — M6 empirical Bayes fitted on train
  * lightgbm_raw — the shipped M8 LightGBM (`full`)
  * lightgbm_recal — the same, times the trailing 3-month level factor
    (clearlane.models.recalibrate); the first test months use validation months,
    all out-of-sample.

Each version is identified by a hash of its fitted parameters. New versions are
evaluated and appended to `reports/test_evaluations.json` (tracked) and
`reports/m8_test.md` is rewritten. A version already in the ledger is refused,
except:
  --verify  recompute the metrics of versions already in the ledger and check
            they match the recorded numbers (reproduction; nothing is appended,
            the report is not rewritten). Fails on any mismatch.
  --auto    verify known versions and record new ones (used by the pipeline).
  --force   re-record known versions (should itself be a recorded decision).
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


LEDGER_METRICS = ("mean_poisson_deviance", "top_decile_capture", "obs_over_pred", "incidents", "predicted_incidents")


def verify(results: dict, recorded: dict, names: set[str], rel: float = 1e-9) -> set[str]:
    """Names whose recomputed metrics differ from the ledger beyond float noise."""
    bad = set()
    for name in names:
        old = recorded[results[name]["version"]]
        for m in LEDGER_METRICS:
            a, b = float(results[name][m]), float(old[m])
            if abs(a - b) > rel * max(1.0, abs(b)):
                bad.add(name)
    return bad


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="re-record versions already in the ledger")
    mode.add_argument("--verify", action="store_true", help="check known versions reproduce; append nothing")
    mode.add_argument("--auto", action="store_true", help="verify known versions, record new ones")
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
    recorded = {e["version"]: e for e in ledger}  # latest entry per version
    known = {k for k, v in versions.items() if v in recorded}
    if args.verify and len(known) < len(versions):
        raise SystemExit(f"--verify: versions not in the ledger: {[versions[k] for k in versions if k not in known]}")
    if known and not (args.force or args.verify or args.auto):
        raise SystemExit(f"test split already evaluated for {[versions[k] for k in known]}; "
                         "use --verify to reproduce, or --force only if that is a recorded decision")
    to_verify = known if (args.verify or args.auto) else set()
    to_record = set(versions) - to_verify

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

    if to_verify:
        mismatches = verify(results, recorded, to_verify)
        for name in sorted(to_verify):
            r = results[name]
            print(f"verify {name:15s} deviance {r['mean_poisson_deviance']:.6g}  top-decile {r['top_decile_capture']:.4f}  "
                  f"obs/pred {r['obs_over_pred']:.4f}  -> {'MISMATCH' if name in mismatches else 'matches ledger'}")
        if mismatches:
            raise SystemExit(f"test metrics do not reproduce for {sorted(mismatches)}")
    if not to_record:
        return

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    ledger += [{"version": results[k]["version"], "evaluated_at": now, "commit": sha,
                "test_months": [test_months[0], test_months[-1]],
                **{m: results[k][m] for m in LEDGER_METRICS}} for k in versions if k in to_record]
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

"""Run the whole ClearLane pipeline, or part of it, in order.

    python -m clearlane.pipeline                  # everything (network pulls + training)
    python -m clearlane.pipeline --from features  # rebuild from cached data onward
    python -m clearlane.pipeline --list
    python -m clearlane.pipeline --refresh        # new data -> served artifact, no training or test eval

Stages read and write `data/` and `reports/`. Network stages are incremental or
reuse cached snapshots (see each module). The test-split stage runs with
`--auto`: model versions already in `reports/test_evaluations.json` are only
re-verified against the recorded numbers; genuinely new versions are evaluated
once and recorded.
"""

from __future__ import annotations

import argparse
import importlib
import time

# (name, module, argv or None for main() without arguments, what it does)
STAGES = [
    ("sr311", "clearlane.ingest.sr311", ["all"], "311 target rows + non-target propensity counts (Socrata)"),
    ("bike-routes", "clearlane.ingest.bike_routes", [], "DOT bike routes snapshot (reuses the latest snapshot)"),
    ("grid", "clearlane.spatial.grid", [], "H3 res-9 network cells and on-network cell-months"),
    ("snap", "clearlane.spatial.snap", [], "snap complaints to active lanes within 40 m"),
    ("panel", "clearlane.panel.build", None, "incidents + zero-filled (cell, hour, month) panel"),
    ("citibike", "clearlane.ingest.citibike", ["--from-year", "2020"], "Citi Bike trips per cell-month (streams ~28 GB once)"),
    ("pluto", "clearlane.ingest.pluto", [], "PLUTO tax lots (reuses the cached release)"),
    ("features", "clearlane.features.build", None, "leak-safe features"),
    ("baseline", "clearlane.models.baseline", None, "empirical-Bayes baseline on validation -> reports/m6_baseline.md"),
    ("lgbm", "clearlane.models.lgbm", None, "LightGBM + PLUTO ablation on validation -> reports/m8_lgbm.md"),
    ("test-eval", "clearlane.models.test_eval", ["--auto"], "test split: verify known versions, record new ones"),
    ("export", "clearlane.serve.export", [], "served artifact for the next month"),
]
NAMES = [s[0] for s in STAGES]
# The weekly refresh: every data stage plus export. The model, its validation reports and the
# test ledger stay as they are; a new model version is a deliberate full run.
REFRESH = [n for n in NAMES if n not in ("baseline", "lgbm", "test-eval")]


def run(start: str | None = None, stop: str | None = None, only: list[str] | None = None) -> None:
    i0 = NAMES.index(start) if start else 0
    i1 = NAMES.index(stop) if stop else len(STAGES) - 1
    for name, module, argv, what in STAGES[i0:i1 + 1]:
        if only is not None and name not in only:
            continue
        print(f"\n=== {name}: {what}", flush=True)
        t = time.time()
        main = importlib.import_module(module).main
        main() if argv is None else main(argv)
        print(f"=== {name} done in {time.time() - t:,.0f} s", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=NAMES)
    ap.add_argument("--to", dest="stop", choices=NAMES)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--refresh", action="store_true", help=f"run only {', '.join(REFRESH)}")
    args = ap.parse_args(argv)
    if args.list:
        for name, _, _, what in STAGES:
            print(f"{name:12s} {what}")
        return
    if args.start and args.stop and NAMES.index(args.start) > NAMES.index(args.stop):
        ap.error("--from comes after --to")
    run(args.start, args.stop, REFRESH if args.refresh else None)


if __name__ == "__main__":
    main()

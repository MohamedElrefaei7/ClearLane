"""LightGBM Poisson model (M8).

    python -m clearlane.models.lgbm

Poisson objective with `init_score = log(exposure) + log(base_rate)`, so the
trees model the log of the rate relative to the training base rate and
exposure is a fixed offset. (With `log(exposure)` alone every slot would start
at 1 incident per hour, ~1,400x the real rate, and the first few hundred trees
would only learn the intercept.) `base_rate` is saved next to the model. Trained on the train
months with early stopping on validation. Two variants are fitted — all
features, and all features except PLUTO (the M7 ablation) — and compared with
the M6 empirical-Bayes baseline on validation.

Decision rules (CLAUDE.md / CONTEXT.md):
  * PLUTO stays only if the full variant has lower validation deviance than
    the no-PLUTO variant.
  * LightGBM ships only if the chosen variant beats the baseline on
    validation deviance; otherwise the baseline ships.
Note that early stopping uses validation, which flatters LightGBM's
validation numbers slightly; the test split (not touched here) settles it.

Writes `reports/m8_lgbm.md`, `data/artifacts/lgbm_<variant>.txt` (trees) and
`lgbm_<variant>.json` (base_rate, best iteration, feature order).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd

from clearlane.config import TRAIN, VALIDATE
from clearlane.features.build import CELL_HOW_PATH, CELL_MONTH_PATH
from clearlane.ingest.sr311 import month_range
from clearlane.models.baseline import EBBaseline
from clearlane.models.eval import SPLITS, calibration_table, evaluate
from clearlane.panel.build import PANEL_PATH, SLOTS
from clearlane.spatial.grid import ROUTES_PATH, SEGMENT_CELLS_PATH, cell_boroughs

REPORT_PATH = Path("reports/m8_lgbm.md")
ARTIFACT_DIR = Path("data/artifacts")
CATEGORICAL = ["boro"]
PARAMS = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 255,
    "num_threads": 8,
    "seed": 0,
    "verbose": -1,
}
NUM_ROUNDS = 3000
EARLY_STOP = 100


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def assemble(panel: pd.DataFrame, cell_month: pd.DataFrame, cell_how: pd.DataFrame,
             drop_prefixes: tuple[str, ...] = ()) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Panel rows (ordered month, cell, hour) -> (X, y, exposure).

    `cell_month` must hold exactly the panel's (cell, month) blocks in the same
    order, and `cell_how` the panel's rows in the same order; both are checked.
    """
    if len(panel) != len(cell_month) * SLOTS or len(cell_how) != len(panel):
        raise ValueError("feature tables do not match the panel")
    blocks = panel.iloc[::SLOTS]
    if not ((blocks["cell"].astype(str).to_numpy() == cell_month["cell"].astype(str).to_numpy()).all()
            and (blocks["month"].astype(str).to_numpy() == cell_month["month"].astype(str).to_numpy()).all()):
        raise ValueError("cell-month features are not aligned with the panel")
    if not ((panel["cell"].astype(str).to_numpy() == cell_how["cell"].astype(str).to_numpy()).all()
            and (panel["hour_of_week"].to_numpy() == cell_how["hour_of_week"].to_numpy()).all()):
        raise ValueError("cell-hour features are not aligned with the panel")

    cm_cols = [c for c in cell_month.columns if c not in ("cell", "month") and not c.startswith(drop_prefixes)]
    X = pd.DataFrame({c: np.repeat(cell_month[c].to_numpy(dtype=np.float32), SLOTS) for c in cm_cols})
    for c in ("inc_cellhow_12m", "inc_cellhow_36m"):
        X[c] = cell_how[c].to_numpy(dtype=np.float32)
    how = panel["hour_of_week"].to_numpy()
    X["hour_of_week"] = how.astype(np.float32)
    X["day_of_week"] = (how // 24).astype(np.float32)
    X["hour"] = (how % 24).astype(np.float32)
    return X, panel["y"].to_numpy(dtype=np.float64), panel["exposure"].to_numpy(dtype=np.float64)


def load_split(name: str, drop_prefixes: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    return load_months(month_range(*SPLITS[name]), drop_prefixes)


def load_months(months: list[str], drop_prefixes: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """(keys, X, y, exposure) for the given panel months."""
    flt = [("month", "in", months)]
    panel = pd.read_parquet(PANEL_PATH, filters=flt)
    panel = panel.assign(cell=panel["cell"].astype(str), month=panel["month"].astype(str))
    cm = pd.read_parquet(CELL_MONTH_PATH)
    cm = cm[cm["month"].isin(months)].reset_index(drop=True)
    ch = pd.read_parquet(CELL_HOW_PATH, filters=flt)
    X, y, e = assemble(panel, cm, ch, drop_prefixes)
    return panel[["cell", "month", "hour_of_week", "y", "exposure"]], X, y, e


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------


def train(X_tr, y_tr, e_tr, X_va, y_va, e_va, params=PARAMS, rounds=NUM_ROUNDS, early_stop=EARLY_STOP):
    """Returns (booster, eval log). `booster.base_rate` holds the offset constant."""
    cats = [c for c in CATEGORICAL if c in X_tr.columns]
    base_rate = float(np.sum(y_tr) / np.sum(e_tr))
    dtr = lgb.Dataset(X_tr, y_tr, init_score=np.log(e_tr * base_rate), categorical_feature=cats, free_raw_data=True)
    dva = lgb.Dataset(X_va, y_va, init_score=np.log(e_va * base_rate), categorical_feature=cats, reference=dtr)
    log: dict = {}
    booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["validate"],
                        callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.record_evaluation(log),
                                   lgb.log_evaluation(100)])
    booster.base_rate = base_rate
    return booster, log


def load_model(variant: str) -> tuple[lgb.Booster, dict]:
    """Saved booster plus its metadata (base_rate, feature order)."""
    booster = lgb.Booster(model_file=str(ARTIFACT_DIR / f"lgbm_{variant}.txt"))
    booster.best_iteration = 0  # file was saved at the best iteration
    return booster, json.loads((ARTIFACT_DIR / f"lgbm_{variant}.json").read_text())


def predict_mu(booster: lgb.Booster, X: pd.DataFrame, exposure: np.ndarray, base_rate: float | None = None) -> np.ndarray:
    """Expected incidents: exp(tree score + log(exposure * base_rate)). The offset is not stored in the model file."""
    base_rate = booster.base_rate if base_rate is None else base_rate
    raw = booster.predict(X, raw_score=True, num_iteration=booster.best_iteration)
    return np.exp(raw + np.log(exposure * base_rate))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)
                                     for v in r.values) + " |")
    return "\n".join(out)


def main() -> None:
    variants = {"full": (), "no_pluto": ("pluto_",)}
    results, mus, boosters = {}, {}, {}
    val_keys = None
    for name, drop in variants.items():
        _, X_tr, y_tr, e_tr = load_split("train", drop)
        val_keys, X_va, y_va, e_va = load_split("validate", drop)
        booster, _ = train(X_tr, y_tr, e_tr, X_va, y_va, e_va)
        del X_tr
        mus[name] = predict_mu(booster, X_va, e_va)
        results[name] = {**evaluate(val_keys, mus[name]), "best_iteration": booster.best_iteration,
                         "base_rate": booster.base_rate,
                         "features": X_va.shape[1]}
        boosters[name] = booster
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(ARTIFACT_DIR / f"lgbm_{name}.txt"), num_iteration=booster.best_iteration)
        (ARTIFACT_DIR / f"lgbm_{name}.json").write_text(json.dumps(
            {"base_rate": booster.base_rate, "best_iteration": booster.best_iteration,
             "features": booster.feature_name()}, indent=1))
        print(name, json.dumps(results[name]), flush=True)

    # Baseline on the same validation rows (refit on train; fast).
    segments = gpd.read_parquet(ROUTES_PATH)
    boroughs = cell_boroughs(segments, pd.read_parquet(SEGMENT_CELLS_PATH))
    tr_panel = pd.read_parquet(PANEL_PATH, columns=["cell", "month", "hour_of_week", "y", "exposure"],
                               filters=[("month", "in", month_range(*TRAIN))])
    base = EBBaseline().fit(tr_panel, boroughs)
    mus["baseline"] = base.predict_rate(val_keys, boroughs) * val_keys["exposure"].to_numpy()
    results["baseline"] = evaluate(val_keys, mus["baseline"])

    keep_pluto = results["full"]["mean_poisson_deviance"] < results["no_pluto"]["mean_poisson_deviance"]
    chosen = "full" if keep_pluto else "no_pluto"
    ships = "lightgbm" if results[chosen]["mean_poisson_deviance"] < results["baseline"]["mean_poisson_deviance"] else "baseline"

    metrics = pd.DataFrame([{"model": k, "mean_poisson_deviance": v["mean_poisson_deviance"],
                             "top_decile_capture": v["top_decile_capture"],
                             "predicted_incidents": v["predicted_incidents"],
                             "best_iteration": v.get("best_iteration", ""), "features": v.get("features", "")}
                            for k, v in [("baseline (M6)", results["baseline"]), ("lightgbm full", results["full"]),
                                         ("lightgbm no_pluto", results["no_pluto"])]])
    imp = pd.DataFrame({"feature": boosters[chosen].feature_name(),
                        "gain": boosters[chosen].feature_importance("gain")})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False).head(20)[["feature", "gain_share"]]
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    d_pluto = results["no_pluto"]["mean_poisson_deviance"] - results["full"]["mean_poisson_deviance"]
    md = "\n".join([
        "# M8 — LightGBM vs baseline on validation",
        "",
        "_Generated by `python -m clearlane.models.lgbm`. Do not hand-edit._ All counts are **reported** "
        "obstruction (311 incidents), not observed obstruction.",
        "",
        f"- Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')} at commit `{sha}`",
        f"- Train {TRAIN[0]} → {TRAIN[1]}; validate {VALIDATE[0]} → {VALIDATE[1]} "
        f"({results['baseline']['rows']:,} rows, {results['baseline']['incidents']:,} incidents). Test split not used.",
        f"- LightGBM: Poisson, init_score = log(exposure), params `{json.dumps({k: v for k, v in PARAMS.items() if k not in ('verbose', 'num_threads')})}`, "
        f"early stopping on validation ({EARLY_STOP} rounds) — validation numbers are slightly optimistic for LightGBM.",
        "",
        "## Validation metrics",
        "",
        _table(metrics),
        "",
        "## Decisions",
        "",
        f"- PLUTO ablation: full − no_pluto deviance = {-d_pluto:+.3g} → "
        + ("**keep PLUTO** (full variant is better)." if keep_pluto else "**drop PLUTO** (it does not improve validation deviance)."),
        f"- Chosen LightGBM variant: `{chosen}`.",
        f"- Ship decision: **{ships}** "
        f"(LightGBM {results[chosen]['mean_poisson_deviance']:.6g} vs baseline {results['baseline']['mean_poisson_deviance']:.6g}).",
        "",
        f"## Calibration by predicted-rate decile — LightGBM `{chosen}` (validation rows)",
        "",
        _table(calibration_table(val_keys, mus[chosen])),
        "",
        "## Calibration by predicted-rate decile — baseline (validation rows)",
        "",
        _table(calibration_table(val_keys, mus["baseline"])),
        "",
        f"## Top 20 features by gain — `{chosen}`",
        "",
        _table(imp),
        "",
    ])
    REPORT_PATH.write_text(md)
    (ARTIFACT_DIR / "m8_decision.json").write_text(json.dumps(
        {"chosen_variant": chosen, "keep_pluto": bool(keep_pluto), "ships": ships,
         "validation": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}}, indent=1, default=float))
    print(metrics.to_string(index=False))
    print(f"keep_pluto={keep_pluto} chosen={chosen} ships={ships}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

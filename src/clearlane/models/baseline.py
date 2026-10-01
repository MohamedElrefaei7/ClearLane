"""Empirical-Bayes baseline (M6).

    python -m clearlane.models.baseline

Rate per (cell, hour_of_week) = Gamma-Poisson posterior mean, shrunk toward a
prior rate with one global strength `b` (pseudo-hours of exposure):

    rate = (y + b * prior) / (E + b)

where y and E are the slot's training incidents and exposure. `b` maximizes
the marginal (negative-binomial) likelihood of the training slots. Priors are
two-level so no prior is ever zero:

    borough x hour_of_week  -> shrunk toward the borough's overall rate
    cell x hour_of_week     -> shrunk toward its borough x hour_of_week rate

The runner fits on train, evaluates on validation against a constant-rate
model, enforces invariant 7, and writes `reports/m6_baseline.md` plus
`data/artifacts/baseline_rates_train.parquet`. The test split is not touched.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from clearlane.config import TRAIN, VALIDATE
from clearlane.models.eval import calibration_table, evaluate, split_mask
from clearlane.panel.build import PANEL_PATH
from clearlane.spatial.grid import BOROUGHS, ROUTES_PATH, SEGMENT_CELLS_PATH, cell_boroughs

REPORT_PATH = Path("reports/m6_baseline.md")
ARTIFACT_PATH = Path("data/artifacts/baseline_rates_train.parquet")
LOG_B_BOUNDS = (0.0, 20.0)  # b between 1 and ~5e8 pseudo-hours


def marginal_loglik(b: float, y: np.ndarray, E: np.ndarray, prior: np.ndarray) -> float:
    """Sum of negative-binomial log-likelihoods of slot totals under Gamma(b*prior, b)."""
    m = E > 0
    y, E, prior = y[m], E[m], prior[m]
    a = b * prior
    return float(np.sum(gammaln(a + y) - gammaln(a) - gammaln(y + 1)
                        + a * np.log(b / (b + E)) + y * np.log(E / (b + E))))


def fit_strength(y: np.ndarray, E: np.ndarray, prior: np.ndarray) -> float:
    res = minimize_scalar(lambda lb: -marginal_loglik(np.exp(lb), y, E, prior),
                          bounds=LOG_B_BOUNDS, method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x))


def posterior_rate(y, E, prior, b):
    return (np.asarray(y) + b * np.asarray(prior)) / (np.asarray(E) + b)


@dataclass
class EBBaseline:
    b_borough: float = np.nan
    b_cell: float = np.nan
    borough_rate: pd.Series = field(default_factory=pd.Series)
    bh_rate: pd.DataFrame = field(default_factory=pd.DataFrame)
    slot_rate: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, train: pd.DataFrame, boroughs: pd.DataFrame) -> "EBBaseline":
        slots = (train.groupby(["cell", "hour_of_week"], observed=True)[["y", "exposure"]].sum().reset_index()
                 .assign(cell=lambda d: d["cell"].astype(str)).merge(boroughs, on="cell", how="left"))
        if slots["boro"].isna().any():
            raise ValueError("cells without a borough in training data")
        bt = slots.groupby("boro")[["y", "exposure"]].sum()
        self.borough_rate = bt["y"] / bt["exposure"]

        bh = slots.groupby(["boro", "hour_of_week"])[["y", "exposure"]].sum().reset_index()
        bh["prior"] = bh["boro"].map(self.borough_rate)
        self.b_borough = fit_strength(bh["y"].to_numpy(), bh["exposure"].to_numpy(), bh["prior"].to_numpy())
        bh["rate"] = posterior_rate(bh["y"], bh["exposure"], bh["prior"], self.b_borough)
        self.bh_rate = bh[["boro", "hour_of_week", "rate"]]

        slots = slots.merge(self.bh_rate.rename(columns={"rate": "prior"}), on=["boro", "hour_of_week"])
        self.b_cell = fit_strength(slots["y"].to_numpy(), slots["exposure"].to_numpy(), slots["prior"].to_numpy())
        slots["rate"] = posterior_rate(slots["y"], slots["exposure"], slots["prior"], self.b_cell)
        self.slot_rate = slots[["cell", "hour_of_week", "boro", "y", "exposure", "prior", "rate"]]
        return self

    def predict_rate(self, rows: pd.DataFrame, boroughs: pd.DataFrame) -> np.ndarray:
        """Rate per panel row; (cell, hour) unseen in training falls back to borough x hour."""
        key = pd.DataFrame({"cell": rows["cell"].astype(str).to_numpy(),
                            "hour_of_week": rows["hour_of_week"].to_numpy()})
        out = key.merge(self.slot_rate[["cell", "hour_of_week", "rate"]], on=["cell", "hour_of_week"], how="left")
        missing = out["rate"].isna()
        if missing.any():
            fb = (key[missing.to_numpy()].merge(boroughs, on="cell", how="left")
                  .merge(self.bh_rate, on=["boro", "hour_of_week"], how="left"))
            out.loc[missing, "rate"] = fb["rate"].to_numpy()
        rate = out["rate"].to_numpy(dtype=float)
        if np.isnan(rate).any():
            raise ValueError("rows with no rate (cell without borough?)")
        return rate


def constant_rate(train: pd.DataFrame) -> float:
    return float(train["y"].sum() / train["exposure"].sum())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _fmt_table(df: pd.DataFrame, floats: str = "{:.4g}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floats.format(v) if isinstance(v, float) else f"{v:,}" if isinstance(v, int)
                                       else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH, columns=["cell", "month", "hour_of_week", "y", "exposure"])
    segments = gpd.read_parquet(ROUTES_PATH)
    boroughs = cell_boroughs(segments, pd.read_parquet(SEGMENT_CELLS_PATH))

    train = panel[split_mask(panel, "train")]
    val = panel[split_mask(panel, "validate")].reset_index(drop=True)
    del panel

    model = EBBaseline().fit(train, boroughs)
    c = constant_rate(train)
    mu_base = model.predict_rate(val, boroughs) * val["exposure"].to_numpy()
    mu_const = np.full(len(val), c) * val["exposure"].to_numpy()
    res_base, res_const = evaluate(val, mu_base), evaluate(val, mu_const)

    if not res_base["mean_poisson_deviance"] < res_const["mean_poisson_deviance"]:
        raise RuntimeError(f"invariant 7 violated: baseline {res_base} vs constant {res_const}")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.slot_rate.to_parquet(ARTIFACT_PATH, index=False)

    cal = calibration_table(val, mu_base)
    val_b = val.assign(cell=val["cell"].astype(str)).merge(boroughs, on="cell", how="left")
    by_boro = (pd.DataFrame({"boro": val_b["boro"], "y": val_b["y"], "mu": mu_base})
               .groupby("boro").agg(observed=("y", "sum"), predicted=("mu", "sum")).reset_index())
    by_boro["borough"] = by_boro["boro"].map(BOROUGHS)
    by_boro["observed_over_predicted"] = by_boro["observed"] / by_boro["predicted"]
    unseen = (~val[["cell"]].astype(str).assign(h=val["hour_of_week"]).set_index(["cell", "h"]).index
              .isin(model.slot_rate.set_index(["cell", "hour_of_week"]).index)).sum()

    metrics = pd.DataFrame([
        {"model": "constant rate", **{k: res_const[k] for k in ("mean_poisson_deviance", "top_decile_capture", "predicted_incidents")}},
        {"model": "EB baseline", **{k: res_base[k] for k in ("mean_poisson_deviance", "top_decile_capture", "predicted_incidents")}},
    ])
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    md = "\n".join([
        "# M6 — baseline on validation",
        "",
        "_Generated by `python -m clearlane.models.baseline`. Do not hand-edit._ All counts are **reported** "
        "obstruction (311 incidents), not observed obstruction.",
        "",
        f"- Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')} at commit `{sha}`",
        f"- Train {TRAIN[0]} → {TRAIN[1]}: {len(train):,} rows, {int(train['y'].sum()):,} incidents",
        f"- Validate {VALIDATE[0]} → {VALIDATE[1]}: {len(val):,} rows, {res_base['incidents']:,} incidents; "
        f"{int(unseen):,} rows in (cell, hour) slots unseen in training (borough × hour fallback)",
        "- Test split: not used.",
        "",
        "## Metrics (validation)",
        "",
        _fmt_table(metrics, "{:.6g}"),
        "",
        f"Invariant 7 (baseline beats constant on validation deviance): **pass** "
        f"({res_base['mean_poisson_deviance']:.6g} < {res_const['mean_poisson_deviance']:.6g}).",
        "Top-decile capture for the constant model is an arbitrary tie-break, shown only for scale.",
        "",
        "## Fitted shrinkage",
        "",
        f"- Constant rate: {c:.6g} incidents per cell-hour",
        f"- Borough rates: " + ", ".join(f"{BOROUGHS[k]} {v:.4g}" for k, v in model.borough_rate.items()),
        f"- b (borough × hour → borough): {model.b_borough:,.0f} pseudo-hours",
        f"- b (cell × hour → borough × hour): {model.b_cell:,.0f} pseudo-hours "
        f"(a slot with E training hours keeps E / (E + b) of its own rate; median training E per slot = "
        f"{model.slot_rate['exposure'].median():,.0f} h)",
        "",
        "## Calibration by predicted-rate decile (validation rows)",
        "",
        _fmt_table(cal),
        "",
        "## Calibration by borough (validation)",
        "",
        _fmt_table(by_boro[["borough", "observed", "predicted", "observed_over_predicted"]]),
        "",
    ])
    REPORT_PATH.write_text(md)
    print(metrics.to_string(index=False))
    print(f"b_borough={model.b_borough:,.0f} b_cell={model.b_cell:,.0f}")
    print(f"wrote {REPORT_PATH} and {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

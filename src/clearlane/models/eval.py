"""Evaluation harness (M6): splits and the three metrics from CLAUDE.md.

A "prediction" is an expected incident count `mu` for each panel row
(rate x exposure). All metrics take the panel rows of one split plus `mu`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clearlane.config import TEST, TRAIN, VALIDATE

SPLITS = {"train": TRAIN, "validate": VALIDATE, "test": TEST}


def split_mask(panel: pd.DataFrame, name: str) -> pd.Series:
    lo, hi = SPLITS[name]
    month = panel["month"].astype(str)
    return (month >= lo) & (month <= hi)


def mean_poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean of 2 * (y log(y/mu) - (y - mu)), with y log y = 0 at y = 0."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
        raise ValueError("predictions must be finite and > 0")
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def slot_table(panel: pd.DataFrame, mu: np.ndarray) -> pd.DataFrame:
    """Aggregate rows to (cell, hour_of_week) slots: y, mu, exposure, predicted rate."""
    df = pd.DataFrame({"cell": panel["cell"].astype(str).to_numpy(),
                       "hour_of_week": panel["hour_of_week"].to_numpy(),
                       "y": panel["y"].to_numpy(), "mu": mu, "exposure": panel["exposure"].to_numpy()})
    slots = df.groupby(["cell", "hour_of_week"], sort=True)[["y", "mu", "exposure"]].sum().reset_index()
    slots["rate"] = slots["mu"] / slots["exposure"]
    return slots


def top_decile_capture(panel: pd.DataFrame, mu: np.ndarray, share: float = 0.10) -> float:
    """Share of incidents in the top `share` of (cell, hour_of_week) slots by predicted rate.

    Ties are broken by (cell, hour_of_week) order, so a model with tied
    predictions (e.g. a constant) gets an arbitrary, roughly `share`-sized result.
    """
    slots = slot_table(panel, mu)
    n_top = int(np.ceil(share * len(slots)))
    top = slots.sort_values("rate", ascending=False, kind="stable").head(n_top)
    total = slots["y"].sum()
    return float(top["y"].sum() / total) if total else float("nan")


def calibration_table(panel: pd.DataFrame, mu: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Rows grouped into equal-size deciles of predicted rate: predicted vs observed incidents."""
    rate = mu / panel["exposure"].to_numpy()
    rank = pd.Series(rate).rank(method="first").to_numpy()
    decile = pd.qcut(rank, bins, labels=False) + 1
    df = pd.DataFrame({"decile": decile, "rate": rate, "mu": mu, "y": panel["y"].to_numpy()})
    out = df.groupby("decile").agg(rows=("y", "size"), mean_pred_rate=("rate", "mean"),
                                   predicted=("mu", "sum"), observed=("y", "sum")).reset_index()
    out["observed_over_predicted"] = out["observed"] / out["predicted"]
    return out


def evaluate(panel: pd.DataFrame, mu: np.ndarray) -> dict:
    return {
        "rows": len(panel),
        "incidents": int(panel["y"].sum()),
        "predicted_incidents": float(np.sum(mu)),
        "mean_poisson_deviance": mean_poisson_deviance(panel["y"].to_numpy(), mu),
        "top_decile_capture": top_decile_capture(panel, mu),
    }

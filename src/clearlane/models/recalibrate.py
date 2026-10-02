"""Monthly level recalibration of LightGBM predictions (M8 follow-up).

The model ranks slots well but its overall level drifts with the city-wide
rate. For month M, predictions are multiplied by

    factor(M) = observed incidents / raw predicted incidents over months M-3 .. M-1

summed over all on-network rows. Only months before M are used (invariant 6).
Where that window reaches into the training months, the raw predictions are
in-sample and the factor is biased toward 1.

    python -m clearlane.models.recalibrate   # evaluates on validation only
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clearlane.config import TRAIN, VALIDATE
from clearlane.ingest.sr311 import month_range
from clearlane.models.eval import calibration_table, evaluate
from clearlane.models.lgbm import load_model, load_months, predict_mu

WINDOW = 3


def monthly_totals(keys: pd.DataFrame, mu: np.ndarray) -> pd.DataFrame:
    return (pd.DataFrame({"month": keys["month"].astype(str).to_numpy(), "observed": keys["y"].to_numpy(),
                          "predicted": mu})
            .groupby("month")[["observed", "predicted"]].sum().sort_index())


def trailing_factor(monthly: pd.DataFrame, window: int = WINDOW) -> pd.Series:
    """factor[M] from months strictly before M; NaN until `window` prior months exist."""
    obs = monthly["observed"].rolling(window).sum().shift(1)
    pred = monthly["predicted"].rolling(window).sum().shift(1)
    return (obs / pred).rename("factor")


def apply_factor(keys: pd.DataFrame, mu: np.ndarray, factor: pd.Series) -> np.ndarray:
    f = keys["month"].astype(str).map(factor).to_numpy(dtype=float)
    if np.isnan(f).any():
        raise ValueError("no recalibration factor for some months (need 3 prior months of predictions)")
    return mu * f


def raw_predictions(months: list[str], variant: str = "full") -> tuple[pd.DataFrame, np.ndarray]:
    booster, meta = load_model(variant)
    drop = () if variant == "full" else ("pluto_",)
    keys, X, _, e = load_months(months, drop)
    return keys, predict_mu(booster, X[meta["features"]], e, meta["base_rate"])


def bias(keys: pd.DataFrame, mu: np.ndarray) -> float:
    return float(keys["y"].sum() / mu.sum())


def main() -> None:
    months = month_range(month_range(TRAIN[0], TRAIN[1])[-WINDOW], VALIDATE[1])
    keys, mu = raw_predictions(months)
    factor = trailing_factor(monthly_totals(keys, mu))
    val = keys["month"].between(*VALIDATE).to_numpy()
    vk, vmu = keys[val].reset_index(drop=True), mu[val]
    vrec = apply_factor(vk, vmu, factor)
    late = (vk["month"] >= "2025-01").to_numpy()
    rows = []
    for label, m in (("validation 2024-10 → 2025-09", np.ones(len(vk), bool)), ("validation 2025-01 → 2025-09", late)):
        k = vk[m].reset_index(drop=True)
        for name, pred in (("raw", vmu[m]), ("recalibrated", vrec[m])):
            r = evaluate(k, pred)
            rows.append({"window": label, "model": name, "obs/pred": round(bias(k, pred), 3),
                         "deviance": round(r["mean_poisson_deviance"], 6), "top_decile": round(r["top_decile_capture"], 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    mt = monthly_totals(vk, vmu).join(factor).assign(recal=monthly_totals(vk, vrec)["predicted"])
    print(mt.assign(raw_ratio=lambda d: d.observed / d.predicted, recal_ratio=lambda d: d.observed / d.recal).round(3).to_string())
    print(calibration_table(vk, vrec).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

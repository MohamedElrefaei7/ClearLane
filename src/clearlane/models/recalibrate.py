"""Level recalibration of LightGBM predictions (M8 follow-up).

The model ranks slots well but its overall level drifts with the city-wide
rate, so predictions are multiplied by a factor = observed / raw predicted
incidents over a trailing window, summed over all on-network rows. Two schemes:

* monthly (the first one shipped, kept for comparison and the ledger): for
  month M, the window is months M-3 .. M-1.
* weekly (shipped since 2026-10-07): refreshed every Monday from the
  `WINDOW_DAYS` days ending `LAG_DAYS` before the refresh (Open Data publishes
  each day about a day late). Chosen on validation: it lags turning points
  less (mean |log monthly obs/pred| 0.109 -> ~0.08) at slightly lower deviance.

Only data before the refresh is used (invariant 6). Where a window reaches into
the training months, the raw predictions are in-sample and the factor is biased
toward 1.

    python -m clearlane.models.recalibrate   # evaluates on validation only
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clearlane.config import TRAIN, VALIDATE
from clearlane.ingest.sr311 import month_range
from clearlane.models.eval import calibration_table, evaluate
from clearlane.models.lgbm import load_model, load_months, predict_mu
from clearlane.panel.build import INCIDENTS_PATH

WINDOW = 3
WINDOW_DAYS = 56
LAG_DAYS = 2
HOURS = 24


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


# ---------------------------------------------------------------------------
# Weekly scheme (day grain)
# ---------------------------------------------------------------------------


def daily_totals(keys: pd.DataFrame, mu: np.ndarray, incidents: pd.DataFrame) -> pd.DataFrame:
    """Observed and raw predicted incidents per calendar day over the months in `keys`.

    Predicted for a day = the month's summed per-hour rate (mu / exposure) over that
    weekday's 24 hour-of-week slots; DST days are treated as 24 hours. Observed = the
    incidents starting that day in the (cell, month) blocks present in `keys`.
    """
    k = pd.DataFrame({"month": keys["month"].astype(str).to_numpy(), "cell": keys["cell"].astype(str).to_numpy(),
                      "how": keys["hour_of_week"].to_numpy(), "mu": mu, "e": keys["exposure"].to_numpy(dtype=float)})
    k = k[k["e"] > 0]
    rate = (k["mu"] / k["e"]).groupby([k["month"], k["how"] // HOURS]).sum()  # (month, weekday) -> per day
    months = sorted(k["month"].unique())
    days = pd.date_range(f"{months[0]}-01", pd.Period(months[-1]).end_time.normalize(), freq="D")
    predicted = rate.reindex(pd.MultiIndex.from_arrays([days.strftime("%Y-%m"), days.dayofweek]), fill_value=0.0).to_numpy()
    blocks = k[["cell", "month"]].drop_duplicates()
    inc = incidents.assign(cell=incidents["cell"].astype(str), month=incidents["month"].astype(str)).merge(blocks)
    observed = inc.groupby(inc["start"].dt.normalize()).size().reindex(days, fill_value=0)
    return pd.DataFrame({"observed": observed.to_numpy(dtype=float), "predicted": predicted}, index=days)


def window_factor(daily: pd.DataFrame, end: pd.Timestamp, window_days: int = WINDOW_DAYS) -> float:
    """observed / predicted over the `window_days` days ending on `end` (inclusive)."""
    span = pd.date_range(end - pd.Timedelta(days=window_days - 1), end, freq="D")
    w = daily.reindex(span)
    if w.isna().any().any():
        raise ValueError(f"recalibration window {span[0].date()} .. {end.date()} is not covered by daily totals")
    return float(w["observed"].sum() / w["predicted"].sum())


def weekly_factor(daily: pd.DataFrame, days: pd.DatetimeIndex, window_days: int = WINDOW_DAYS,
                  lag_days: int = LAG_DAYS) -> pd.Series:
    """Factor in force on each of `days`: refreshed each Monday from the window ending `lag_days` before it."""
    mondays = days - pd.to_timedelta(days.dayofweek, unit="D")
    per_monday = {m: window_factor(daily, m - pd.Timedelta(days=lag_days), window_days) for m in mondays.unique()}
    return pd.Series([per_monday[m] for m in mondays], index=days, name="factor")


def apply_daily_factor(keys: pd.DataFrame, mu: np.ndarray, factor: pd.Series) -> np.ndarray:
    """Scale each (cell, hour, month) row by the mean factor over that hour's occurrences in the month."""
    f = pd.Series(factor.to_numpy(), index=factor.index)
    per = f.groupby([f.index.strftime("%Y-%m"), f.index.dayofweek]).mean()
    idx = pd.MultiIndex.from_arrays([keys["month"].astype(str).to_numpy(), keys["hour_of_week"].to_numpy() // HOURS])
    out = per.reindex(idx).to_numpy(dtype=float)
    if np.isnan(out).any():
        raise ValueError("no recalibration factor for some (month, weekday)")
    return mu * out


def weekly_recalibrated(keys: pd.DataFrame, mu: np.ndarray, target_months: list[str],
                        incidents: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.Series]:
    """Weekly-recalibrated predictions for the rows of `target_months`; earlier months in `keys` feed the windows."""
    incidents = pd.read_parquet(INCIDENTS_PATH) if incidents is None else incidents
    daily = daily_totals(keys, mu, incidents)
    days = daily.index[daily.index.strftime("%Y-%m").isin(target_months)]
    factor = weekly_factor(daily, days)
    m = keys["month"].astype(str).isin(target_months).to_numpy()
    return apply_daily_factor(keys[m], mu[m], factor), factor


# ---------------------------------------------------------------------------


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
    val_months = month_range(*VALIDATE)
    val = keys["month"].isin(val_months).to_numpy()
    vk, vmu = keys[val].reset_index(drop=True), mu[val]
    vrec = apply_factor(vk, vmu, factor)
    vweek, _ = weekly_recalibrated(keys, mu, val_months)
    late = (vk["month"] >= "2025-01").to_numpy()
    rows = []
    for label, m in (("validation 2024-10 → 2025-09", np.ones(len(vk), bool)), ("validation 2025-01 → 2025-09", late)):
        k = vk[m].reset_index(drop=True)
        for name, pred in (("raw", vmu[m]), ("monthly 3m", vrec[m]), (f"weekly {WINDOW_DAYS}d", vweek[m])):
            r = evaluate(k, pred)
            ratio = monthly_totals(k, pred).pipe(lambda d: d["observed"] / d["predicted"])
            rows.append({"window": label, "model": name, "obs/pred": round(bias(k, pred), 3),
                         "deviance": round(r["mean_poisson_deviance"], 6), "top_decile": round(r["top_decile_capture"], 4),
                         "mean |log month obs/pred|": round(float(np.abs(np.log(ratio)).mean()), 3)})
    print(pd.DataFrame(rows).to_string(index=False))
    mt = (monthly_totals(vk, vmu).join(factor).assign(monthly=monthly_totals(vk, vrec)["predicted"],
                                                      weekly=monthly_totals(vk, vweek)["predicted"]))
    print(mt.assign(raw_ratio=lambda d: d.observed / d.predicted, monthly_ratio=lambda d: d.observed / d.monthly,
                    weekly_ratio=lambda d: d.observed / d.weekly).round(3).to_string())
    print(calibration_table(vk, vweek).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

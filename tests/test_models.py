import numpy as np
import pandas as pd
import pytest

from clearlane.models.baseline import EBBaseline, constant_rate, fit_strength, posterior_rate
from clearlane.models.eval import (calibration_table, evaluate, mean_poisson_deviance, split_mask,
                                   top_decile_capture)
from clearlane.spatial.grid import cell_boroughs

# --- metrics ---------------------------------------------------------------------


def test_deviance_zero_when_perfect_and_known_values():
    assert mean_poisson_deviance([1, 2, 3], [1, 2, 3]) == pytest.approx(0)
    assert mean_poisson_deviance([0], [0.5]) == pytest.approx(1.0)              # 2 * mu
    assert mean_poisson_deviance([2], [1]) == pytest.approx(2 * (2 * np.log(2) - 1))


def test_deviance_rejects_nonpositive_predictions():
    with pytest.raises(ValueError):
        mean_poisson_deviance([0, 1], [0.0, 1.0])


def slot_panel(rates_by_slot, y_by_slot, months=1):
    rows = []
    for (cell, how), y in y_by_slot.items():
        rows.append({"cell": cell, "hour_of_week": how, "month": "2025-01", "y": y, "exposure": 4,
                     "rate": rates_by_slot[(cell, how)]})
    df = pd.DataFrame(rows)
    return df, (df["rate"] * df["exposure"]).to_numpy()


def test_top_decile_capture_counts_incidents_in_top_slots():
    rates = {(f"c{i}", 0): float(i) for i in range(20)}       # c19 and c18 are the top 10%
    ys = {(f"c{i}", 0): (5 if i >= 18 else 1 if i == 0 else 0) for i in range(20)}
    df, mu = slot_panel(rates, ys)
    assert top_decile_capture(df, mu) == pytest.approx(10 / 11)


def test_calibration_totals_match():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.poisson(0.2, 1000), "exposure": 4})
    mu = rng.uniform(0.1, 2, 1000)
    cal = calibration_table(df, mu)
    assert len(cal) == 10 and cal["rows"].sum() == 1000
    assert cal["predicted"].sum() == pytest.approx(mu.sum())
    assert cal["observed"].sum() == df["y"].sum()
    assert cal["mean_pred_rate"].is_monotonic_increasing


def test_split_mask_uses_config_months():
    df = pd.DataFrame({"month": ["2020-12", "2021-01", "2024-09", "2024-10", "2025-09", "2025-10", "2026-09"]})
    assert split_mask(df, "train").tolist() == [False, True, True, False, False, False, False]
    assert split_mask(df, "validate").tolist() == [False, False, False, True, True, False, False]
    assert split_mask(df, "test").tolist() == [False, False, False, False, False, True, True]


# --- empirical Bayes ---------------------------------------------------------------


def test_posterior_rate_limits():
    assert posterior_rate(0, 0, 0.01, 100) == pytest.approx(0.01)            # no data -> prior
    assert posterior_rate(5000, 1e9, 0.01, 100) == pytest.approx(5e-6, rel=1e-3)  # lots of data -> MLE
    assert posterior_rate(3, 100, 0.01, 100) == pytest.approx((3 + 1) / 200)


def test_fit_strength_recovers_true_b():
    rng = np.random.default_rng(1)
    b_true, n = 300.0, 20_000
    prior = rng.choice([0.002, 0.005, 0.01], n)
    lam = rng.gamma(shape=b_true * prior, scale=1 / b_true)
    E = np.full(n, 500.0)
    y = rng.poisson(lam * E)
    assert fit_strength(y, E, prior) == pytest.approx(b_true, rel=0.25)


def synthetic_panel(seed=0, cells=60, months=("2024-01", "2024-02", "2024-03", "2024-10", "2024-11")):
    """Cells with very different true rates, stable over time."""
    rng = np.random.default_rng(seed)
    true = {f"c{i}": rng.gamma(0.5, 0.004) for i in range(cells)}
    rows = []
    for m in months:
        for c, r in true.items():
            for h in range(0, 168, 12):
                e = 4
                rows.append({"cell": c, "month": m, "hour_of_week": h, "exposure": e,
                             "y": rng.poisson(r * e * (3 if 7 <= h % 24 <= 9 else 1))})
    df = pd.DataFrame(rows)
    boroughs = pd.DataFrame({"cell": list(true), "boro": ["1" if i % 2 else "3" for i in range(cells)]}).astype("string")
    return df, boroughs


def test_invariant7_baseline_beats_constant_on_heldout_months():
    df, boroughs = synthetic_panel()
    train, val = df[df["month"] < "2024-10"], df[df["month"] >= "2024-10"].reset_index(drop=True)
    model = EBBaseline().fit(train, boroughs)
    mu_b = model.predict_rate(val, boroughs) * val["exposure"].to_numpy()
    mu_c = constant_rate(train) * val["exposure"].to_numpy()
    assert evaluate(val, mu_b)["mean_poisson_deviance"] < evaluate(val, mu_c)["mean_poisson_deviance"]


def test_unseen_slot_falls_back_to_borough_hour_rate():
    df, boroughs = synthetic_panel(cells=10)
    allb = pd.concat([boroughs, pd.DataFrame({"cell": ["new"], "boro": ["3"]}).astype("string")])
    model = EBBaseline().fit(df, allb)
    rate = model.predict_rate(pd.DataFrame({"cell": ["new"], "hour_of_week": [12]}), allb)
    expected = model.bh_rate.set_index(["boro", "hour_of_week"]).loc[("3", 12), "rate"]
    assert rate[0] == pytest.approx(expected)
    assert (model.slot_rate["rate"] > 0).all()


def test_cell_boroughs_majority_and_tie_break():
    segs = pd.DataFrame({"route_id": ["a", "b", "c", "d", "e"], "boro": ["1", "3", "3", "4", "2"],
                         "usable": True, "facilitycl": "II", "onoffst": "ON"})
    seg_cells = pd.DataFrame({"route_id": ["a", "b", "c", "d", "e"], "cell": ["x", "x", "x", "y", "y"]})
    out = cell_boroughs(segs, seg_cells).set_index("cell")["boro"].to_dict()
    assert out == {"x": "3", "y": "2"}

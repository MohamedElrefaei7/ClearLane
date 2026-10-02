import numpy as np
import pandas as pd
import pytest

from clearlane.models.lgbm import assemble, predict_mu, train
from clearlane.panel.build import SLOTS


def tables(cells=("a", "b"), months=("2024-01", "2024-02")):
    cm = pd.DataFrame([(c, m) for m in months for c in cells], columns=["cell", "month"])
    cm["inc_cell_12m"] = np.arange(len(cm), dtype=float)
    cm["pluto_com_lot"] = 7.0
    cm["boro"] = 1
    panel = pd.DataFrame([(c, m, h) for m in months for c in cells for h in range(SLOTS)],
                         columns=["cell", "month", "hour_of_week"])
    panel["y"] = 0
    panel["exposure"] = 4
    ch = panel[["cell", "month", "hour_of_week"]].assign(inc_cellhow_12m=1.0, inc_cellhow_36m=2.0)
    return panel, cm, ch


def test_assemble_repeats_cell_month_features_and_adds_calendar():
    panel, cm, ch = tables()
    X, y, e = assemble(panel, cm, ch)
    assert len(X) == len(panel) == 4 * SLOTS
    assert X["inc_cell_12m"].iloc[SLOTS * 3] == 3.0           # 4th (cell, month) block
    row = X.iloc[SLOTS + 37]                                   # block 2, hour 37 = Tue 13:00
    assert (row["hour_of_week"], row["day_of_week"], row["hour"]) == (37, 1, 13)
    assert {"inc_cellhow_12m", "inc_cellhow_36m", "pluto_com_lot"} <= set(X.columns)
    assert e.tolist() == [4] * len(panel)


def test_assemble_drops_pluto_on_request():
    panel, cm, ch = tables()
    X, _, _ = assemble(panel, cm, ch, drop_prefixes=("pluto_",))
    assert not any(c.startswith("pluto_") for c in X.columns)


def test_assemble_rejects_misaligned_tables():
    panel, cm, ch = tables()
    with pytest.raises(ValueError, match="cell-month"):
        assemble(panel, cm.iloc[::-1].reset_index(drop=True), ch)
    with pytest.raises(ValueError, match="cell-hour"):
        assemble(panel, cm, ch.sample(frac=1, random_state=0).reset_index(drop=True))


def test_exposure_is_an_offset():
    """Rates depend on a feature; exposure varies independently. The model must
    recover rates and scale predictions linearly with exposure."""
    rng = np.random.default_rng(0)
    n = 40_000
    X = pd.DataFrame({"f": rng.integers(0, 4, n).astype(np.float32)})
    rate = np.array([0.01, 0.02, 0.05, 0.1])[X["f"].astype(int)]
    e = rng.choice([1.0, 5.0, 20.0], n)
    y = rng.poisson(rate * e).astype(float)
    params = {"objective": "poisson", "metric": "poisson", "learning_rate": 0.1, "num_leaves": 4,
              "min_data_in_leaf": 50, "verbose": -1, "seed": 0, "num_threads": 2}
    booster, _ = train(X[:30_000], y[:30_000], e[:30_000], X[30_000:], y[30_000:], e[30_000:],
                       params=params, rounds=300, early_stop=30)
    probe = pd.DataFrame({"f": np.arange(4, dtype=np.float32)})
    est = predict_mu(booster, probe, np.ones(4))
    assert est == pytest.approx([0.01, 0.02, 0.05, 0.1], rel=0.15)
    assert predict_mu(booster, probe, np.full(4, 10.0)) == pytest.approx(10 * est)


def test_recalibration_factor_uses_only_prior_months():
    from clearlane.models.recalibrate import trailing_factor
    monthly = pd.DataFrame({"observed": [80.0, 90, 100, 70, 999], "predicted": [100.0, 100, 100, 100, 100]},
                           index=["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"])
    f = trailing_factor(monthly)
    assert f.iloc[:3].isna().all()
    assert f["2025-04"] == pytest.approx(270 / 300)
    assert f["2025-05"] == pytest.approx(260 / 300)
    changed = monthly.copy()
    changed.loc["2025-04", "observed"] = 5000.0  # month M's own data
    assert trailing_factor(changed)["2025-04"] == f["2025-04"]


def test_apply_factor_scales_by_month_and_requires_coverage():
    from clearlane.models.recalibrate import apply_factor
    keys = pd.DataFrame({"month": ["2025-04", "2025-05"]})
    out = apply_factor(keys, np.array([1.0, 1.0]), pd.Series({"2025-04": 0.5, "2025-05": 2.0}))
    assert out.tolist() == [0.5, 2.0]
    with pytest.raises(ValueError):
        apply_factor(keys, np.array([1.0, 1.0]), pd.Series({"2025-04": 0.5}))

import pandas as pd
import pytest

from clearlane.panel.build import SLOTS, build_panel, exposure_table
from clearlane.panel.incidents import assign_incidents, build_incidents, hour_of_week

TS = pd.Timestamp


def kept(rows):
    """rows: (unique_key, cell, created_date)."""
    return pd.DataFrame(rows, columns=["unique_key", "cell", "created_date"]).assign(
        created_date=lambda d: pd.to_datetime(d["created_date"]).astype("datetime64[us]"))


def n_incidents(rows):
    return assign_incidents(kept(rows)).nunique()


# --- invariant 4 -------------------------------------------------------------------


def test_invariant4_five_minutes_apart_is_one_incident():
    assert n_incidents([("a", "c1", "2024-05-06 08:00"), ("b", "c1", "2024-05-06 08:05")]) == 1


def test_invariant4_twenty_minutes_apart_is_two_incidents():
    assert n_incidents([("a", "c1", "2024-05-06 08:00"), ("b", "c1", "2024-05-06 08:20")]) == 2


def test_window_is_inclusive_at_fifteen_minutes():
    assert n_incidents([("a", "c1", "2024-05-06 08:00:00"), ("b", "c1", "2024-05-06 08:15:00")]) == 1
    assert n_incidents([("a", "c1", "2024-05-06 08:00:00"), ("b", "c1", "2024-05-06 08:15:01")]) == 2


def test_window_is_anchored_not_chained():
    rows = [("a", "c1", "2024-05-06 08:00"), ("b", "c1", "2024-05-06 08:10"), ("c", "c1", "2024-05-06 08:20")]
    assert n_incidents(rows) == 2
    assert assign_incidents(kept(rows), chained=True).nunique() == 1


def test_different_cells_never_merge():
    assert n_incidents([("a", "c1", "2024-05-06 08:00"), ("b", "c2", "2024-05-06 08:00")]) == 2


def test_incident_takes_first_complaint_time_and_size():
    inc = build_incidents(kept([("b", "c1", "2024-05-06 08:05"), ("a", "c1", "2024-05-06 08:00"),
                                ("c", "c1", "2024-05-06 09:00")]))
    first = inc.sort_values("start").iloc[0]
    assert first["start"] == TS("2024-05-06 08:00") and first["n_complaints"] == 2
    assert first["hour_of_week"] == 8 and first["month"] == "2024-05"  # 2024-05-06 is a Monday


def test_hour_of_week_endpoints():
    s = pd.Series(pd.to_datetime(["2024-05-06 00:00", "2024-05-12 23:59"]))  # Mon, Sun
    assert hour_of_week(s).tolist() == [0, 167]


# --- invariant 5 -------------------------------------------------------------------


@pytest.mark.parametrize("month, hours", [("2024-03", 743), ("2024-11", 721), ("2024-02", 696), ("2024-05", 744)])
def test_exposure_counts_real_hours(month, hours):
    e = exposure_table([month])
    assert len(e) == SLOTS
    assert e["exposure"].sum() == hours
    assert (e["exposure"] > 0).all()


def test_dst_slots_adjusted_not_dropped():
    mar = exposure_table(["2024-03"]).set_index("hour_of_week")["exposure"]
    nov = exposure_table(["2024-11"]).set_index("hour_of_week")["exposure"]
    sun = 6 * 24
    assert (mar[sun + 2], mar[sun + 3]) == (4, 5)  # 2024-03-10 02:00 never happened
    assert (nov[sun + 1], nov[sun + 2]) == (5, 4)  # 2024-11-03 01:00 happened twice


def test_invariant5_every_cell_month_has_168_rows_including_dst():
    network = pd.DataFrame({"cell": ["c1", "c1", "c1", "c2"], "month": ["2024-02", "2024-03", "2024-11", "2024-03"]})
    k = kept([("a", "c1", "2024-03-10 03:30"), ("b", "c1", "2024-03-10 03:35"), ("c", "c2", "2024-03-11 08:00"),
              ("d", "c1", "2024-11-03 01:30")])
    panel, stats = build_panel(network, build_incidents(k), k)
    rows = panel.groupby(["cell", "month"], observed=True).size()
    assert len(rows) == 4 and (rows == SLOTS).all()
    assert sorted(panel.groupby(["cell", "month"], observed=True)["hour_of_week"].nunique().unique()) == [SLOTS]
    assert panel["y"].sum() == 3 and panel["y_raw"].sum() == 4
    assert (panel["y"] == 0).sum() == len(panel) - 3  # zeros explicit
    hit = panel[(panel["cell"] == "c1") & (panel["month"] == "2024-03") & (panel["y"] > 0)]
    assert hit["hour_of_week"].tolist() == [6 * 24 + 3] and hit["y_raw"].tolist() == [2]
    assert stats["incidents_dropped_off_network"] == 0


def test_off_network_and_out_of_window_incidents_are_counted():
    network = pd.DataFrame({"cell": ["c1"], "month": ["2024-05"]})
    k = kept([("a", "c1", "2024-05-06 08:00"),   # in panel
              ("b", "c9", "2024-05-06 08:00"),   # cell not on network
              ("c", "c1", "2024-06-03 08:00")])  # month outside panel
    panel, stats = build_panel(network, build_incidents(k), k)
    assert panel["y"].sum() == 1
    assert stats["incidents_dropped_off_network"] == 1
    assert stats["incidents_outside_panel_months"] == 1
    assert stats["incidents_in_panel"] + stats["incidents_dropped_off_network"] + stats["incidents_outside_panel_months"] == 3

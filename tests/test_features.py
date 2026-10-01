import h3
import numpy as np
import pandas as pd
import pytest

from clearlane.features.build import (cell_how_history, history_features, lane_features, lane_type,
                                      pluto_features, propensity_cells, ring_sum, trailing)
from clearlane.ingest.bike_routes import normalize
from clearlane.ingest.sr311 import month_range
from clearlane.spatial.grid import segment_cells

MONTHS = month_range("2023-01", "2023-12")
C0 = h3.latlng_to_cell(40.7314, -73.9820, 9)
C1 = sorted(set(h3.grid_disk(C0, 1)) - {C0})[0]   # a neighbour of C0
C2 = h3.latlng_to_cell(40.70, -73.90, 9)      # far away


def test_trailing_excludes_current_month():
    mat = np.array([[1.0, 2.0, 4.0, 8.0]])
    assert trailing(mat, 1).tolist() == [[0, 1, 2, 4]]
    assert trailing(mat, 2).tolist() == [[0, 1, 3, 6]]
    assert trailing(mat, 12).tolist() == [[0, 1, 3, 7]]


def test_ring_sum_with_and_without_self():
    cells = {C0: 0, C1: 1, C2: 2}
    mat = np.array([[1.0], [10.0], [100.0]])
    assert ring_sum(mat, cells, include_self=False)[:, 0].tolist() == [10, 1, 0]
    assert ring_sum(mat, cells, include_self=True)[:, 0].tolist() == [11, 11, 100]


# --- invariant 6 ------------------------------------------------------------------


def inputs():
    rows = pd.DataFrame([(c, m) for m in MONTHS for c in (C0, C2)], columns=["cell", "month"])
    incidents = pd.DataFrame({"cell": [C0, C1, C2, C0], "month": ["2023-02", "2023-03", "2023-04", "2023-05"],
                              "hour_of_week": [8, 9, 10, 8]})
    network = pd.DataFrame([(c, m) for m in MONTHS for c in (C0, C2)], columns=["cell", "month"])
    prop = pd.DataFrame({"cell": [C0, C1, C2], "month": ["2023-02", "2023-02", "2023-03"], "n": [50, 20, 5]})
    cb = pd.DataFrame({"cell": [C0, C2], "month": ["2023-03", "2023-04"], "starts": [100, 7], "ends": [90, 3]})
    boroughs = pd.DataFrame({"cell": [C0, C1, C2], "boro": ["1", "1", "3"]})
    return rows, incidents, network, prop, cb, boroughs


def spike(month):
    """Large values at `month` and later in every time-stamped input."""
    later = [m for m in MONTHS if m >= month]
    inc = pd.DataFrame([(c, m, h) for m in later for c in (C0, C1, C2) for h in range(0, 168, 7)],
                       columns=["cell", "month", "hour_of_week"])
    prop = pd.DataFrame([(c, m, 10_000) for m in later for c in (C0, C1, C2)], columns=["cell", "month", "n"])
    cb = pd.DataFrame([(c, m, 50_000, 50_000) for m in later for c in (C0, C1, C2)],
                      columns=["cell", "month", "starts", "ends"])
    return inc, prop, cb


def features(rows, inc, net, prop, cb, boro):
    return history_features(rows, inc, net, prop, cb, boro, MONTHS).set_index(["cell", "month"]).sort_index()


@pytest.mark.parametrize("m", ["2023-03", "2023-06", "2023-12"])
def test_invariant6_future_spike_leaves_features_unchanged(m):
    rows, inc, net, prop, cb, boro = inputs()
    base = features(rows, inc, net, prop, cb, boro)
    s_inc, s_prop, s_cb = spike(m)
    spiked = features(rows, pd.concat([inc, s_inc]), net, pd.concat([prop, s_prop]), pd.concat([cb, s_cb]), boro)
    upto = base.index.get_level_values("month") <= m
    pd.testing.assert_frame_equal(base[upto], spiked[upto])
    later = base.index.get_level_values("month") > m
    if later.any():  # the spike is visible from the next month on (the test can fail)
        assert not base[later].equals(spiked[later])


def test_previous_month_data_does_reach_features():
    rows, inc, net, prop, cb, boro = inputs()
    f = features(rows, inc, net, prop, cb, boro)
    assert f.loc[(C0, "2023-02"), "inc_cell_1m"] == 0
    assert f.loc[(C0, "2023-03"), "inc_cell_1m"] == 1        # Feb incident
    assert f.loc[(C0, "2023-04"), "inc_ring1_12m"] == 1      # C1's March incident
    assert f.loc[(C0, "2023-03"), "prop311_ring1_12m"] == 70  # own 50 + neighbour 20
    assert f.loc[(C0, "2023-04"), "citibike_cell_1m"] == 190
    assert f.loc[(C0, "2023-06"), "inc_boro_12m"] == 3        # C0 Feb + C1 Mar + C0 May (borough 1)
    assert f.loc[(C2, "2023-06"), "inc_city_3m"] == 3         # Mar, Apr, May


def test_invariant6_cell_how_history():
    _, inc, *_ = inputs()
    keys = pd.DataFrame([(C0, m, h) for m in MONTHS for h in range(168)], columns=["cell", "month", "hour_of_week"])
    base = cell_how_history(keys, inc, MONTHS)
    s_inc, _, _ = spike("2023-06")
    spiked = cell_how_history(keys, pd.concat([inc, s_inc]), MONTHS)
    upto = keys["month"] <= "2023-06"
    pd.testing.assert_frame_equal(base[upto], spiked[upto])
    row = base[(base["month"] == "2023-06") & (base["hour_of_week"] == 8)]
    assert row["inc_cellhow_12m"].item() == 2  # Feb and May incidents at hour 8


SEG = [[-73.9826, 40.7314], [-73.9812, 40.7314]]


def seg(rid, inst, cls="II", ft=None, tf=None, ret=None, status="Current"):
    return {":id": rid, "status": status, "facilitycl": cls, "onoffst": "ON", "instdate": inst, "ret_date": ret,
            "ft_facilit": ft, "tf_facilit": tf,
            "the_geom": {"type": "MultiLineString", "coordinates": [SEG]}}


def test_invariant6_lane_state_as_of_first_instant():
    segs = normalize([seg("a", "2010-01-01T00:00:00.000"),
                      seg("b", "2023-06-15T00:00:00.000", cls="I")])  # opens mid-June
    rows = pd.DataFrame({"cell": [C0] * 3, "month": ["2023-06", "2023-07", "2023-08"]})
    f = lane_features(segs, segment_cells(segs), rows).set_index("month")
    assert f.loc["2023-06", "lane_m_protected"] == 0
    assert f.loc["2023-07", "lane_m_protected"] > 0
    assert f.loc["2023-07", "newest_lane_age_months"] == 1
    assert f.loc["2023-06", "lane_m_conventional"] == pytest.approx(f.loc["2023-07", "lane_m_conventional"])


def test_lane_type_mapping():
    segs = pd.DataFrame({"facilitycl": ["I", "II", "II", "II", "II", "II"],
                         "ft_facilit": ["Protected", "Curbside Buffered", None, "Conventional Buffered", "Shared", None],
                         "tf_facilit": [None, None, "Curbside", None, None, None]})
    assert lane_type(segs).tolist() == ["protected", "curbside", "curbside", "buffered", "conventional", "conventional"]


def test_pluto_ring_includes_off_network_neighbours():
    lat0, lng0 = h3.cell_to_latlng(C0)
    lat1, lng1 = h3.cell_to_latlng(C1)
    lots = pd.DataFrame({"latitude": [lat0, lat1, lat1], "longitude": [lng0, lng1, lng1],
                         "landuse": ["5", "4", "1"], "lotfront": [20.0, 30.0, 99.0],
                         "retailarea": [100.0, 0.0, 0.0], "comarea": [500.0, 300.0, 0.0]})
    f = pluto_features(lots, [C0]).set_index("cell")
    assert list(f.index) == [C0]
    assert f.loc[C0, "pluto_com_lot"] == 1 and f.loc[C0, "pluto_com_lot_ring1"] == 2
    assert f.loc[C0, "pluto_com_frontage_ft_ring1"] == 50  # residential lot's frontage excluded


def test_propensity_points_map_to_cells():
    lat, lng = h3.cell_to_latlng(C0)
    prop = pd.DataFrame({"month": ["2023-01", "2023-01", "2023-01"], "grid_lat": [lat, lat, None],
                         "grid_lon": [lng, lng, None], "n": [3, 4, 9]})
    out = propensity_cells(prop)
    assert out.to_dict("records") == [{"cell": C0, "month": "2023-01", "n": 7}]

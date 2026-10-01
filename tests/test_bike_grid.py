import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from clearlane.ingest.bike_routes import normalize
from clearlane.spatial.activity import active_at, active_in_month, in_scope
from clearlane.spatial.grid import cell_polygon, line_cells, network_cell_months, segment_cells

TS = pd.Timestamp


def raw(rid, coords, inst, ret=None, status="Current", cls="II", onoff="ON"):
    return {":id": rid, "status": status, "facilitycl": cls, "onoffst": onoff, "instdate": inst,
            "ret_date": ret, "the_geom": {"type": "MultiLineString", "coordinates": [coords]}}


# A short east-west segment on 1st Ave-ish coordinates, ~120 m long.
SEG = [[-73.9826, 40.7314], [-73.9812, 40.7314]]


@pytest.fixture
def segs():
    return normalize([
        raw("a", SEG, "2023-06-15T00:00:00.000"),                                  # opens mid-June
        raw("b", SEG, "2010-01-01T00:00:00.000", "2023-03-20T00:00:00.000", "Retired"),
        raw("c", SEG, "2010-01-01T00:00:00.000", None, "Retired"),                 # no ret_date
        raw("d", SEG, "2015-01-01T00:00:00.000", "2014-01-01T00:00:00.000", "Retired"),
        raw("e", SEG, "1900-01-01T00:00:00.000", cls="III"),                       # shared route
        raw("f", SEG, "1900-01-01T00:00:00.000", cls="I", onoff="OFF"),            # park path
    ])


def test_normalize_flags_undeterminable_dates(segs):
    reasons = dict(zip(segs["route_id"], segs["exclude_reason"]))
    assert reasons["c"] == "retired_no_ret_date"
    assert reasons["d"] == "ret_before_inst"
    assert segs.set_index("route_id")["usable"].to_dict() == {
        "a": True, "b": True, "c": False, "d": False, "e": True, "f": True}
    assert segs.crs.to_epsg() == 4326


def test_active_at_bounds(segs):
    a = segs[segs["route_id"] == "a"]
    b = segs[segs["route_id"] == "b"]
    assert not active_at(a, TS("2023-06-14 23:59:59")).item()
    assert active_at(a, TS("2023-06-15 00:00:00")).item()
    assert active_at(b, TS("2023-03-19 23:59:59")).item()
    assert not active_at(b, TS("2023-03-20 00:00:00")).item()  # ret_date exclusive


def test_unusable_segments_never_active(segs):
    bad = segs[~segs["usable"]]
    for when in (TS("2000-01-01"), TS("2016-06-01"), TS("2026-01-01")):
        assert not active_at(bad, when).any()
    for month in ("2016-06", "2026-01"):
        assert not active_in_month(bad, month).any()


def test_invariant2_complaint_before_install_is_excluded(segs):
    """A complaint dated before its lane's install date cannot be attributed to it,
    and the lane's cell is not on network for the complaint's month."""
    lane = segs[segs["route_id"] == "a"]
    before, after = TS("2023-06-14 08:00"), TS("2023-06-20 08:00")
    assert not active_at(lane, before).any()
    assert active_at(lane, after).any()
    cells = segment_cells(lane)
    net = network_cell_months(lane, cells, ["2023-05", "2023-06", "2023-07"], "all")
    assert set(net["month"]) == {"2023-07"}  # mid-month opening: on network from the next full month


def test_month_activity_requires_whole_month(segs):
    b = segs[segs["route_id"] == "b"]  # retired 2023-03-20
    assert active_in_month(b, "2023-02").item()
    assert not active_in_month(b, "2023-03").item()
    a_first = normalize([raw("x", SEG, "2023-06-01T00:00:00.000")])
    assert active_in_month(a_first, "2023-06").item()  # installed at the first instant


def test_scope_filters_class_and_street(segs):
    on = in_scope(segs, "lanes_on_street")
    assert dict(zip(segs["route_id"], on)) == {"a": True, "b": True, "c": True, "d": True, "e": False, "f": False}
    assert in_scope(segs, "all").all()
    with pytest.raises(ValueError):
        in_scope(segs, "bogus")


def test_line_cells_matches_brute_force():
    # ~2 km diagonal across Midtown: crosses many cells and cell corners.
    line = MultiLineString([LineString([(-73.9900, 40.7500), (-73.9700, 40.7620)])])
    got = line_cells(line)
    start = h3.latlng_to_cell(40.7500, -73.9900, 9)
    brute = {c for c in h3.grid_disk(start, 20) if cell_polygon(c).intersects(line)}
    assert got == brute
    assert len(got) >= 6  # res-9 cells are ~350 m across


def test_line_cells_endpoint_cells_included():
    line = LineString(SEG)
    cells = line_cells(line)
    for lng, lat in SEG:
        assert h3.latlng_to_cell(lat, lng, 9) in cells


def test_segment_cells_skips_unusable(segs):
    cells = segment_cells(segs)
    assert set(cells["route_id"]) == {"a", "b", "e", "f"}

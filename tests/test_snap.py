import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from clearlane.ingest.bike_routes import normalize
from clearlane.spatial.grid import segment_cells
from clearlane.spatial.snap import METRIC_CRS, check_cells_on_segments, drop_stats, snap

TS = pd.Timestamp
CELL = h3.latlng_to_cell(40.7314, -73.9820, 9)  # Lower Manhattan


def to_ll(xy):
    p = gpd.GeoSeries([Point(*xy)], crs=METRIC_CRS).to_crs(4326).iloc[0]
    return p.x, p.y


def edge_frame():
    """Metric frame at the midpoint of one edge of CELL: centre c, edge midpoint m, outward unit u."""
    lat, lng = h3.cell_to_latlng(CELL)
    b = h3.cell_to_boundary(CELL)
    m_ll = ((b[0][1] + b[1][1]) / 2, (b[0][0] + b[1][0]) / 2)
    c, m = (gpd.GeoSeries([Point(lng, lat), Point(*m_ll)], crs=4326).to_crs(METRIC_CRS))
    u = np.array([m.x - c.x, m.y - c.y])
    u /= np.linalg.norm(u)
    return np.array([m.x, m.y]), u, np.array([-u[1], u[0]])


def lane(rid, centre, along, inst="2010-01-01T00:00:00.000", ret=None, status="Current", half=10.0):
    a, b = to_ll(centre - half * along), to_ll(centre + half * along)
    return {":id": rid, "status": status, "facilitycl": "II", "onoffst": "ON", "instdate": inst,
            "ret_date": ret, "the_geom": {"type": "MultiLineString", "coordinates": [[list(a), list(b)]]}}


def complaint(key, xy, when="2023-06-20T08:00:00"):
    lng, lat = to_ll(xy)
    return {"unique_key": key, "created_date": TS(when), "latitude": lat, "longitude": lng}


def run(lanes, complaints):
    return snap(pd.DataFrame(complaints), normalize(lanes), scope="lanes_on_street").set_index("unique_key")


# --- invariant 3 ------------------------------------------------------------------


def test_invariant3_adjacent_hex_point_assigned_to_lane_hex():
    m, u, v = edge_frame()
    lanes = [lane("A", m - 5 * u, v)]           # 5 m inside CELL, parallel to the edge
    out = run(lanes, [complaint("q", m + 5 * u)])  # 5 m outside: 10 m from the lane
    q = out.loc["q"]
    assert q["raw_cell"] != CELL                  # raw point is in the neighbouring hex
    assert q["cell"] == CELL                      # assigned to the lane's hex
    assert q["snap_dist_m"] == pytest.approx(10, abs=0.5)
    assert pd.isna(q["drop_reason"])


def test_invariant3_point_100m_from_every_lane_excluded():
    m, u, v = edge_frame()
    out = run([lane("A", m - 5 * u, v)], [complaint("far", m - 5 * u + 100 * u)])
    assert out.loc["far", "drop_reason"] == "no_lane_within_tol"
    assert pd.isna(out.loc["far", "cell"])


def test_tolerance_boundary():
    m, u, v = edge_frame()
    out = run([lane("A", m, v)], [complaint("in", m + 39 * u), complaint("out", m + 41 * u)])
    assert pd.isna(out.loc["in", "drop_reason"])
    assert out.loc["out", "drop_reason"] == "no_lane_within_tol"


# --- invariant 2 (nearest-lane form) ----------------------------------------------


def test_invariant2_nearest_lane_not_yet_installed_excludes():
    m, u, v = edge_frame()
    lanes = [lane("X", m, v, inst="2023-07-01T00:00:00.000")]
    out = run(lanes, [complaint("q", m + 10 * u, "2023-06-20T08:00:00")])
    assert out.loc["q", "drop_reason"] == "lane_not_installed"
    assert pd.isna(out.loc["q", "route_id"])


def test_invariant2_falls_through_to_active_lane_within_tolerance():
    m, u, v = edge_frame()
    lanes = [lane("X", m, v, inst="2023-07-01T00:00:00.000"),  # nearest, not yet built
             lane("Y", m + 30 * u, v)]                          # 20 m away, active
    out = run(lanes, [complaint("q", m + 10 * u, "2023-06-20T08:00:00")])
    assert out.loc["q", "route_id"] == "Y"
    assert out.loc["q", "snap_dist_m"] == pytest.approx(20, abs=0.5)


def test_retired_lane_drops_as_not_active():
    m, u, v = edge_frame()
    lanes = [lane("R", m, v, ret="2023-01-01T00:00:00.000", status="Retired")]
    out = run(lanes, [complaint("q", m + 10 * u, "2023-06-20T08:00:00")])
    assert out.loc["q", "drop_reason"] == "lane_not_active"


# --- other ------------------------------------------------------------------------


def test_nearest_of_two_active_lanes_wins_and_out_of_scope_ignored():
    m, u, v = edge_frame()
    shared = lane("S", m + 2 * u, v) | {"facilitycl": "III"}   # closer, but a shared route
    lanes = [shared, lane("near", m + 5 * u, v), lane("far", m + 25 * u, v)]
    out = run(lanes, [complaint("q", m)])
    assert out.loc["q", "route_id"] == "near"


def test_no_coords_and_stats():
    m, u, v = edge_frame()
    rows = [complaint("ok", m + 3 * u), {"unique_key": "nc", "created_date": TS("2023-06-20"),
                                          "latitude": None, "longitude": None}]
    out = snap(pd.DataFrame(rows), normalize([lane("A", m, v)]))
    st = drop_stats(out)
    assert out.set_index("unique_key").loc["nc", "drop_reason"] == "no_coords"
    assert (st["complaints"], st["kept"], st["drop_no_coords"]) == (2, 1, 1)
    assert st["dropped_fraction"] == 0.5


def test_snapped_cells_lie_on_their_segment():
    m, u, v = edge_frame()
    lanes = [lane("A", m - 5 * u, v, half=60)]
    pts = [complaint(str(i), m + (i - 10) * 3 * u + (i - 10) * 5 * v) for i in range(21)]
    segs = normalize(lanes)
    out = snap(pd.DataFrame(pts), segs)
    assert out["cell"].notna().sum() > 5
    assert check_cells_on_segments(out, segment_cells(segs)) == 0

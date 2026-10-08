import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from clearlane.serve import app as app_module
from clearlane.serve.export import propensity_index, validate_artifact, write

SLOTS = 168
CELLS = ["892a100d2c3ffff", "892a100d2c7ffff"]


def artifact_frames():
    preds = pd.DataFrame({"cell": np.repeat(CELLS, SLOTS), "hour_of_week": np.tile(np.arange(SLOTS), 2)})
    preds["predicted"] = np.where(preds["hour_of_week"] == 8, 0.5, 0.01) * np.where(preds["cell"] == CELLS[0], 2, 1)
    preds["adjusted"] = preds["predicted"] / 2
    preds["historical"] = 0.0
    preds["historical_hours"] = 52.0
    cells = pd.DataFrame({"cell": CELLS, "lat": [40.75, 40.76], "lng": [-73.99, -73.98], "borough": "Manhattan",
                          "lane_m_protected": [60.0, 0.0], "lane_m_curbside": 0.0, "lane_m_buffered": 0.0,
                          "lane_m_conventional": [40.0, 100.0], "lane_m_total": 100.0, "propensity_index": 2.0})
    meta = {"month": "2026-10", "units": "per week", "caveat": "REPORTED obstruction", "cells": 2}
    return preds, cells, meta


# --- invariant 8 ----------------------------------------------------------------


def test_invariant8_valid_artifact_passes():
    preds, cells, _ = artifact_frames()
    validate_artifact(preds, cells["cell"])


@pytest.mark.parametrize("break_it, msg", [
    (lambda p: p.drop(index=5), "168"),                                          # a slot missing
    (lambda p: p[p["cell"] == CELLS[0]], "missing"),                             # a whole cell missing
    (lambda p: p.assign(predicted=p["predicted"].where(p.index != 3)), "NaN"),   # NaN
    (lambda p: p.assign(adjusted=p["adjusted"].where(p.index != 3, np.inf)), "infinite"),
    (lambda p: p.assign(historical=p["historical"].where(p.index != 3, -0.1)), "negative"),
    (lambda p: pd.concat([p, p.iloc[[0]]]), "168"),                              # duplicated slot
])
def test_invariant8_violations_are_caught(break_it, msg):
    preds, cells, _ = artifact_frames()
    with pytest.raises(ValueError, match=msg):
        validate_artifact(break_it(preds), cells["cell"])


def test_write_refuses_invalid_artifact(tmp_path):
    preds, cells, meta = artifact_frames()
    with pytest.raises(ValueError):
        write(preds.drop(index=0), cells, meta, tmp_path)
    assert not (tmp_path / "predictions.parquet").exists()


def test_propensity_index_normalized_and_floored():
    idx = propensity_index(pd.Series([0.0] + [10.0] * 18 + [100.0]))
    assert idx.median() == pytest.approx(1.0)
    assert idx.min() > 0
    with pytest.raises(ValueError):
        propensity_index(pd.Series([0.0] * 10 + [1.0]))


# --- API ----------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    preds, cells, meta = artifact_frames()
    write(preds, cells, meta, tmp_path, land=["892a100d2cbffff"])
    monkeypatch.setenv("CLEARLANE_SERVING_DIR", str(tmp_path))
    app_module.artifact.cache_clear()
    yield TestClient(app_module.app)
    app_module.artifact.cache_clear()


def test_slot_by_day_hour_and_by_how_agree(client):
    a = client.get("/api/slot", params={"day": "Mon", "hour": 8}).json()
    b = client.get("/api/slot", params={"how": 8}).json()
    assert a == b
    assert a["hour_of_week"] == 8 and a["layer"] == "predicted" and "REPORTED" in a["caveat"]
    assert {c["cell"]: c["value"] for c in a["cells"]} == {CELLS[0]: 1.0, CELLS[1]: 0.5}


def test_adjusted_layer_flagged_heuristic(client):
    r = client.get("/api/slot", params={"how": 8, "layer": "adjusted"}).json()
    assert r["heuristic"] is True
    assert client.get("/api/slot", params={"how": 8}).json()["heuristic"] is False


@pytest.mark.parametrize("params", [{"how": 168}, {"day": "Mon"}, {"day": "Xyz", "hour": 1},
                                    {"how": 1, "day": "Mon", "hour": 1}, {"how": 1, "layer": "bogus"}])
def test_bad_slot_requests_rejected(client, params):
    assert client.get("/api/slot", params=params).status_code in (400, 422)


def test_cell_detail(client):
    r = client.get(f"/api/cell/{CELLS[0]}").json()
    assert len(r["series"]["predicted"]) == SLOTS
    assert r["lane_share"]["protected"] == 0.6
    assert r["weekly"]["predicted"] == pytest.approx(2 * (0.5 + 0.01 * 167))
    assert client.get("/api/cell/89ffffffffffff").status_code == 404


def test_grid_lists_network_and_grey_cells(client):
    g = client.get("/api/grid").json()
    assert g["network"] == CELLS and g["off_network"] == ["892a100d2cbffff"]


def test_scale_edges_from_positive_values():
    from clearlane.serve.export import SCALE_QUANTILES, scale_edges
    edges = scale_edges(np.array([0.0] * 50 + list(range(1, 101))))
    assert len(edges) == len(SCALE_QUANTILES) and edges == sorted(edges) and edges[0] > 0


def test_map_page_served_from_same_origin(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>ClearLane</title>" in r.text
    assert "reported" in r.text.lower()
    assert client.get("/map.js").status_code == 200


def test_app_reloads_when_the_artifact_is_rewritten(client, tmp_path):
    import os
    assert client.get("/api/slot", params={"how": 8}).json()["cells"][0]["value"] == 1.0
    preds, cells, meta = artifact_frames()
    preds["predicted"] = preds["predicted"] * 2
    write(preds, cells, {**meta, "month": "next"}, tmp_path, land=["892a100d2cbffff"])
    st = (tmp_path / "meta.json").stat()
    os.utime(tmp_path / "meta.json", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))  # coarse-clock filesystems
    assert client.get("/api/meta").json()["month"] == "next"
    assert client.get("/api/slot", params={"how": 8}).json()["cells"][0]["value"] == 2.0

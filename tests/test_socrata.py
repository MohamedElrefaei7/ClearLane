import pytest

from clearlane.audit import socrata
from clearlane.audit.socrata import (
    TruncatedResponseError,
    build_params,
    fetch,
    in_nyc_bbox,
    to_hour_of_week,
)

SR_311 = "erm2-nwe9"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    """Patch requests.get; returns a dict holding the call count and payload."""
    state = {"calls": 0, "payload": [{"complaint_type": "X", "n": "1"}]}

    def _get(url, params=None, headers=None, timeout=None):
        state["calls"] += 1
        return FakeResponse(state["payload"])

    monkeypatch.setattr(socrata.requests, "get", _get)
    return state


def test_build_params_always_sets_limit():
    params = build_params(select="count(*)")
    assert "$limit" in params
    assert int(params["$limit"]) == socrata.DEFAULT_LIMIT


def test_build_params_explicit_limit():
    assert build_params(limit=5)["$limit"] == "5"


def test_grouped_response_at_limit_raises(fake_get, tmp_path):
    params = build_params(select="complaint_type, count(*) as n", group="complaint_type", limit=3)
    fake_get["payload"] = [{"complaint_type": str(i), "n": "1"} for i in range(3)]
    with pytest.raises(TruncatedResponseError):
        fetch(SR_311, params, cache_dir=tmp_path / "a")


def test_grouped_response_below_limit_ok(fake_get, tmp_path):
    params = build_params(select="complaint_type, count(*) as n", group="complaint_type", limit=3)
    fake_get["payload"] = [{"complaint_type": str(i), "n": "1"} for i in range(2)]
    assert len(fetch(SR_311, params, cache_dir=tmp_path)) == 2


def test_cache_round_trip(fake_get, tmp_path):
    params = build_params(select="count(*)", where="1=1")
    first = fetch(SR_311, params, cache_dir=tmp_path)
    second = fetch(SR_311, params, cache_dir=tmp_path)
    assert first == second
    assert fake_get["calls"] == 1
    fetch(SR_311, params, cache_dir=tmp_path, refresh=True)
    assert fake_get["calls"] == 2


def test_cache_key_ignores_param_order(fake_get, tmp_path):
    a = {"$limit": "10", "$select": "count(*)", "$where": "1=1"}
    b = {"$where": "1=1", "$select": "count(*)", "$limit": "10"}
    assert list(a) != list(b)
    fetch(SR_311, a, cache_dir=tmp_path)
    fetch(SR_311, b, cache_dir=tmp_path)
    assert fake_get["calls"] == 1


@pytest.mark.parametrize("sunday_is_zero, monday_dow, sunday_dow", [(True, 1, 0), (False, 0, 6)])
def test_to_hour_of_week_endpoints(sunday_is_zero, monday_dow, sunday_dow):
    assert to_hour_of_week(monday_dow, 0, sunday_is_zero) == 0
    assert to_hour_of_week(sunday_dow, 23, sunday_is_zero) == 167


@pytest.mark.parametrize("sunday_is_zero", [True, False])
def test_to_hour_of_week_bijective(sunday_is_zero):
    out = {to_hour_of_week(d, h, sunday_is_zero) for d in range(7) for h in range(24)}
    assert out == set(range(168))


def test_in_nyc_bbox():
    assert in_nyc_bbox(40.7580, -73.9855)  # Times Square
    # Newark, NJ is inside the bbox: a known limitation of a rectangular check.
    # Acceptable for an audit-level coordinate sanity check.
    assert in_nyc_bbox(40.7357, -74.1724)
    assert not in_nyc_bbox(42.3601, -71.0589)  # Boston
    assert not in_nyc_bbox(None, None)


@pytest.mark.live
def test_live_311_sample(tmp_path):
    rows = fetch(SR_311, build_params(limit=5), cache_dir=tmp_path)
    assert len(rows) == 5
    cols = set().union(*(r.keys() for r in rows))
    for col in ["unique_key", "created_date", "complaint_type", "descriptor", "latitude", "longitude"]:
        assert col in cols, col

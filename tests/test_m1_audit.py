import pandas as pd
import pytest

from clearlane.audit.m1_audit import SECTIONS, mom_changes, render_report, sparsity, month_chunks, filter_pairs

OLD, NEW = "76ig-c548", "erm2-nwe9"
TW = {"complaint_type": "Illegal Parking", "descriptor": "Blocked Bike Lane"}


def test_sparsity_unit_case():
    annual = 168 * 12 * 1000
    assert annual == 2_016_000
    assert sparsity(annual, 1000) == pytest.approx(1.0)
    assert sparsity(annual, 2000) == pytest.approx(0.5)


def test_sparsity_three_hour_bins_triple():
    assert sparsity(50_000, 4000, 56) == pytest.approx(3 * sparsity(50_000, 4000, 168))


def test_month_chunks():
    chunks = month_chunks("2019-11-15", "2020-02-01")
    assert chunks == [("2019-11-15", "2019-12-01"), ("2019-12-01", "2020-01-01"), ("2020-01-01", "2020-02-01")]
    assert len(month_chunks("2016-01-01", "2026-10-01")) == 129


def test_filter_pairs_matches_like_semantics():
    rows = {"x": [{"complaint_type": "Illegal Parking", "descriptor": "Blocked Bike Lane", "n": "1"},
                  {"complaint_type": "Bike Rack", "n": "1"},
                  {"complaint_type": "Illegal Parking", "descriptor": "Double Parked Blocking Traffic", "n": "1"},
                  {"complaint_type": "Noise", "descriptor": "Loud Music", "n": "1"}]}
    assert [r.get("descriptor") for r in filter_pairs(rows, "BIKE")["x"]] == ["Blocked Bike Lane", None]
    assert [r["descriptor"] for r in filter_pairs(rows, "DOUBLE PARK")["x"]] == ["Double Parked Blocking Traffic"]


def test_mom_largest_drop():
    s = pd.Series([100, 100, 10, 100], index=["2020-01", "2020-02", "2020-03", "2020-04"])
    drops, rises = mom_changes(s)
    assert len(drops) == 1
    assert drops.loc[0, "month"] == "2020-03"
    assert drops.loc[0, "change"] == -90
    assert drops.loc[0, "pct"] == pytest.approx(-90.0)
    assert rises.loc[0, "month"] == "2020-04"
    assert rises.loc[0, "change"] == 90


def _catalog(rows):
    return {"results": [{"resource": r} for r in rows]}


@pytest.fixture
def responses():
    """Small fixture of named cached responses, shaped like collect() output."""
    months = [f"{y}-{m:02d}-01T00:00:00.000" for y in (2024, 2025) for m in range(1, 13)]
    return {
        "as_of": "2026-09-30",
        "taxonomy_bike": {
            OLD: [{**TW, "n": "100", "first": "2016-11-02T08:00:00.000", "last": "2019-12-31T10:00:00.000"}],
            NEW: [{**TW, "n": "900", "first": "2020-01-01T09:00:00.000", "last": "2026-09-28T10:00:00.000"},
                  {"complaint_type": "Snow or Ice", "descriptor": "Bike Lane", "n": "5"}],
        },
        "taxonomy_double_park": {
            NEW: [{"complaint_type": "Illegal Parking", "descriptor": "Double Parked Blocking Traffic", "n": "50"}],
        },
        "sample_311": [{"unique_key": "1"}],
        "yearly": {OLD: [{"year": "2019", "n": "100"}],
                   NEW: [{"year": "2024", "n": "400"}, {"year": "2025", "n": "500"}]},
        "monthly": {NEW: [{"month": m, "n": str(30 + i)} for i, m in enumerate(months)]
                    + [{"month": "2026-09-01T00:00:00.000", "n": "5"}]},
        "coord_total": {NEW: [{"year": "2025", "n": "500"}]},
        "coord_null": {NEW: [{"year": "2025", "n": "5"}]},
        "coord_outside": {NEW: []},
        "dow_check": {"scope": "target pair", "rows": {NEW: [{"dow": "1", "n": "7"}]}},
        "how": {NEW: [{"dow": "1", "hh": "8", "n": "20"}, {"dow": "0", "hh": "23", "n": "3"}]},
        "resolution": {NEW: [{"resolution_description": "Closed", "n": "800"}, {"n": "100"}]},
        "bike_catalog": _catalog([{"id": "mzxg-pwib", "name": "New York City Bike Routes", "type": "dataset",
                                   "data_updated_at": "2026-07-24T00:00:00Z",
                                   "description": "current and historic bicycle routes as line segments"}]),
        "bike_id": "mzxg-pwib",
        "bike_meta": _catalog([{"id": "mzxg-pwib", "name": "x", "description": "d",
                                "columns_field_name": ["status", "instdate", "ret_date", "ft_facilit"],
                                "columns_datatype": ["text", "calendar_date", "calendar_date", "text"]}]),
        "bike_sample": [{"status": "Current", "instdate": "2010-01-01", "ft_facilit": "Protected"}],
        "bike_status": [{"status": "Current", "n": "10", "n_with_ret_date": "0", "min_instdate": "2000-01-01",
                         "max_instdate": "2026-01-01"}],
        "pv_catalog": _catalog([{"id": "pvqr-7yc4", "name": "Parking Violations Issued - Fiscal Year 2027",
                                 "type": "dataset", "data_updated_at": "2026-09-17T00:00:00Z"}]),
        "pv_id": "pvqr-7yc4",
        "pv_meta": _catalog([{"id": "pvqr-7yc4", "name": "x",
                              "columns_field_name": ["summons_number", "street_name"],
                              "columns_datatype": ["text", "text"]}]),
        "pv_sample": [{"summons_number": "1", "street_name": "BROADWAY"}],
    }


META = {"generated": "t", "git_commit": "abc", "cache_hits": 3, "cache_misses": 0, "cache_dir": "x"}


def test_render_contains_all_sections(responses, tmp_path):
    md = render_report(responses, META, tmp_path)
    for title in SECTIONS:
        assert f"## {title}" in md
    assert "MISSING" not in md
    assert (tmp_path / "m1_monthly.png").exists()
    assert "Verified convention" in md


def test_render_missing_fixture_marks_section(responses, tmp_path):
    del responses["bike_sample"]
    md = render_report(responses, META, tmp_path)
    for title in SECTIONS:
        assert f"## {title}" in md
    bike = md.split("## 10. DOT bike-route dataset")[1].split("## 11.")[0]
    assert "MISSING: response 'bike_sample' not available" in bike
    # other sections unaffected
    assert "MISSING" not in md.split("## 10.")[0]

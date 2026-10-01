import pandas as pd
import pytest

from clearlane.ingest import sr311
from clearlane.ingest.sr311 import (
    PROPENSITY_STORE,
    TARGET_STORE,
    default_start,
    ingest_propensity,
    ingest_target,
    month_range,
    read_manifest,
    read_store,
)

NOW = pd.Timestamp("2023-04-15T12:00:00", tz="UTC")


def row(key, created, status="Open", lat="40.75", lon="-73.99"):
    return {"unique_key": str(key), "created_date": created, "complaint_type": "Illegal Parking",
            "descriptor": "Blocked Bike Lane", "status": status, "latitude": lat, "longitude": lon}


class FakeSource:
    """In-memory 311: rows keyed by created month; records each call."""

    def __init__(self, target=None, propensity=None):
        self.target = target or {}
        self.propensity = propensity or {}
        self.calls = []

    def target_rows(self, dataset, lo, hi):
        self.calls.append(("target", dataset, lo, hi))
        return [dict(r) for r in self.target.get(lo[:7], [])]

    def propensity_rows(self, dataset, lo, hi):
        self.calls.append(("propensity", dataset, lo, hi))
        return [dict(r) for r in self.propensity.get(lo[:7], [])]


@pytest.fixture
def src():
    return FakeSource(target={
        "2023-01": [row(1, "2023-01-03T08:15:00.000"), row(2, "2023-01-31T23:59:59.000")],
        "2023-02": [row(3, "2023-02-01T00:00:00.000")],
        "2023-03": [row(4, "2023-03-12T02:30:00.000"), row(5, "2023-03-30T17:00:00.000")],
    })


def store_rows(d):
    return read_store(d / TARGET_STORE)


# --- invariant 1 -----------------------------------------------------------


def test_overlapping_reingest_does_not_change_row_counts(src, tmp_path):
    ingest_target(["2023-01", "2023-02"], tmp_path, src, NOW)
    assert len(store_rows(tmp_path)) == 3
    ingest_target(["2023-02", "2023-03"], tmp_path, src, NOW)  # Feb overlaps
    first = store_rows(tmp_path)
    assert len(first) == 5
    assert first["unique_key"].is_unique
    ingest_target(month_range("2023-01", "2023-03"), tmp_path, src, NOW)  # full overlap
    again = store_rows(tmp_path)
    assert len(again) == 5
    pd.testing.assert_frame_equal(first, again)


def test_update_replaces_row_not_duplicates(src, tmp_path):
    ingest_target(["2023-02"], tmp_path, src, NOW)
    src.target["2023-02"][0]["status"] = "Closed"
    stats = ingest_target(["2023-02"], tmp_path, src, NOW)
    df = store_rows(tmp_path)
    assert len(df) == 1 and df.loc[0, "status"] == "Closed"
    assert (stats[0]["new"], stats[0]["updated"], stats[0]["rows"]) == (0, 1, 1)


def test_value_becoming_null_counts_as_update(src, tmp_path):
    ingest_target(["2023-02"], tmp_path, src, NOW)
    src.target["2023-02"][0]["latitude"] = None
    stats = ingest_target(["2023-02"], tmp_path, src, NOW)
    assert stats[0]["updated"] == 1
    assert store_rows(tmp_path)["latitude"].isna().all()


def test_row_missing_from_source_is_reported_not_deleted(src, tmp_path):
    ingest_target(["2023-03"], tmp_path, src, NOW)
    src.target["2023-03"].pop()
    stats = ingest_target(["2023-03"], tmp_path, src, NOW)
    assert stats[0]["missing_from_source"] == 1
    assert len(store_rows(tmp_path)) == 2


def test_rows_stored_by_created_month_and_typed(src, tmp_path):
    ingest_target(month_range("2023-01", "2023-03"), tmp_path, src, NOW)
    files = sorted(p.name for p in (tmp_path / TARGET_STORE).glob("*.parquet"))
    assert files == ["2023-01.parquet", "2023-02.parquet", "2023-03.parquet"]
    jan = pd.read_parquet(tmp_path / TARGET_STORE / "2023-01.parquet")
    assert list(jan["unique_key"]) == ["1", "2"]
    assert jan["created_date"].dtype == "datetime64[us]"  # naive NYC wall clock, not shifted
    assert jan.loc[1, "created_date"] == pd.Timestamp("2023-01-31 23:59:59")
    assert jan["latitude"].dtype == "float64"


def test_row_outside_requested_month_raises(src, tmp_path):
    src.target["2023-02"].append(row(9, "2023-03-01T00:00:00.000"))
    with pytest.raises(ValueError, match="outside"):
        ingest_target(["2023-02"], tmp_path, src, NOW)


def test_empty_month_written_and_in_manifest(src, tmp_path):
    ingest_target(["2022-12"], tmp_path, src, NOW)
    assert (tmp_path / TARGET_STORE / "2022-12.parquet").exists()
    assert read_manifest(tmp_path / TARGET_STORE)["2022-12"]["rows"] == 0


def test_routing_across_dataset_split(src, tmp_path):
    ingest_target(["2019-12", "2020-01"], tmp_path, src, NOW, workers=1)
    assert [c[1] for c in src.calls] == ["76ig-c548", "erm2-nwe9"]
    assert src.calls[0][2:] == ("2019-12-01", "2020-01-01")


def test_manifest_marks_current_month_incomplete(src, tmp_path):
    ingest_target(["2023-03", "2023-04"], tmp_path, src, NOW)
    m = read_manifest(tmp_path / TARGET_STORE)
    assert m["2023-03"]["complete"] is True
    assert m["2023-04"]["complete"] is False


def test_completeness_uses_nyc_date():
    # 2023-05-01 02:00 UTC is still April 30 in New York.
    assert not sr311.is_complete("2023-04", pd.Timestamp("2023-05-01T02:00:00", tz="UTC"))
    assert sr311.is_complete("2023-04", pd.Timestamp("2023-05-01T05:00:00", tz="UTC"))


# --- propensity --------------------------------------------------------------


def test_propensity_month_replaced_idempotently(tmp_path):
    src = FakeSource(propensity={"2023-03": [
        {"g": {"type": "Point", "coordinates": [-73.91, 40.846]}, "n": "7"},
        {"n": "3"},  # no location
    ]})
    ingest_propensity(["2023-03"], tmp_path, src, NOW)
    first = read_store(tmp_path / PROPENSITY_STORE)
    ingest_propensity(["2023-03"], tmp_path, src, NOW)
    again = read_store(tmp_path / PROPENSITY_STORE)
    pd.testing.assert_frame_equal(first, again)
    assert len(again) == 2 and again["n"].sum() == 10
    located = again.dropna(subset=["grid_lat"]).iloc[0]
    assert (located["grid_lat"], located["grid_lon"]) == (40.846, -73.91)
    src.propensity["2023-03"] = [{"n": "1"}]
    ingest_propensity(["2023-03"], tmp_path, src, NOW)
    assert read_store(tmp_path / PROPENSITY_STORE)["n"].tolist() == [1]


def test_propensity_excludes_target_with_null_safe_complement():
    w = sr311.not_target_where(sr311.TARGET)
    assert w.startswith("complaint_type IS NULL OR descriptor IS NULL OR NOT (")
    assert sr311.target_where(sr311.TARGET) in w


# --- months --------------------------------------------------------------------


def test_month_range_inclusive_across_year():
    assert month_range("2022-11", "2023-02") == ["2022-11", "2022-12", "2023-01", "2023-02"]
    assert month_range("2023-02", "2023-01") == []


def test_default_start_overlaps_latest_months():
    assert default_start([]) == sr311.DEFAULT_START
    assert default_start(["2026-07", "2026-08"]) == "2026-07"
    assert default_start(["2026-01"], overlap=3) == "2025-11"


# --- live ----------------------------------------------------------------------


@pytest.mark.live
def test_live_one_day_each_store():
    s = sr311.SocrataSource()
    rows = s.target_rows("erm2-nwe9", "2025-06-02", "2025-06-03")
    assert rows and set(rows[0]) <= set(sr311.TARGET_COLUMNS)
    assert {"unique_key", "created_date"} <= set(rows[0])
    grid = s.propensity_rows("erm2-nwe9", "2025-06-02", "2025-06-03")
    assert sum(int(r["n"]) for r in grid) > 1000
    assert any("g" in r for r in grid)

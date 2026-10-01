"""Thin Socrata (SODA) client plus pure helpers.

Network code is confined to `fetch` / `fetch_url` (cached; used by the M1
audit) and `fetch_pages` (uncached; used by ingestion, whose output store is
Parquet). Everything else is pure so it can be tested offline. Cached
responses are raw JSON under `cache_dir`, keyed by a stable hash of
(url, sorted params), so the audit report can be regenerated offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://data.cityofnewyork.us/resource/{dataset_id}.json"
DISCOVERY_URL = "https://api.us.socrata.com/api/catalog/v1"
DEFAULT_LIMIT = 50_000
DEFAULT_CACHE_DIR = Path("data/raw/m1")
TOKEN_ENV = "SOCRATA_APP_TOKEN"
TIMEOUT_S = 180
RETRIES = 5
RETRY_BACKOFF_S = 30

# NYC bounding box used for coordinate-quality checks (audit only).
NYC_LAT_MIN, NYC_LAT_MAX = 40.47, 40.93
NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68

# Cache hit/miss counters for the current process; the audit reports these.
STATS: dict[str, int] = {"hits": 0, "misses": 0}
_STATS_LOCK = threading.Lock()


class TruncatedResponseError(RuntimeError):
    """A grouped query returned exactly `$limit` rows, so results were cut off."""


def reset_stats() -> None:
    STATS["hits"] = 0
    STATS["misses"] = 0


def build_params(
    *,
    select: str | None = None,
    where: str | None = None,
    group: str | None = None,
    order: str | None = None,
    limit: int = DEFAULT_LIMIT,
    **extra: Any,
) -> dict[str, str]:
    """Build SoQL query params. `$limit` is always set (Socrata defaults to 1000)."""
    params: dict[str, str] = {"$limit": str(int(limit))}
    for key, val in (("$select", select), ("$where", where), ("$group", group), ("$order", order)):
        if val is not None:
            params[key] = val
    for key, val in extra.items():
        params[key] = str(val)
    return params


def cache_key(url: str, params: dict[str, Any]) -> str:
    """Stable hash of url + params; independent of dict insertion order."""
    payload = json.dumps({"url": url, "params": sorted((str(k), str(v)) for k, v in params.items())})
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def check_truncation(rows: Any, params: dict[str, Any]) -> None:
    """Raise if a grouped response came back with exactly `$limit` rows."""
    if "$group" not in params or "$limit" not in params:
        return
    if isinstance(rows, list) and len(rows) == int(params["$limit"]):
        raise TruncatedResponseError(
            f"grouped query returned exactly $limit={params['$limit']} rows; results truncated"
        )


def _headers() -> dict[str, str]:
    token = os.environ.get(TOKEN_ENV)
    return {"X-App-Token": token} if token else {}


def _get_with_retry(url: str, params: dict[str, Any]) -> Any:
    """GET with retries on connection errors, timeouts and 5xx (Socrata is flaky on heavy aggregates)."""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT_S)
            if resp.status_code < 500 or attempt == RETRIES:
                resp.raise_for_status()
                return resp.json()
            log.warning("HTTP %s on attempt %d/%d", resp.status_code, attempt, RETRIES)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == RETRIES:
                raise
            log.warning("%s on attempt %d/%d", type(e).__name__, attempt, RETRIES)
        time.sleep(RETRY_BACKOFF_S * attempt)
    raise AssertionError("unreachable")


def fetch_url(
    url: str,
    params: dict[str, Any],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> Any:
    """GET `url` with `params`, caching the raw JSON body on disk."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{cache_key(url, params)}.json"
    if path.exists() and not refresh:
        with _STATS_LOCK:
            STATS["hits"] += 1
        log.debug("cache hit %s", path.name)
        data = json.loads(path.read_text())
    else:
        with _STATS_LOCK:
            STATS["misses"] += 1
        log.info("cache miss %s -> GET %s %s", path.name, url, params)
        data = _get_with_retry(url, params)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)  # atomic: an interrupted run never leaves a partial cache file
    check_truncation(data, params)
    return data


def fetch(
    dataset_id: str,
    params: dict[str, Any],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> Any:
    """Query an NYC Open Data dataset by id. See `fetch_url`."""
    return fetch_url(BASE_URL.format(dataset_id=dataset_id), params, cache_dir, refresh)


def fetch_live(dataset_id: str, params: dict[str, Any]) -> Any:
    """Uncached single query (with retries and the grouped-truncation check)."""
    data = _get_with_retry(BASE_URL.format(dataset_id=dataset_id), params)
    check_truncation(data, params)
    return data


def fetch_pages(dataset_id: str, params: dict[str, Any], order: str) -> list[dict]:
    """Uncached, paged raw-row query. `order` must be a unique, stable key.

    Requests pages of `$limit` rows with increasing `$offset` until a short
    page comes back. Not for grouped queries (use `fetch`, which raises on
    truncation).
    """
    if "$group" in params:
        raise ValueError("fetch_pages is for raw rows; grouped queries go through fetch()")
    url = BASE_URL.format(dataset_id=dataset_id)
    limit = int(params["$limit"])
    rows: list[dict] = []
    offset = 0
    while True:
        page = _get_with_retry(url, {**params, "$order": order, "$offset": str(offset)})
        rows.extend(page)
        if len(page) < limit:
            return rows
        offset += limit


def to_hour_of_week(socrata_dow: int, hour: int, sunday_is_zero: bool) -> int:
    """Map a Socrata day-of-week + hour to hour-of-week (0 = Monday 00:00, 167 = Sunday 23:00)."""
    if not 0 <= socrata_dow <= 6:
        raise ValueError(f"dow out of range: {socrata_dow}")
    if not 0 <= hour <= 23:
        raise ValueError(f"hour out of range: {hour}")
    monday_based = (socrata_dow - 1) % 7 if sunday_is_zero else socrata_dow
    return monday_based * 24 + hour


def in_nyc_bbox(lat: float | None, lon: float | None) -> bool:
    """True if (lat, lon) lies in the NYC bounding box. Nulls are False."""
    if lat is None or lon is None:
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return NYC_LAT_MIN <= lat <= NYC_LAT_MAX and NYC_LON_MIN <= lon <= NYC_LON_MAX

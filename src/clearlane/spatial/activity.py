"""Bike-lane activity over time (M3).

A segment exists from `instdate` (inclusive) to `ret_date` (exclusive);
rows with `usable=False` (undeterminable dates) are never active.

* Point-in-time (`active_at`): used to decide whether a complaint can be
  attributed to a segment. A complaint dated before a lane's install date
  cannot be (invariant 2).
* Month-level (`active_in_month`): a segment counts for month M only if it
  was installed on or before the first instant of M and not retired before
  the first instant of M+1, i.e. present for the whole month. A cell is on
  network in M if any in-scope segment touching it is active in M.

The month rule is stricter than the point rule: a complaint on a lane that
opened mid-month is attributable, but its cell-month is not on network, so
it is dropped (and counted) when the panel is built.
"""

from __future__ import annotations

import pandas as pd

from clearlane.ingest.sr311 import month_bounds

# Which segments make up "the bike network". See CONTEXT.md for the choice.
SCOPES = {
    "all": "every usable segment (CLAUDE.md as written)",
    "lanes_on_street": "on-street class I (protected) and II (painted) only",
}


def in_scope(segments: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=segments.index)
    if scope == "lanes_on_street":
        return segments["facilitycl"].isin(["I", "II"]) & (segments["onoffst"] == "ON")
    raise ValueError(f"unknown scope {scope!r}; expected one of {sorted(SCOPES)}")


def active_at(segments: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    """Segments that existed at wall-clock time `when`."""
    ret = segments["ret_date"]
    return segments["usable"] & (segments["instdate"] <= when) & (ret.isna() | (when < ret))


def active_in_month(segments: pd.DataFrame, month: str) -> pd.Series:
    """Segments present for the whole of `month` ('YYYY-MM')."""
    lo, hi = (pd.Timestamp(d) for d in month_bounds(month))
    ret = segments["ret_date"]
    return segments["usable"] & (segments["instdate"] <= lo) & (ret.isna() | (ret >= hi))

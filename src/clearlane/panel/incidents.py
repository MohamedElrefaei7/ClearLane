"""Deduplicate kept complaints into incidents (M5).

An incident is a run of kept complaints in the same cell whose window opens at
its first complaint and covers everything up to `WINDOW` later (inclusive).
The next complaint after that opens a new incident. Windows are anchored, not
chained: a lane reported every 10 minutes for two hours is several incidents,
not one of unbounded length.

The incident's time is its first complaint's `created_date` (NYC wall clock);
that sets its hour-of-week and month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(minutes=15)


def hour_of_week(ts: pd.Series) -> pd.Series:
    """0 = Monday 00:00 ... 167 = Sunday 23:00, from naive wall-clock timestamps."""
    return (ts.dt.dayofweek * 24 + ts.dt.hour).astype("int16")


def assign_incidents(kept: pd.DataFrame, window: pd.Timedelta = WINDOW, chained: bool = False) -> pd.Series:
    """Incident number (0..n-1) for each row of `kept` (needs cell, created_date).

    `chained=True` uses gap-to-previous instead of the anchored window; it exists
    only to report how much the definition matters.
    """
    order = kept.sort_values(["cell", "created_date", "unique_key"], kind="stable").index
    cells = kept.loc[order, "cell"].to_numpy()
    times = kept.loc[order, "created_date"].to_numpy()
    w = np.timedelta64(window)
    ids = np.empty(len(order), dtype=np.int64)
    current, start, prev, prev_cell = -1, None, None, None
    for i in range(len(order)):
        t, c = times[i], cells[i]
        if c != prev_cell:
            new = True
        elif chained:
            new = t - prev > w
        else:
            new = t - start > w
        if new:
            current += 1
            start = t
        ids[i] = current
        prev, prev_cell = t, c
    return pd.Series(ids, index=order).reindex(kept.index)


def build_incidents(kept: pd.DataFrame, window: pd.Timedelta = WINDOW) -> pd.DataFrame:
    """One row per incident: incident_id, cell, start, month, hour_of_week, n_complaints."""
    k = kept.assign(incident_id=assign_incidents(kept, window))
    inc = (k.groupby("incident_id")
             .agg(cell=("cell", "first"), start=("created_date", "min"), n_complaints=("unique_key", "size"))
             .reset_index())
    inc["month"] = inc["start"].dt.strftime("%Y-%m").astype("string")
    inc["hour_of_week"] = hour_of_week(inc["start"])
    return inc

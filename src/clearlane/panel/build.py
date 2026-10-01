"""Zero-filled (cell, hour_of_week, month) panel (M5).

    python -m clearlane.panel.build

Inputs: `complaints_snapped.parquet` (M4) and `network_cell_months.parquet`
(M3). Outputs:

* `data/interim/incidents.parquet` — one row per incident (see incidents.py).
* `data/interim/panel.parquet` — exactly 168 rows for every on-network
  (cell, month): `y` = incidents starting in that slot, `y_raw` = kept raw
  complaints in that slot (by their own timestamp), `exposure` = real hours of
  that wall-clock hour-of-week in that month. Zeros are explicit.

Exposure is counted in elapsed time on the America/New_York clock, so the hour
skipped at the spring-forward transition contributes 0 and the repeated
fall-back hour contributes 2 (e.g. March 2024: Sun 02:00 occurs 4 times, not 5;
the month totals 743 hours). Every slot still has a row (invariant 5).

Incidents and raw complaints whose (cell, month) is not on network (e.g. a lane
opened mid-month) are dropped and counted; so are those outside the panel months.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clearlane.config import PANEL_MONTHS
from clearlane.ingest.sr311 import month_bounds, month_range
from clearlane.panel.incidents import build_incidents, hour_of_week
from clearlane.spatial.grid import NETWORK_PATH
from clearlane.spatial.snap import OUT_PATH as SNAPPED_PATH

TZ = "America/New_York"
SLOTS = 168
INCIDENTS_PATH = Path("data/interim/incidents.parquet")
PANEL_PATH = Path("data/interim/panel.parquet")
RUN_LOG = Path("data/interim/panel_runs.jsonl")


def exposure_table(months: list[str]) -> pd.DataFrame:
    """(month, hour_of_week, exposure): elapsed hours of each wall-clock slot per month."""
    frames = []
    for month in months:
        lo, hi = (pd.Timestamp(d, tz=TZ) for d in month_bounds(month))
        hours = pd.date_range(lo.tz_convert("UTC"), hi.tz_convert("UTC"), freq="h", inclusive="left").tz_convert(TZ)
        how = hours.dayofweek * 24 + hours.hour
        counts = np.bincount(how, minlength=SLOTS)
        frames.append(pd.DataFrame({"month": month, "hour_of_week": np.arange(SLOTS, dtype="int16"),
                                    "exposure": counts.astype("int16")}))
    return pd.concat(frames, ignore_index=True)


def build_panel(network: pd.DataFrame, incidents: pd.DataFrame, kept: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Panel rows for every on-network (cell, month) x 168 slots, plus drop counts."""
    cm = network[["cell", "month"]].drop_duplicates().sort_values(["month", "cell"]).reset_index(drop=True)
    cm_index = pd.Series(np.arange(len(cm)), index=pd.MultiIndex.from_frame(cm))
    months = sorted(cm["month"].unique())

    def slot_counts(df: pd.DataFrame, time_col: str) -> tuple[np.ndarray, int, int]:
        in_months = df["month"].isin(months)
        key = pd.MultiIndex.from_arrays([df["cell"], df["month"]])
        pos = cm_index.reindex(key).to_numpy()
        on = in_months.to_numpy() & ~np.isnan(pos)
        flat = pos[on].astype(np.int64) * SLOTS + hour_of_week(df.loc[on, time_col]).to_numpy()
        return np.bincount(flat, minlength=len(cm) * SLOTS), int((~in_months).sum()), int((in_months & ~on).sum())

    kept = kept.assign(month=kept["created_date"].dt.strftime("%Y-%m"))
    y, inc_out, inc_off = slot_counts(incidents, "start")
    y_raw, raw_out, raw_off = slot_counts(kept, "created_date")

    expo = exposure_table(months).set_index(["month", "hour_of_week"])["exposure"]
    month_rep = np.repeat(cm["month"].to_numpy(), SLOTS)
    slot_rep = np.tile(np.arange(SLOTS, dtype="int16"), len(cm))
    panel = pd.DataFrame({
        "cell": pd.Categorical(np.repeat(cm["cell"].to_numpy(), SLOTS)),
        "month": pd.Categorical(month_rep),
        "hour_of_week": slot_rep,
        "y": y.astype("int16"),
        "y_raw": y_raw.astype("int16"),
        "exposure": expo.reindex(pd.MultiIndex.from_arrays([month_rep, slot_rep])).to_numpy().astype("int16"),
    })
    stats = {
        "cell_months": len(cm),
        "rows": len(panel),
        "incidents_total": len(incidents),
        "incidents_in_panel": int(y.sum()),
        "incidents_outside_panel_months": inc_out,
        "incidents_dropped_off_network": inc_off,
        "raw_total": len(kept),
        "raw_in_panel": int(y_raw.sum()),
        "raw_outside_panel_months": raw_out,
        "raw_dropped_off_network": raw_off,
    }
    return panel, stats


def main() -> None:
    snapped = pd.read_parquet(SNAPPED_PATH)
    kept = snapped[snapped["cell"].notna()].reset_index(drop=True)
    network = pd.read_parquet(NETWORK_PATH)
    expected = month_range(*PANEL_MONTHS)
    if sorted(network["month"].unique()) != expected:
        raise RuntimeError("network_cell_months does not cover the panel months; rerun clearlane.spatial.grid")

    incidents = build_incidents(kept)
    incidents.to_parquet(INCIDENTS_PATH, index=False)
    panel, stats = build_panel(network, incidents, kept)
    panel.to_parquet(PANEL_PATH, index=False)

    record = {"run_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
              "panel_months": list(PANEL_MONTHS), **stats,
              "dedup_ratio_in_panel": round(stats["incidents_in_panel"] / stats["raw_in_panel"], 4)}
    with RUN_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=1))
    print(f"wrote {INCIDENTS_PATH} and {PANEL_PATH}")


if __name__ == "__main__":
    main()

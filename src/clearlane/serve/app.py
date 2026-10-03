"""FastAPI app serving the precomputed artifact (M9).

    uvicorn clearlane.serve.app:app --port 8000

Everything is loaded once at startup from `data/artifacts/serving/` (or
`CLEARLANE_SERVING_DIR`); requests only index into precomputed arrays.

    GET /api/meta                                  artifact metadata + caveat
    GET /api/slot?day=Mon&hour=8&layer=predicted   one slot, every on-network cell
    GET /api/slot?how=8&layer=adjusted             same, by hour-of-week (0 = Mon 00:00)
    GET /api/cell/{cell}                           detail + 168-slot series per layer
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from clearlane.serve.export import LAYERS, SERVING_DIR

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SLOTS = 168


class Artifact:
    def __init__(self, directory: Path):
        self.meta = json.loads((directory / "meta.json").read_text())
        preds = pd.read_parquet(directory / "predictions.parquet").sort_values(["cell", "hour_of_week"])
        self.cells = preds["cell"].iloc[::SLOTS].astype(str).tolist()
        self.index = {c: i for i, c in enumerate(self.cells)}
        n = len(self.cells)
        self.layers = {l: preds[l].to_numpy(dtype=float).reshape(n, SLOTS) for l in LAYERS}
        self.hours = preds["historical_hours"].to_numpy(dtype=float).reshape(n, SLOTS)
        self.detail = pd.read_parquet(directory / "cells.parquet").set_index("cell")


@lru_cache(maxsize=1)
def artifact() -> Artifact:
    return Artifact(Path(os.environ.get("CLEARLANE_SERVING_DIR", SERVING_DIR)))


app = FastAPI(title="ClearLane", description="Reported blocked-bike-lane risk (NYC 311). Not observed obstruction.")


def resolve_how(how: int | None, day: str | None, hour: int | None) -> int:
    if how is not None:
        if day is not None or hour is not None:
            raise HTTPException(400, "give either how, or day and hour")
        if not 0 <= how < SLOTS:
            raise HTTPException(400, "how must be 0..167")
        return how
    if day is None or hour is None:
        raise HTTPException(400, "give how, or day and hour")
    if day.title()[:3] not in DAYS:
        raise HTTPException(400, f"day must be one of {DAYS}")
    if not 0 <= hour <= 23:
        raise HTTPException(400, "hour must be 0..23")
    return DAYS.index(day.title()[:3]) * 24 + hour


@app.get("/api/meta")
def meta() -> dict:
    return artifact().meta


@app.get("/api/slot")
def slot(how: int | None = None, day: str | None = None, hour: int | None = None,
         layer: str = Query("predicted", pattern="^(predicted|adjusted|historical)$")) -> dict:
    a = artifact()
    h = resolve_how(how, day, hour)
    values = a.layers[layer][:, h]
    return {
        "month": a.meta["month"],
        "hour_of_week": h,
        "day": DAYS[h // 24],
        "hour": h % 24,
        "layer": layer,
        "units": a.meta["units"],
        "heuristic": layer == "adjusted",
        "caveat": a.meta["caveat"],
        "max": float(values.max()),
        "cells": [{"cell": c, "value": float(v)} for c, v in zip(a.cells, values)],
    }


@app.get("/api/cell/{cell}")
def cell(cell: str) -> dict:
    a = artifact()
    if cell not in a.index:
        raise HTTPException(404, "not an on-network cell (no bike lane)")
    i = a.index[cell]
    d = a.detail.loc[cell]
    lane_cols = [c for c in d.index if c.startswith("lane_m_") and c != "lane_m_total"]
    total = float(d["lane_m_total"])
    return {
        "cell": cell,
        "month": a.meta["month"],
        "lat": float(d["lat"]),
        "lng": float(d["lng"]),
        "borough": d["borough"],
        "lane_metres": {c.removeprefix("lane_m_"): round(float(d[c]), 1) for c in lane_cols},
        "lane_share": {c.removeprefix("lane_m_"): (round(float(d[c]) / total, 3) if total else 0.0) for c in lane_cols},
        "propensity_index": float(d["propensity_index"]),
        "weekly": {l: float(a.layers[l][i].sum()) for l in LAYERS},
        "series": {l: np.round(a.layers[l][i], 6).tolist() for l in LAYERS},
        "historical_hours": a.hours[i].tolist(),
        "caveat": a.meta["caveat"],
    }

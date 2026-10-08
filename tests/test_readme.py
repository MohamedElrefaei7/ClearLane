"""The README's test-split table must match the tracked ledger (guards against stale numbers)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_test_metrics_match_ledger():
    readme = (ROOT / "README.md").read_text()
    ledger = json.loads((ROOT / "reports" / "test_evaluations.json").read_text())
    latest = {}
    for e in ledger:
        latest[e["version"].split(":")[0]] = e
    rows = {"Empirical-Bayes baseline": "baseline", "LightGBM": "lgbm_full",
            "LightGBM + 3-month recalibration (shipped until 2026-10-07)": "lgbm_full+recal3m",
            "LightGBM + weekly recalibration (shipped)": "lgbm_full+recalw56d"}
    for label, key in rows.items():
        e = latest[key]
        cells = [f"{e['mean_poisson_deviance']:.6f}", f"{100 * e['top_decile_capture']:.1f}%", f"{e['obs_over_pred']:.3f}"]
        pattern = r"\|\s*\**" + re.escape(label) + r"\**\s*\|\s*\**" + r"\**\s*\|\s*\**".join(map(re.escape, cells)) + r"\**\s*\|"
        assert re.search(pattern, readme), f"README row for {label} does not match ledger {cells}"
    assert f"{latest['baseline']['incidents']:,} incidents" in readme

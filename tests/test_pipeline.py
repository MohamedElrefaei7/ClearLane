import importlib

import pytest

from clearlane import pipeline
from clearlane.models.test_eval import verify


def test_every_stage_has_an_importable_main():
    for name, module, argv, _ in pipeline.STAGES:
        assert callable(importlib.import_module(module).main), name


def test_run_respects_from_and_to(monkeypatch):
    called = []
    fake = type("M", (), {"main": staticmethod(lambda argv=None: called.append(argv))})
    monkeypatch.setattr(pipeline.importlib, "import_module", lambda m: fake)
    pipeline.run("features", "lgbm")
    assert len(called) == 3  # features, baseline, lgbm


def test_reversed_range_rejected():
    with pytest.raises(SystemExit):
        pipeline.main(["--from", "lgbm", "--to", "snap"])


def test_verify_flags_only_real_differences():
    rec = {"v1": {"mean_poisson_deviance": 0.021, "top_decile_capture": 0.85, "obs_over_pred": 0.98,
                  "incidents": 100, "predicted_incidents": 101.5}}
    same = {"m": {"version": "v1", **rec["v1"]}}
    assert verify(same, rec, {"m"}) == set()
    off = {"m": {**same["m"], "top_decile_capture": 0.8501}}
    assert verify(off, rec, {"m"}) == {"m"}


def test_refresh_skips_training_and_test_eval(monkeypatch):
    called = []
    monkeypatch.setattr(pipeline.importlib, "import_module",
                        lambda m: type("M", (), {"main": staticmethod(lambda argv=None, m=m: called.append(m))}))
    pipeline.main(["--refresh"])
    names = [n for n, mod, _, _ in pipeline.STAGES if mod in called]
    assert names == [n for n in pipeline.NAMES if n not in ("baseline", "lgbm", "test-eval")]
    assert names[0] == "sr311" and names[-1] == "export"

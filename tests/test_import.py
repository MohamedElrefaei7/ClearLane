import importlib

import pytest

SUBPACKAGES = ["ingest", "spatial", "panel", "features", "models", "serve", "audit"]


def test_import_root():
    import clearlane  # noqa: F401


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_import_subpackage(name):
    importlib.import_module(f"clearlane.{name}")

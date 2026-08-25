"""Smoke tests: every plugin module must import cleanly from the venv."""

import importlib

import pytest

MODULES = [
    "napari_peredox._segment",
    "napari_peredox._measure",
    "napari_peredox._learning",
    "napari_peredox._io",
    "napari_peredox._host",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)

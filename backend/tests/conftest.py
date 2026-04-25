from __future__ import annotations

import os
from pathlib import Path

import pytest

import embeddings

DEFAULT_MODEL_PATH = Path.home() / "models" / "fasttext" / "cc.en.300.bin"


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests that load the fastText model (~70s)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow (loads fastText model)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def _model_path() -> Path:
    return Path(os.environ.get("FASTTEXT_MODEL_PATH", DEFAULT_MODEL_PATH))


@pytest.fixture(scope="session")
def loaded_model():
    """Load fastText once per test session (~60s). Skip suite if model missing."""
    path = _model_path()
    if not path.exists():
        pytest.skip(
            f"fastText model not at {path}. "
            f"Set FASTTEXT_MODEL_PATH or place the file there."
        )
    embeddings.load_model(path)
    return embeddings

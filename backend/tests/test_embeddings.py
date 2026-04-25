from __future__ import annotations

import numpy as np
import pytest

import embeddings


def test_raises_when_not_loaded(monkeypatch):
    monkeypatch.setattr(embeddings, "_model", None)
    with pytest.raises(RuntimeError, match="not loaded"):
        embeddings.get_vector("apple")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        embeddings.load_model(tmp_path / "nope.bin")


@pytest.mark.slow
def test_in_vocab_shape_and_dtype(loaded_model):
    v = loaded_model.get_vector("apple")
    assert v.shape == (300,)
    assert v.dtype == np.float32


@pytest.mark.slow
def test_vectors_are_unit_normalized(loaded_model):
    v = loaded_model.get_vector("apple")
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_get_vectors_batch_shape(loaded_model):
    mat = loaded_model.get_vectors(["apple", "orange", "car"])
    assert mat.shape == (3, 300)
    assert mat.dtype == np.float32


@pytest.mark.slow
def test_semantic_ordering(loaded_model):
    """Related pairs should score higher than unrelated pairs."""
    apple = loaded_model.get_vector("apple")
    orange = loaded_model.get_vector("orange")
    car = loaded_model.get_vector("car")
    truck = loaded_model.get_vector("truck")

    fruit_sim = float(np.dot(apple, orange))
    vehicle_sim = float(np.dot(car, truck))
    cross_sim = float(np.dot(apple, car))

    assert fruit_sim > cross_sim
    assert vehicle_sim > cross_sim


@pytest.mark.slow
def test_oov_subword_fallback(loaded_model):
    """fastText's subwords should produce a real vector for made-up words."""
    v = loaded_model.get_vector("fjordcatbus")
    assert v.shape == (300,)
    assert np.any(v != 0)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.slow
def test_case_sensitivity_reasonable(loaded_model):
    """Case variants shouldn't produce wildly different vectors (via subwords)."""
    lower = loaded_model.get_vector("apple")
    upper = loaded_model.get_vector("APPLE")
    sim = float(np.dot(lower, upper))
    assert sim > 0.3, f"case variants cos={sim:.3f} too low"

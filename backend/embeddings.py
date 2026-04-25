from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from gensim.models.fasttext import load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors

log = logging.getLogger(__name__)

_model: KeyedVectors | None = None
_dim: int | None = None


def load_model(path: str | Path) -> KeyedVectors:
    """Load fastText .bin into memory. Call once at startup."""
    global _model, _dim
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fastText model not found at {path}")
    log.info("Loading fastText vectors from %s (this takes ~30s)...", path)
    _model = load_facebook_vectors(str(path))
    _dim = _model.vector_size
    log.info("Loaded fastText: %d-dim, %d in-vocab tokens", _dim, len(_model))
    return _model


def is_loaded() -> bool:
    return _model is not None


def get_vector(word: str) -> np.ndarray:
    """
    Return a unit-normalized vector for `word`.

    fastText's subword model produces a vector for any string (even OOV).
    If the word is still missing (shouldn't happen with subwords), returns zeros.
    """
    if _model is None:
        raise RuntimeError("fastText model not loaded. Call load_model() first.")
    try:
        vec = _model.get_vector(word, norm=True)
    except KeyError:
        log.warning("No vector for %r (even with subwords); using zeros", word)
        return np.zeros(_dim, dtype=np.float32)
    return vec.astype(np.float32, copy=False)


def get_vectors(words: list[str]) -> np.ndarray:
    """Return an (N, dim) matrix of unit-normalized vectors."""
    return np.stack([get_vector(w) for w in words])

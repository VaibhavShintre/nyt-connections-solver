"""
Quick smoke test for embeddings.py. Run:

    python verify_embeddings.py /path/to/cc.en.300.bin

Checks: model loads, in-vocab lookup, OOV (subword) lookup, cosine sanity.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import embeddings


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(model_path: str) -> None:
    t0 = time.time()
    embeddings.load_model(model_path)
    print(f"Loaded in {time.time() - t0:.1f}s")

    in_vocab = ["apple", "orange", "banana", "car", "truck", "bicycle"]
    oov = ["xyzzy", "fjordcatbus"]

    vecs = embeddings.get_vectors(in_vocab)
    print(f"In-vocab shape: {vecs.shape}")

    pairs = [
        ("apple", "orange"),
        ("apple", "car"),
        ("car", "truck"),
        ("car", "banana"),
    ]
    for a, b in pairs:
        va = embeddings.get_vector(a)
        vb = embeddings.get_vector(b)
        print(f"cos({a!r:>10}, {b!r:>10}) = {cosine(va, vb):+.3f}")

    print("\nOOV (subword fallback):")
    for w in oov:
        v = embeddings.get_vector(w)
        norm = np.linalg.norm(v)
        print(f"  {w!r:>15}: norm={norm:.3f}, nonzero={int(np.any(v))}")

    print("\nOK")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_embeddings.py /path/to/cc.en.300.bin")
        sys.exit(1)
    main(sys.argv[1])

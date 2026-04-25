"""
CLI for the solver. Usage:

    python solve_cli.py apple orange banana grape car truck bicycle motorcycle \\
                        red blue green yellow dog cat horse rabbit

Loads the fastText model, computes similarity, prints the top-5 partitions.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import embeddings
import solver

DEFAULT_MODEL_PATH = Path.home() / "models" / "fasttext" / "cc.en.300.bin"


def main(words: list[str]) -> None:
    if len(words) != 16:
        print(f"need exactly 16 words, got {len(words)}", file=sys.stderr)
        sys.exit(2)

    model_path = Path(os.environ.get("FASTTEXT_MODEL_PATH", DEFAULT_MODEL_PATH))
    t0 = time.time()
    embeddings.load_model(model_path)
    print(f"model loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    result = solver.solve(words, k=5)
    print(f"solved in {time.time() - t0:.2f}s")

    best = result["best_partition"]
    print(f"\nbest (score={best['score']:.3f}, confidence={result['confidence']:.3f}):")
    for i, group in enumerate(best["groups"], 1):
        print(f"  {i}. {group}")

    print("\nalternatives:")
    for rank, alt in enumerate(result["alternatives"], 2):
        print(f"  #{rank} (score={alt['score']:.3f})")
        for group in alt["groups"]:
            print(f"     {group}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1:])

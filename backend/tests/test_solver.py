from __future__ import annotations

import time

import numpy as np
import pytest

import solver


def _block_diagonal_sim() -> np.ndarray:
    """16x16 similarity where {0..3}, {4..7}, {8..11}, {12..15} are perfect blocks."""
    sim = np.zeros((16, 16), dtype=np.float32)
    for start in (0, 4, 8, 12):
        for i in range(start, start + 4):
            for j in range(start, start + 4):
                sim[i, j] = 1.0
    return sim


def test_partition_cache_shape():
    """16 items into 4 unordered groups of 4 = 16! / (4!^4 * 4!) = 2,627,625."""
    ids, subsets = solver._partition_cache()
    assert ids.shape == (2_627_625, 4)
    assert len(subsets) == 1820  # C(16, 4)


def test_partition_cache_is_canonical():
    """Each partition should have every index {0..15} present exactly once."""
    ids, subsets = solver._partition_cache()
    # Spot check a handful of rows (full check would be slow).
    for row in ids[::100_000]:
        seen = set()
        for gid in row:
            seen.update(subsets[int(gid)])
        assert seen == set(range(16))


def test_find_best_partition_block_diagonal():
    sim = _block_diagonal_sim()
    groups, score = solver.find_best_partition(sim)
    # Each group has 6 pairs at similarity 1.0, 4 groups => score = 24.
    assert score == pytest.approx(24.0)
    sorted_groups = sorted(sorted(g) for g in groups)
    assert sorted_groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]


def test_find_top_k_partitions_length_and_order():
    sim = _block_diagonal_sim()
    top = solver.find_top_k_partitions(sim, k=5)
    assert len(top) == 5
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(24.0)


def test_find_top_k_k_equals_one():
    sim = _block_diagonal_sim()
    top = solver.find_top_k_partitions(sim, k=1)
    assert len(top) == 1


@pytest.mark.slow
def test_compute_similarity_matrix_properties(loaded_model):
    words = ["apple", "orange", "banana", "car", "truck", "bicycle"]
    sim = solver.compute_similarity_matrix(words)
    n = len(words)
    assert sim.shape == (n, n)
    assert np.allclose(sim, sim.T, atol=1e-6)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
    # fruit-fruit should beat fruit-vehicle
    assert sim[0, 1] > sim[0, 3]


@pytest.mark.slow
def test_solve_on_clean_four_category_puzzle(loaded_model):
    """Sanity check end-to-end: clearly separable categories should group correctly."""
    words = [
        "apple", "orange", "banana", "grape",
        "car", "truck", "bicycle", "motorcycle",
        "red", "blue", "green", "yellow",
        "dog", "cat", "horse", "rabbit",
    ]
    result = solver.solve(words, k=5)
    best_groups = [sorted(g) for g in result["best_partition"]["groups"]]
    expected = [
        sorted(["apple", "orange", "banana", "grape"]),
        sorted(["car", "truck", "bicycle", "motorcycle"]),
        sorted(["red", "blue", "green", "yellow"]),
        sorted(["dog", "cat", "horse", "rabbit"]),
    ]
    assert sorted(best_groups) == sorted(expected)
    assert result["confidence"] > 0.0
    assert len(result["alternatives"]) == 4


def test_solver_warmup_builds_cache():
    """Cache build is a one-time cost; must complete in a reasonable time."""
    # Reset the cache to force a rebuild so we measure the cold path honestly.
    solver._PARTITION_GROUP_IDS = None
    solver._GROUP_SUBSETS = None
    t0 = time.time()
    solver.warmup()
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"warmup took {elapsed:.2f}s"


def test_solver_solve_is_fast():
    """After warmup, solving on a random matrix should be well under 1s."""
    solver.warmup()  # ensure cache is built (no-op if already built)
    rng = np.random.default_rng(42)
    sim = rng.standard_normal((16, 16)).astype(np.float32)
    sim = (sim + sim.T) / 2
    t0 = time.time()
    solver.find_best_partition(sim)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"solve took {elapsed:.2f}s"

from __future__ import annotations

from itertools import combinations

import numpy as np

from embeddings import get_vectors

GROUP_SIZE = 4
NUM_GROUPS = 4
N = GROUP_SIZE * NUM_GROUPS  # 16
NUM_PARTITIONS = 2_627_625   # 16! / (4!^4 * 4!)
NUM_SUBSETS = 1_820          # C(16, 4)


def compute_similarity_matrix(words: list[str]) -> np.ndarray:
    """Cosine similarity matrix. Vectors from embeddings.py are already unit-normalized."""
    if len(words) < 2:
        raise ValueError("need at least 2 words")
    vecs = get_vectors(words)
    return (vecs @ vecs.T).astype(np.float32)


# ---- Partition structure cache ---------------------------------------------
# The structure of all 2,627,625 partitions of 16 items into 4 groups of 4
# depends only on N, not on the similarity matrix. Build once, reuse forever.

_PARTITION_GROUP_IDS: np.ndarray | None = None       # (NUM_PARTITIONS, 4) int16
_GROUP_SUBSETS: tuple[tuple[int, ...], ...] | None = None  # len = NUM_SUBSETS


def _build_partition_cache() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    subsets = tuple(combinations(range(N), GROUP_SIZE))
    subset_id = {s: i for i, s in enumerate(subsets)}
    ids = np.empty((NUM_PARTITIONS, NUM_GROUPS), dtype=np.int16)
    idx = 0

    # Canonical form: smallest remaining index always starts the next group.
    # Three nested loops over combinations of size 3 enumerate all partitions.
    all_but_zero = tuple(range(1, N))
    for c1 in combinations(all_but_zero, 3):
        gid1 = subset_id[(0, *c1)]
        s1 = set(c1)
        remaining1 = tuple(i for i in all_but_zero if i not in s1)  # 12 items
        first2, rest2 = remaining1[0], remaining1[1:]
        for c2 in combinations(rest2, 3):
            gid2 = subset_id[(first2, *c2)]
            s2 = set(c2)
            remaining2 = tuple(i for i in rest2 if i not in s2)  # 8 items
            first3, rest3 = remaining2[0], remaining2[1:]
            for c3 in combinations(rest3, 3):
                gid3 = subset_id[(first3, *c3)]
                s3 = set(c3)
                remaining3 = tuple(i for i in rest3 if i not in s3)  # 4 items
                gid4 = subset_id[remaining3]
                ids[idx, 0] = gid1
                ids[idx, 1] = gid2
                ids[idx, 2] = gid3
                ids[idx, 3] = gid4
                idx += 1

    assert idx == NUM_PARTITIONS, idx
    return ids, subsets


def warmup() -> None:
    """Build the partition cache eagerly. Call at server startup."""
    _partition_cache()


def _partition_cache() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    global _PARTITION_GROUP_IDS, _GROUP_SUBSETS
    if _PARTITION_GROUP_IDS is None:
        _PARTITION_GROUP_IDS, _GROUP_SUBSETS = _build_partition_cache()
    return _PARTITION_GROUP_IDS, _GROUP_SUBSETS


def _group_scores(sim: np.ndarray, subsets: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Sum of the 6 pairwise similarities within each 4-subset. Shape (NUM_SUBSETS,)."""
    scores = np.empty(len(subsets), dtype=np.float64)
    for i, (a, b, c, d) in enumerate(subsets):
        scores[i] = (
            sim[a, b] + sim[a, c] + sim[a, d]
            + sim[b, c] + sim[b, d]
            + sim[c, d]
        )
    return scores


def _partition_scores(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Score every partition. Returns (scores, partition_group_ids, subsets)."""
    if sim.shape != (N, N):
        raise ValueError(f"expected ({N},{N}) similarity matrix, got {sim.shape}")
    ids, subsets = _partition_cache()
    gscores = _group_scores(sim, subsets)
    pscores = gscores[ids].sum(axis=1)  # (NUM_PARTITIONS,)
    return pscores, ids, subsets


def find_best_partition(sim: np.ndarray) -> tuple[list[list[int]], float]:
    """Partition indices into 4 groups of 4, maximizing total intra-group similarity."""
    pscores, ids, subsets = _partition_scores(sim)
    best = int(pscores.argmax())
    groups = [list(subsets[int(gid)]) for gid in ids[best]]
    return groups, float(pscores[best])


def find_top_k_partitions(
    sim: np.ndarray, k: int = 5
) -> list[tuple[list[list[int]], float]]:
    """Top-k partitions by score, sorted descending."""
    if k < 1:
        raise ValueError("k must be >= 1")
    pscores, ids, subsets = _partition_scores(sim)
    k = min(k, pscores.size)
    # argpartition gets unordered top-k quickly, then sort just those.
    top_idx = np.argpartition(-pscores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-pscores[top_idx])]
    return [
        ([list(subsets[int(gid)]) for gid in ids[p]], float(pscores[p]))
        for p in top_idx
    ]


def solve(words: list[str], k: int = 5) -> dict:
    """Full pipeline: words -> similarity matrix + top-k partitions (as word groups)."""
    if len(words) != N:
        raise ValueError(f"expected {N} words, got {len(words)}")
    sim = compute_similarity_matrix(words)
    top = find_top_k_partitions(sim, k=k)
    best_idx_groups, best_score = top[0]
    alts = top[1:]
    second = alts[0][1] if alts else best_score
    confidence = (
        max(0.0, min(1.0, (best_score - second) / best_score))
        if best_score > 0
        else 0.0
    )

    def to_words(groups: list[list[int]]) -> list[list[str]]:
        return [[words[i] for i in g] for g in groups]

    return {
        "similarity_matrix": sim.tolist(),
        "best_partition": {"groups": to_words(best_idx_groups), "score": best_score},
        "alternatives": [{"groups": to_words(g), "score": s} for g, s in alts],
        "confidence": confidence,
    }

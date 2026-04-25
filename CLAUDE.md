# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

NYT Connections puzzle solver using fastText word embeddings. The build is staged in phases (see `backend/` for current state):

- **Phase 1–2 done**: solver core + FastAPI `/api/solve` endpoint.
- **Phase 3**: `nyt_client.py` + `/api/puzzle/today` endpoint (stub exists).
- **Phase 4–5**: React/Vite frontend with d3-force visualization (not started).

## Commands (run from `backend/`)

All commands assume the venv at `backend/.venv`.

```bash
# Tests — fast suite (default, ~8s, skips model-loading tests)
.venv/bin/pytest

# Tests — full suite (~70s, loads fastText)
.venv/bin/pytest --run-slow

# Single test
.venv/bin/pytest tests/test_solver.py::test_partition_cache_shape -v

# Run dev server (loads fastText at startup, ~70s before ready)
.venv/bin/uvicorn main:app --port 8000

# CLI solver (16 words as args)
.venv/bin/python solve_cli.py apple orange banana grape ... rabbit
```

The fastText model lives **outside the repo** at `~/models/fasttext/cc.en.300.bin` (6.7 GB). Override via `FASTTEXT_MODEL_PATH` env var.

## Architecture

Three modules cooperate; understanding their boundary is the key to being productive:

**`embeddings.py`** holds a module-level `_model` global populated by `load_model()`. `get_vector(word)` returns a unit-normalized `(300,)` float32 — fastText's subword model handles OOV words natively, so we don't fall back to zeros except for pathological cases. This module is stateful by design: the model is multi-GB, loaded once, reused everywhere.

**`solver.py`** has two distinct caches with very different lifetimes:
- The **similarity matrix** is per-puzzle (cheap: 16 vector lookups + a 16×16 dot product).
- The **partition structure** (`_PARTITION_GROUP_IDS`, shape `(2_627_625, 4)` int16) is precomputed once because it depends only on N=16, not on the puzzle. `warmup()` builds it eagerly. This is *the* reason `find_best_partition` runs in <100ms after startup: scoring becomes a vectorized `group_scores[ids].sum(axis=1)`. Don't reintroduce per-call partition enumeration.

**`main.py`** wires both into FastAPI. The `lifespan` async context loads the model and calls `solver.warmup()` *before* the server accepts traffic, so cold-start cost (~70s) is paid once. Per-request latency is ~70ms warm.

## Performance model — important

The 70s startup cost is **not** in the user's path: in production the server stays warm and amortizes it. Three places pay it repeatedly: tests, `solve_cli.py`, and `uvicorn --reload` on file save. Don't optimize the model load unless dev velocity in one of those becomes painful — adding pickled-model caching has real complexity (gensim format compatibility, subword preservation) and provides zero benefit to end users.

## Testing conventions

- Model-dependent tests are marked `@pytest.mark.slow`. Default `pytest` skips them via the `--run-slow` flag wired in `tests/conftest.py`. Tests that exercise pure algorithms use synthetic similarity matrices (e.g. block-diagonal) so they don't need fastText.
- The `loaded_model` fixture is session-scoped; the model loads once across the entire `--run-slow` invocation.
- When adding a new test that touches the algorithm, prefer a synthetic similarity matrix over loading the model. Reserve `@slow` for behaviors that are specifically about fastText (subword OOV, real semantic ordering).

## API contract

`POST /api/solve` takes `{"words": [16 strings]}` and returns `{similarity_matrix, best_partition: {groups, score}, alternatives: [...], confidence}`. Pydantic enforces exactly 16 words and returns 422 otherwise. Confidence is `(best - second_best) / best`, clamped to `[0, 1]`.

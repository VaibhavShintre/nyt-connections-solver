from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import embeddings
import solver

DEFAULT_MODEL_PATH = Path.home() / "models" / "fasttext" / "cc.en.300.bin"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("connections")


class SolveRequest(BaseModel):
    words: list[str] = Field(..., min_length=16, max_length=16)


class GroupResult(BaseModel):
    groups: list[list[str]]
    score: float


class SolveResponse(BaseModel):
    similarity_matrix: list[list[float]]
    best_partition: GroupResult
    alternatives: list[GroupResult]
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("FASTTEXT_MODEL_PATH", DEFAULT_MODEL_PATH))
    embeddings.load_model(model_path)
    solver.warmup()
    log.info("Server ready")
    yield


app = FastAPI(title="NYT Connections Solver", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": embeddings.is_loaded()}


@app.post("/api/solve", response_model=SolveResponse)
def solve_endpoint(req: SolveRequest):
    try:
        return solver.solve(req.words, k=5)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

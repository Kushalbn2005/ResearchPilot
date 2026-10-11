from sqlalchemy import select

from .bm25 import bm25_search
from .config import settings
from .db import Chunk
from .indexing import vector_search
from .rerank import rerank_scores

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). No score normalisation needed."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _load_chunks(db, ids: list[str]) -> dict[str, dict]:
    rows = db.scalars(select(Chunk).where(Chunk.id.in_(ids))).all()
    return {c.id: {"chunk_id": c.id, "document_id": c.document_id, "section": c.section,
                   "page_start": c.page_start, "page_end": c.page_end, "text": c.text} for c in rows}


def search(db, query: str, k: int = 8, document_ids: list[str] | None = None,
           mode: str = "hybrid_rerank") -> list[dict]:
    """Retrieve chunks. Modes exist so you can compare retrievers (your evaluation baselines):
    vector | bm25 | hybrid (RRF) | hybrid_rerank (RRF then cross-encoder)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    n = settings.rerank_candidates

    vec = vector_search(query, k=n, document_ids=document_ids) if mode != "bm25" else []
    bm = bm25_search(query, k=n, document_ids=document_ids) if mode != "vector" else []
    vrank = {h["chunk_id"]: r for r, h in enumerate(vec, start=1)}
    brank = {h["chunk_id"]: r for r, h in enumerate(bm, start=1)}

    rrf_scores: dict[str, float] = {}
    if mode == "vector":
        ordered = [(h["chunk_id"], h["score"]) for h in vec]
    elif mode == "bm25":
        ordered = [(h["chunk_id"], h["score"]) for h in bm]
    else:
        ordered = rrf([[h["chunk_id"] for h in vec], [h["chunk_id"] for h in bm]])
        rrf_scores = dict(ordered)

    candidate_ids = [cid for cid, _ in ordered[: n if mode == "hybrid_rerank" else k]]
    chunks = _load_chunks(db, candidate_ids)  # drops ids missing from Postgres (stale vectors)
    candidate_ids = [cid for cid in candidate_ids if cid in chunks]
    score_by_id = dict(ordered)

    rerank_by_id: dict[str, float] = {}
    if mode == "hybrid_rerank" and candidate_ids:
        scores = rerank_scores(query, [chunks[c]["text"] for c in candidate_ids])
        rerank_by_id = dict(zip(candidate_ids, scores))
        candidate_ids.sort(key=lambda c: -rerank_by_id[c])

    results = []
    for cid in candidate_ids[:k]:
        results.append({
            **chunks[cid],
            "score": round(rerank_by_id.get(cid, score_by_id[cid]), 4),
            "vector_rank": vrank.get(cid),
            "bm25_rank": brank.get(cid),
            "rrf_score": round(rrf_scores[cid], 5) if cid in rrf_scores else None,
            "rerank_score": round(rerank_by_id[cid], 4) if cid in rerank_by_id else None,
        })
    return results
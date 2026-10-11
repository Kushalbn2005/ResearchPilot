import os
import sys
import types

os.environ.setdefault("DATABASE_URL", "postgresql://x:x@localhost/x")

try:  # use the real package if installed; otherwise stub so these tests stay light
    import sentence_transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = object
    stub.CrossEncoder = object
    sys.modules["sentence_transformers"] = stub

from app import retrieval
from app.bm25 import BM25Index, tokenize

ROWS = [
    ("c1", "d1", "We trained XGBoost and Random Forest on the Karnataka rainfall dataset."),
    ("c2", "d1", "Evaluation used RMSE and MAE over five cross-validation folds."),
    ("c3", "d2", "LSTM networks capture temporal dependencies in long sequences."),
    ("c4", "d2", "The weather stations provide daily precipitation records."),
]


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("What is the XGBoost model?") == ["xgboost", "model"]


def test_bm25_finds_exact_term_that_embeddings_often_miss():
    idx = BM25Index(ROWS)
    assert idx.search("XGBoost")[0]["chunk_id"] == "c1"


def test_bm25_document_filter_and_empty_cases():
    idx = BM25Index(ROWS)
    assert all(h["chunk_id"] in {"c3", "c4"} for h in idx.search("precipitation LSTM", document_ids=["d2"]))
    assert BM25Index([]).search("anything") == []
    assert idx.search("the of and") == []  # only stopwords


def test_rrf_rewards_agreement_between_rankers():
    fused = [cid for cid, _ in retrieval.rrf([["a", "b", "c"], ["b", "c", "d"]])]
    assert fused == ["b", "c", "a", "d"]


def _patch(monkeypatch, vec, bm):
    chunks = {c: {"chunk_id": c, "document_id": "d", "section": "S", "page_start": 1,
                  "page_end": 1, "text": f"text {c}"} for c in "abcd"}
    monkeypatch.setattr(retrieval, "vector_search", lambda q, k, document_ids=None: [{"chunk_id": c, "score": 0.9} for c in vec])
    monkeypatch.setattr(retrieval, "bm25_search", lambda q, k, document_ids=None: [{"chunk_id": c, "score": 5.0} for c in bm])
    monkeypatch.setattr(retrieval, "_load_chunks", lambda db, ids: {i: chunks[i] for i in ids if i in chunks})


def test_hybrid_reports_ranks_from_each_retriever(monkeypatch):
    _patch(monkeypatch, vec=["a", "b"], bm=["b", "c"])
    res = retrieval.search(None, "q", k=3, mode="hybrid")
    assert res[0]["chunk_id"] == "b"
    assert res[0]["vector_rank"] == 2 and res[0]["bm25_rank"] == 1
    assert res[0]["rerank_score"] is None


def test_rerank_can_reorder_fused_candidates(monkeypatch):
    _patch(monkeypatch, vec=["a", "b"], bm=["b", "c"])
    monkeypatch.setattr(retrieval, "rerank_scores", lambda q, texts: [0.1 if t.endswith("b") else 0.9 if t.endswith("c") else 0.5 for t in texts])
    res = retrieval.search(None, "q", k=2, mode="hybrid_rerank")
    assert [r["chunk_id"] for r in res] == ["c", "a"]


def test_unknown_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        retrieval.search(None, "q", mode="magic")
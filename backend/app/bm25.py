import re
import threading

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from .db import Chunk, SessionLocal

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
    "be", "by", "with", "as", "at", "from", "that", "this", "it", "its", "which", "we", "our",
    "what", "how", "did", "do", "does", "used", "use", "using",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    """In-memory BM25 over chunk text. rows: iterable of (chunk_id, document_id, text)."""

    def __init__(self, rows):
        rows = list(rows)
        self.ids = [r[0] for r in rows]
        self.docs = [r[1] for r in rows]
        corpus = [tokenize(r[2]) for r in rows]
        # BM25Okapi divides by corpus size, so an empty corpus is handled separately.
        self.bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None

    def search(self, query: str, k: int = 30, document_ids: list[str] | None = None) -> list[dict]:
        terms = tokenize(query)
        if self.bm25 is None or not terms:
            return []
        scores = self.bm25.get_scores(terms)
        allowed = set(document_ids) if document_ids else None
        hits = [(i, s) for i, s in enumerate(scores)
                if s > 0 and (allowed is None or self.docs[i] in allowed)]
        hits.sort(key=lambda x: -x[1])
        return [{"chunk_id": self.ids[i], "score": float(s)} for i, s in hits[:k]]


# ---- process-wide cache, rebuilt lazily after documents are added or deleted ----
_lock = threading.Lock()
_index: BM25Index | None = None
_dirty = True


def mark_dirty() -> None:
    global _dirty
    with _lock:
        _dirty = True


def _get_index() -> BM25Index:
    global _index, _dirty
    with _lock:
        if _dirty or _index is None:
            with SessionLocal() as db:
                rows = db.execute(select(Chunk.id, Chunk.document_id, Chunk.text)).all()
            _index = BM25Index(rows)
            _dirty = False
        return _index


def bm25_search(query: str, k: int = 30, document_ids: list[str] | None = None) -> list[dict]:
    return _get_index().search(query, k=k, document_ids=document_ids)
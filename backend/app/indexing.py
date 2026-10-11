import os
import threading

# Must be set before sentence-transformers/tokenizers are imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from qdrant_client import QdrantClient, models  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from .config import settings  # noqa: E402

COLLECTION = "chunks"
DIM = 384  # bge-small-en-v1.5
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_client: QdrantClient | None = None
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()   # protects model loading
_encode_lock = threading.Lock()  # one encode at a time: avoids thread crashes on macOS


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def model() -> SentenceTransformer:
    global _model
    with _model_lock:
        if _model is None:
            # CPU on purpose: Apple's MPS backend is unstable when called from worker threads,
            # and a 384-dim small model is fast enough on CPU.
            _model = SentenceTransformer(settings.embedding_model, device="cpu")
        return _model


def warm_up() -> None:
    """Load the embedding model once, at startup, in the main thread."""
    model()


def _encode(texts, **kw):
    m = model()
    with _encode_lock:
        return m.encode(texts, normalize_embeddings=True, **kw)


def ensure_collection() -> None:
    c = client()
    if not c.collection_exists(COLLECTION):
        c.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        )
        c.create_payload_index(COLLECTION, "document_id", models.PayloadSchemaType.KEYWORD)


def index_chunks(rows: list) -> None:
    """rows: Chunk ORM objects (already have ids)."""
    vectors = _encode([r.text for r in rows], batch_size=32)
    points = [
        models.PointStruct(
            id=r.id,
            vector=v.tolist(),
            payload={
                "chunk_id": r.id,
                "document_id": r.document_id,
                "section": r.section,
                "page_start": r.page_start,
                "page_end": r.page_end,
            },
        )
        for r, v in zip(rows, vectors)
    ]
    client().upsert(COLLECTION, points=points)


def vector_search(query: str, k: int = 10, document_ids: list[str] | None = None) -> list[dict]:
    vec = _encode(QUERY_PREFIX + query).tolist()
    flt = None
    if document_ids:
        flt = models.Filter(must=[models.FieldCondition(
            key="document_id", match=models.MatchAny(any=document_ids))])
    res = client().query_points(COLLECTION, query=vec, limit=k, query_filter=flt, with_payload=True)
    return [{"chunk_id": p.payload["chunk_id"], "score": p.score} for p in res.points]


def delete_document_vectors(document_id: str) -> None:
    client().delete(COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])))
import os
import threading

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import CrossEncoder  # noqa: E402

from .config import settings  # noqa: E402
from .indexing import _encode_lock as torch_lock  # noqa: E402  (one torch call at a time)

_model: CrossEncoder | None = None
_load_lock = threading.Lock()


def reranker() -> CrossEncoder:
    global _model
    with _load_lock:
        if _model is None:
            _model = CrossEncoder(settings.reranker_model, device="cpu", max_length=512)
        return _model


def warm_up() -> None:
    reranker()


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Cross-encoder relevance of each (query, passage) pair; higher is more relevant."""
    if not texts:
        return []
    m = reranker()
    with torch_lock:
        scores = m.predict([(query, t) for t in texts], batch_size=16)
    return [float(s) for s in scores]
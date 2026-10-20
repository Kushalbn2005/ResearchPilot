import os
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import qa, retrieval
from .bm25 import mark_dirty
from .config import settings
from .db import Base, Chunk, Document, ExtractionRecord, SessionLocal, StressTestRecord, engine
from .extraction.compare import compare_extractions
from .extraction.extract import run_extraction
from .indexing import delete_document_vectors, ensure_collection, warm_up
from .ingestion.validate import InvalidPDF, validate_pdf
from .pipeline import ingest_document
from .stress.engine import run_stress_test
from .rerank import warm_up as warm_up_reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)  # MVP only; switch to Alembic migrations later
    # Anything still queued/processing was interrupted by a restart or crash.
    with SessionLocal() as db:
        db.query(Document).filter(Document.status.in_(["queued", "processing"])).update(
            {"status": "failed", "error": "Interrupted by a server restart; delete and re-upload."},
            synchronize_session=False)
        db.commit()
    warm_up()           # embedding model, loaded once in the main thread
    warm_up_reranker()  # cross-encoder, same reason
    ensure_collection()
    os.makedirs(settings.upload_dir, exist_ok=True)
    yield


app = FastAPI(title="ResearchPilot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    with SessionLocal() as db:
        yield db


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", status_code=202)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...),
                          db: Session = Depends(get_db)):
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    try:
        pages = validate_pdf(data)
    except InvalidPDF as e:
        raise HTTPException(status_code=400, detail=str(e))

    doc_id = str(uuid.uuid4())
    path = os.path.join(settings.upload_dir, f"{doc_id}.pdf")  # generated name, never user input
    with open(path, "wb") as f:
        f.write(data)
    db.add(Document(id=doc_id, filename=(file.filename or "upload.pdf")[:255],
                    stored_path=path, page_count=pages, status="queued"))
    db.commit()
    background.add_task(ingest_document, doc_id, data)
    return {"id": doc_id, "status": "queued"}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [{"id": d.id, "filename": d.filename, "title": d.title, "status": d.status,
             "error": d.error, "page_count": d.page_count} for d in docs]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    d = db.get(Document, doc_id)
    if not d:
        raise HTTPException(404, "Document not found")
    return {"id": d.id, "filename": d.filename, "title": d.title, "status": d.status,
            "error": d.error, "page_count": d.page_count}


@app.post("/documents/{doc_id}/reingest", status_code=202)
def reingest_document(doc_id: str, background: BackgroundTasks, db: Session = Depends(get_db)):
    """Re-run ingestion from the stored PDF (e.g. after a failure or a chunking/extraction fix)."""
    d = db.get(Document, doc_id)
    if not d:
        raise HTTPException(404, "Document not found")
    if d.status in ("queued", "processing"):
        raise HTTPException(409, f"Document is already {d.status}")
    if not os.path.exists(d.stored_path):
        raise HTTPException(410, "Stored PDF is gone; please upload it again")
    delete_document_vectors(doc_id)
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
    db.query(ExtractionRecord).filter(ExtractionRecord.document_id == doc_id).delete()
    d.status, d.error = "queued", None
    db.commit()
    mark_dirty()
    with open(d.stored_path, "rb") as f:
        data = f.read()
    background.add_task(ingest_document, doc_id, data)
    return {"id": doc_id, "status": "queued"}


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    d = db.get(Document, doc_id)
    if not d:
        raise HTTPException(404, "Document not found")
    delete_document_vectors(doc_id)
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
    db.query(ExtractionRecord).filter(ExtractionRecord.document_id == doc_id).delete()
    if os.path.exists(d.stored_path):
        os.remove(d.stored_path)
    db.delete(d)
    db.commit()
    mark_dirty()


def _check_mode(mode: str) -> None:
    if mode not in retrieval.MODES:
        raise HTTPException(400, f"mode must be one of {list(retrieval.MODES)}")


@app.get("/search")
def search(q: str, k: int = 10, document_id: str | None = None,
           mode: str = "hybrid_rerank", db: Session = Depends(get_db)):
    """Retrieval inspection. Try mode=vector | bm25 | hybrid | hybrid_rerank on the same query
    and compare; each result shows its rank in each retriever."""
    _check_mode(mode)
    return retrieval.search(db, q, k=min(k, 30),
                            document_ids=[document_id] if document_id else None, mode=mode)


class AskRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None
    k: int = 6
    mode: str = "hybrid_rerank"


@app.post("/ask")
def ask(req: AskRequest, db: Session = Depends(get_db)):
    _check_mode(req.mode)
    passages = retrieval.search(db, req.question, k=min(req.k, 12),
                                document_ids=req.document_ids, mode=req.mode)
    if not passages:
        return {"answer": "No indexed documents to search.", "sufficient_evidence": False, "citations": []}
    try:
        result = qa.answer(req.question, passages)
    except Exception as e:  # provider errors (bad model name, rate limit, network)
        raise HTTPException(status_code=502, detail=f"LLM call failed ({type(e).__name__}): {str(e)[:300]}")
    result["retrieved"] = [{k: p[k] for k in ("chunk_id", "score", "vector_rank", "bm25_rank", "rerank_score")}
                           for p in passages]
    return result


# ---------------- Phase 4: structured extraction ----------------

@app.post("/documents/{doc_id}/extract")
def extract_document(doc_id: str, force: bool = False, db: Session = Depends(get_db)):
    """Extract problem/method/datasets/metrics/results/baselines/limitations, each with citations."""
    d = db.get(Document, doc_id)
    if not d:
        raise HTTPException(404, "Document not found")
    if d.status != "ready":
        raise HTTPException(409, f"Document is {d.status}; extraction needs status 'ready'")
    rec = db.get(ExtractionRecord, doc_id)
    if rec and not force:
        return rec.data
    data = run_extraction(db, doc_id)
    if rec:
        rec.data, rec.model = data, settings.llm_model
    else:
        db.add(ExtractionRecord(document_id=doc_id, data=data, model=settings.llm_model))
    db.commit()
    return data


@app.get("/documents/{doc_id}/extraction")
def get_extraction(doc_id: str, db: Session = Depends(get_db)):
    rec = db.get(ExtractionRecord, doc_id)
    if not rec:
        raise HTTPException(404, "No extraction yet; call POST /documents/{id}/extract first")
    return rec.data


@app.get("/matrix")
def matrix(document_ids: list[str] | None = Query(None), db: Session = Depends(get_db)):
    """Comparison matrix: one row per extracted paper (all of them, or the ids you pass)."""
    q = select(ExtractionRecord, Document).join(Document, Document.id == ExtractionRecord.document_id)
    if document_ids:
        q = q.where(ExtractionRecord.document_id.in_(document_ids))
    return [{"document_id": d.id, "title": d.title, "filename": d.filename, "fields": rec.data["fields"], "warnings": rec.data.get("warnings", [])}
            for rec, d in db.execute(q).all()]


@app.get("/compare")
def compare(doc_a: str, doc_b: str, db: Session = Depends(get_db)):
    a, b = db.get(ExtractionRecord, doc_a), db.get(ExtractionRecord, doc_b)
    if not a or not b:
        raise HTTPException(404, "Both documents need an extraction first")
    return compare_extractions(a.data, b.data)


# ---------------- Phase 5: evidence stress test ----------------

class StressTestRequest(BaseModel):
    claim: str = Field(min_length=10, max_length=1000)
    document_ids: list[str] | None = None
    k: int = Field(default=8, ge=2, le=12)  # passages sent to the model


@app.post("/stress-tests")
def create_stress_test(req: StressTestRequest, db: Session = Depends(get_db)):
    """Audit a claim against the retrieved evidence: per-passage relationship with a verified quote,
    comparability between supporting and challenging papers, and a rule-based overall assessment."""
    try:
        report = run_stress_test(db, req.claim, req.document_ids, req.k)
    except Exception as e:  # provider errors (bad model name, rate limit, network)
        raise HTTPException(status_code=502, detail=f"Stress test failed ({type(e).__name__}): {str(e)[:300]}")
    rec = StressTestRecord(claim=req.claim, data=report)
    db.add(rec)
    db.commit()
    report["id"] = rec.id
    rec.data = report
    db.commit()
    return report


@app.get("/stress-tests/{test_id}")
def get_stress_test(test_id: str, db: Session = Depends(get_db)):
    rec = db.get(StressTestRecord, test_id)
    if not rec:
        raise HTTPException(404, "Stress test not found")
    return rec.data


@app.get("/stress-tests")
def list_stress_tests(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.scalars(select(StressTestRecord).order_by(StressTestRecord.created_at.desc()).limit(min(limit, 100))).all()
    return [{"id": r.id, "claim": r.claim, "overall_assessment": r.data.get("overall_assessment"),
             "created_at": r.created_at} for r in rows]

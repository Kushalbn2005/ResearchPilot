import os
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import Base, Chunk, Document, SessionLocal, engine
from .indexing import delete_document_vectors, ensure_collection, vector_search, warm_up
from .ingestion.validate import InvalidPDF, validate_pdf
from .pipeline import ingest_document
from . import qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)  # MVP only; switch to Alembic migrations later
    # Anything still queued/processing was interrupted by a restart or crash.
    with SessionLocal() as db:
        db.query(Document).filter(Document.status.in_(["queued", "processing"])).update(
            {"status": "failed", "error": "Interrupted by a server restart; delete and re-upload."},
            synchronize_session=False)
        db.commit()
    warm_up()  # load the embedding model once, in the main thread
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


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    d = db.get(Document, doc_id)
    if not d:
        raise HTTPException(404, "Document not found")
    delete_document_vectors(doc_id)
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
    if os.path.exists(d.stored_path):
        os.remove(d.stored_path)
    db.delete(d)
    db.commit()


def _hydrate(hits: list[dict], db: Session) -> list[dict]:
    ids = [h["chunk_id"] for h in hits]
    rows = {c.id: c for c in db.scalars(select(Chunk).where(Chunk.id.in_(ids)))}
    return [{
        "chunk_id": h["chunk_id"], "score": round(h["score"], 4),
        "document_id": rows[h["chunk_id"]].document_id,
        "section": rows[h["chunk_id"]].section,
        "page_start": rows[h["chunk_id"]].page_start,
        "page_end": rows[h["chunk_id"]].page_end,
        "text": rows[h["chunk_id"]].text,
    } for h in hits if h["chunk_id"] in rows]


@app.get("/search")
def search(q: str, k: int = 10, document_id: str | None = None, db: Session = Depends(get_db)):
    """Retrieval inspection endpoint: look at what the retriever returns BEFORE adding an LLM."""
    hits = vector_search(q, k=min(k, 30), document_ids=[document_id] if document_id else None)
    return _hydrate(hits, db)


class AskRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None
    k: int = 6


@app.post("/ask")
def ask(req: AskRequest, db: Session = Depends(get_db)):
    hits = vector_search(req.question, k=min(req.k, 12), document_ids=req.document_ids)
    passages = _hydrate(hits, db)
    if not passages:
        return {"answer": "No indexed documents to search.", "sufficient_evidence": False, "citations": []}
    result = qa.answer(req.question, passages)
    result["retrieved"] = [{"chunk_id": p["chunk_id"], "score": p["score"]} for p in passages]
    return result
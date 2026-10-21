# ResearchPilot

**An evidence-grounded assistant for reading and comparing research papers.**
Upload scientific PDFs, ask questions, and get answers that point to the exact page and quoted text. Test a claim against several papers and see which ones support it, which challenge it, and whether their results can even be compared.

Author: Kushal ([@Kushalbn2005](https://github.com/Kushalbn2005))

---

## Why this exists

Language models often give fluent answers that are wrong or unsourced. For research, that is not acceptable, and comparing papers is harder still: two papers can report different numbers for good reasons (different data, different metrics), and calling that a "contradiction" is misleading.

ResearchPilot is built around three rules:

1. **Every answer must cite its source.** The model never writes page numbers; the server attaches them from the database.
2. **Nothing the model says is trusted blindly.** Citations, extracted facts and quotes are checked against the retrieved text.
3. **Different setups are not contradictions.** Before two papers are called conflicting, the system checks whether they share a dataset and a metric.

## Features

| Feature | What it does |
|---|---|
| **PDF ingestion** | Validates uploads, extracts text page by page, removes repeated headers and footers, detects sections, stops at the reference list, and cleans characters that break the database. |
| **Hybrid search** | Combines vector search (meaning), BM25 (exact words) and a cross-encoder reranker. Each retriever can be used on its own for comparison. |
| **Cited Q&A** | Answers only from retrieved passages, refuses when evidence is missing, and drops any citation it was not given. |
| **Structured extraction** | Pulls problem, method, datasets, metrics, results, baselines and limitations from each paper, and marks each item as verified only if its text appears in the cited passage. |
| **Comparison matrix** | Lines up extracted fields across papers and checks whether two papers share a verified dataset and metric. |
| **Evidence stress test** | Takes a claim and labels retrieved passages as supporting, challenging, contextualizing, unrelated or insufficient, each with a quote that is checked against the passage. The overall verdict comes from plain rules, not the model. |
| **Evaluation harness** | Measures retrieval (Recall@k, MRR), answer citations, refusal on unanswerable questions and stress-test accuracy against questions you label by hand. |

## How it works

```mermaid
flowchart LR
    A[PDF upload] --> B[Validate]
    B --> C[Extract text per page<br/>PyMuPDF]
    C --> D[Section-aware chunks]
    D --> E[(PostgreSQL<br/>chunks + metadata)]
    D --> F[(Qdrant<br/>vectors)]

    Q[Question or claim] --> G[Vector search]
    Q --> H[BM25 search]
    F --> G
    E --> H
    G --> I[Reciprocal Rank Fusion]
    H --> I
    I --> J[Cross-encoder rerank]
    J --> K[Cited answer<br/>+ citation validator]
    J --> L[Structured extraction<br/>+ verification]
    J --> M[Stress test<br/>+ quote checks]
    L --> N[Comparability check]
    N --> M
```

**Ingestion.** A PDF is checked (real PDF header, size and page limits, not encrypted), then processed in the background. Text is split into chunks of about 300 words with 40 words of overlap, never crossing a section boundary. Every chunk keeps its section and page range.

**Retrieval.** A query is searched two ways: vector similarity (BAAI/bge-small-en-v1.5, cosine) and BM25 keyword matching. The two ranked lists are merged with Reciprocal Rank Fusion, and the top 30 are re-scored by a cross-encoder that reads the question and each passage together.

**Answering.** The top passages are labelled P1, P2 and so on. The model returns JSON naming the passages it used. The server maps those labels back to real documents, sections and pages, drops any label it did not supply, and returns "insufficient evidence" if nothing valid is cited.

**Stress test.** For a claim, two searches run: one for the claim and one for counter-evidence. Each retrieved passage is labelled by the model with a verbatim quote. A quote that does not appear in the passage downgrades the label to `INSUFFICIENT`. Supporting and challenging papers are then compared using their extracted datasets and metrics. The final verdict (`supported`, `challenged`, `mixed`, `insufficient`) is decided by code, and a `mixed` result between non-comparable papers is reported as a comparability limitation, not a contradiction.

## Tech stack

Python 3.12, FastAPI, PostgreSQL (SQLAlchemy, psycopg 3), Qdrant, PyMuPDF, sentence-transformers (embeddings and reranker), rank-bm25, Groq (`openai/gpt-oss-120b`), Docker Compose, pytest, httpx.

## Project structure

```
ResearchPilot/
├── docker-compose.yml          # PostgreSQL + Qdrant
├── .env.example
├── backend/
│   ├── requirements.txt
│   ├── list_models.py          # prints Groq models your key can use
│   ├── app/
│   │   ├── main.py             # API endpoints and startup
│   │   ├── config.py           # settings from .env
│   │   ├── db.py               # tables: documents, chunks, extractions, stress_tests
│   │   ├── pipeline.py         # background ingestion job
│   │   ├── indexing.py         # embeddings + Qdrant
│   │   ├── bm25.py             # keyword search
│   │   ├── rerank.py           # cross-encoder reranker
│   │   ├── retrieval.py        # fusion and search modes
│   │   ├── qa.py               # prompt + citation validator
│   │   ├── llm.py              # the only file that calls the LLM
│   │   ├── ingestion/          # validate.py, extract.py, chunk.py
│   │   ├── extraction/         # schema.py, extract.py, compare.py
│   │   └── stress/             # engine.py
│   ├── evaluation/             # metrics.py, run_eval.py, show_misses.py, example data
│   └── tests/
└── frontend/                   # not built yet
```

## Getting started

**Prerequisites:** Python 3.12, Docker, a free [Groq](https://console.groq.com) API key.

```bash
git clone https://github.com/Kushalbn2005/<repo-name>.git
cd <repo-name>

cp .env.example .env            # then edit the password and add your Groq key
docker compose up -d            # PostgreSQL + Qdrant

cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
python -m pip install -r requirements.txt
python list_models.py           # pick a model your key can use, set LLM_MODEL in .env

python -m uvicorn app.main:app --reload --reload-dir app
```

Open `http://localhost:8000/docs` for the interactive API. The first start downloads the embedding model and the reranker, so it is slow once.

**Key settings in `.env`:**

```
POSTGRES_USER=researchpilot
POSTGRES_PASSWORD=change_me
POSTGRES_DB=researchpilot
DATABASE_URL=postgresql://researchpilot:change_me@localhost:5432/researchpilot
QDRANT_URL=http://localhost:6333
LLM_API_KEY=your_groq_key
LLM_MODEL=openai/gpt-oss-120b
```

**Tip:** always use `python -m pip` and `python -m uvicorn` so the packages and the server come from the same virtual environment.

## Using it

1. `POST /documents` with a PDF, then poll `GET /documents/{id}` until `status` is `ready`.
2. `GET /search?q=...&mode=hybrid_rerank` to inspect retrieval, or `POST /ask` for a cited answer.
3. `POST /documents/{id}/extract` for each paper, then `GET /matrix` and `GET /compare?doc_a=&doc_b=`.
4. `POST /stress-tests` with `{"claim": "..."}` to audit a claim across your papers.

### API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload a PDF (processed in the background) |
| GET | `/documents`, `/documents/{id}` | List documents, check status |
| POST | `/documents/{id}/reingest` | Re-run ingestion from the stored PDF |
| DELETE | `/documents/{id}` | Remove a document and its data |
| GET | `/search` | Retrieval with `mode=vector\|bm25\|hybrid\|hybrid_rerank` |
| POST | `/ask` | Cited answer, or "insufficient evidence" |
| POST | `/documents/{id}/extract` | Structured extraction (`?force=true` to redo) |
| GET | `/documents/{id}/extraction` | Read a saved extraction |
| GET | `/matrix` | Comparison matrix across extracted papers |
| GET | `/compare` | Do two papers share a verified dataset and metric? |
| POST | `/stress-tests` | Audit a claim against the evidence |
| GET | `/stress-tests`, `/stress-tests/{id}` | List or read saved reports |

## Testing

```bash
cd backend
python -m pytest tests -q
```

The unit tests need no database, models or API key. They cover PDF validation, extraction cleanup, chunking, the citation validator, BM25 and fusion, field verification, comparability, stress-test validation and aggregation, and the evaluation metrics (including a full evaluation run against a fake server).

## Evaluation

Quality is measured, not assumed. The harness in `backend/evaluation/` runs against your own hand-labelled questions and claims. See [`backend/evaluation/README.md`](backend/evaluation/README.md) for how to label data.

```bash
python -m evaluation.run_eval --skip-llm          # retrieval only
python -m evaluation.run_eval --sleep 6           # full run
python -m evaluation.show_misses --mode hybrid_rerank --split dev
```

**Results** (fill in from your final run on the held-out `test` split; report the number of questions next to every figure):

| Retrieval mode | n | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| vector | | | | |
| bm25 | | | | |
| hybrid | | | | |
| hybrid + rerank | | | | |

## Design decisions

- **Hybrid retrieval.** Embeddings capture meaning but can miss exact terms such as model names; BM25 matches exact terms but misses paraphrases. Evaluation on papers from one topic showed that exact-term matching matters a lot.
- **Verify, don't trust.** Items count as verified only if their text appears, as whole words, in the cited passage. This catches hallucinations such as a model name appearing only inside another word.
- **Rules decide the verdict.** The model labels passages; plain code aggregates them. That keeps the final result explainable and testable.
- **Comparability before contradiction.** Results on different datasets or metrics are flagged as not comparable instead of being called conflicts.
- **Postgres plus Qdrant.** Text and metadata live in Postgres, vectors in Qdrant; chunk ids tie them together.
- **CPU-only embeddings, one model load.** Models load once at startup and run one call at a time, which avoids crashes seen with multi-threaded PyTorch on macOS.

## Known limitations

- Tables are extracted as flat text, so per-method result numbers can be misattributed. Treat `verified` as "this text appears in the cited passage", not "this is correct".
- Section detection is heuristic (font size, bold text, known headings) and can mislabel a journal banner as a section.
- No OCR: scanned PDFs are rejected.
- Comparability checks cover dataset and metric only, not splits, preprocessing or baselines.
- Single user, no authentication, and the BM25 index is rebuilt in memory after changes, so this targets small collections.
- Groq's free tier has rate limits; the client retries but long evaluation runs need `--sleep`.

## Roadmap

- Parse tables directly for reliable per-method results
- Web UI (React): upload, search, comparison matrix, evidence audit view
- Orchestration of the agent workflow (e.g. LangGraph) with bounded retries
- Compare splits and baselines in the comparability check
- Authentication, database migrations, deployment

## License

Add a license before publishing (MIT is a common choice).

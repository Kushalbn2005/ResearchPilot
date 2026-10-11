import os

os.environ.setdefault("DATABASE_URL", "postgresql://x:x@localhost/x")

from app.qa import parse_and_validate

PASSAGES = [
    {"document_id": "d1", "section": "Results", "page_start": 4, "page_end": 4, "text": "..."},
    {"document_id": "d2", "section": "Methods", "page_start": 2, "page_end": 3, "text": "..."},
]


def test_valid_citation_gets_metadata_from_server():
    raw = '{"answer": "RF won.", "citations": ["P1"], "sufficient_evidence": true}'
    out = parse_and_validate(raw, PASSAGES)
    assert out["sufficient_evidence"] and out["citations"][0]["page_start"] == 4


def test_invented_citation_ids_are_dropped_and_answer_refused():
    raw = '{"answer": "x", "citations": ["P9", "abc"], "sufficient_evidence": true}'
    assert parse_and_validate(raw, PASSAGES)["sufficient_evidence"] is False


def test_garbage_and_fenced_json():
    assert parse_and_validate("not json", PASSAGES)["sufficient_evidence"] is False
    fenced = '```json\n{"answer": "ok", "citations": ["P2"], "sufficient_evidence": true}\n```'
    assert parse_and_validate(fenced, PASSAGES)["citations"][0]["id"] == "P2"
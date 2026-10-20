import json

import httpx
import pytest

from evaluation.metrics import (aggregate, citation_support, classification_report, keyword_hit,
                                matches_gold, retrieval_scores, validate_dataset)
from evaluation.run_eval import evaluate, render_markdown

FILES = {"dA": "a.pdf", "dB": "b.pdf"}
filename_of = FILES.get


def chunk(doc, start, end, text=""):
    return {"document_id": doc, "page_start": start, "page_end": end, "text": text}


def test_matches_gold_by_page_phrase_and_file():
    g = {"file": "a.pdf", "pages": [7], "phrases": ["Diebold-Mariano"]}
    assert matches_gold(chunk("dA", 6, 7), g, filename_of)                       # page overlap
    assert matches_gold(chunk("dA", 1, 1, "we ran the Diebold Mariano test"), g, filename_of)  # phrase
    assert not matches_gold(chunk("dA", 1, 2, "nothing"), g, filename_of)
    assert not matches_gold(chunk("dB", 7, 7), g, filename_of)                   # wrong file


def test_retrieval_scores_use_rank_and_gold_coverage():
    golds = [{"file": "a.pdf", "pages": [3]}, {"file": "b.pdf", "pages": [9]}]
    results = [chunk("dA", 1, 1)] * 2 + [chunk("dA", 3, 3)] + [chunk("dA", 1, 1)] * 3 + [chunk("dB", 9, 9)]
    s = retrieval_scores(results, golds, filename_of)
    assert s["rr"] == pytest.approx(1 / 3)
    assert s["recall@5"] == 0.5 and s["recall@10"] == 1.0
    assert retrieval_scores([], [golds[0]], filename_of) == {"recall@5": 0.0, "recall@10": 0.0, "rr": 0.0}
    assert aggregate([{"recall@5": 1, "recall@10": 1, "rr": 1}, {"recall@5": 0, "recall@10": 1, "rr": 0.5}])["rr"] == 0.75


def test_keyword_and_citation_helpers():
    assert keyword_hit("SVR was best, with an RMSE of 0.29", ["SVR", "RMSE"]) is True
    assert keyword_hit("The Informer model", ["SVR"]) is False
    assert keyword_hit("anything", None) is None
    golds = [{"file": "a.pdf", "pages": [3]}]
    cites = [{"document_id": "dA", "page_start": 3, "page_end": 4}, {"document_id": "dB", "page_start": 3, "page_end": 3}]
    assert citation_support(cites, golds, filename_of) == (1, 2)
    assert citation_support(cites, [{"file": "a.pdf", "phrases": ["x"]}], filename_of) is None


def test_classification_report():
    r = classification_report(["supported", "supported", "mixed", "insufficient"],
                              ["supported", "mixed", "mixed", "insufficient"])
    assert r["accuracy"] == 0.75 and r["confusion"]["supported"]["mixed"] == 1
    assert r["per_class"]["mixed"]["precision"] == 0.5 and r["per_class"]["mixed"]["recall"] == 1.0
    assert r["per_class"]["challenged"]["support"] == 0


def test_validate_dataset_catches_common_mistakes():
    bad = [{"id": "q1", "question": "x", "answerable": True},                                     # no gold
           {"id": "q1", "question": "y", "answerable": False, "gold": [{"file": "a.pdf", "pages": [1]}]},  # dup id + gold
           {"id": "q2", "question": "z", "answerable": True, "gold": [{"file": "a.pdf"}]},          # gold w/o pages/phrases
           {"id": "q3", "question": "w", "answerable": True, "type": "weird", "gold": [{"file": "a.pdf", "pages": [1]}]}]
    problems = validate_dataset(bad, [{"id": "c1", "claim": "short", "expected": "maybe"}])
    assert len(problems) >= 6
    assert validate_dataset([{"id": "q", "question": "ok", "answerable": False}]) == []


def fake_server():
    """A tiny fake ResearchPilot API: 'vector' ranks the right chunk 2nd, everything else ranks it 1st."""
    docs = [{"id": "dA", "filename": "a.pdf", "status": "ready"}, {"id": "dB", "filename": "b.pdf", "status": "ready"}]
    good, bad = chunk("dA", 3, 3, "SVR achieved the best RMSE"), chunk("dB", 1, 1, "unrelated")

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/documents":
            return httpx.Response(200, json=docs)
        if path == "/search":
            return httpx.Response(200, json=[bad, good] if req.url.params["mode"] == "vector" else [good, bad])
        if path == "/ask":
            q = json.loads(req.content)["question"]
            if "ImageNet" in q:
                return httpx.Response(200, json={"answer": "Insufficient evidence.", "sufficient_evidence": False, "citations": []})
            return httpx.Response(200, json={"answer": "SVR was best.", "sufficient_evidence": True,
                                             "citations": [{"document_id": "dA", "page_start": 3, "page_end": 3}]})
        if path == "/stress-tests":
            claim = json.loads(req.content)["claim"]
            return httpx.Response(200, json={"overall_assessment": "supported" if "SVR" in claim else "insufficient",
                                             "direct_conflict": False, "evidence": []})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://fake")


def test_evaluate_end_to_end_with_fake_server():
    questions = [
        {"id": "q1", "question": "Which model was best?", "answerable": True,
         "gold": [{"file": "a.pdf", "pages": [3]}], "answer_keywords": ["SVR"]},
        {"id": "q2", "question": "ImageNet accuracy?", "answerable": False},
    ]
    claims = [{"id": "c1", "claim": "SVR is the best model for forecasting.", "expected": "supported", "expected_direct_conflict": False},
              {"id": "c2", "claim": "Random forests beat every deep model.", "expected": "mixed"}]
    res = evaluate(fake_server(), questions, claims, ask_modes=("hybrid_rerank",), sleep=0, retry_wait=0)

    assert res["retrieval"]["vector"]["overall"]["rr"] == 0.5
    assert res["retrieval"]["hybrid_rerank"]["overall"]["rr"] == 1.0
    g = res["generation"]["hybrid_rerank"]
    assert g["keyword_accuracy"] == 1.0 and g["citation_precision"] == 1.0 and g["abstention_accuracy"] == 1.0
    assert res["stress"]["accuracy"] == 0.5 and res["stress"]["false_contradictions"] == 0
    md = render_markdown(res)
    assert "| hybrid_rerank |" in md and "Confusion matrix" in md


def test_evaluate_skip_llm_and_missing_gold_file_warning():
    q = [{"id": "q1", "question": "x?", "answerable": True, "gold": [{"file": "missing.pdf", "pages": [1]}]}]
    res = evaluate(fake_server(), q, skip_llm=True)
    assert "generation" not in res and any("missing.pdf" in w for w in res["warnings"])

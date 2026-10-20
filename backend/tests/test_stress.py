import json

from app.stress import engine
from app.stress.engine import aggregate, attach_notes, build_pairs, interleave, select_passages, validate_stress

P = lambda doc, cid, text, page=1: {"document_id": doc, "chunk_id": cid, "section": "Results", "page_start": page,
                                    "page_end": page, "text": text, "score": 0.5, "rerank_score": 0.9}
PASSAGES = [
    P("A", "c1", "On the Karnataka rainfall data, Random Forest achieved an RMSE of 0.42, better than LSTM."),
    P("B", "c2", "In our experiments on the Yellow River data, LSTM clearly outperformed Random Forest on F1."),
    P("B", "c3", "The weather stations are located along the river basin."),
]


def raw(entries, follow=None):
    return json.dumps({"evidence": entries, "follow_up_questions": follow or []})


def entry(i, rel, quote="", why="because"):
    return {"id": f"P{i}", "relationship": rel, "quote": quote, "rationale": why}


# ---------------- selection

def test_select_passages_caps_each_paper():
    hits = [P("A", f"a{i}", "x") for i in range(5)] + [P("B", "b1", "y")]
    out = select_passages(hits, per_doc=2, max_total=8)
    assert [h["chunk_id"] for h in out] == ["a0", "a1", "b1"]


def test_interleave_dedupes_and_alternates():
    a, b = [P("A", "1", ""), P("A", "2", "")], [P("A", "2", ""), P("B", "3", "")]
    assert [h["chunk_id"] for h in interleave(a, b)] == ["1", "2", "3"]


# ---------------- validation

def test_verified_quote_keeps_label_and_citation_data():
    ev, _, warnings = validate_stress(raw([
        entry(1, "SUPPORTS", "Random Forest achieved an RMSE of 0.42"),
        entry(2, "CHALLENGES", "LSTM clearly outperformed Random Forest on F1"),
        entry(3, "UNRELATED"),
    ]), PASSAGES)
    assert [e["relationship"] for e in ev] == ["SUPPORTS", "CHALLENGES", "UNRELATED"]
    assert ev[0]["quote_verified"] is True and ev[0]["page_number"] == 1 and ev[0]["document_id"] == "A"
    assert warnings == []


def test_fabricated_quote_downgrades_to_insufficient():
    ev, _, warnings = validate_stress(raw([entry(1, "SUPPORTS", "Random Forest beats everything by a mile")]), PASSAGES)
    assert ev[0]["relationship"] == "INSUFFICIENT" and ev[0]["downgraded_from"] == "SUPPORTS"
    assert ev[0]["quote"] == "" and any("downgraded" in w for w in warnings)


def test_missing_short_or_invalid_entries_are_handled():
    ev, _, warnings = validate_stress(raw([entry(1, "SUPPORTS", "RMSE"),          # quote too short
                                           entry(2, "MAYBE", "x y z w"),           # invalid label
                                           entry(9, "SUPPORTS", "ignored id")]),   # id we never supplied
                                      PASSAGES)
    assert [e["relationship"] for e in ev] == ["INSUFFICIENT"] * 3
    assert len(warnings) == 3


def test_garbage_output_gives_all_insufficient():
    ev, _, warnings = validate_stress("nope", PASSAGES)
    assert all(e["relationship"] == "INSUFFICIENT" for e in ev) and warnings


def test_ligatures_and_hyphen_variants_do_not_break_quote_check():
    passages = [P("A", "c1", "The \ufb01nal model reached state\u2011of\u2011the\u2011art accuracy on the Karnataka data.")]
    ev, _, _ = validate_stress(raw([entry(1, "SUPPORTS", "The final model reached state-of-the-art accuracy")]), passages)
    assert ev[0]["relationship"] == "SUPPORTS"


# ---------------- comparability + aggregation

def ext(datasets, metrics):
    mk = lambda xs: {"value": [{"text": t, "verified": True} for t in xs], "citations": []}
    return {"fields": {"datasets": mk(datasets), "metrics": mk(metrics)}}


def evidence_two_sided():
    ev, _, _ = validate_stress(raw([
        entry(1, "SUPPORTS", "Random Forest achieved an RMSE of 0.42"),
        entry(2, "CHALLENGES", "LSTM clearly outperformed Random Forest on F1"),
        entry(3, "UNRELATED")]), PASSAGES)
    return ev


def test_mixed_but_not_comparable_is_not_called_a_contradiction():
    ev = evidence_two_sided()
    exts = {"A": ext(["Karnataka rainfall dataset"], ["RMSE"]), "B": ext(["Yellow River data"], ["F1"])}
    pairs = build_pairs(ev, exts)
    attach_notes(ev, pairs, exts, {"A": "Paper A", "B": "Paper B"})
    assessment, direct, limits = aggregate(ev, pairs, papers_searched=2)
    assert assessment == "mixed" and direct is False
    assert any("not a demonstrated contradiction" in l for l in limits)
    assert "Not shown to be comparable with 'Paper B'" in ev[0]["comparability_notes"][0]
    assert ev[2]["comparability_notes"] == []


def test_comparable_pair_is_flagged_as_direct_conflict():
    ev = evidence_two_sided()
    exts = {"A": ext(["Yellow River data"], ["F1"]), "B": ext(["Yellow River dataset"], ["F1"])}
    pairs = build_pairs(ev, exts)
    attach_notes(ev, pairs, exts, {"A": "Paper A", "B": "Paper B"})
    assessment, direct, limits = aggregate(ev, pairs, 2)
    assert assessment == "mixed" and direct is True
    assert not any("not a demonstrated contradiction" in l for l in limits)
    assert ev[0]["comparability_notes"][0].startswith("Comparable with 'Paper B'")


def test_missing_extraction_means_unknown_not_conflict():
    ev = evidence_two_sided()
    pairs = build_pairs(ev, {"A": ext(["x"], ["y"])})
    assert pairs[0]["comparable"] is None
    assert aggregate(ev, pairs, 2)[1] is False


def test_other_assessments():
    only_support, _, _ = validate_stress(raw([entry(1, "SUPPORTS", "Random Forest achieved an RMSE of 0.42")]), PASSAGES)
    a, _, limits = aggregate(only_support, [], 1)
    assert a == "supported" and any("single paper" in l for l in limits)
    nothing, _, _ = validate_stress(raw([entry(1, "UNRELATED"), entry(2, "INSUFFICIENT")]), PASSAGES)
    assert aggregate(nothing, [], 1)[0] == "insufficient"
    only_chal, _, _ = validate_stress(raw([entry(2, "CHALLENGES", "LSTM clearly outperformed Random Forest on F1")]), PASSAGES)
    assert aggregate(only_chal, [], 1)[0] == "challenged"


# ---------------- end to end with fakes (no DB, no models, no LLM)

def test_run_stress_test_end_to_end(monkeypatch):
    from app import llm, retrieval

    monkeypatch.setattr(retrieval, "search", lambda db, q, k, document_ids=None, mode="hybrid_rerank": PASSAGES)
    monkeypatch.setattr(llm, "complete", lambda system, user, **kw: raw([
        entry(1, "SUPPORTS", "Random Forest achieved an RMSE of 0.42"),
        entry(2, "CHALLENGES", "LSTM clearly outperformed Random Forest on F1"),
        entry(3, "UNRELATED")], follow=["Which model wins on a shared dataset?"]))
    monkeypatch.setattr(engine, "_load_extractions",
                        lambda db, ids: {"A": ext(["Karnataka rainfall dataset"], ["RMSE"]), "B": ext(["Yellow River data"], ["F1"])})
    monkeypatch.setattr(engine, "_load_titles", lambda db, ids: {"A": "Paper A", "B": "Paper B"})
    monkeypatch.setattr(engine, "_count_papers", lambda db, ids: 2)

    report = engine.run_stress_test(None, "Random Forest consistently outperforms LSTM for drought prediction.")
    assert report["overall_assessment"] == "mixed" and report["direct_conflict"] is False
    assert report["follow_up_questions"] == ["Which model wins on a shared dataset?"]
    assert report["evidence"][0]["document_title"] == "Paper A"
    assert report["comparability"][0]["comparable"] is False

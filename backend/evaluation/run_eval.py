"""Run the evaluation against a running ResearchPilot server.

    python -m evaluation.run_eval                      # everything
    python -m evaluation.run_eval --skip-llm           # retrieval only (no Groq calls)
    python -m evaluation.run_eval --ask-modes vector,hybrid_rerank --sleep 5

Requires the server (uvicorn) to be running and your papers to be uploaded with status 'ready'.
"""
import argparse
import datetime
import json
import pathlib
import time

import httpx

from .metrics import (ASSESSMENTS, aggregate, citation_support, classification_report, keyword_hit, mean,
                      retrieval_scores, validate_dataset)

MODES = ["vector", "bm25", "hybrid", "hybrid_rerank"]


class EvalError(RuntimeError):
    pass


def _request(client, method, path, retry_wait=10.0, **kw):
    """Call the API; retry rate limits / provider hiccups a couple of times."""
    last = None
    for attempt in range(3):
        resp = client.request(method, path, **kw)
        if resp.status_code in (429, 502, 503):
            last = f"{resp.status_code}: {resp.text[:200]}"
            time.sleep(retry_wait * (attempt + 1))
            continue
        if resp.status_code >= 400:
            raise EvalError(f"{method} {path} -> {resp.status_code}: {resp.text[:200]}")
        return resp.json()
    raise EvalError(f"{method} {path} kept failing ({last})")


def _split_rows(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return {k: aggregate(v) for k, v in out.items()}


def evaluate(client, questions, claims=None, ask_modes=("vector", "hybrid_rerank"),
             skip_llm=False, sleep=0.0, retry_wait=10.0) -> dict:
    problems = validate_dataset(questions, claims)
    if problems:
        raise ValueError("Dataset problems:\n- " + "\n- ".join(problems))
    for q in questions:
        q.setdefault("type", "factual")
        q.setdefault("split", "dev")

    docs = _request(client, "GET", "/documents", retry_wait)
    id_to_file = {d["id"]: d["filename"] for d in docs if d["status"] == "ready"}
    filename_of = id_to_file.get
    file_to_id = {v: k for k, v in id_to_file.items()}
    wanted = {g["file"] for q in questions if q["answerable"] for g in q["gold"]}
    warnings = [f"Gold file not found among ready documents: {f}" for f in sorted(wanted - set(file_to_id))]

    results: dict = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                     "collection": sorted(file_to_id), "warnings": warnings,
                     "n_questions": len(questions), "n_answerable": sum(q["answerable"] for q in questions)}

    # ---------------- retrieval (no LLM)
    retrieval = {}
    for mode in MODES:
        rows = []
        for q in questions:
            if not q["answerable"]:
                continue
            hits = _request(client, "GET", "/search", retry_wait, params={"q": q["question"], "k": 10, "mode": mode})
            rows.append({"id": q["id"], "type": q["type"], "split": q["split"],
                         **retrieval_scores(hits, q["gold"], filename_of)})
        retrieval[mode] = {"overall": aggregate(rows), "by_type": _split_rows(rows, "type"),
                           "by_split": _split_rows(rows, "split"), "rows": rows}
    results["retrieval"] = retrieval
    if skip_llm:
        return results

    # ---------------- answers and citations
    generation = {}
    for mode in ask_modes:
        rows, review = [], []
        for q in questions:
            try:
                r = _request(client, "POST", "/ask", retry_wait, json={"question": q["question"], "k": 6, "mode": mode})
            except EvalError as e:
                rows.append({"id": q["id"], "error": str(e)})
                continue
            time.sleep(sleep)
            answered = bool(r.get("sufficient_evidence"))
            row = {"id": q["id"], "type": q["type"], "split": q["split"], "answerable": q["answerable"],
                   "answered": answered}
            if q["answerable"] and answered:
                row["keyword_hit"] = keyword_hit(r.get("answer", ""), q.get("answer_keywords"))
                row["citation_support"] = citation_support(r.get("citations", []), q["gold"], filename_of)
            rows.append(row)
            review.append({"id": q["id"], "question": q["question"], "answerable": q["answerable"],
                           "answer": r.get("answer"), "citations": r.get("citations"),
                           "your_grade": None})
        ok = [r for r in rows if "error" not in r]
        ans = [r for r in ok if r["answerable"]]
        unans = [r for r in ok if not r["answerable"]]
        kw = [r["keyword_hit"] for r in ans if r.get("keyword_hit") is not None]
        sup = [r["citation_support"] for r in ans if r.get("citation_support")]
        generation[mode] = {
            "answerable_n": len(ans), "answered_rate": mean(r["answered"] for r in ans),
            "keyword_accuracy": mean(kw), "keyword_n": len(kw),
            "citation_precision": (sum(s for s, _ in sup) / sum(t for _, t in sup)) if sum(t for _, t in sup) else None,
            "citation_n": sum(t for _, t in sup),
            "unanswerable_n": len(unans), "abstention_accuracy": mean(not r["answered"] for r in unans),
            "errors": len(rows) - len(ok), "rows": rows, "review": review,
        }
    results["generation"] = generation

    # ---------------- stress tests
    if claims:
        rows = []
        for c in claims:
            ids = [file_to_id[f] for f in c.get("documents", []) if f in file_to_id] or None
            try:
                rep = _request(client, "POST", "/stress-tests", retry_wait,
                               json={"claim": c["claim"], "document_ids": ids, "k": 8})
            except EvalError as e:
                rows.append({"id": c["id"], "expected": c["expected"], "error": str(e)})
                continue
            time.sleep(sleep)
            rows.append({"id": c["id"], "expected": c["expected"], "predicted": rep["overall_assessment"],
                         "direct_conflict": rep.get("direct_conflict"),
                         "expected_direct_conflict": c.get("expected_direct_conflict"),
                         "downgrades": sum(1 for e in rep["evidence"] if e.get("downgraded_from"))})
        ok = [r for r in rows if "error" not in r]
        report = classification_report([r["expected"] for r in ok], [r["predicted"] for r in ok])
        labelled = [r for r in ok if r["expected_direct_conflict"] is not None]
        report["false_contradictions"] = sum(1 for r in labelled if not r["expected_direct_conflict"] and r["direct_conflict"])
        report["false_contradiction_n"] = len(labelled)
        report["errors"] = len(rows) - len(ok)
        report["rows"] = rows
        results["stress"] = report
    return results


# ---------------------------------------------------------------- report

def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_markdown(res: dict) -> str:
    L = [f"# ResearchPilot evaluation ({res['created']})", "",
         f"Collection: {len(res['collection'])} paper(s); {res['n_questions']} questions "
         f"({res['n_answerable']} answerable).", ""]
    for w in res["warnings"]:
        L.append(f"> Warning: {w}")
    L += ["", "## Retrieval (answerable questions)", "",
          "| Mode | n | Recall@5 | Recall@10 | MRR@10 |", "|---|---|---|---|---|"]
    for mode, r in res["retrieval"].items():
        o = r["overall"]
        L.append(f"| {mode} | {o['n']} | {_pct(o['recall@5'])} | {_pct(o['recall@10'])} | "
                 f"{'n/a' if o['rr'] is None else format(o['rr'], '.3f')} |")
    types = sorted({t for r in res["retrieval"].values() for t in r["by_type"]})
    if types:
        L += ["", "Recall@5 by question type:", "", "| Type | " + " | ".join(res["retrieval"]) + " |",
              "|---|" + "---|" * len(res["retrieval"])]
        for t in types:
            L.append(f"| {t} | " + " | ".join(
                _pct(r["by_type"].get(t, {}).get("recall@5")) for r in res["retrieval"].values()) + " |")
    if "generation" in res:
        L += ["", "## Answers and citations", "",
              "| Retrieval mode | Answered (answerable) | Keyword accuracy* | Citation precision** | Abstained correctly (unanswerable) |",
              "|---|---|---|---|---|"]
        for mode, g in res["generation"].items():
            L.append(f"| {mode} | {_pct(g['answered_rate'])} (n={g['answerable_n']}) | {_pct(g['keyword_accuracy'])} (n={g['keyword_n']}) "
                     f"| {_pct(g['citation_precision'])} ({g['citation_n']} cites) | {_pct(g['abstention_accuracy'])} (n={g['unanswerable_n']}) |")
        L += ["", "*Keyword accuracy is a proxy: all expected keywords appear in the answer. Read the answers too.",
              "**Share of citations whose file and page match a gold page. Needs gold pages."]
    if "stress" in res:
        s = res["stress"]
        L += ["", "## Evidence stress test", "",
              f"Claims: {s['n']}  |  Accuracy: {_pct(s['accuracy'])}  |  Macro-F1: {_pct(s['macro_f1'])}", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for lab in ASSESSMENTS:
            c = s["per_class"][lab]
            L.append(f"| {lab} | {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['f1'])} | {c['support']} |")
        L += ["", "Confusion matrix (rows = expected, columns = predicted):", "",
              "| expected \\ predicted | " + " | ".join(ASSESSMENTS) + " |", "|---|" + "---|" * len(ASSESSMENTS)]
        for t in ASSESSMENTS:
            L.append(f"| {t} | " + " | ".join(str(s["confusion"][t][p]) for p in ASSESSMENTS) + " |")
        if s["false_contradiction_n"]:
            L += ["", f"False contradictions (non-comparable setups called a direct conflict): "
                      f"{s['false_contradictions']} of {s['false_contradiction_n']} labelled claims."]
    L += ["", "Small samples give noisy numbers; report n next to every figure."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--questions", default="evaluation/questions.json")
    ap.add_argument("--claims", default="evaluation/claims.json")
    ap.add_argument("--skip-llm", action="store_true", help="retrieval metrics only")
    ap.add_argument("--ask-modes", default="vector,hybrid_rerank")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between LLM calls (rate limits)")
    ap.add_argument("--out", default="evaluation/results")
    a = ap.parse_args()

    qpath, cpath = pathlib.Path(a.questions), pathlib.Path(a.claims)
    if not qpath.exists():
        raise SystemExit(f"{qpath} not found. Copy evaluation/questions.example.json to {qpath} and fill it in.")
    questions = json.loads(qpath.read_text())["questions"]
    claims = json.loads(cpath.read_text())["claims"] if cpath.exists() and not a.skip_llm else None

    with httpx.Client(base_url=a.base, timeout=240) as client:
        res = evaluate(client, questions, claims, tuple(a.ask_modes.split(",")), a.skip_llm, a.sleep)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}.json").write_text(json.dumps(res, indent=2))
    md = render_markdown(res)
    (out / f"{stamp}.md").write_text(md)
    print(md)
    print(f"\nSaved {out / (stamp + '.json')} and .md")


if __name__ == "__main__":
    main()

"""Pure metric functions for the evaluation harness (no network, no models), so they are unit-tested."""
from app.extraction.schema import contains_words, words

QUESTION_TYPES = ("factual", "exact_term", "comparison", "unanswerable")
ASSESSMENTS = ("supported", "challenged", "mixed", "insufficient")
SPLITS = ("dev", "test")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


# ------------------------------------------------------------------ dataset validation

def validate_dataset(questions: list[dict], claims: list[dict] | None = None) -> list[str]:
    problems: list[str] = []
    seen = set()
    for q in questions:
        qid = q.get("id", "(no id)")
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        if not str(q.get("question", "")).strip():
            problems.append(f"{qid}: empty question")
        if q.get("type", "factual") not in QUESTION_TYPES:
            problems.append(f"{qid}: type must be one of {QUESTION_TYPES}")
        if q.get("split", "dev") not in SPLITS:
            problems.append(f"{qid}: split must be one of {SPLITS}")
        if not isinstance(q.get("answerable"), bool):
            problems.append(f"{qid}: 'answerable' must be true or false")
            continue
        if q["answerable"]:
            golds = q.get("gold") or []
            if not golds:
                problems.append(f"{qid}: answerable questions need at least one 'gold' entry")
            for g in golds:
                if not g.get("file") or not (g.get("pages") or g.get("phrases")):
                    problems.append(f"{qid}: each gold entry needs 'file' and 'pages' and/or 'phrases'")
        elif q.get("gold"):
            problems.append(f"{qid}: unanswerable questions should not have 'gold'")
    for c in claims or []:
        cid = c.get("id", "(no id)")
        if len(str(c.get("claim", ""))) < 10:
            problems.append(f"{cid}: claim is too short")
        if c.get("expected") not in ASSESSMENTS:
            problems.append(f"{cid}: expected must be one of {ASSESSMENTS}")
    return problems


# ------------------------------------------------------------------ retrieval

def matches_gold(chunk: dict, gold: dict, filename_of) -> bool:
    """A retrieved chunk is relevant to a gold entry if it is from the right file AND either overlaps one of the
    gold pages or contains one of the gold phrases (whole words). Page numbers are PDF page indexes (1-based)."""
    if filename_of(chunk["document_id"]) != gold["file"]:
        return False
    pages = gold.get("pages") or []
    if any(chunk["page_start"] <= p <= chunk["page_end"] for p in pages):
        return True
    text_words = words(chunk.get("text", ""))
    return any(contains_words(text_words, ph) for ph in gold.get("phrases") or [])


def retrieval_scores(results: list[dict], golds: list[dict], filename_of) -> dict:
    """Recall@k = share of gold entries covered by the top-k results (for single-gold questions this is hit rate).
    rr = reciprocal rank of the first result relevant to any gold entry within the top 10."""
    def covered(k: int) -> float:
        return sum(any(matches_gold(r, g, filename_of) for r in results[:k]) for g in golds) / len(golds)

    rr = 0.0
    for rank, r in enumerate(results[:10], start=1):
        if any(matches_gold(r, g, filename_of) for g in golds):
            rr = 1.0 / rank
            break
    return {"recall@5": covered(5), "recall@10": covered(10), "rr": rr}


def aggregate(rows: list[dict], keys=("recall@5", "recall@10", "rr")) -> dict:
    return {"n": len(rows), **{k: mean(r[k] for r in rows) for k in keys}}


# ------------------------------------------------------------------ answers and citations

def keyword_hit(answer: str, keywords: list[str] | None):
    """True when every expected keyword appears in the answer (a cheap proxy for correctness; not a substitute
    for reading the answers). None when the question has no keywords."""
    if not keywords:
        return None
    aw = words(answer)
    return all(contains_words(aw, k) for k in keywords)


def citation_support(citations: list[dict], golds: list[dict], filename_of):
    """(supported, total) citations whose file and pages match a gold entry. None if no gold entry has pages
    (we cannot judge page-level support without them)."""
    page_golds = [g for g in golds if g.get("pages")]
    if not page_golds:
        return None
    supported = 0
    for c in citations:
        if any(filename_of(c["document_id"]) == g["file"]
               and any(c["page_start"] <= p <= c["page_end"] for p in g["pages"]) for g in page_golds):
            supported += 1
    return supported, len(citations)


# ------------------------------------------------------------------ classification (stress test)

def classification_report(y_true: list[str], y_pred: list[str], labels=ASSESSMENTS) -> dict:
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1
    per_class = {}
    for lab in labels:
        tp = confusion[lab][lab]
        fp = sum(confusion[t][lab] for t in labels if t != lab)
        fn = sum(confusion[lab][p] for p in labels if p != lab)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if tp + fp + fn else None)
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    n = len(y_true)
    correct = sum(confusion[l][l] for l in labels)
    return {"n": n, "accuracy": correct / n if n else None,
            "macro_f1": mean(v["f1"] for v in per_class.values() if v["f1"] is not None),
            "per_class": per_class, "confusion": confusion}

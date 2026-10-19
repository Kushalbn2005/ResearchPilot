import re

from .schema import squash

_STOP = {"dataset", "datasets", "data", "the", "a", "of", "and"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


def similar(a: str, b: str) -> bool:
    """Same name after normalising, or mostly the same words (e.g. 'Karnataka rainfall data' ~ 'Karnataka rainfall dataset')."""
    if squash(a) == squash(b):
        return True
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.6


def verified_texts(extraction: dict, field: str) -> list[str]:
    items = extraction.get("fields", {}).get(field, {}).get("value") or []
    return [i["text"] for i in items if i.get("verified")]


def _shared(xs: list[str], ys: list[str]) -> list[str]:
    out: list[str] = []
    for x in xs:
        if any(similar(x, y) for y in ys) and not any(similar(x, o) for o in out):
            out.append(x)
    return out


def compare_extractions(a: dict, b: dict) -> dict:
    """First-pass comparability: do two papers report on the same dataset AND the same metric?
    Only verified items count. Split, preprocessing and baseline checks come in Phase 5."""
    datasets = _shared(verified_texts(a, "datasets"), verified_texts(b, "datasets"))
    metrics = _shared(verified_texts(a, "metrics"), verified_texts(b, "metrics"))
    reasons = []
    if not datasets:
        reasons.append("No shared verified dataset")
    if not metrics:
        reasons.append("No shared verified metric")
    return {"shared_datasets": datasets, "shared_metrics": metrics,
            "comparable": not reasons, "reasons": reasons}
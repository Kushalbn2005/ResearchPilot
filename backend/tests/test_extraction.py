import json

from app.extraction.compare import compare_extractions, similar
from app.extraction.extract import validate_group
from app.extraction.schema import canonical

PASSAGES = [
    {"chunk_id": "c1", "section": "Methods", "page_start": 3, "page_end": 3,
     "text": "We evaluate on the Karnataka rainfall dataset using Random Forest and LSTM."},
    {"chunk_id": "c2", "section": "Results", "page_start": 5, "page_end": 5,
     "text": "Random Forest reached an RMSE of 0.42 while LSTM reached an RMSE of 0.51."},
]
FIELDS = ("datasets", "metrics", "results", "baselines")


def run(obj, fields=FIELDS):
    return validate_group(json.dumps(obj), PASSAGES, fields)


def test_aliases_and_cleanup():
    assert canonical("RF") == "Random Forest"
    assert canonical(" rmse. ") == "RMSE"
    assert canonical("Some New Model") == "Some New Model"


def test_valid_extraction_gets_citations_and_verification():
    out, _ = run({
        "datasets": {"value": ["Karnataka rainfall dataset"], "evidence": ["P1"]},
        "metrics": {"value": ["RMSE"], "evidence": ["P2"]},
        "results": {"value": [{"method": "RF", "dataset": None, "metric": "rmse", "value": "0.42"}],
                    "evidence": ["P2"]},
        "baselines": {"value": ["LSTM"], "evidence": ["P1"]},
    })
    assert out["datasets"]["value"][0]["verified"] is True
    assert out["datasets"]["citations"][0]["page_start"] == 3  # from the database, not the model
    r = out["results"]["value"][0]
    assert r["method"] == "Random Forest" and r["metric"] == "RMSE" and r["verified"] is True


def test_hallucinated_item_is_flagged_unverified():
    out, warnings = run({"datasets": {"value": ["ImageNet"], "evidence": ["P1"]}}, ("datasets",))
    assert out["datasets"]["value"][0]["verified"] is False
    assert any("not found verbatim" in w for w in warnings)


def test_item_not_in_any_passage_with_invented_evidence_is_dropped():
    out, warnings = run({"metrics": {"value": ["Pearson correlation"], "evidence": ["P9"]}}, ("metrics",))
    assert out["metrics"]["value"] == [] and any("dropped" in w for w in warnings)


def test_wrong_evidence_id_does_not_lose_a_correct_item():
    # model cites a non-existent passage, but "RMSE" really is in P2: keep it, cite P2
    out, _ = run({"metrics": {"value": ["RMSE"], "evidence": ["P9"]}}, ("metrics",))
    item = out["metrics"]["value"][0]
    assert item["verified"] is True and out["metrics"]["citations"][0]["id"] == "P2"


def test_bare_list_shape_is_recovered_with_a_warning():
    out, warnings = run({"datasets": ["Karnataka rainfall dataset"]}, ("datasets",))
    assert out["datasets"]["value"][0]["verified"] is True
    assert out["datasets"]["citations"][0]["page_start"] == 3
    assert any("unexpected output shape" in w for w in warnings)


def test_empty_fields_always_explain_themselves():
    out, warnings = run({"datasets": {"value": [], "evidence": []}}, ("datasets",))
    assert out["datasets"]["value"] == []
    assert warnings == ["datasets: nothing found in the retrieved passages"]


def test_value_with_wrong_number_is_unverified():
    out, _ = run({"results": {"value": [{"method": "LSTM", "dataset": None, "metric": "RMSE", "value": "0.99"}],
                              "evidence": ["P2"]}}, ("results",))
    assert out["results"]["value"][0]["verified"] is False


def test_garbage_output_gives_empty_fields():
    out, warnings = validate_group("not json", PASSAGES, ("problem", "datasets"))
    assert out["problem"]["value"] is None and out["datasets"]["value"] == [] and warnings


def test_text_field_needs_evidence():
    out, _ = run({"problem": {"value": "Predict drought.", "evidence": ["P1"]}}, ("problem",))
    assert out["problem"]["value"] == "Predict drought."
    out, _ = run({"problem": {"value": "Predict drought.", "evidence": []}}, ("problem",))
    assert out["problem"]["value"] is None


def _extraction(datasets, metrics):
    mk = lambda xs: {"value": [{"text": t, "verified": v} for t, v in xs], "citations": []}
    return {"fields": {"datasets": mk(datasets), "metrics": mk(metrics)}}


def test_similar_names():
    assert similar("Karnataka rainfall data", "Karnataka rainfall dataset")
    assert not similar("ImageNet", "CIFAR-10")


def test_compare_needs_shared_dataset_and_metric():
    a = _extraction([("Karnataka rainfall dataset", True)], [("RMSE", True)])
    b = _extraction([("Karnataka rainfall data", True)], [("RMSE", True), ("MAE", True)])
    assert compare_extractions(a, b)["comparable"] is True
    c = _extraction([("Karnataka rainfall dataset", True)], [("F1", True)])
    res = compare_extractions(a, c)
    assert res["comparable"] is False and res["reasons"] == ["No shared verified metric"]


def test_compare_ignores_unverified_items():
    a = _extraction([("ImageNet", False)], [("RMSE", True)])
    b = _extraction([("ImageNet", True)], [("RMSE", True)])
    assert compare_extractions(a, b)["shared_datasets"] == []


def test_failed_group_is_reported_and_other_group_survives(monkeypatch):
    from app import llm
    from app.extraction import extract as ex

    monkeypatch.setattr(ex, "gather_passages", lambda db, doc_id, fields: PASSAGES)

    def fake_complete(system, user, **kw):
        if "keys: datasets" in system:  # the word also appears in the shared rules text
            raise RuntimeError("rate limit")
        return json.dumps({"problem": {"value": "Predict drought.", "evidence": ["P1"]},
                           "method": {"value": None, "evidence": []},
                           "limitations": {"value": [], "evidence": []}})

    monkeypatch.setattr(llm, "complete", fake_complete)
    result = ex.run_extraction(None, "doc1")
    assert result["fields"]["problem"]["value"] == "Predict drought."
    assert any("experiments: LLM call failed" in w and "rate limit" in w for w in result["warnings"])
    assert result["debug"]["experiments"]["error"].startswith("RuntimeError")
    assert result["debug"]["overview"]["passages"][0]["id"] == "P1"


def test_short_names_do_not_match_inside_longer_words():
    from app.extraction.schema import contains_words, words
    body = words("The annual rainfall over the channel region")
    assert not contains_words(body, "ANN")          # used to "verify" because of 'annual'/'channel'
    assert contains_words(words("We also trained an ANN baseline."), "ANN")


def test_numbers_match_whole_values_only():
    from app.extraction.schema import contains_words, words
    body = words("RMSE 0.031 and 10.214")
    assert contains_words(body, "0.031")
    assert not contains_words(body, "0.31")         # not inside 0.031
    assert not contains_words(body, "0.214")        # not inside 10.214


def test_result_value_must_appear_as_a_whole_number():
    out, _ = run({"results": {"value": [{"method": "LSTM", "dataset": None, "metric": "RMSE", "value": "0.5"}],
                              "evidence": ["P2"]}}, ("results",))
    assert out["results"]["value"][0]["verified"] is False   # "0.5" is only inside "0.51"
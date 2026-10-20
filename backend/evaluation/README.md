# Evaluation

1. Upload your papers (status `ready`) and run `POST /documents/{id}/extract` on each.
2. Copy `questions.example.json` to `questions.json` and `claims.example.json` to `claims.json`; replace the examples with your own, checked against the PDFs.
3. Run from `backend/` with the server running:

```bash
python -m evaluation.run_eval --skip-llm      # retrieval only, no Groq calls
python -m evaluation.run_eval                 # everything (add --sleep 6 if you hit rate limits)
```

Results are written to `evaluation/results/<timestamp>.json` and `.md`.

## Writing good questions
- ~40-50 questions over 4-6 papers: factual (one paper), exact_term (a model name, metric or number), comparison (two papers; list two gold entries), unanswerable (not in any paper; no gold).
- Gold = where the answer is: `file`, `pages` (PDF viewer index), and/or exact `phrases`.
- Mark about a third as `"split": "test"`. Look at dev results while improving the system; run test results only to report.
- ~15 stress-test claims across supported / challenged / mixed / insufficient.

## What is measured
- Retrieval: Recall@5, Recall@10, MRR@10 for vector, bm25, hybrid, hybrid_rerank.
- Answers: answered rate, keyword accuracy (proxy), citation precision (file + page match), abstention accuracy on unanswerable questions.
- Stress test: accuracy, per-class precision/recall/F1, confusion matrix, false contradictions.

Also read 30-40 answers from the JSON `review` list and grade them yourself. Keyword matching is only a proxy.

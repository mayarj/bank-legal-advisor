# Evaluation Harness

Measures how well the RAG retriever and the agents perform against a labelled dataset, and
prints full-length score reports. This is a **standalone harness**, separate from the unit
test suite (`tests/`) — it needs real data and, for the agent evals, the API key.

For the metric definitions and target scores, see
[../BENCHMARK.md](../BENCHMARK.md). The math here is unit-tested in
`tests/test_eval_metrics.py` (runs with no data or API key).

---

## What each evaluation measures

| Command | Measures | Metrics printed | Needs |
|---|---|---|---|
| `retrieval` | RAG ranking quality | Precision@k, Recall@k, F1@k, Hit-rate@k, MRR, MAP | embedding model + populated vector store |
| `legal` | Legal agent answer grounding | Citation precision / recall / F1 (per query + average) | API key + vector store + DB |
| `loan` | Loan agent decisions | Risk-level accuracy, confusion matrix, per-class & macro P/R/F1 | API key + DB with the referenced loans |

`retrieval` does **not** need the API key (embeddings are local; query rewrite is off by
default). The two agent evals invoke the real agents, so they need `ANTHROPIC_API_KEY`.

---

## 1. Prepare the data

Each dataset has an `.example.json` template in `datasets/`. Copy it (drop the `.example`)
and replace the rows with your own:

```bash
cp evaluation/datasets/retrieval.example.json evaluation/datasets/retrieval.json
cp evaluation/datasets/agent.example.json     evaluation/datasets/agent.json
cp evaluation/datasets/loan.example.json      evaluation/datasets/loan.json
```

- **Retrieval / legal:** `relevant` / `expected_citations` are lists of
  `{"legislation_code": "...", "article_number": "..."}` — the articles a domain expert
  confirms are correct for that query.
- **Loan:** each case is a `loan_id` already present in the database plus its
  `expected_risk` (`low` | `medium` | `high`).

The vector store and database must contain the legislation/loans you reference (ingest them
first via the normal pipeline).

## 2. Run

From the project root (`bank-legal-advisor/`):

```bash
# RAG retrieval — no API key needed
python -m evaluation.harness retrieval --dataset evaluation/datasets/retrieval.json
python -m evaluation.harness retrieval --dataset evaluation/datasets/retrieval.json --method semantic
python -m evaluation.harness retrieval --dataset evaluation/datasets/retrieval.json --k 1 3 5 10 20

# Legal agent citations — needs API key
python -m evaluation.harness legal --dataset evaluation/datasets/agent.json

# Loan agent risk levels — needs API key + populated DB
python -m evaluation.harness loan --dataset evaluation/datasets/loan.json
```

### Useful retrieval flags
- `--method {hybrid,semantic}` — which retriever path to score (default `hybrid`).
- `--k 1 3 5 10` — cutoffs to report.
- `--rewrite` — enable LLM query rewriting (needs the API key).
- `--include-out-of-force` — search all articles instead of only in-force ones.
- `--no-per-query` — print aggregates only.

---

## Notes

- Citation precision/recall (the `legal` eval) is fully objective — it checks which articles
  the answer cited against the expected set. It does **not** judge whether the prose answer is
  semantically correct; that needs a human or an LLM judge (see BENCHMARK.md §3.1).
- Loan cases that pause for clarification are reported as skipped and excluded from accuracy,
  so use self-contained loans for automated scoring.
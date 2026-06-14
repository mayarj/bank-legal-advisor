"""Evaluation harness for the RAG retriever and the agents.

Run from the project root (``bank-legal-advisor/``):

    python -m evaluation.harness retrieval --dataset evaluation/datasets/retrieval.json
    python -m evaluation.harness legal     --dataset evaluation/datasets/agent.json
    python -m evaluation.harness loan      --dataset evaluation/datasets/loan.json

`retrieval` needs only the local embedding model (no API key) but a populated vector store.
`legal` and `loan` invoke the real agents and therefore need the ANTHROPIC_API_KEY and a
populated database. Project (`src.*`) imports are done lazily inside each runner so the
metrics/report code stays importable without a configured environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from evaluation.metrics import (
    RankingSample,
    aggregate_retrieval,
    classification_report,
    set_prf,
)
from evaluation.report import (
    render_citation_results,
    render_classification,
    render_header,
    render_per_query_ranking,
    render_retrieval_aggregate,
)


# ── Dataset helpers ───────────────────────────────────────────────────────────-

def _load(dataset_path: str) -> dict:
    path = Path(dataset_path)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}\nFill in a dataset first (see evaluation/README.md).")
    return json.loads(path.read_text())


def _article_id(entry: dict) -> str:
    """Normalize a {legislation_code, article_number} entry to the canonical article id."""
    return f"{entry['legislation_code']}_article_{entry['article_number']}"


# ── Retrieval evaluation (RAG ranking) ────────────────────────────────────────-

def run_retrieval(
    dataset_path: str,
    method: str = "hybrid",
    ks: tuple[int, ...] = (1, 3, 5, 10),
    rewrite: bool = False,
    in_force_only: bool = True,
    show_per_query: bool = True,
) -> None:
    from src.rag.retriever import retrieve, retrieve_hybrid

    data = _load(dataset_path)
    queries = data["queries"]
    max_k = max(ks)
    filters = {"is_in_force": True} if in_force_only else None

    samples: list[RankingSample] = []
    for q in queries:
        relevant = {_article_id(r) for r in q.get("relevant", [])}
        if method == "hybrid":
            results = retrieve_hybrid(q["query"], n_results=max_k, filters=filters, rewrite=rewrite)
        else:
            results = retrieve(q["query"], n_results=max_k, filters=filters, rewrite=rewrite)
        ranked = [f"{r.legislation_code}_article_{r.article_number}" for r in results]
        samples.append(RankingSample(id=str(q.get("id", q["query"][:20])), ranked=ranked, relevant=relevant))

    agg = aggregate_retrieval(samples, ks)

    print(render_header(f"RETRIEVAL EVALUATION  (method={method}, rewrite={rewrite}, in_force_only={in_force_only})"))
    if show_per_query:
        print(render_per_query_ranking(samples, primary_k=min(5, max_k)))
    print(render_retrieval_aggregate(agg))


# ── Legal agent evaluation (citation precision / recall) ──────────────────────-

async def run_legal(dataset_path: str) -> None:
    from src.mcp.client import mcp_client
    from src.agents.legal_agent import build_legal_agent, _parse_article_refs

    data = _load(dataset_path)
    queries = data["queries"]

    print(render_header("LEGAL AGENT EVALUATION  (citation precision / recall / F1)"))
    rows: list[dict] = []
    async with mcp_client() as tools:
        for q in queries:
            agent = build_legal_agent(tools)
            result = await agent.ainvoke({"query": q["query"], "messages": []})
            answer = result.get("draft_answer", "") or ""
            predicted = {f"{code}_article_{num}" for code, num in _parse_article_refs(answer)}
            expected = {_article_id(e) for e in q.get("expected_citations", [])}
            precision, recall, f1 = set_prf(predicted, expected)
            rows.append({
                "id": str(q.get("id", q["query"][:20])),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

    print(render_citation_results(rows))


# ── Loan agent evaluation (risk-level classification) ─────────────────────────-

async def run_loan(dataset_path: str) -> None:
    from src.mcp.client import mcp_client
    from src.agents.loan_agent import build_loan_agent

    data = _load(dataset_path)
    cases = data["cases"]

    print(render_header("LOAN AGENT EVALUATION  (risk-level accuracy)"))
    y_true: list[str] = []
    y_pred: list[str] = []
    async with mcp_client() as tools:
        for case in cases:
            agent = build_loan_agent(tools)
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            result = await agent.ainvoke({"loan_id": case["loan_id"], "messages": []}, config=config)
            predicted = result.get("risk_level")
            if not predicted or not result.get("assessment_saved"):
                print(f"  [skipped loan {case['loan_id']}: agent did not complete (clarification needed?)]")
                continue
            y_true.append(case["expected_risk"])
            y_pred.append(predicted)

    if not y_true:
        print("  No completed cases to score.")
        return

    report = classification_report(y_true, y_pred, labels=["low", "medium", "high"])
    print(render_classification(report))


# ── CLI ───────────────────────────────────────────────────────────────────────-

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and the agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ret = sub.add_parser("retrieval", help="RAG ranking metrics (precision/recall/F1/hit-rate/MRR/MAP)")
    p_ret.add_argument("--dataset", required=True)
    p_ret.add_argument("--method", choices=["hybrid", "semantic"], default="hybrid")
    p_ret.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    p_ret.add_argument("--rewrite", action="store_true", help="LLM query rewrite (needs API key)")
    p_ret.add_argument("--include-out-of-force", action="store_true", help="search all articles, not just in-force")
    p_ret.add_argument("--no-per-query", action="store_true")

    p_legal = sub.add_parser("legal", help="Legal agent citation precision/recall (needs API key)")
    p_legal.add_argument("--dataset", required=True)

    p_loan = sub.add_parser("loan", help="Loan agent risk-level accuracy (needs API key + DB)")
    p_loan.add_argument("--dataset", required=True)

    args = parser.parse_args()

    if args.command == "retrieval":
        run_retrieval(
            args.dataset,
            method=args.method,
            ks=tuple(args.k),
            rewrite=args.rewrite,
            in_force_only=not args.include_out_of_force,
            show_per_query=not args.no_per_query,
        )
    elif args.command == "legal":
        asyncio.run(run_legal(args.dataset))
    elif args.command == "loan":
        asyncio.run(run_loan(args.dataset))


if __name__ == "__main__":
    main()
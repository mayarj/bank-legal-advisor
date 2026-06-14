"""Full-length text rendering of evaluation results — nothing is truncated."""

from __future__ import annotations

import math
from typing import Sequence

from evaluation.metrics import ClassificationReport, RankingSample


def _fmt(value: float) -> str:
    return "  n/a" if isinstance(value, float) and math.isnan(value) else f"{value:6.3f}"


def _rule(char: str = "─", width: int = 72) -> str:
    return char * width


def render_header(title: str) -> str:
    return f"\n{_rule('=')}\n{title}\n{_rule('=')}"


# ── Retrieval ─────────────────────────────────────────────────────────────────-

def render_per_query_ranking(
    samples: Sequence[RankingSample],
    primary_k: int,
) -> str:
    """One row per query: hits, precision/recall at the primary cutoff, RR — all shown."""
    from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank

    lines = [
        f"\nPer-query detail (k={primary_k}):",
        f"{'query id':<24} {'#rel':>5} {'#ret':>5} {'P@k':>7} {'R@k':>7} {'RR':>7}",
        _rule(),
    ]
    for s in samples:
        lines.append(
            f"{s.id:<24} {len(s.relevant):>5} {len(s.ranked):>5} "
            f"{_fmt(precision_at_k(s.ranked, s.relevant, primary_k))} "
            f"{_fmt(recall_at_k(s.ranked, s.relevant, primary_k))} "
            f"{_fmt(reciprocal_rank(s.ranked, s.relevant))}"
        )
    return "\n".join(lines)


def render_retrieval_aggregate(agg: dict) -> str:
    lines = [
        f"\nAggregate over {agg['n_queries']} queries:",
        f"{'k':>4} {'Precision':>11} {'Recall':>9} {'F1':>9} {'HitRate':>9}",
        _rule(),
    ]
    for k in agg["k_values"]:
        m = agg["per_k"][k]
        lines.append(
            f"{k:>4} {_fmt(m['precision']):>11} {_fmt(m['recall']):>9} "
            f"{_fmt(m['f1']):>9} {_fmt(m['hit_rate']):>9}"
        )
    lines += [
        _rule(),
        f"MRR (mean reciprocal rank): {_fmt(agg['mrr']).strip()}",
        f"MAP (mean average precision): {_fmt(agg['map']).strip()}",
    ]
    return "\n".join(lines)


# ── Citations (set PRF, averaged) ─────────────────────────────────────────────-

def render_citation_results(rows: list[dict]) -> str:
    """rows: [{id, precision, recall, f1, predicted, expected}]"""
    lines = [
        f"\nPer-query citation scores:",
        f"{'query id':<24} {'Prec':>7} {'Recall':>7} {'F1':>7}",
        _rule(),
    ]
    for r in rows:
        lines.append(
            f"{r['id']:<24} {_fmt(r['precision'])} {_fmt(r['recall'])} {_fmt(r['f1'])}"
        )
    lines.append(_rule())
    n = len(rows)
    avg_p = sum(r["precision"] for r in rows) / n if n else float("nan")
    avg_r_vals = [r["recall"] for r in rows if not (isinstance(r["recall"], float) and math.isnan(r["recall"]))]
    avg_r = sum(avg_r_vals) / len(avg_r_vals) if avg_r_vals else float("nan")
    avg_f_vals = [r["f1"] for r in rows if not (isinstance(r["f1"], float) and math.isnan(r["f1"]))]
    avg_f = sum(avg_f_vals) / len(avg_f_vals) if avg_f_vals else float("nan")
    lines.append(f"{'AVERAGE':<24} {_fmt(avg_p)} {_fmt(avg_r)} {_fmt(avg_f)}")
    return "\n".join(lines)


# ── Classification (risk level) ───────────────────────────────────────────────-

def render_classification(report: ClassificationReport) -> str:
    labels = report.labels
    lines = [
        f"\nClassification over {report.n} cases:",
        f"Accuracy: {_fmt(report.accuracy).strip()}",
        "",
        "Confusion matrix (rows = true, cols = predicted):",
    ]
    header = f"{'true\\pred':<12}" + "".join(f"{str(l):>10}" for l in labels)
    lines.append(header)
    lines.append(_rule())
    for t in labels:
        row = f"{str(t):<12}" + "".join(f"{report.confusion[t][p]:>10}" for p in labels)
        lines.append(row)

    lines += ["", "Per-class precision / recall / F1:",
              f"{'class':<12} {'Prec':>7} {'Recall':>7} {'F1':>7} {'support':>9}", _rule()]
    for label in labels:
        c = report.per_class[label]
        lines.append(
            f"{str(label):<12} {_fmt(c['precision'])} {_fmt(c['recall'])} "
            f"{_fmt(c['f1'])} {c['support']:>9}"
        )
    lines.append(_rule())
    lines.append(
        f"{'macro avg':<12} {_fmt(report.macro['precision'])} "
        f"{_fmt(report.macro['recall'])} {_fmt(report.macro['f1'])}"
    )
    return "\n".join(lines)
"""Pure evaluation metrics — no I/O, no project imports, no external deps.

Everything here operates on plain hashable IDs (strings or tuples), so it is trivially
unit-testable and reusable for both retrieval ranking and classification (risk level).

Conventions:
- A *ranked* result is an ordered list, best first.
- *relevant* / *expected* are sets of the IDs that count as correct.
- Metrics that are undefined for an input (e.g. recall when there is nothing relevant)
  return ``float('nan')`` and are skipped by the aggregators rather than counted as 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Sequence


# ── Per-query ranking metrics ─────────────────────────────────────────────────-

def precision_at_k(ranked: Sequence[Hashable], relevant: set, k: int) -> float:
    """Fraction of the top-k results that are relevant (divided by k)."""
    if k <= 0:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[Hashable], relevant: set, k: int) -> float:
    """Fraction of all relevant items that appear in the top-k. NaN if nothing relevant."""
    if not relevant:
        return float("nan")
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(ranked: Sequence[Hashable], relevant: set, k: int) -> float:
    """Harmonic mean of precision@k and recall@k."""
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    if math.isnan(r):
        return float("nan")
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate_at_k(ranked: Sequence[Hashable], relevant: set, k: int) -> float:
    """1.0 if at least one relevant item is in the top-k, else 0.0 (a.k.a. success@k)."""
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[Hashable], relevant: set) -> float:
    """1 / rank of the first relevant item (0.0 if none found). Mean → MRR."""
    for index, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (index + 1)
    return 0.0


def average_precision(ranked: Sequence[Hashable], relevant: set) -> float:
    """Average precision for one query (area under its precision-recall curve).
    Mean over queries → MAP. NaN if nothing relevant."""
    if not relevant:
        return float("nan")
    hits = 0
    running = 0.0
    for index, item in enumerate(ranked):
        if item in relevant:
            hits += 1
            running += hits / (index + 1)
    return running / len(relevant)


# ── Aggregation across queries ────────────────────────────────────────────────-

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _nanmean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return _mean(clean)


@dataclass
class RankingSample:
    """One evaluated query: the IDs the system returned (ranked) vs the relevant IDs."""
    id: str
    ranked: list
    relevant: set


def aggregate_retrieval(
    samples: Sequence[RankingSample],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """Aggregate ranking metrics over all queries, at each cutoff k, plus MRR and MAP."""
    per_k: dict[int, dict] = {}
    for k in ks:
        per_k[k] = {
            "precision": _mean([precision_at_k(s.ranked, s.relevant, k) for s in samples]),
            "recall": _nanmean([recall_at_k(s.ranked, s.relevant, k) for s in samples]),
            "f1": _nanmean([f1_at_k(s.ranked, s.relevant, k) for s in samples]),
            "hit_rate": _mean([hit_rate_at_k(s.ranked, s.relevant, k) for s in samples]),
        }
    return {
        "n_queries": len(samples),
        "k_values": list(ks),
        "per_k": per_k,
        "mrr": _mean([reciprocal_rank(s.ranked, s.relevant) for s in samples]),
        "map": _nanmean([average_precision(s.ranked, s.relevant) for s in samples]),
    }


# ── Set-based PRF (citations: predicted set vs expected set, order-free) ───────-

def set_prf(predicted: set, expected: set) -> tuple[float, float, float]:
    """Precision, recall, F1 for an unordered prediction vs an expected set.
    A correct empty prediction for an empty expectation scores 1/1/1 (true negative)."""
    if not predicted and not expected:
        return (1.0, 1.0, 1.0)
    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else float("nan")
    if math.isnan(recall) or (precision + recall) == 0:
        f1 = 0.0 if not math.isnan(recall) else float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


# ── Classification metrics (risk level: low / medium / high) ───────────────────-

@dataclass
class ClassificationReport:
    labels: list
    accuracy: float
    confusion: dict            # confusion[true][pred] = count
    per_class: dict            # label → {precision, recall, f1, support}
    macro: dict                # {precision, recall, f1}
    n: int = 0


def classification_report(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> ClassificationReport:
    """Accuracy, confusion matrix, and per-class + macro precision/recall/F1."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)
    labels = list(labels)

    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else float("nan")

    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    per_class: dict = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[t][label] for t in labels if t != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        support = sum(confusion[label][p] for p in labels)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro = {
        "precision": _nanmean([per_class[c]["precision"] for c in labels]),
        "recall": _nanmean([per_class[c]["recall"] for c in labels]),
        "f1": _nanmean([per_class[c]["f1"] for c in labels]),
    }

    return ClassificationReport(
        labels=labels,
        accuracy=accuracy,
        confusion=confusion,
        per_class=per_class,
        macro=macro,
        n=n,
    )
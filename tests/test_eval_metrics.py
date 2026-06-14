import math

from evaluation.metrics import (
    RankingSample,
    aggregate_retrieval,
    average_precision,
    classification_report,
    f1_at_k,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    set_prf,
)

# A reference ranking used across several tests:
#   ranked = [a, b, c, d], relevant = {b, d}
RANKED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d"}


class TestRankingMetrics:

    def test_precision_at_k(self):
        assert precision_at_k(RANKED, RELEVANT, 1) == 0.0          # a
        assert precision_at_k(RANKED, RELEVANT, 2) == 0.5          # a, b
        assert precision_at_k(RANKED, RELEVANT, 4) == 0.5          # 2 of 4

    def test_recall_at_k(self):
        assert recall_at_k(RANKED, RELEVANT, 2) == 0.5             # found b of {b, d}
        assert recall_at_k(RANKED, RELEVANT, 4) == 1.0             # found both

    def test_recall_is_nan_when_nothing_relevant(self):
        assert math.isnan(recall_at_k(RANKED, set(), 3))

    def test_f1_at_k(self):
        # p@2 = 0.5, r@2 = 0.5 → f1 = 0.5
        assert f1_at_k(RANKED, RELEVANT, 2) == 0.5

    def test_hit_rate_at_k(self):
        assert hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0
        assert hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(RANKED, RELEVANT) == 0.5           # first hit at position 2
        assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0       # no hit

    def test_average_precision(self):
        # hits at ranks 2 and 4 → (1/2 + 2/4) / 2 = 0.5
        assert average_precision(RANKED, RELEVANT) == 0.5

    def test_perfect_ranking_scores_one(self):
        ranked = ["b", "d", "a", "c"]
        assert precision_at_k(ranked, RELEVANT, 2) == 1.0
        assert recall_at_k(ranked, RELEVANT, 2) == 1.0
        assert reciprocal_rank(ranked, RELEVANT) == 1.0
        assert average_precision(ranked, RELEVANT) == 1.0


class TestAggregateRetrieval:

    def test_aggregates_two_queries(self):
        samples = [
            RankingSample(id="q1", ranked=["b", "d", "a", "c"], relevant={"b", "d"}),  # perfect
            RankingSample(id="q2", ranked=["a", "b", "c", "d"], relevant={"b", "d"}),  # the reference
        ]
        agg = aggregate_retrieval(samples, ks=(1, 2, 4))

        assert agg["n_queries"] == 2
        # MRR = mean(1.0, 0.5) = 0.75
        assert agg["mrr"] == 0.75
        # MAP = mean(1.0, 0.5) = 0.75
        assert agg["map"] == 0.75
        # precision@2 = mean(1.0, 0.5) = 0.75
        assert agg["per_k"][2]["precision"] == 0.75
        # recall@4 = mean(1.0, 1.0) = 1.0
        assert agg["per_k"][4]["recall"] == 1.0

    def test_nan_recall_is_skipped_in_aggregate(self):
        samples = [
            RankingSample(id="q1", ranked=["a", "b"], relevant={"b"}),   # recall@2 = 1.0
            RankingSample(id="q2", ranked=["a", "b"], relevant=set()),   # recall = NaN, skipped
        ]
        agg = aggregate_retrieval(samples, ks=(2,))
        assert agg["per_k"][2]["recall"] == 1.0


class TestSetPRF:

    def test_partial_overlap(self):
        p, r, f = set_prf(predicted={"a", "b"}, expected={"b", "c"})
        assert p == 0.5            # 1 of 2 predicted correct
        assert r == 0.5            # 1 of 2 expected found
        assert f == 0.5

    def test_perfect_match(self):
        assert set_prf({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)

    def test_empty_prediction_and_expectation_is_true_negative(self):
        assert set_prf(set(), set()) == (1.0, 1.0, 1.0)

    def test_predicting_when_nothing_expected_scores_zero_precision(self):
        p, r, f = set_prf(predicted={"a"}, expected=set())
        assert p == 0.0
        assert math.isnan(r)


class TestClassificationReport:

    def test_accuracy_and_confusion(self):
        y_true = ["low", "low", "high", "medium"]
        y_pred = ["low", "high", "high", "medium"]

        report = classification_report(y_true, y_pred, labels=["low", "medium", "high"])

        assert report.n == 4
        assert report.accuracy == 0.75
        assert report.confusion["low"]["low"] == 1
        assert report.confusion["low"]["high"] == 1
        assert report.confusion["high"]["high"] == 1
        assert report.confusion["medium"]["medium"] == 1

    def test_per_class_precision_recall(self):
        y_true = ["low", "low", "high", "medium"]
        y_pred = ["low", "high", "high", "medium"]

        report = classification_report(y_true, y_pred, labels=["low", "medium", "high"])

        # low: 1 predicted, 1 correct → precision 1.0; 2 actual, 1 found → recall 0.5
        assert report.per_class["low"]["precision"] == 1.0
        assert report.per_class["low"]["recall"] == 0.5
        # high: 2 predicted, 1 correct → precision 0.5; 1 actual, 1 found → recall 1.0
        assert report.per_class["high"]["precision"] == 0.5
        assert report.per_class["high"]["recall"] == 1.0
        assert report.per_class["low"]["support"] == 2

    def test_perfect_classification(self):
        report = classification_report(["low", "high"], ["low", "high"], labels=["low", "high"])
        assert report.accuracy == 1.0
        assert report.macro["f1"] == 1.0
"""Unit tests for the evaluation module's metric computations."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.evaluation.evaluate import (
    build_tier_thresholds,
    compute_classification_metrics,
    compute_confusion_counts,
    compute_tier_counts,
    format_tier_counts_line,
    per_rule_analysis,
)


# ===========================================================================
# 1. compute_confusion_counts
# ===========================================================================

class TestConfusionCounts:
    """Tests for compute_confusion_counts."""

    def test_perfect_predictions(self):
        """All predictions are correct."""
        y_true = pd.Series([True, True, False, False])
        y_pred = pd.Series([True, True, False, False])
        counts = compute_confusion_counts(y_true, y_pred)
        assert counts == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    def test_all_wrong_predictions(self):
        """Every prediction is inverted."""
        y_true = pd.Series([True, True, False, False])
        y_pred = pd.Series([False, False, True, True])
        counts = compute_confusion_counts(y_true, y_pred)
        assert counts == {"tp": 0, "fp": 2, "tn": 0, "fn": 2}

    def test_all_positive_predictions(self):
        """Predict fraud for every row."""
        y_true = pd.Series([True, False, False, False, True])
        y_pred = pd.Series([True, True, True, True, True])
        counts = compute_confusion_counts(y_true, y_pred)
        assert counts == {"tp": 2, "fp": 3, "tn": 0, "fn": 0}

    def test_all_negative_predictions(self):
        """Predict normal for every row."""
        y_true = pd.Series([True, True, False, False])
        y_pred = pd.Series([False, False, False, False])
        counts = compute_confusion_counts(y_true, y_pred)
        assert counts == {"tp": 0, "fp": 0, "tn": 2, "fn": 2}

    def test_empty_series(self):
        """Empty input should yield all zeros."""
        y_true = pd.Series([], dtype=bool)
        y_pred = pd.Series([], dtype=bool)
        counts = compute_confusion_counts(y_true, y_pred)
        assert counts == {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    def test_single_tp(self):
        """Single true-positive row."""
        counts = compute_confusion_counts(pd.Series([True]), pd.Series([True]))
        assert counts == {"tp": 1, "fp": 0, "tn": 0, "fn": 0}


# ===========================================================================
# 2. compute_classification_metrics
# ===========================================================================

class TestClassificationMetrics:
    """Tests for precision, recall, F1, and FPR computation."""

    def test_perfect_metrics(self):
        """Perfect predictions yield precision=1, recall=1, F1=1, FPR=0."""
        m = compute_classification_metrics({"tp": 10, "fp": 0, "tn": 90, "fn": 0})
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["fpr"] == 0.0

    def test_typical_case(self):
        """TP=80, FP=20, TN=880, FN=20."""
        m = compute_classification_metrics({"tp": 80, "fp": 20, "tn": 880, "fn": 20})
        assert m["precision"] == pytest.approx(0.8)
        assert m["recall"] == pytest.approx(0.8)
        assert m["f1"] == pytest.approx(0.8)
        assert m["fpr"] == pytest.approx(20 / 900)

    def test_zero_positive_predictions(self):
        """No positive predictions: precision=None, recall=0/None depending on FN."""
        m = compute_classification_metrics({"tp": 0, "fp": 0, "tn": 90, "fn": 10})
        assert m["precision"] is None  # 0 / 0
        assert m["recall"] == 0.0       # 0 / 10
        assert m["f1"] is None          # precision is None
        assert m["fpr"] == 0.0          # 0 / 90

    def test_no_true_positives_in_ground_truth(self):
        """Ground truth has no fraud rows: recall denominator is zero."""
        m = compute_classification_metrics({"tp": 0, "fp": 5, "tn": 95, "fn": 0})
        assert m["precision"] == 0.0    # 0 / 5
        assert m["recall"] is None      # 0 / 0 (no actual positives)
        assert m["f1"] is None
        assert m["fpr"] == pytest.approx(5 / 100)

    def test_all_zeros(self):
        """All counts are zero (empty dataset)."""
        m = compute_classification_metrics({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        assert m["precision"] is None
        assert m["recall"] is None
        assert m["f1"] is None
        assert m["fpr"] is None

    def test_fpr_no_negatives(self):
        """All rows are actually fraud; FPR denominator (FP+TN) is zero."""
        m = compute_classification_metrics({"tp": 10, "fp": 0, "tn": 0, "fn": 5})
        assert m["fpr"] is None

    def test_f1_with_zero_precision_and_recall(self):
        """Both precision and recall are zero → F1 is None (0/0)."""
        m = compute_classification_metrics({"tp": 0, "fp": 5, "tn": 0, "fn": 5})
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] is None  # (0 + 0) in denominator

    def test_f1_formula(self):
        """Verify the harmonic mean formula: 2 * P * R / (P + R)."""
        m = compute_classification_metrics({"tp": 60, "fp": 40, "tn": 860, "fn": 40})
        p = 60 / 100  # 0.6
        r = 60 / 100  # 0.6
        expected_f1 = 2 * p * r / (p + r)  # 0.6
        assert m["f1"] == pytest.approx(expected_f1)


# ===========================================================================
# 3. build_tier_thresholds
# ===========================================================================

class TestBuildTierThresholds:
    """Tests for tier threshold generation from config."""

    def test_standard_four_tiers(self):
        tiers = [
            {"name": "Low", "min_score": 0, "max_score": 24},
            {"name": "Medium", "min_score": 25, "max_score": 49},
            {"name": "High", "min_score": 50, "max_score": 74},
            {"name": "Critical", "min_score": 75, "max_score": 100},
        ]
        thresholds = build_tier_thresholds(tiers)
        assert len(thresholds) == 3

        assert thresholds[0]["label"] == "Medium and above"
        assert thresholds[0]["included_tiers"] == {"Medium", "High", "Critical"}

        assert thresholds[1]["label"] == "High and above"
        assert thresholds[1]["included_tiers"] == {"High", "Critical"}

        assert thresholds[2]["label"] == "Critical only"
        assert thresholds[2]["included_tiers"] == {"Critical"}

    def test_two_tiers(self):
        tiers = [
            {"name": "Normal", "min_score": 0, "max_score": 49},
            {"name": "Alert", "min_score": 50, "max_score": 100},
        ]
        thresholds = build_tier_thresholds(tiers)
        assert len(thresholds) == 1
        assert thresholds[0]["label"] == "Alert only"
        assert thresholds[0]["included_tiers"] == {"Alert"}

    def test_unsorted_input_is_handled(self):
        """Tiers given in reverse order should produce the same result."""
        tiers = [
            {"name": "Critical", "min_score": 75, "max_score": 100},
            {"name": "Low", "min_score": 0, "max_score": 24},
            {"name": "High", "min_score": 50, "max_score": 74},
            {"name": "Medium", "min_score": 25, "max_score": 49},
        ]
        thresholds = build_tier_thresholds(tiers)
        assert thresholds[0]["included_tiers"] == {"Medium", "High", "Critical"}


# ===========================================================================
# 4. per_rule_analysis
# ===========================================================================

class TestPerRuleAnalysis:
    """Tests for per-rule firing statistics."""

    def _build_df(self) -> pd.DataFrame:
        """Build a hand-crafted DataFrame for per-rule tests."""
        return pd.DataFrame({
            "is_fraud_actual": [True, True, False, False, True, False],
            "triggered_rules": [
                "high_velocity,large_amount_deviation",
                "large_amount_deviation",
                "large_amount_deviation",
                "",
                "high_velocity",
                "high_velocity",
            ],
        })

    def test_rule_counts(self):
        df = self._build_df()
        results = per_rule_analysis(df, ["high_velocity", "large_amount_deviation"])

        hv = next(r for r in results if r["rule"] == "high_velocity")
        assert hv["fraud_fired"] == 2   # rows 0, 4
        assert hv["normal_fired"] == 1  # row 5

        la = next(r for r in results if r["rule"] == "large_amount_deviation")
        assert la["fraud_fired"] == 2   # rows 0, 1
        assert la["normal_fired"] == 1  # row 2

    def test_rule_precision_recall(self):
        df = self._build_df()
        results = per_rule_analysis(df, ["high_velocity"])

        hv = results[0]
        # precision = 2 / 3
        assert hv["precision"] == pytest.approx(2 / 3)
        # recall = 2 / 3  (3 total fraud rows)
        assert hv["recall"] == pytest.approx(2 / 3)

    def test_rule_never_fires(self):
        df = self._build_df()
        results = per_rule_analysis(df, ["nonexistent_rule"])
        r = results[0]
        assert r["fraud_fired"] == 0
        assert r["normal_fired"] == 0
        assert r["precision"] is None  # 0/0
        assert r["recall"] == 0.0       # 0/3

    def test_no_fraud_in_dataset(self):
        """Dataset with no fraud rows: recall denominator is zero."""
        df = pd.DataFrame({
            "is_fraud_actual": [False, False, False],
            "triggered_rules": ["rule_a", "rule_a", ""],
        })
        results = per_rule_analysis(df, ["rule_a"])
        r = results[0]
        assert r["normal_fired"] == 2
        assert r["fraud_fired"] == 0
        assert r["precision"] == 0.0    # 0 / 2
        assert r["recall"] is None      # 0 fraud total


# ===========================================================================
# 5. compute_tier_counts and format_tier_counts_line
# ===========================================================================

class TestTierCounts:
    """Tests for tier counts computation and report line formatting."""

    def test_compute_tier_counts(self):
        s = pd.Series(["Low", "Low", "Medium", "High", "Critical", "Low"])
        counts = compute_tier_counts(s)
        assert counts == {
            "Medium": 1,
            "High": 1,
            "Critical": 1,
            "Low": 3,
            "total": 6,
        }

    def test_compute_tier_counts_missing_tiers(self):
        """Tiers not present in data should default to 0."""
        s = pd.Series(["Low", "Low"])
        counts = compute_tier_counts(s)
        assert counts == {
            "Medium": 0,
            "High": 0,
            "Critical": 0,
            "Low": 2,
            "total": 2,
        }

    def test_format_tier_counts_line(self):
        counts = {
            "Medium": 635,
            "High": 422,
            "Critical": 44,
            "Low": 48899,
            "total": 50000,
        }
        line = format_tier_counts_line(counts)
        assert line == "Tier counts: Medium: 635, High: 422, Critical: 44, Low: 48,899, Total: 50,000"


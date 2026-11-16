"""Evaluate rule engine predictions against ground-truth is_fraud_actual labels.

This module is for **development-time validation only**.  In a real system the
ground-truth column would not exist; operational metrics come from analyst
review outcomes via the DBManager.

Usage:
    python -m src.evaluation.evaluate                 # default paths
    python -m src.evaluation.evaluate --csv path.csv  # custom CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.figure_factory as ff
import yaml

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_SCORED_CSV = _REPO_ROOT / "data" / "processed" / "transactions_scored.csv"
_DEFAULT_RULES_CONFIG = _REPO_ROOT / "src" / "rules_engine" / "rules_config.yaml"
_DEFAULT_REPORTS_DIR = _REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Core metric computation — pure functions, no I/O
# ---------------------------------------------------------------------------

def compute_confusion_counts(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, int]:
    """Compute TP, FP, TN, FN from boolean Series.

    Args:
        y_true: Ground-truth boolean labels.
        y_pred: Predicted boolean labels.

    Returns:
        Dictionary with keys 'tp', 'fp', 'tn', 'fn'.
    """
    tp = ((y_true == True) & (y_pred == True)).sum()   # noqa: E712
    fp = ((y_true == False) & (y_pred == True)).sum()  # noqa: E712
    tn = ((y_true == False) & (y_pred == False)).sum() # noqa: E712
    fn = ((y_true == True) & (y_pred == False)).sum()  # noqa: E712
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_classification_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    """Derive precision, recall, F1, and FPR from confusion counts.

    Returns None for any metric whose denominator is zero.
    """
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    precision: float | None = tp / (tp + fp) if (tp + fp) > 0 else None
    recall: float | None = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1: float | None = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    fpr: float | None = fp / (fp + tn) if (fp + tn) > 0 else None

    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


# ---------------------------------------------------------------------------
# Tier threshold helpers
# ---------------------------------------------------------------------------

def load_risk_tiers(config_path: Path = _DEFAULT_RULES_CONFIG) -> list[dict[str, Any]]:
    """Load risk_tiers list from rules_config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["risk_tiers"]


def load_enabled_rules(config_path: Path = _DEFAULT_RULES_CONFIG) -> list[str]:
    """Return names of enabled rules from rules_config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [r["name"] for r in cfg.get("rules", []) if r.get("enabled", True)]


def build_tier_thresholds(tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a list of threshold definitions from the tier list.

    For N tiers sorted by min_score ascending, produces N-1 thresholds:
      tier[1] and above, tier[2] and above, ... , tier[N-1] only.

    Each entry has:
      - label: e.g. "Medium and above", "Critical only"
      - included_tiers: set of tier names counted as positive
    """
    sorted_tiers = sorted(tiers, key=lambda t: t["min_score"])
    names = [t["name"] for t in sorted_tiers]
    thresholds: list[dict[str, Any]] = []

    for i in range(1, len(names)):
        included = set(names[i:])
        if len(included) == 1:
            label = f"{names[i]} only"
        else:
            label = f"{names[i]} and above"
        thresholds.append({"label": label, "included_tiers": included})

    return thresholds


# ---------------------------------------------------------------------------
# Per-rule analysis
# ---------------------------------------------------------------------------

def compute_tier_counts(risk_tier_series: pd.Series) -> dict[str, int]:
    """Compute row count for each risk_tier value plus total.

    Args:
        risk_tier_series: Series containing risk_tier strings.

    Returns:
        Dictionary with counts for 'Medium', 'High', 'Critical', 'Low', and 'total'.
    """
    counts = risk_tier_series.value_counts().to_dict()
    return {
        "Medium": counts.get("Medium", 0),
        "High": counts.get("High", 0),
        "Critical": counts.get("Critical", 0),
        "Low": counts.get("Low", 0),
        "total": len(risk_tier_series),
    }


def format_tier_counts_line(tier_counts: dict[str, int]) -> str:
    """Format Tier counts line showing Medium, High, Critical, Low, plus total."""
    med = tier_counts.get("Medium", 0)
    high = tier_counts.get("High", 0)
    crit = tier_counts.get("Critical", 0)
    low = tier_counts.get("Low", 0)
    total = tier_counts.get("total", med + high + crit + low)
    return f"Tier counts: Medium: {med:,}, High: {high:,}, Critical: {crit:,}, Low: {low:,}, Total: {total:,}"


def per_rule_analysis(
    df: pd.DataFrame,
    rule_names: list[str],
) -> list[dict[str, Any]]:
    """Compute per-rule firing stats against ground truth.

    For each rule: how many normal vs fraud rows it fired on, plus
    precision and recall (treating the single rule as the only signal).
    """
    results: list[dict[str, Any]] = []
    is_fraud = df["is_fraud_actual"].astype(bool)

    for rule_name in rule_names:
        fired = df["triggered_rules"].str.contains(
            rf"(?:^|,)\s*{rule_name}\s*(?:,|$)", regex=True, na=False,
        )
        fraud_fired = (fired & is_fraud).sum()
        normal_fired = (fired & ~is_fraud).sum()
        total_fired = fraud_fired + normal_fired
        total_fraud = is_fraud.sum()

        precision = fraud_fired / total_fired if total_fired > 0 else None
        recall = fraud_fired / total_fraud if total_fraud > 0 else None

        results.append({
            "rule": rule_name,
            "normal_fired": normal_fired,
            "fraud_fired": fraud_fired,
            "precision": precision,
            "recall": recall,
        })

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(value: float | None, pct: bool = True) -> str:
    """Format a metric for display."""
    if value is None:
        return "N/A (zero denominator)"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def generate_report(
    scored_csv: Path = _DEFAULT_SCORED_CSV,
    config_path: Path = _DEFAULT_RULES_CONFIG,
    reports_dir: Path = _DEFAULT_REPORTS_DIR,
) -> Path:
    """Run the full evaluation and write a markdown report.

    Args:
        scored_csv: Path to transactions_scored.csv.
        config_path: Path to rules_config.yaml.
        reports_dir: Directory to write the report into.

    Returns:
        Path to the written markdown report.
    """
    # 1. Load data
    df = pd.read_csv(scored_csv, keep_default_na=False, na_values=[""])
    df["is_fraud_actual"] = df["is_fraud_actual"].astype(bool)
    df["triggered_rules"] = df["triggered_rules"].fillna("").astype(str)

    total_rows = len(df)
    total_fraud = int(df["is_fraud_actual"].sum())
    total_normal = total_rows - total_fraud

    tier_counts = compute_tier_counts(df["risk_tier"])
    tier_counts_line = format_tier_counts_line(tier_counts)

    # 2. Load tier thresholds from config
    tiers = load_risk_tiers(config_path)
    thresholds = build_tier_thresholds(tiers)
    enabled_rules = load_enabled_rules(config_path)

    # 3. Per-threshold evaluation
    threshold_results: list[dict[str, Any]] = []
    for th in thresholds:
        y_pred = df["risk_tier"].isin(th["included_tiers"])
        y_true = df["is_fraud_actual"]
        counts = compute_confusion_counts(y_true, y_pred)
        metrics = compute_classification_metrics(counts)
        threshold_results.append({
            "label": th["label"],
            "included_tiers": th["included_tiers"],
            "counts": counts,
            "metrics": metrics,
        })

    # 4. Per-rule analysis
    rule_results = per_rule_analysis(df, enabled_rules)

    # 5. Print summary
    print(f"\n{'=' * 70}")
    print("EVALUATION REPORT — Rule Engine vs Ground Truth")
    print(f"{'=' * 70}")
    print(f"Total rows: {total_rows:,}  |  Fraud: {total_fraud:,}  |  Normal: {total_normal:,}")
    print(tier_counts_line)
    print()

    for tr in threshold_results:
        c = tr["counts"]
        m = tr["metrics"]
        print(f"--- Threshold: {tr['label']} ---")
        print(f"  TP: {c['tp']:,}  FP: {c['fp']:,}  TN: {c['tn']:,}  FN: {c['fn']:,}")
        print(f"  Precision: {_fmt(m['precision'])}  Recall: {_fmt(m['recall'])}")
        print(f"  F1: {_fmt(m['f1'])}  FPR: {_fmt(m['fpr'])}")
        print(f"  Normal rows flagged: {c['fp']:,}")
        print()

    print("--- Per-Rule Analysis ---")
    for rr in rule_results:
        print(
            f"  {rr['rule']:35s}  normal_fired={rr['normal_fired']:5d}  "
            f"fraud_fired={rr['fraud_fired']:5d}  "
            f"precision={_fmt(rr['precision'])}  recall={_fmt(rr['recall'])}"
        )
    print()

    # 6. Save confusion matrix heatmap for the first threshold (broadest)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for tr in threshold_results:
        c = tr["counts"]
        z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
        x_labels = ["Predicted Normal", "Predicted Fraud"]
        y_labels = ["Actual Normal", "Actual Fraud"]

        fig = ff.create_annotated_heatmap(
            z,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(
            title_text=f"Confusion Matrix — {tr['label']}",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(l=80, r=40, t=60, b=40),
        )
        safe_label = tr["label"].replace(" ", "_").replace("/", "_").lower()
        fig_path = reports_dir / f"confusion_matrix_{safe_label}.html"
        fig.write_html(str(fig_path))
        print(f"  Saved confusion matrix chart: {fig_path}")

    # 7. Write markdown report
    run_date = date.today().isoformat()
    report_path = reports_dir / f"evaluation_{run_date}.md"

    lines: list[str] = []
    lines.append(f"# Evaluation Report — {run_date}")
    lines.append("")
    lines.append("> **Note:** These figures describe one synthetic dataset generated on this date.")
    lines.append("> Ground-truth (`is_fraud_actual`) is used here **only** for development validation.")
    lines.append("> In production, this column would not exist.")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(tier_counts_line)
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total rows | {total_rows:,} |")
    lines.append(f"| Fraud rows | {total_fraud:,} ({total_fraud / total_rows * 100:.2f}%) |")
    lines.append(f"| Normal rows | {total_normal:,} |")
    lines.append(f"| Tier counts | Medium: {tier_counts['Medium']:,}, High: {tier_counts['High']:,}, Critical: {tier_counts['Critical']:,}, Low: {tier_counts['Low']:,}, Total: {tier_counts['total']:,} |")
    lines.append("")

    # Per-threshold sections
    lines.append("## Threshold Evaluation")
    lines.append("")

    for tr in threshold_results:
        c = tr["counts"]
        m = tr["metrics"]
        lines.append(f"### {tr['label']}")
        lines.append(f"Tiers included: {', '.join(sorted(tr['included_tiers']))}")
        lines.append("")
        lines.append("**Confusion Matrix**")
        lines.append("")
        lines.append("| | Predicted Normal | Predicted Fraud |")
        lines.append("|---|---|---|")
        lines.append(f"| **Actual Normal** | {c['tn']:,} (TN) | {c['fp']:,} (FP) |")
        lines.append(f"| **Actual Fraud** | {c['fn']:,} (FN) | {c['tp']:,} (TP) |")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Precision | {_fmt(m['precision'])} |")
        lines.append(f"| Recall | {_fmt(m['recall'])} |")
        lines.append(f"| F1 Score | {_fmt(m['f1'])} |")
        lines.append(f"| False-Positive Rate | {_fmt(m['fpr'])} |")
        lines.append(f"| Normal rows flagged | {c['fp']:,} |")
        lines.append("")

    # Per-rule section
    lines.append("## Per-Rule Analysis")
    lines.append("")
    lines.append("| Rule | Normal Fired | Fraud Fired | Precision | Recall |")
    lines.append("|------|-------------|-------------|-----------|--------|")
    for rr in rule_results:
        lines.append(
            f"| {rr['rule']} | {rr['normal_fired']:,} | {rr['fraud_fired']:,} "
            f"| {_fmt(rr['precision'])} | {_fmt(rr['recall'])} |"
        )
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- **Synthetic data**: all transactions are generated, not real.")
    lines.append("- **No legitimate high-risk-merchant purchases**: the `high_risk_merchant` rule "
                 "fires on 0 normal rows because the generator only places high-risk merchant "
                 "categories in fraud-injected transactions.")
    lines.append("- **No legitimate rapid bursts**: all rapid-fire (5 txns in 5 min) patterns "
                 "are injected as fraud; real data would contain some legitimate bursts.")
    lines.append("- **Amount distribution narrower than real data**: per-card amounts are drawn "
                 "from a single log-normal; real spending has heavier tails and seasonal variation.")
    lines.append("- **Warm-up rows are neutral by design**: the first 5 transactions per card "
                 "establish the home country and are never fraud-injected, inflating TN count.")
    lines.append("- **Grey-list travel is the only legitimate high-tier activity**: ~4% of cards "
                 "get one travel episode to a high-tier grey-list country; this is the sole source "
                 "of false positives for the `foreign_high_risk_country` rule.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate rule engine against ground truth.")
    parser.add_argument("--csv", type=Path, default=_DEFAULT_SCORED_CSV, help="Path to scored CSV.")
    parser.add_argument("--config", type=Path, default=_DEFAULT_RULES_CONFIG, help="Path to rules_config.yaml.")
    parser.add_argument("--reports-dir", type=Path, default=_DEFAULT_REPORTS_DIR, help="Reports output directory.")
    args = parser.parse_args()

    generate_report(
        scored_csv=args.csv,
        config_path=args.config,
        reports_dir=args.reports_dir,
    )

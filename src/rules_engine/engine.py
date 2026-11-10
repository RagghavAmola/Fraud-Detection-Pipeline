"""Execution engine for evaluating fraud detection rules on enriched transactions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.rules_engine.rule_definitions import (
    Condition,
    Rule,
    RulesConfig,
    load_rules_config,
    validate_rules_against_df,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "rules_config.yaml"
_DEFAULT_ENRICHED_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "transactions_enriched.csv"
)
_DEFAULT_SCORED_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "transactions_scored.csv"
)


class RuleEngine:
    """Evaluates declarative fraud rules on transaction DataFrames."""

    def __init__(
        self,
        config: RulesConfig | str | Path | dict[str, Any] | None = None,
    ) -> None:
        """Initialize RuleEngine with given configuration or default YAML file."""
        if config is None:
            config = _DEFAULT_CONFIG_PATH

        if isinstance(config, RulesConfig):
            self.config = config
        else:
            self.config = load_rules_config(config)

        self.enabled_rules: list[Rule] = [r for r in self.config.rules if r.enabled]

    def assert_dtypes(self, df: pd.DataFrame) -> None:
        """Assert that every column referenced by any rule exists and matches expected types.

        All rules in the configuration (including disabled ones) are verified.
        """
        validate_rules_against_df(self.config.rules, df)

    def load_transactions(self, file_path: str | Path = _DEFAULT_ENRICHED_PATH) -> pd.DataFrame:
        """Load enriched transactions from CSV and assert field dtypes.

        Uses keep_default_na=False and na_values=[''] to preserve string values like 'N/A'.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transactions file not found: {path}")

        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        self.assert_dtypes(df)
        return df

    def evaluate_condition(self, df: pd.DataFrame, condition: Condition) -> pd.Series:
        """Evaluate a single condition vectorially on a DataFrame.

        Returns a boolean Series of shape (len(df),).
        """
        col = df[condition.field]
        op = condition.operator
        val = condition.threshold

        if op == "==":
            return col == val
        elif op == "!=":
            return col != val
        elif op == ">":
            return col > val
        elif op == ">=":
            return col >= val
        elif op == "<":
            return col < val
        elif op == "<=":
            return col <= val
        elif op == "in":
            return col.isin(val)
        else:
            raise ValueError(f"Unsupported operator: '{op}'")

    def evaluate_rule(self, df: pd.DataFrame, rule: Rule) -> pd.Series:
        """Evaluate a rule vectorially across all of its conditions.

        For single-condition rules, evaluates that condition.
        For multi-condition ('all') rules, performs logical AND across all conditions.

        Returns a boolean Series of shape (len(df),).
        """
        if not rule.conditions:
            return pd.Series(False, index=df.index, dtype=bool)

        mask = pd.Series(True, index=df.index, dtype=bool)
        for cond in rule.conditions:
            cond_mask = self.evaluate_condition(df, cond)
            mask = mask & cond_mask
        return mask

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score transactions using all enabled rules.

        Adds three columns:
          - fraud_score (int): sum of weights of triggered enabled rules, capped at score_cap
          - risk_tier (str): tier label according to configured boundaries
          - triggered_rules (str): comma-separated rule names that fired on this transaction

        Original columns are preserved.
        """
        self.assert_dtypes(df)
        out_df = df.copy()
        n_rows = len(out_df)

        # Vectorized evaluation of each enabled rule
        rule_masks: dict[str, np.ndarray] = {}
        for rule in self.enabled_rules:
            mask_series = self.evaluate_rule(out_df, rule)
            rule_masks[rule.name] = mask_series.to_numpy()

        # 1. Compute fraud score vectorially
        fraud_score = np.zeros(n_rows, dtype=int)
        for rule in self.enabled_rules:
            mask = rule_masks[rule.name]
            fraud_score += mask.astype(int) * rule.weight

        # Cap score
        fraud_score = np.clip(fraud_score, 0, self.config.score_cap)
        out_df["fraud_score"] = fraud_score

        # 2. Assign risk tier vectorially via np.select
        tier_conditions = [
            (fraud_score >= tier.min_score) & (fraud_score <= tier.max_score)
            for tier in self.config.risk_tiers
        ]
        tier_choices = [tier.name for tier in self.config.risk_tiers]
        out_df["risk_tier"] = np.select(tier_conditions, tier_choices, default="Unknown")

        # 3. Build triggered_rules vectorially (per-rule string concatenation, no row loop)
        triggered = np.full(n_rows, "", dtype=object)
        for rule in self.enabled_rules:
            mask = rule_masks[rule.name]
            triggered = np.where(
                mask,
                np.where(triggered == "", rule.name, triggered + "," + rule.name),
                triggered,
            )
        out_df["triggered_rules"] = pd.Series(triggered, index=out_df.index, dtype=str)

        return out_df


def generate_rule_report(scored_df: pd.DataFrame, engine: RuleEngine) -> None:
    """Print rule evaluation counts and risk tier distributions split by is_fraud_actual.

    Note: This reporting function is the only code in rules_engine that accesses is_fraud_actual.
    """
    if "is_fraud_actual" not in scored_df.columns:
        print("Note: 'is_fraud_actual' column not found; skipping ground-truth evaluation report.")
        return

    fraud_mask = scored_df["is_fraud_actual"].to_numpy(dtype=bool)
    normal_mask = ~fraud_mask

    n_total = len(scored_df)
    n_normal = int(np.sum(normal_mask))
    n_fraud = int(np.sum(fraud_mask))

    print("\n" + "=" * 80)
    print("RULE ENGINE EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Transactions: {n_total:,} (Normal: {n_normal:,}, Fraud: {n_fraud:,})\n")

    # Table 1: Per-Rule Trigger Counts
    print("--- Per-Rule Trigger Counts (Enabled Rules) ---")
    header = f"{'Rule Name':<30} | {'Weight':<6} | {'Normal Fired':<14} | {'Fraud Fired':<14} | {'Total Fired':<14}"
    print(header)
    print("-" * len(header))

    for rule in engine.enabled_rules:
        mask = engine.evaluate_rule(scored_df, rule).to_numpy()
        normal_fired = int(np.sum(mask & normal_mask))
        fraud_fired = int(np.sum(mask & fraud_mask))
        total_fired = int(np.sum(mask))
        print(
            f"{rule.name:<30} | {rule.weight:<6} | {normal_fired:<14,d} | {fraud_fired:<14,d} | {total_fired:<14,d}"
        )

    # Table 2: Risk Tier Distribution
    print("\n--- Risk Tier Distribution ---")
    tier_header = f"{'Risk Tier':<12} | {'Score Range':<12} | {'Normal Count (%)':<22} | {'Fraud Count (%)':<22} | {'Total Count (%)':<22}"
    print(tier_header)
    print("-" * len(tier_header))

    tier_series = scored_df["risk_tier"]
    for tier in engine.config.risk_tiers:
        tier_mask = (tier_series == tier.name).to_numpy()
        norm_count = int(np.sum(tier_mask & normal_mask))
        fraud_count = int(np.sum(tier_mask & fraud_mask))
        total_count = int(np.sum(tier_mask))

        norm_pct = (norm_count / n_normal * 100) if n_normal > 0 else 0.0
        fraud_pct = (fraud_count / n_fraud * 100) if n_fraud > 0 else 0.0
        total_pct = (total_count / n_total * 100) if n_total > 0 else 0.0

        score_range = f"{tier.min_score}-{tier.max_score}"
        norm_str = f"{norm_count:,} ({norm_pct:5.2f}%)"
        fraud_str = f"{fraud_count:,} ({fraud_pct:5.2f}%)"
        total_str = f"{total_count:,} ({total_pct:5.2f}%)"

        print(
            f"{tier.name:<12} | {score_range:<12} | {norm_str:<22} | {fraud_str:<22} | {total_str:<22}"
        )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = RuleEngine()
    logger.info("Loading enriched data from %s", _DEFAULT_ENRICHED_PATH)
    enriched_df = engine.load_transactions(_DEFAULT_ENRICHED_PATH)

    logger.info("Evaluating rules and scoring transactions...")
    scored_df = engine.score_transactions(enriched_df)

    _DEFAULT_SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(_DEFAULT_SCORED_PATH, index=False)
    logger.info("Saved %d scored transactions to %s", len(scored_df), _DEFAULT_SCORED_PATH)

    # Ground-truth reporting
    generate_rule_report(scored_df, engine)

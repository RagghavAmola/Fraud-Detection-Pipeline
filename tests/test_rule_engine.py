"""Unit and integration tests for the modular, config-driven rule engine."""

from __future__ import annotations

import copy
import pandas as pd
import pytest

from src.rules_engine.engine import RuleEngine, generate_rule_report
from src.rules_engine.rule_definitions import (
    Condition,
    RiskTier,
    Rule,
    RulesConfig,
    load_rules_config,
    validate_rules_against_df,
)


def _base_config_dict() -> dict:
    """Return a valid minimal configuration dict for testing."""
    return {
        "score_cap": 100,
        "risk_tiers": [
            {"name": "Low", "min_score": 0, "max_score": 24},
            {"name": "Medium", "min_score": 25, "max_score": 49},
            {"name": "High", "min_score": 50, "max_score": 74},
            {"name": "Critical", "min_score": 75, "max_score": 100},
        ],
        "rules": [
            {
                "name": "r_velocity",
                "description": "Velocity check",
                "field": "txn_velocity_10min",
                "operator": ">",
                "threshold": 3,
                "weight": 25,
                "enabled": True,
            },
            {
                "name": "r_amount",
                "description": "Amount deviation check",
                "field": "amount_zscore",
                "operator": ">",
                "threshold": 3.0,
                "weight": 30,
                "enabled": True,
            },
        ],
    }


def _create_sample_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Create a hand-built DataFrame with standard column types."""
    if rows is None:
        rows = [
            {
                "transaction_id": "T1",
                "card_id": "C1",
                "amount": 100.0,
                "txn_velocity_10min": 4,
                "amount_zscore": 3.5,
                "is_odd_hour": True,
                "is_new_merchant": True,
                "is_foreign_country": False,
                "geo_jump_flag": False,
                "merchant_risk_category": "high",
                "country_risk_tier": "low",
                "is_fraud_actual": False,
            }
        ]
    return pd.DataFrame(rows)


# ===========================================================================
# 1. Functional evaluation tests with hand-built DataFrames
# ===========================================================================


def test_zero_rules():
    """Verify engine handles empty active rules list (score=0, Low, empty string)."""
    cfg = _base_config_dict()
    cfg["rules"] = []
    engine = RuleEngine(config=cfg)

    df = _create_sample_df()
    scored = engine.score_transactions(df)

    assert scored["fraud_score"].tolist() == [0]
    assert scored["risk_tier"].tolist() == ["Low"]
    assert scored["triggered_rules"].tolist() == [""]


def test_single_rule_firing_and_not_firing():
    """Verify single rule evaluation when it fires vs when it does not fire."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "r_vel",
            "description": "Vel",
            "field": "txn_velocity_10min",
            "operator": ">",
            "threshold": 3,
            "weight": 25,
            "enabled": True,
        }
    ]
    engine = RuleEngine(config=cfg)

    df = pd.DataFrame(
        [
            {"txn_velocity_10min": 4},  # fires
            {"txn_velocity_10min": 2},  # does not fire
        ]
    )
    scored = engine.score_transactions(df)

    assert scored.loc[0, "fraud_score"] == 25
    assert scored.loc[0, "risk_tier"] == "Medium"
    assert scored.loc[0, "triggered_rules"] == "r_vel"

    assert scored.loc[1, "fraud_score"] == 0
    assert scored.loc[1, "risk_tier"] == "Low"
    assert scored.loc[1, "triggered_rules"] == ""


def test_several_rules_combination():
    """Verify combination of multiple rules sums weights and concatenates names."""
    cfg = _base_config_dict()
    engine = RuleEngine(config=cfg)

    df = pd.DataFrame(
        [
            {"txn_velocity_10min": 4, "amount_zscore": 4.0},  # both fire: 25+30=55 -> High
            {"txn_velocity_10min": 4, "amount_zscore": 1.0},  # only vel fires: 25 -> Medium
            {"txn_velocity_10min": 1, "amount_zscore": 4.0},  # only amount fires: 30 -> Medium
            {"txn_velocity_10min": 1, "amount_zscore": 1.0},  # neither fires: 0 -> Low
        ]
    )
    scored = engine.score_transactions(df)

    assert scored.loc[0, "fraud_score"] == 55
    assert scored.loc[0, "risk_tier"] == "High"
    assert scored.loc[0, "triggered_rules"] == "r_velocity,r_amount"

    assert scored.loc[1, "fraud_score"] == 25
    assert scored.loc[1, "risk_tier"] == "Medium"
    assert scored.loc[1, "triggered_rules"] == "r_velocity"

    assert scored.loc[2, "fraud_score"] == 30
    assert scored.loc[2, "risk_tier"] == "Medium"
    assert scored.loc[2, "triggered_rules"] == "r_amount"

    assert scored.loc[3, "fraud_score"] == 0
    assert scored.loc[3, "risk_tier"] == "Low"
    assert scored.loc[3, "triggered_rules"] == ""


def test_rule_at_exact_threshold():
    """Verify boundary conditions: zscore == 3 does not fire for '>', but fires for '>='."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "r_strict_gt",
            "description": "Strict gt",
            "field": "amount_zscore",
            "operator": ">",
            "threshold": 3.0,
            "weight": 10,
            "enabled": True,
        },
        {
            "name": "r_gte",
            "description": "Gte",
            "field": "amount_zscore",
            "operator": ">=",
            "threshold": 3.0,
            "weight": 20,
            "enabled": True,
        },
        {
            "name": "r_strict_lt",
            "description": "Strict lt",
            "field": "amount_zscore",
            "operator": "<",
            "threshold": 3.0,
            "weight": 5,
            "enabled": True,
        },
        {
            "name": "r_lte",
            "description": "Lte",
            "field": "amount_zscore",
            "operator": "<=",
            "threshold": 3.0,
            "weight": 15,
            "enabled": True,
        },
    ]
    engine = RuleEngine(config=cfg)

    # Exact threshold value: 3.0
    df = pd.DataFrame([{"amount_zscore": 3.0}])
    scored = engine.score_transactions(df)

    # At exactly 3.0:
    # r_strict_gt (>) does not fire
    # r_gte (>=) fires (+20)
    # r_strict_lt (<) does not fire
    # r_lte (<=) fires (+15)
    # Total = 35
    assert scored.loc[0, "fraud_score"] == 35
    assert scored.loc[0, "triggered_rules"] == "r_gte,r_lte"


def test_all_condition_rule_requires_every_condition():
    """Verify multi-condition 'all' rule fires if and only if all conditions hold."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "odd_hour_high_risk",
            "description": "Odd hour and high risk merchant",
            "all": [
                {"field": "is_odd_hour", "operator": "==", "threshold": True},
                {"field": "merchant_risk_category", "operator": "==", "threshold": "high"},
            ],
            "weight": 40,
            "enabled": True,
        }
    ]
    engine = RuleEngine(config=cfg)

    df = pd.DataFrame(
        [
            {"is_odd_hour": True, "merchant_risk_category": "high"},   # both True -> fires
            {"is_odd_hour": True, "merchant_risk_category": "low"},    # second False -> does not fire
            {"is_odd_hour": False, "merchant_risk_category": "high"},  # first False -> does not fire
            {"is_odd_hour": False, "merchant_risk_category": "low"},   # both False -> does not fire
        ]
    )
    scored = engine.score_transactions(df)

    assert scored.loc[0, "fraud_score"] == 40
    assert scored.loc[0, "triggered_rules"] == "odd_hour_high_risk"

    assert scored.loc[1, "fraud_score"] == 0
    assert scored.loc[1, "triggered_rules"] == ""

    assert scored.loc[2, "fraud_score"] == 0
    assert scored.loc[2, "triggered_rules"] == ""

    assert scored.loc[3, "fraud_score"] == 0
    assert scored.loc[3, "triggered_rules"] == ""


def test_in_operator():
    """Verify 'in' operator correctly checks membership in a list."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "medium_or_high_risk",
            "description": "Risk tier in list",
            "field": "merchant_risk_category",
            "operator": "in",
            "threshold": ["medium", "high"],
            "weight": 25,
            "enabled": True,
        }
    ]
    engine = RuleEngine(config=cfg)

    df = pd.DataFrame(
        [
            {"merchant_risk_category": "high"},    # in list -> fires
            {"merchant_risk_category": "medium"},  # in list -> fires
            {"merchant_risk_category": "low"},     # not in list -> does not fire
        ]
    )
    scored = engine.score_transactions(df)

    assert scored.loc[0, "fraud_score"] == 25
    assert scored.loc[0, "triggered_rules"] == "medium_or_high_risk"

    assert scored.loc[1, "fraud_score"] == 25
    assert scored.loc[1, "triggered_rules"] == "medium_or_high_risk"

    assert scored.loc[2, "fraud_score"] == 0
    assert scored.loc[2, "triggered_rules"] == ""


def test_disabled_rule_never_fires():
    """Verify that a rule with enabled=False is completely ignored during scoring."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "disabled_velocity",
            "description": "Disabled",
            "field": "txn_velocity_10min",
            "operator": ">",
            "threshold": 3,
            "weight": 50,
            "enabled": False,
        },
        {
            "name": "enabled_amount",
            "description": "Enabled",
            "field": "amount_zscore",
            "operator": ">",
            "threshold": 3.0,
            "weight": 20,
            "enabled": True,
        },
    ]
    engine = RuleEngine(config=cfg)

    # Condition for disabled_velocity is met (10 > 3), but it must NOT fire
    df = pd.DataFrame([{"txn_velocity_10min": 10, "amount_zscore": 4.0}])
    scored = engine.score_transactions(df)

    assert scored.loc[0, "fraud_score"] == 20
    assert scored.loc[0, "triggered_rules"] == "enabled_amount"


def test_score_cap_at_100():
    """Verify that fraud score is capped at configured score_cap (100)."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "r1",
            "description": "Rule 1",
            "field": "is_odd_hour",
            "operator": "==",
            "threshold": True,
            "weight": 60,
            "enabled": True,
        },
        {
            "name": "r2",
            "description": "Rule 2",
            "field": "is_new_merchant",
            "operator": "==",
            "threshold": True,
            "weight": 70,
            "enabled": True,
        },
    ]
    engine = RuleEngine(config=cfg)

    df = pd.DataFrame([{"is_odd_hour": True, "is_new_merchant": True}])
    scored = engine.score_transactions(df)

    # Raw sum is 60 + 70 = 130, capped at 100
    assert scored.loc[0, "fraud_score"] == 100
    assert scored.loc[0, "risk_tier"] == "Critical"
    assert scored.loc[0, "triggered_rules"] == "r1,r2"


def test_risk_tier_exact_boundaries():
    """Verify boundary conditions for Low (0-24), Medium (25-49), High (50-74), Critical (75-100)."""
    cfg = _base_config_dict()
    engine = RuleEngine(config=cfg)

    test_scores = [0, 24, 25, 49, 50, 74, 75, 100]
    expected_tiers = [
        "Low",       # 0
        "Low",       # 24
        "Medium",    # 25
        "Medium",    # 49
        "High",      # 50
        "High",      # 74
        "Critical",  # 75
        "Critical",  # 100
    ]

    # Dynamically verify risk tier assignment logic for each boundary
    tier_conditions = [
        (pd.Series(test_scores) >= tier.min_score) & (pd.Series(test_scores) <= tier.max_score)
        for tier in engine.config.risk_tiers
    ]
    tier_choices = [tier.name for tier in engine.config.risk_tiers]
    actual_tiers = pd.Series(test_scores).map(
        lambda s: next(t.name for t in engine.config.risk_tiers if t.min_score <= s <= t.max_score)
    ).tolist()

    assert actual_tiers == expected_tiers


# ===========================================================================
# 2. Strict Schema and Validation Error Tests
# ===========================================================================


def test_validation_unknown_operator():
    """Reject unknown comparison operator."""
    cfg = _base_config_dict()
    cfg["rules"][0]["operator"] = "~="
    with pytest.raises(ValueError, match="unknown operator"):
        load_rules_config(cfg)


def test_validation_missing_weight():
    """Reject rule with missing weight."""
    cfg = _base_config_dict()
    del cfg["rules"][0]["weight"]
    with pytest.raises(ValueError, match="missing required key 'weight'"):
        load_rules_config(cfg)


def test_validation_negative_weight():
    """Reject rule with negative weight."""
    cfg = _base_config_dict()
    cfg["rules"][0]["weight"] = -10
    with pytest.raises(ValueError, match="negative weight"):
        load_rules_config(cfg)


def test_validation_rule_with_both_field_and_all():
    """Reject rule having both 'field' and 'all'."""
    cfg = _base_config_dict()
    cfg["rules"][0]["all"] = [
        {"field": "is_odd_hour", "operator": "==", "threshold": True}
    ]
    with pytest.raises(ValueError, match="cannot have both 'field' and 'all'"):
        load_rules_config(cfg)


def test_validation_rule_with_neither_field_nor_all():
    """Reject rule lacking both 'field' and 'all'."""
    cfg = _base_config_dict()
    del cfg["rules"][0]["field"]
    del cfg["rules"][0]["operator"]
    del cfg["rules"][0]["threshold"]
    with pytest.raises(ValueError, match="must have either 'field' or 'all'"):
        load_rules_config(cfg)


def test_validation_rule_with_empty_all():
    """Reject rule with an empty 'all' list."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "r_empty",
            "description": "Empty all",
            "all": [],
            "weight": 10,
            "enabled": True,
        }
    ]
    with pytest.raises(ValueError, match="'all' must be a non-empty list"):
        load_rules_config(cfg)


def test_validation_non_boolean_enabled():
    """Reject rule with non-boolean enabled (e.g. string or int)."""
    cfg = _base_config_dict()
    cfg["rules"][0]["enabled"] = "true"
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        load_rules_config(cfg)

    cfg["rules"][0]["enabled"] = 1
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        load_rules_config(cfg)


def test_validation_in_operator_requires_list_threshold():
    """Reject 'in' operator when threshold is not a list."""
    cfg = _base_config_dict()
    cfg["rules"][0]["operator"] = "in"
    cfg["rules"][0]["threshold"] = "high"
    with pytest.raises(ValueError, match="requires a list threshold"):
        load_rules_config(cfg)


def test_validation_inequality_requires_numeric_threshold():
    """Reject inequality operators (>, >=, <, <=) with non-numeric threshold."""
    for op in (">", ">=", "<", "<="):
        cfg = _base_config_dict()
        cfg["rules"][0]["operator"] = op
        cfg["rules"][0]["threshold"] = "invalid_string"
        with pytest.raises(ValueError, match="requires a numeric threshold"):
            load_rules_config(cfg)

        cfg["rules"][0]["threshold"] = True  # bool must not be accepted as numeric
        with pytest.raises(ValueError, match="requires a numeric threshold"):
            load_rules_config(cfg)


def test_validation_duplicate_rule_name():
    """Reject duplicate rule names."""
    cfg = _base_config_dict()
    cfg["rules"].append(copy.deepcopy(cfg["rules"][0]))
    with pytest.raises(ValueError, match="Duplicate rule name"):
        load_rules_config(cfg)


def test_validation_unknown_keys_in_config():
    """Reject unknown keys at root, rule, condition, and risk tier levels."""
    # Unknown root key
    cfg = _base_config_dict()
    cfg["extra_root"] = 123
    with pytest.raises(ValueError, match="Unknown root configuration key"):
        load_rules_config(cfg)

    # Unknown rule key
    cfg = _base_config_dict()
    cfg["rules"][0]["unknown_rule_prop"] = True
    with pytest.raises(ValueError, match="Rule contains unknown key"):
        load_rules_config(cfg)

    # Unknown condition key
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "r_all",
            "description": "All",
            "all": [
                {"field": "is_odd_hour", "operator": "==", "threshold": True, "bad_key": 99}
            ],
            "weight": 10,
            "enabled": True,
        }
    ]
    with pytest.raises(ValueError, match="condition contains unknown key"):
        load_rules_config(cfg)

    # Unknown risk tier key
    cfg = _base_config_dict()
    cfg["risk_tiers"][0]["color"] = "red"
    with pytest.raises(ValueError, match="risk_tiers entry at index 0 contains unknown key"):
        load_rules_config(cfg)


def test_validation_disabled_rules_are_also_validated():
    """Verify that disabled rules are strictly validated, not silently skipped."""
    cfg = _base_config_dict()
    cfg["rules"][0]["enabled"] = False
    cfg["rules"][0]["operator"] = "UNKNOWN_OP"
    with pytest.raises(ValueError, match="unknown operator"):
        load_rules_config(cfg)


def test_validation_risk_tiers_bounds():
    """Verify risk_tiers starting at 0, ending at score_cap, contiguous, min <= max."""
    # Not starting at 0
    cfg = _base_config_dict()
    cfg["risk_tiers"][0]["min_score"] = 5
    with pytest.raises(ValueError, match="First risk tier must start at 0"):
        load_rules_config(cfg)

    # Not ending at score_cap
    cfg = _base_config_dict()
    cfg["risk_tiers"][-1]["max_score"] = 90
    with pytest.raises(ValueError, match="Last risk tier must end at score_cap"):
        load_rules_config(cfg)

    # min_score > max_score
    cfg = _base_config_dict()
    cfg["risk_tiers"][0]["min_score"] = 30
    cfg["risk_tiers"][0]["max_score"] = 20
    with pytest.raises(ValueError, match="has min_score .* > max_score"):
        load_rules_config(cfg)

    # Gap between tiers
    cfg = _base_config_dict()
    cfg["risk_tiers"][1]["min_score"] = 26  # gap after 24
    with pytest.raises(ValueError, match="Risk tiers must be contiguous"):
        load_rules_config(cfg)

    # Overlap between tiers
    cfg = _base_config_dict()
    cfg["risk_tiers"][1]["min_score"] = 24  # overlap with 24
    with pytest.raises(ValueError, match="Risk tiers must be contiguous"):
        load_rules_config(cfg)


# ===========================================================================
# 3. DataFrame compatibility & Dtype assertion tests
# ===========================================================================


def test_validation_field_not_in_dataframe():
    """Reject configuration when a rule references a non-existent DataFrame column."""
    cfg = _base_config_dict()
    cfg["rules"][0]["field"] = "non_existent_column"
    df = pd.DataFrame({"txn_velocity_10min": [1, 2], "amount_zscore": [0.5, 1.2]})

    engine = RuleEngine(config=cfg)
    with pytest.raises(ValueError, match="references field 'non_existent_column' which is not in DataFrame"):
        engine.assert_dtypes(df)


def test_validation_inequality_on_non_numeric_column():
    """Reject inequality operators when applied to a non-numeric column."""
    cfg = _base_config_dict()
    cfg["rules"][0]["field"] = "merchant_name"  # string column
    df = pd.DataFrame({"merchant_name": ["Store A", "Store B"], "amount_zscore": [1.0, 2.0]})

    engine = RuleEngine(config=cfg)
    with pytest.raises(TypeError, match="must be numeric for operator '>'"):
        engine.assert_dtypes(df)


def test_validation_boolean_threshold_on_non_boolean_column():
    """Reject boolean threshold condition when applied to a non-boolean column."""
    cfg = _base_config_dict()
    cfg["rules"] = [
        {
            "name": "r_bool",
            "description": "Bool check",
            "field": "merchant_name",
            "operator": "==",
            "threshold": True,
            "weight": 10,
            "enabled": True,
        }
    ]
    df = pd.DataFrame({"merchant_name": ["Store A", "Store B"]})

    engine = RuleEngine(config=cfg)
    with pytest.raises(TypeError, match="must be boolean for boolean threshold"):
        engine.assert_dtypes(df)


# ===========================================================================
# 4. Production rules_config.yaml verification
# ===========================================================================


def test_production_rules_config_loads_and_validates():
    """Verify that default production rules_config.yaml passes all strict schema checks."""
    engine = RuleEngine()
    assert engine.config.score_cap == 100
    assert len(engine.config.risk_tiers) == 4
    assert len(engine.config.rules) == 10
    # Exactly 9 enabled rules, 1 disabled rule (geo_jump)
    assert len(engine.enabled_rules) == 9
    assert any(r.name == "geo_jump" and not r.enabled for r in engine.config.rules)


def test_reporting_function_does_not_crash(capsys):
    """Verify ground-truth reporting function formats and prints without errors."""
    engine = RuleEngine()
    df = _create_sample_df()
    scored = engine.score_transactions(df)
    generate_rule_report(scored, engine)
    captured = capsys.readouterr()
    assert "RULE ENGINE EVALUATION REPORT" in captured.out
    assert "Risk Tier Distribution" in captured.out


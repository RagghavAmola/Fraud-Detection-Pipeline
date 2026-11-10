"""Rule definitions and configuration loader for fraud detection rule engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml


ALLOWED_ROOT_KEYS = frozenset({"score_cap", "risk_tiers", "rules"})
ALLOWED_RULE_KEYS = frozenset(
    {"name", "description", "weight", "enabled", "field", "operator", "threshold", "all"}
)
ALLOWED_CONDITION_KEYS = frozenset({"field", "operator", "threshold"})
ALLOWED_TIER_KEYS = frozenset({"name", "min_score", "max_score"})
ALLOWED_OPERATORS = frozenset({"==", "!=", ">", ">=", "<", "<=", "in"})
INEQUALITY_OPERATORS = frozenset({">", ">=", "<", "<="})


@dataclass(frozen=True)
class Condition:
    """A single rule condition."""

    field: str
    operator: str
    threshold: Any


@dataclass(frozen=True)
class Rule:
    """A fraud detection rule comprising one or more conditions."""

    name: str
    description: str
    weight: int
    enabled: bool
    conditions: list[Condition]


@dataclass(frozen=True)
class RiskTier:
    """A risk tier defined by score bounds."""

    name: str
    min_score: int
    max_score: int


@dataclass(frozen=True)
class RulesConfig:
    """Full configuration including score cap, risk tiers, and rules."""

    score_cap: int
    risk_tiers: list[RiskTier]
    rules: list[Rule]


def _validate_condition(
    cond_dict: dict[str, Any], rule_name: str
) -> Condition:
    """Validate a single condition dictionary and return a Condition object."""
    unknown = set(cond_dict.keys()) - ALLOWED_CONDITION_KEYS
    if unknown:
        raise ValueError(
            f"Rule '{rule_name}' condition contains unknown key(s): {sorted(unknown)}"
        )

    for req in ("field", "operator", "threshold"):
        if req not in cond_dict:
            raise ValueError(f"Rule '{rule_name}' condition missing required key '{req}'")

    field = cond_dict["field"]
    if not isinstance(field, str) or not field.strip():
        raise ValueError(f"Rule '{rule_name}' condition 'field' must be a non-empty string")

    operator = cond_dict["operator"]
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(
            f"Rule '{rule_name}' condition for field '{field}' has unknown operator: '{operator}'. "
            f"Allowed operators: {sorted(ALLOWED_OPERATORS)}"
        )

    threshold = cond_dict["threshold"]

    if operator == "in":
        if not isinstance(threshold, list):
            raise ValueError(
                f"Rule '{rule_name}' condition for field '{field}' with 'in' operator requires a list threshold, "
                f"got {type(threshold).__name__}"
            )
    elif operator in INEQUALITY_OPERATORS:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"Rule '{rule_name}' condition for field '{field}' with '{operator}' operator requires a numeric threshold, "
                f"got {type(threshold).__name__}"
            )

    return Condition(field=field, operator=operator, threshold=threshold)


def _validate_risk_tiers(tiers_data: Any, score_cap: int) -> list[RiskTier]:
    """Validate risk tier definitions."""
    if not isinstance(tiers_data, list) or len(tiers_data) == 0:
        raise ValueError("risk_tiers must be a non-empty list")

    parsed_tiers: list[RiskTier] = []
    for idx, tier_dict in enumerate(tiers_data):
        if not isinstance(tier_dict, dict):
            raise ValueError(f"risk_tiers entry at index {idx} must be a dictionary")

        unknown = set(tier_dict.keys()) - ALLOWED_TIER_KEYS
        if unknown:
            raise ValueError(
                f"risk_tiers entry at index {idx} contains unknown key(s): {sorted(unknown)}"
            )

        for req in ("name", "min_score", "max_score"):
            if req not in tier_dict:
                raise ValueError(f"risk_tiers entry at index {idx} missing required key '{req}'")

        name = tier_dict["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"risk_tier name at index {idx} must be a non-empty string")

        min_score = tier_dict["min_score"]
        max_score = tier_dict["max_score"]

        if isinstance(min_score, bool) or not isinstance(min_score, int):
            raise ValueError(f"risk_tier '{name}' min_score must be an integer, got {type(min_score).__name__}")
        if isinstance(max_score, bool) or not isinstance(max_score, int):
            raise ValueError(f"risk_tier '{name}' max_score must be an integer, got {type(max_score).__name__}")

        if min_score > max_score:
            raise ValueError(
                f"risk_tier '{name}' has min_score ({min_score}) > max_score ({max_score})"
            )

        parsed_tiers.append(RiskTier(name=name, min_score=min_score, max_score=max_score))

    # Boundary checks
    if parsed_tiers[0].min_score != 0:
        raise ValueError(f"First risk tier must start at 0, got {parsed_tiers[0].min_score}")

    if parsed_tiers[-1].max_score != score_cap:
        raise ValueError(
            f"Last risk tier must end at score_cap ({score_cap}), got {parsed_tiers[-1].max_score}"
        )

    # Contiguity check
    for i in range(1, len(parsed_tiers)):
        prev = parsed_tiers[i - 1]
        curr = parsed_tiers[i]
        if curr.min_score != prev.max_score + 1:
            raise ValueError(
                f"Risk tiers must be contiguous: gap or overlap between '{prev.name}' (max {prev.max_score}) "
                f"and '{curr.name}' (min {curr.min_score})"
            )

    return parsed_tiers


def _validate_rule(rule_dict: Any, seen_names: set[str]) -> Rule:
    """Validate a single rule dictionary and return a Rule object."""
    if not isinstance(rule_dict, dict):
        raise ValueError("Each rule entry must be a dictionary")

    unknown = set(rule_dict.keys()) - ALLOWED_RULE_KEYS
    if unknown:
        raise ValueError(f"Rule contains unknown key(s): {sorted(unknown)}")

    if "name" not in rule_dict:
        raise ValueError("Rule missing required key 'name'")
    name = rule_dict["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Rule 'name' must be a non-empty string")

    if name in seen_names:
        raise ValueError(f"Duplicate rule name: '{name}'")
    seen_names.add(name)

    description = rule_dict.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Rule '{name}' description must be a string")

    if "weight" not in rule_dict:
        raise ValueError(f"Rule '{name}' missing required key 'weight'")
    weight = rule_dict["weight"]
    if isinstance(weight, bool) or not isinstance(weight, int):
        raise ValueError(f"Rule '{name}' weight must be an integer, got {type(weight).__name__}")
    if weight < 0:
        raise ValueError(f"Rule '{name}' has negative weight: {weight}")

    if "enabled" not in rule_dict:
        raise ValueError(f"Rule '{name}' missing required key 'enabled'")
    enabled = rule_dict["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(f"Rule '{name}' enabled must be a boolean, got {type(enabled).__name__}")

    has_field = "field" in rule_dict
    has_all = "all" in rule_dict

    if has_field and has_all:
        raise ValueError(f"Rule '{name}' cannot have both 'field' and 'all'")
    if not has_field and not has_all:
        raise ValueError(f"Rule '{name}' must have either 'field' or 'all'")

    conditions: list[Condition] = []
    if has_field:
        # Single condition rule: must have operator and threshold
        cond_keys = {"field", "operator", "threshold"}
        cond_dict = {k: rule_dict[k] for k in cond_keys if k in rule_dict}
        conditions.append(_validate_condition(cond_dict, name))
    else:
        # Multi-condition 'all' rule
        if "operator" in rule_dict or "threshold" in rule_dict:
            raise ValueError(f"Rule '{name}' with 'all' cannot have top-level 'operator' or 'threshold'")

        all_list = rule_dict["all"]
        if not isinstance(all_list, list) or len(all_list) == 0:
            raise ValueError(f"Rule '{name}' 'all' must be a non-empty list")

        for idx, cond_item in enumerate(all_list):
            if not isinstance(cond_item, dict):
                raise ValueError(f"Rule '{name}' 'all' item at index {idx} must be a dictionary")
            conditions.append(_validate_condition(cond_item, name))

    return Rule(
        name=name,
        description=description,
        weight=weight,
        enabled=enabled,
        conditions=conditions,
    )


def validate_rules_against_df(
    rules: Sequence[Rule], df: pd.DataFrame
) -> None:
    """Validate that every field referenced by ANY rule exists in df and has compatible dtypes.

    All rules (including disabled ones) are validated against df.
    """
    for rule in rules:
        for cond in rule.conditions:
            if cond.field not in df.columns:
                raise ValueError(
                    f"Rule '{rule.name}' references field '{cond.field}' which is not in DataFrame columns: "
                    f"{list(df.columns)}"
                )

            col = df[cond.field]
            if cond.operator in INEQUALITY_OPERATORS:
                if not pd.api.types.is_numeric_dtype(col):
                    raise TypeError(
                        f"Rule '{rule.name}' field '{cond.field}' must be numeric for operator '{cond.operator}', "
                        f"got dtype '{col.dtype}'"
                    )

            if isinstance(cond.threshold, bool):
                if not pd.api.types.is_bool_dtype(col):
                    raise TypeError(
                        f"Rule '{rule.name}' field '{cond.field}' must be boolean for boolean threshold, "
                        f"got dtype '{col.dtype}'"
                    )


def load_rules_config(
    config_source: str | Path | dict[str, Any],
    df: pd.DataFrame | None = None,
) -> RulesConfig:
    """Load and strictly validate rule engine configuration.

    Validates ALL rules (including disabled ones). If a DataFrame is provided,
    also validates that all referenced fields exist and match expected dtypes.

    Args:
        config_source: Path to YAML config file or pre-loaded dictionary.
        df: Optional DataFrame to validate field existence and dtypes against.

    Returns:
        Validated RulesConfig object.
    """
    if isinstance(config_source, (str, Path)):
        config_path = Path(config_source)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        raw_config = config_source
    else:
        raise TypeError(f"config_source must be a Path, str, or dict, got {type(config_source).__name__}")

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration must be a dictionary at the root level")

    unknown_root = set(raw_config.keys()) - ALLOWED_ROOT_KEYS
    if unknown_root:
        raise ValueError(f"Unknown root configuration key(s): {sorted(unknown_root)}")

    for req in ("score_cap", "risk_tiers", "rules"):
        if req not in raw_config:
            raise ValueError(f"Configuration missing required root key '{req}'")

    score_cap = raw_config["score_cap"]
    if isinstance(score_cap, bool) or not isinstance(score_cap, int) or score_cap <= 0:
        raise ValueError(f"score_cap must be a positive integer, got {score_cap}")

    risk_tiers = _validate_risk_tiers(raw_config["risk_tiers"], score_cap)

    rules_data = raw_config["rules"]
    if not isinstance(rules_data, list):
        raise ValueError("rules must be a list")

    seen_names: set[str] = set()
    parsed_rules = [_validate_rule(r, seen_names) for r in rules_data]

    if df is not None:
        validate_rules_against_df(parsed_rules, df)

    return RulesConfig(
        score_cap=score_cap,
        risk_tiers=risk_tiers,
        rules=parsed_rules,
    )

"""Feature Report Script.

Computes and reports feature distribution rates across normal and fraudulent transactions
from data/processed/transactions_enriched.csv. Compares measured rates against Phase 2 baselines
and flags any significant discrepancies or weak separation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = _REPO_ROOT / "data" / "processed" / "transactions_enriched.csv"

# Expected baselines (normal_rate_pct, fraud_rate_pct) in percent values
BASELINES: dict[str, tuple[float, float]] = {
    "is_odd_hour": (1.4, 30.6),
    "is_new_merchant": (10.3, 71.8),
    "amount_zscore > 3": (0.6, 28.9),
    "is_foreign_country": (0.7, 13.6),
    "is_foreign_country AND country_risk_tier == high": (0.1, 3.9),
    "txn_velocity_10min >= 4": (0.0, 8.6),
    "geo_jump_flag": (0.1, 0.6),
}


def generate_feature_report(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed transactions file not found at: {data_path}")

    df = pd.read_csv(data_path, keep_default_na=False)

    # Derived boolean flags for reporting
    df["amount_zscore > 3"] = df["amount_zscore"] > 3
    df["txn_velocity_10min >= 4"] = df["txn_velocity_10min"] >= 4
    df["is_foreign_country AND country_risk_tier == high"] = (
        df["is_foreign_country"] & (df["country_risk_tier"] == "high")
    )

    normal = df[~df["is_fraud_actual"]]
    fraud = df[df["is_fraud_actual"]]

    total_txns = len(df)
    n_normal = len(normal)
    n_fraud = len(fraud)

    print("=" * 80)
    print("FEATURE DISTRIBUTION REPORT: NORMAL VS FRAUD")
    print(f"Total Transactions: {total_txns:,} (Normal: {n_normal:,}, Fraud: {n_fraud:,})")
    print("=" * 80)

    TOLERANCE_PCT = 0.1  # 0.1 percentage points tolerance on percent values
    report_rows = []
    mismatches = []

    for feat, (exp_norm, exp_fraud) in BASELINES.items():
        norm_rate = float(normal[feat].mean())
        fraud_rate = float(fraud[feat].mean())
        norm_pct = norm_rate * 100.0
        fraud_pct = fraud_rate * 100.0
        lift = (fraud_rate / norm_rate) if norm_rate > 0 else float("inf")

        # Compare abs(measured - expected) against tolerance of 0.1 percentage points
        norm_diff = abs(norm_pct - exp_norm)
        fraud_diff = abs(fraud_pct - exp_fraud)
        is_mismatch = (norm_diff > TOLERANCE_PCT) or (fraud_diff > TOLERANCE_PCT)

        if is_mismatch:
            flag = "MISMATCH"
            mismatches.append(
                f"{feat}: measured (normal={norm_pct:.2f}%, fraud={fraud_pct:.2f}%) "
                f"differs from expected (normal={exp_norm:.1f}%, fraud={exp_fraud:.1f}%) "
                f"by more than {TOLERANCE_PCT} pp."
            )
        else:
            flag = "OK"

        report_rows.append({
            "Feature": feat,
            "Normal Rate": f"{norm_pct:.2f}%",
            "Fraud Rate": f"{fraud_pct:.2f}%",
            "Exp Normal": f"{exp_norm:.1f}%",
            "Exp Fraud": f"{exp_fraud:.1f}%",
            "Lift": f"{lift:.1f}x" if lift != float("inf") else "inf",
            "Status": flag,
        })

    report_df = pd.DataFrame(report_rows)
    print(report_df.to_string(index=False))
    print("=" * 80)

    if mismatches:
        print("\nMismatching features:")
        for disc in mismatches:
            print(f"  - {disc}")
    else:
        print("\nAll feature rates align closely with expected Phase 2 baselines.")

    return report_df


if __name__ == "__main__":
    generate_feature_report()

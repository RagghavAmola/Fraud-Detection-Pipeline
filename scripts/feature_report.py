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

# Expected baselines (normal_rate, fraud_rate) from Phase 2 specifications
BASELINES: dict[str, tuple[float, float]] = {
    "is_odd_hour": (0.014, 0.325),
    "is_new_merchant": (0.095, 0.730),
    "amount_zscore > 3": (0.006, 0.300),
    "is_foreign_country": (0.000, 0.145),
    "txn_velocity_10min >= 4": (0.000, 0.086),
    "geo_jump_flag": (0.000, 0.007),
}


def generate_feature_report(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed transactions file not found at: {data_path}")

    df = pd.read_csv(data_path, keep_default_na=False)

    # Derived boolean flags for reporting
    df["amount_zscore > 3"] = df["amount_zscore"] > 3
    df["txn_velocity_10min >= 4"] = df["txn_velocity_10min"] >= 4

    normal = df[~df["is_fraud_actual"]]
    fraud = df[df["is_fraud_actual"]]

    total_txns = len(df)
    n_normal = len(normal)
    n_fraud = len(fraud)

    print("=" * 80)
    print("FEATURE DISTRIBUTION REPORT: NORMAL VS FRAUD")
    print(f"Total Transactions: {total_txns:,} (Normal: {n_normal:,}, Fraud: {n_fraud:,})")
    print("=" * 80)

    report_rows = []
    discrepancies = []

    for feat, (exp_norm, exp_fraud) in BASELINES.items():
        norm_rate = float(normal[feat].mean())
        fraud_rate = float(fraud[feat].mean())
        lift = (fraud_rate / norm_rate) if norm_rate > 0 else float("inf")

        # Check for discrepancies:
        # 1. Normal vs Fraud similarity check (weak separation check)
        # Note: geo_jump_flag is weak by design (~0% vs ~0.7%)
        is_similar = abs(fraud_rate - norm_rate) < 0.005 and feat != "geo_jump_flag"

        # 2. Difference from expected baseline: threshold > 0.05 (5 percentage points)
        norm_diff = abs(norm_rate - exp_norm)
        fraud_diff = abs(fraud_rate - exp_fraud)
        differs_a_lot = norm_diff > 0.05 or fraud_diff > 0.05

        flag = "OK"
        if is_similar:
            flag = "FLAG: SIMILAR"
            discrepancies.append(f"{feat}: Normal ({norm_rate:.1%}) and Fraud ({fraud_rate:.1%}) are too similar.")
        elif differs_a_lot:
            flag = "FLAG: DRIFT"
            discrepancies.append(
                f"{feat}: Measured (norm={norm_rate:.1%}, fraud={fraud_rate:.1%}) "
                f"differs from expected (norm={exp_norm:.1%}, fraud={exp_fraud:.1%})."
            )

        report_rows.append({
            "Feature": feat,
            "Normal Rate": f"{norm_rate * 100:.2f}%",
            "Fraud Rate": f"{fraud_rate * 100:.2f}%",
            "Exp Normal": f"{exp_norm * 100:.1f}%",
            "Exp Fraud": f"{exp_fraud * 100:.1f}%",
            "Lift": f"{lift:.1f}x" if lift != float("inf") else "inf",
            "Status": flag,
        })

    report_df = pd.DataFrame(report_rows)
    print(report_df.to_string(index=False))
    print("=" * 80)

    if discrepancies:
        print("\nFlagged Discrepancies:")
        for disc in discrepancies:
            print(f"  - {disc}")
    else:
        print("\nAll feature rates align closely with expected Phase 2 baselines.")

    return report_df


if __name__ == "__main__":
    generate_feature_report()

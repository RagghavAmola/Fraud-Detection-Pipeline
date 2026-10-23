"""Unit and integration tests for feature engineering pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd

from src.transformation.feature_engineering import FeatureEngineer


def _build_base_transaction(
    txn_id: str,
    card_id: str,
    ts: datetime,
    amount: float = 50.0,
    merchant_name: str = "Test Store",
    merchant_category: str = "Grocery",
    transaction_country: str = "US",
    channel: str = "online",
    device_id: str = "DEV-123456",
    is_fraud_actual: bool = False,
) -> dict:
    """Helper to construct a single transaction dictionary with all required schema columns."""
    return {
        "transaction_id": txn_id,
        "card_id": card_id,
        "cardholder_name": f"User {card_id}",
        "timestamp": ts,
        "amount": amount,
        "merchant_name": merchant_name,
        "merchant_category": merchant_category,
        "merchant_country": transaction_country,
        "transaction_city": "TestCity",
        "transaction_country": transaction_country,
        "channel": channel,
        "device_id": device_id,
        "is_fraud_actual": is_fraud_actual,
    }


# ===========================================================================
# 1. Warm-up and std == 0 tests
# ===========================================================================

def test_warmup_handling():
    """Verify each card's first 5 transactions are flagged as warm-up and neutral values assigned."""
    engineer = FeatureEngineer(config={"warmup_length_txns": 5})
    start = datetime(2026, 1, 1, 10, 0, 0)
    
    # 7 transactions for CARD-1
    rows = [
        _build_base_transaction(
            txn_id=f"T{i}",
            card_id="CARD-1",
            ts=start + timedelta(days=i),
            amount=10.0 * (i + 1),
            merchant_name=f"Merchant-{i}",
            transaction_country="US" if i < 6 else "FR",
        )
        for i in range(7)
    ]
    df = pd.DataFrame(rows)
    enriched = engineer.compute_features(df)

    # First 5 rows are warm-up (indices 0..4)
    assert enriched["is_warmup"].tolist() == [True, True, True, True, True, False, False]

    # Warm-up rows must have neutral values, not NaNs
    warmup_rows = enriched.iloc[:5]
    assert (warmup_rows["amount_zscore"] == 0.0).all()
    assert (warmup_rows["is_new_merchant"] == False).all()
    assert (warmup_rows["is_foreign_country"] == False).all()

    # Post-warmup rows have active features
    post_warmup = enriched.iloc[5:]
    assert not post_warmup["is_warmup"].any()
    assert post_warmup["amount_zscore"].notna().all()


def test_std_zero_guard():
    """Verify that when historical std is 0 (all prior amounts identical), amount_zscore is 0.0."""
    engineer = FeatureEngineer(config={"warmup_length_txns": 5})
    start = datetime(2026, 1, 1, 10, 0, 0)

    # First 5 transactions have identical amounts of 50.0 (historical std = 0)
    # Row 6 has amount 100.0, Row 7 has amount 50.0
    amounts = [50.0, 50.0, 50.0, 50.0, 50.0, 100.0, 50.0]
    rows = [
        _build_base_transaction(
            txn_id=f"T{i}",
            card_id="CARD-1",
            ts=start + timedelta(hours=i),
            amount=amounts[i],
        )
        for i in range(len(amounts))
    ]
    df = pd.DataFrame(rows)
    enriched = engineer.compute_features(df)

    # At row 5 (6th transaction), prior amounts are [50, 50, 50, 50, 50], so std == 0.0
    # Guard must ensure amount_zscore is 0.0, avoiding division by zero or NaN/Inf
    assert enriched.loc[5, "is_warmup"] == False
    assert enriched.loc[5, "amount_zscore"] == 0.0
    z = cast(float, enriched.loc[5, "amount_zscore"])
    assert np.isfinite(z)

    # At row 6 (7th transaction), prior amounts are [50, 50, 50, 50, 50, 100] (std > 0)
    # Here amount_zscore should be computed normally
    assert enriched.loc[6, "is_warmup"] == False
    assert enriched.loc[6, "amount_zscore"] != 0.0


# ===========================================================================
# 2. Velocity windows tests
# ===========================================================================

def test_velocity_windows_and_current_row_inclusion():
    """Verify trailing 1h and 10min velocity counting, including current row inclusion."""
    engineer = FeatureEngineer()
    t0 = datetime(2026, 1, 1, 12, 0, 0)

    rows = [
        # T0: isolated txn -> count must be 1 (current row counted)
        _build_base_transaction("T0", "CARD-1", t0),
        # T1: 5 minutes later -> within 10m & within 1h -> v10=2, v1h=2
        _build_base_transaction("T1", "CARD-1", t0 + timedelta(minutes=5)),
        # T2: 12 minutes after T0 (7m after T1) -> within 10m of T1, within 1h of T0 & T1 -> v10=2, v1h=3
        _build_base_transaction("T2", "CARD-1", t0 + timedelta(minutes=12)),
        # T3: 75 minutes after T0 (63m after T2) -> outside 1h window -> v10=1, v1h=1
        _build_base_transaction("T3", "CARD-1", t0 + timedelta(minutes=75)),
        # T_other: different card at same timestamp as T0 -> separate group -> v10=1, v1h=1
        _build_base_transaction("T_other", "CARD-2", t0),
    ]
    df = pd.DataFrame(rows)
    enriched = engineer.compute_features(df)

    c1 = enriched[enriched["card_id"] == "CARD-1"].reset_index(drop=True)
    assert c1.loc[0, "txn_velocity_10min"] == 1
    assert c1.loc[0, "txn_velocity_1h"] == 1

    assert c1.loc[1, "txn_velocity_10min"] == 2
    assert c1.loc[1, "txn_velocity_1h"] == 2

    assert c1.loc[2, "txn_velocity_10min"] == 2
    assert c1.loc[2, "txn_velocity_1h"] == 3

    assert c1.loc[3, "txn_velocity_10min"] == 1
    assert c1.loc[3, "txn_velocity_1h"] == 1

    c2 = enriched[enriched["card_id"] == "CARD-2"].reset_index(drop=True)
    assert c2.loc[0, "txn_velocity_10min"] == 1
    assert c2.loc[0, "txn_velocity_1h"] == 1


# ===========================================================================
# 3. Geographic jump flag tests
# ===========================================================================

def test_geo_jump_flag():
    """Verify geo_jump_flag triggers on country change within 3h, and not outside 3h or on same country."""
    engineer = FeatureEngineer(config={"geo_window_hours": 3})
    t0 = datetime(2026, 1, 1, 10, 0, 0)

    rows = [
        # T0: Initial US txn -> no prior txn -> False
        _build_base_transaction("T0", "CARD-1", t0, transaction_country="US"),
        # T1: 2 hours later in FR (country changed, 2h < 3h) -> True
        _build_base_transaction("T1", "CARD-1", t0 + timedelta(hours=2), transaction_country="FR"),
        # T2: 5 hours after T1 in US (country changed, 5h >= 3h) -> False
        _build_base_transaction("T2", "CARD-1", t0 + timedelta(hours=7), transaction_country="US"),
        # T3: 1 hour after T2 in US (same country, 1h < 3h) -> False
        _build_base_transaction("T3", "CARD-1", t0 + timedelta(hours=8), transaction_country="US"),
    ]
    df = pd.DataFrame(rows)
    enriched = engineer.compute_features(df)

    assert enriched.loc[0, "geo_jump_flag"] == False
    assert enriched.loc[1, "geo_jump_flag"] == True
    assert enriched.loc[2, "geo_jump_flag"] == False
    assert enriched.loc[3, "geo_jump_flag"] == False


# ===========================================================================
# 4. Odd-hour tests
# ===========================================================================

def test_is_odd_hour():
    """Verify is_odd_hour at boundary hours 0, 1, 4, and 5."""
    engineer = FeatureEngineer(config={"odd_hour_window": {"start_hour": 1, "end_hour": 4}})
    
    rows = [
        _build_base_transaction("T0", "CARD-1", datetime(2026, 1, 1, 0, 30, 0)),
        _build_base_transaction("T1", "CARD-1", datetime(2026, 1, 1, 1, 0, 0)),
        _build_base_transaction("T2", "CARD-1", datetime(2026, 1, 1, 4, 59, 59)),
        _build_base_transaction("T3", "CARD-1", datetime(2026, 1, 1, 5, 0, 0)),
    ]
    df = pd.DataFrame(rows)
    enriched = engineer.compute_features(df)

    assert enriched.loc[0, "is_odd_hour"] == False  # Hour 0
    assert enriched.loc[1, "is_odd_hour"] == True   # Hour 1
    assert enriched.loc[2, "is_odd_hour"] == True   # Hour 4
    assert enriched.loc[3, "is_odd_hour"] == False  # Hour 5


# ===========================================================================
# 5. Foreign country & tie-breaking tests
# ===========================================================================

def test_is_foreign_country_and_alphabetical_tie_break():
    """Verify home country inference, False on warm-up, and deterministic alphabetical tie-breaking."""
    engineer = FeatureEngineer(config={"warmup_length_txns": 5})
    t0 = datetime(2026, 1, 1, 10, 0, 0)

    # CARD-1: warm-up countries: ["US", "US", "US", "CA", "CA"] -> mode is clearly "US"
    # Row 5 (post-warmup) in "US" -> False; Row 6 (post-warmup) in "FR" -> True
    c1_countries = ["US", "US", "US", "CA", "CA", "US", "FR"]
    c1_rows = [
        _build_base_transaction(f"C1_T{i}", "CARD-1", t0 + timedelta(days=i), transaction_country=c1_countries[i])
        for i in range(len(c1_countries))
    ]

    # CARD-2: warm-up countries: ["US", "US", "GB", "GB", "CA"]
    # Tied between "GB" (count 2) and "US" (count 2). Alphabetical tie-break chooses "GB".
    # Row 5 (post-warmup) in "GB" -> False; Row 6 (post-warmup) in "US" -> True
    c2_countries = ["US", "US", "GB", "GB", "CA", "GB", "US"]
    c2_rows = [
        _build_base_transaction(f"C2_T{i}", "CARD-2", t0 + timedelta(days=i), transaction_country=c2_countries[i])
        for i in range(len(c2_countries))
    ]

    df = pd.DataFrame(c1_rows + c2_rows)
    enriched = engineer.compute_features(df)

    c1_enriched = enriched[enriched["card_id"] == "CARD-1"].reset_index(drop=True)
    # Warm-up rows must be False
    assert (c1_enriched.loc[:4, "is_foreign_country"] == False).all()
    # Post-warmup: US is home -> False; FR is foreign -> True
    assert c1_enriched.loc[5, "is_foreign_country"] == False
    assert c1_enriched.loc[6, "is_foreign_country"] == True

    c2_enriched = enriched[enriched["card_id"] == "CARD-2"].reset_index(drop=True)
    # Warm-up rows must be False
    assert (c2_enriched.loc[:4, "is_foreign_country"] == False).all()
    # Alphabetical tie-breaker picked "GB" as home country:
    # Post-warmup GB is home -> False; US is foreign -> True
    assert c2_enriched.loc[5, "is_foreign_country"] == False
    assert c2_enriched.loc[6, "is_foreign_country"] == True


# ===========================================================================
# 6. Device ID "N/A" preservation test
# ===========================================================================

def test_device_id_na_preservation(tmp_path):
    """Verify device_id 'N/A' string is preserved and not converted to NaN on load."""
    csv_file = tmp_path / "test_txns.csv"
    data = {
        "transaction_id": ["TXN-1", "TXN-2"],
        "card_id": ["CARD-1", "CARD-1"],
        "cardholder_name": ["Alice", "Alice"],
        "timestamp": ["2026-01-01 10:00:00", "2026-01-01 11:00:00"],
        "amount": [10.5, 20.0],
        "merchant_name": ["Shop A", "Shop B"],
        "merchant_category": ["Grocery", "Retail"],
        "merchant_country": ["US", "US"],
        "transaction_city": ["City A", "City B"],
        "transaction_country": ["US", "US"],
        "channel": ["in-store", "online"],
        "device_id": ["N/A", "DEV-999"],
        "is_fraud_actual": [False, False],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    engineer = FeatureEngineer()
    loaded_df = engineer.load_raw_data(csv_file)

    assert loaded_df.loc[0, "device_id"] == "N/A"
    assert isinstance(loaded_df.loc[0, "device_id"], str)
    assert not pd.isna(loaded_df.loc[0, "device_id"])


# ===========================================================================
# 7. No-leakage test (future changes do not affect earlier features)
# ===========================================================================

def test_no_leakage_on_later_transactions():
    """Verify changing later transactions does NOT alter earlier rows' prior-only features."""
    engineer = FeatureEngineer(config={"warmup_length_txns": 3})
    t0 = datetime(2026, 1, 1, 10, 0, 0)

    # 6 initial transactions
    base_rows = [
        _build_base_transaction(
            txn_id=f"T{i}",
            card_id="CARD-1",
            ts=t0 + timedelta(days=i),
            amount=50.0 + i * 10,
            merchant_name=f"Store-{i % 3}",
            transaction_country="US",
        )
        for i in range(6)
    ]
    df_baseline = pd.DataFrame(base_rows)
    enriched_base = engineer.compute_features(df_baseline)

    # Modified dataset: change rows 4 and 5 drastically, and append row 6
    modified_rows = [dict(r) for r in base_rows]
    modified_rows[4]["amount"] = 9999.0
    modified_rows[4]["merchant_name"] = "Extreme Fraud Store"
    modified_rows[4]["transaction_country"] = "NG"
    modified_rows[5]["amount"] = 5555.0

    # Append row 6
    modified_rows.append(
        _build_base_transaction(
            txn_id="T6",
            card_id="CARD-1",
            ts=t0 + timedelta(days=6),
            amount=1234.0,
            merchant_name="Unknown New Shop",
            transaction_country="RU",
        )
    )

    df_modified = pd.DataFrame(modified_rows)
    enriched_modified = engineer.compute_features(df_modified)

    # Check earlier rows (0, 1, 2, 3) across all prior-only features
    prior_only_features = [
        "txn_velocity_1h",
        "txn_velocity_10min",
        "seconds_since_prev_txn",
        "amount_zscore",
        "is_new_merchant",
        "geo_jump_flag",
        "is_foreign_country",
    ]

    for feat in prior_only_features:
        base_earlier = cast(pd.Series, enriched_base.loc[:3, feat])
        mod_earlier =  cast(pd.Series, enriched_modified.loc[:3, feat])
        pd.testing.assert_series_equal(
            base_earlier,
            mod_earlier,
            check_names=False,
            obj=f"Leakage detected in feature: {feat}",
        )

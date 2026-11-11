"""Unit tests for SQLite database manager, models, and queries."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import inspect

from src.database.db_manager import DBManager
from src.database.models import (
    CountryRisk,
    FlaggedTransaction,
    RuleAudit,
    Transaction,
)


def _build_sample_scored_df(n: int = 6) -> pd.DataFrame:
    """Construct a small hand-built DataFrame with scored transactions."""
    rows = []
    tiers = ["Low", "Low", "Medium", "Medium", "High", "Critical"]
    amounts = [25.0, 50.0, 100.0, 150.0, 500.0, 1200.0]
    scores = [10, 20, 30, 45, 60, 85]
    rules_list = [
        "",
        "foreign_low_risk_country",
        "large_amount_deviation",
        "large_amount_deviation,odd_hour",
        "high_velocity,large_amount_deviation",
        "foreign_high_risk_country,high_velocity,large_amount_deviation",
    ]

    base_time = datetime(2026, 1, 1, 10, 0, 0)

    for i in range(min(n, len(tiers))):
        rows.append(
            {
                "transaction_id": f"TXN-{i + 1:04d}",
                "card_id": f"CARD-{(i % 2) + 1}",
                "cardholder_name": f"User {(i % 2) + 1}",
                "timestamp": base_time + timedelta(days=i * 5),
                "amount": amounts[i],
                "merchant_name": f"Merchant {i + 1}",
                "merchant_category": "Retail",
                "merchant_country": "US",
                "transaction_city": "Cityville",
                "transaction_country": "US",
                "channel": "in-store",
                "device_id": "N/A" if i % 2 == 0 else f"DEV-{1000 + i}",
                "is_fraud_actual": (i >= 4),
                "txn_velocity_1h": 1,
                "txn_velocity_10min": 1 if i < 4 else 4,
                "seconds_since_prev_txn": None if i == 0 else 86400.0 * i,
                "is_odd_hour": (i == 3),
                "merchant_risk_category": "low",
                "is_warmup": False,
                "amount_zscore": 0.5 if i < 2 else 3.5,
                "is_new_merchant": False,
                "geo_jump_flag": False,
                "is_foreign_country": False,
                "country_risk_tier": "low",
                "fraud_score": scores[i],
                "risk_tier": tiers[i],
                "triggered_rules": rules_list[i],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def memory_db():
    """Fixture providing a fresh in-memory SQLite DBManager instance."""
    db = DBManager(database_url="sqlite:///:memory:")
    db.reset_database()
    return db


# ===========================================================================
# 1. Schema & Naming Convention Tests (Item 12)
# ===========================================================================


def test_schema_table_and_column_names(memory_db):
    """Verify that all tables and columns use the strict 3-letter uppercase prefix naming."""
    inspector = inspect(memory_db.engine)
    tables = inspector.get_table_names()

    # Exact table names
    expected_tables = {
        "TXN_TRANSACTION",
        "FTX_FLAGGEDTRANSACTION",
        "RAU_RULEAUDIT",
        "CNR_COUNTRYRISK",
    }
    assert expected_tables.issubset(set(tables)), f"Missing tables: {expected_tables - set(tables)}"

    # Prefix enforcement on columns
    for table_name in expected_tables:
        prefix = table_name[:3] + "_"
        cols = inspector.get_columns(table_name)
        for col in cols:
            col_name = col["name"]
            assert col_name.startswith(prefix), f"Column {col_name} in {table_name} does not start with {prefix}"

    # Verify indexes
    txn_indexes = {idx["name"] for idx in inspector.get_indexes("TXN_TRANSACTION")}
    assert "ix_txn_card_id" in txn_indexes
    assert "ix_txn_timestamp" in txn_indexes
    assert "ix_txn_risk_tier" in txn_indexes
    assert "ix_txn_card_timestamp" in txn_indexes

    ftx_indexes = {idx["name"] for idx in inspector.get_indexes("FTX_FLAGGEDTRANSACTION")}
    assert "ix_ftx_transaction_id" in ftx_indexes
    assert "ix_ftx_risk_tier" in ftx_indexes


# ===========================================================================
# 2. Bulk Load & Row Counts (Item 3, 4, 7)
# ===========================================================================


def test_bulk_load_row_counts(memory_db):
    """Verify row counts in TXN_TRANSACTION, FTX_FLAGGEDTRANSACTION, and RAU_RULEAUDIT after load."""
    df = _build_sample_scored_df(6)
    stats = memory_db.bulk_load_transactions(df)

    assert stats["transactions_loaded"] == 6
    # Flagged tiers: Medium (2), High (1), Critical (1) -> 4 rows
    assert stats["flagged_loaded"] == 4

    with memory_db.get_session() as session:
        txn_count = session.query(Transaction).count()
        assert txn_count == 6

        ftx_count = session.query(FlaggedTransaction).count()
        assert ftx_count == 4

        # Verify only Medium, High, Critical were flagged
        ftx_tiers = {r.risk_tier for r in session.query(FlaggedTransaction).all()}
        assert ftx_tiers == {"Medium", "High", "Critical"}

        # Verify RuleAudit
        audits = session.query(RuleAudit).all()
        audit_dict = {a.rule_name: a.times_triggered for a in audits}
        assert audit_dict["large_amount_deviation"] == 4
        assert audit_dict["high_velocity"] == 2
        assert audit_dict["foreign_low_risk_country"] == 1


def test_null_and_string_preservation(memory_db):
    """Verify NULLs for empty numeric fields and string preservation for device_id 'N/A'."""
    df = _build_sample_scored_df(2)
    memory_db.bulk_load_transactions(df)

    with memory_db.get_session() as session:
        first_txn = session.query(Transaction).filter_by(transaction_id="TXN-0001").one()
        assert first_txn.seconds_since_prev_txn is None
        assert first_txn.device_id == "N/A"

        second_txn = session.query(Transaction).filter_by(transaction_id="TXN-0002").one()
        assert second_txn.seconds_since_prev_txn == 86400.0
        assert second_txn.device_id == "DEV-1001"


# ===========================================================================
# 3. Querying Flagged Transactions (Item 8)
# ===========================================================================


def test_query_flagged_by_tier(memory_db):
    """Query flagged transactions by single and multiple risk tiers."""
    df = _build_sample_scored_df(6)
    memory_db.bulk_load_transactions(df)

    # Query single tier
    high_flagged = memory_db.get_flagged(tier="High")
    assert len(high_flagged) == 1
    assert high_flagged[0].transaction_id == "TXN-0005"
    assert high_flagged[0].risk_tier == "High"

    # Query multiple tiers
    med_crit = memory_db.get_flagged(tier=["Medium", "Critical"])
    assert len(med_crit) == 3
    assert {f.risk_tier for f in med_crit} == {"Medium", "Critical"}

    # Query non-flagged tier returns empty
    low_flagged = memory_db.get_flagged(tier="Low")
    assert len(low_flagged) == 0


def test_query_flagged_by_date_range(memory_db):
    """Query flagged transactions by date range."""
    df = _build_sample_scored_df(6)
    # Timestamps in df:
    # TXN-0001 (Low): 2026-01-01
    # TXN-0002 (Low): 2026-01-06
    # TXN-0003 (Med): 2026-01-11
    # TXN-0004 (Med): 2026-01-16
    # TXN-0005 (High): 2026-01-21
    # TXN-0006 (Crit): 2026-01-26
    memory_db.bulk_load_transactions(df)

    # Query range between Jan 10 and Jan 22 -> TXN-0003, TXN-0004, TXN-0005
    results = memory_db.get_flagged(date_from="2026-01-10", date_to="2026-01-22")
    assert len(results) == 3
    result_ids = {r.transaction_id for r in results}
    assert result_ids == {"TXN-0003", "TXN-0004", "TXN-0005"}


# ===========================================================================
# 4. Review Outcome & Metrics (Item 8)
# ===========================================================================


def test_update_review_outcome(memory_db):
    """Verify updating review outcome sets reviewed=True and persists outcome."""
    df = _build_sample_scored_df(6)
    memory_db.bulk_load_transactions(df)

    # Update TXN-0003
    updated = memory_db.update_review_outcome("TXN-0003", "false_positive")
    assert updated is True

    # Update TXN-0005
    updated = memory_db.update_review_outcome("TXN-0005", "true_positive")
    assert updated is True

    # Non-existent txn
    assert memory_db.update_review_outcome("NON-EXISTENT", "true_positive") is False

    # Invalid outcome rejects
    with pytest.raises(ValueError, match="Invalid review outcome"):
        memory_db.update_review_outcome("TXN-0003", "invalid_outcome")

    with memory_db.get_session() as session:
        f3 = session.query(FlaggedTransaction).filter_by(transaction_id="TXN-0003").one()
        assert f3.reviewed is True
        assert f3.review_outcome == "false_positive"

        f5 = session.query(FlaggedTransaction).filter_by(transaction_id="TXN-0005").one()
        assert f5.reviewed is True
        assert f5.review_outcome == "true_positive"


def test_false_positive_rate_calculation(memory_db):
    """Verify false-positive rate returns None when unreviewed and FP / (TP + FP) when reviewed."""
    df = _build_sample_scored_df(6)
    memory_db.bulk_load_transactions(df)

    # Initially zero reviewed
    metrics_initial = memory_db.get_metrics()
    assert metrics_initial["total_flagged"] == 4
    assert metrics_initial["total_reviewed"] == 0
    assert metrics_initial["false_positive_rate"] is None

    # Review 1 TP and 3 FP: rate = 3 / (1 + 3) = 0.75
    memory_db.update_review_outcome("TXN-0003", "false_positive")
    memory_db.update_review_outcome("TXN-0004", "false_positive")
    memory_db.update_review_outcome("TXN-0005", "true_positive")
    memory_db.update_review_outcome("TXN-0006", "false_positive")

    metrics_reviewed = memory_db.get_metrics()
    assert metrics_reviewed["total_reviewed"] == 4
    assert metrics_reviewed["true_positives"] == 1
    assert metrics_reviewed["false_positives"] == 3
    assert pytest.approx(metrics_reviewed["false_positive_rate"], 0.001) == 0.75

    # Check exposure by tier
    exposure = metrics_reviewed["amount_exposure_by_tier"]
    assert exposure["Medium"] == 250.0  # 100 + 150
    assert exposure["High"] == 500.0
    assert exposure["Critical"] == 1200.0


# ===========================================================================
# 5. Database Reset & Recreate (Item 5)
# ===========================================================================


def test_reset_database_drops_and_recreates_tables(memory_db):
    """Verify reset_database cleanly wipes existing records while recreating empty tables."""
    df = _build_sample_scored_df(6)
    memory_db.bulk_load_transactions(df)

    with memory_db.get_session() as session:
        assert session.query(Transaction).count() == 6

    # Reset
    memory_db.reset_database()

    with memory_db.get_session() as session:
        assert session.query(Transaction).count() == 0
        assert session.query(FlaggedTransaction).count() == 0
        assert session.query(RuleAudit).count() == 0


# ===========================================================================
# 6. CountryRisk Reference Table (Item 9)
# ===========================================================================


def test_country_risk_loading(memory_db):
    """Verify loading jurisdiction reference data into CNR_COUNTRYRISK."""
    count = memory_db.load_country_risk()
    assert count == 50

    with memory_db.get_session() as session:
        cr_records = session.query(CountryRisk).all()
        assert len(cr_records) == 50
        codes = {c.iso_code for c in cr_records}
        assert "US" in codes
        assert "KP" in codes
        assert "AO" in codes


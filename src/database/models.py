"""SQLAlchemy ORM models for fraud detection and credit risk data."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base declarative class for database models."""


class Transaction(Base):
    """Transaction model storing raw and enriched feature data."""

    __tablename__ = "TXN_TRANSACTION"

    transaction_id = Column("TXN_TRANSACTION_ID", String, primary_key=True)
    card_id = Column("TXN_CARD_ID", String, nullable=False, index=True)
    cardholder_name = Column("TXN_CARDHOLDER_NAME", String, nullable=False)
    timestamp = Column("TXN_TIMESTAMP", DateTime, nullable=False, index=True)
    amount = Column("TXN_AMOUNT", Float, nullable=False)
    merchant_name = Column("TXN_MERCHANT_NAME", String, nullable=False)
    merchant_category = Column("TXN_MERCHANT_CATEGORY", String, nullable=False)
    merchant_country = Column("TXN_MERCHANT_COUNTRY", String, nullable=False)
    transaction_city = Column("TXN_TRANSACTION_CITY", String, nullable=False)
    transaction_country = Column("TXN_TRANSACTION_COUNTRY", String, nullable=False)
    channel = Column("TXN_CHANNEL", String, nullable=False)
    device_id = Column("TXN_DEVICE_ID", String, nullable=False)

    # Ground truth for evaluation only; operational metrics must never use it.
    is_fraud_actual = Column("TXN_IS_FRAUD_ACTUAL", Boolean, nullable=False)

    txn_velocity_1h = Column("TXN_TXN_VELOCITY_1H", Integer, nullable=False)
    txn_velocity_10min = Column("TXN_TXN_VELOCITY_10MIN", Integer, nullable=False)
    seconds_since_prev_txn = Column("TXN_SECONDS_SINCE_PREV_TXN", Float, nullable=True)
    is_odd_hour = Column("TXN_IS_ODD_HOUR", Boolean, nullable=False)
    merchant_risk_category = Column("TXN_MERCHANT_RISK_CATEGORY", String, nullable=False)
    is_warmup = Column("TXN_IS_WARMUP", Boolean, nullable=False)
    amount_zscore = Column("TXN_AMOUNT_ZSCORE", Float, nullable=False)
    is_new_merchant = Column("TXN_IS_NEW_MERCHANT", Boolean, nullable=False)
    geo_jump_flag = Column("TXN_GEO_JUMP_FLAG", Boolean, nullable=False)
    is_foreign_country = Column("TXN_IS_FOREIGN_COUNTRY", Boolean, nullable=False)
    country_risk_tier = Column("TXN_COUNTRY_RISK_TIER", String, nullable=False)
    fraud_score = Column("TXN_FRAUD_SCORE", Integer, nullable=False)
    risk_tier = Column("TXN_RISK_TIER", String, nullable=False, index=True)
    triggered_rules = Column("TXN_TRIGGERED_RULES", String, nullable=False)

    flagged_records = relationship("FlaggedTransaction", back_populates="transaction")

    __table_args__ = (
        Index("ix_txn_card_id", "TXN_CARD_ID"),
        Index("ix_txn_timestamp", "TXN_TIMESTAMP"),
        Index("ix_txn_risk_tier", "TXN_RISK_TIER"),
        Index("ix_txn_card_timestamp", "TXN_CARD_ID", "TXN_TIMESTAMP"),
    )


class FlaggedTransaction(Base):
    """Transactions flagged for analyst review based on configured risk tier thresholds."""

    __tablename__ = "FTX_FLAGGEDTRANSACTION"

    flagged_id = Column("FTX_FLAGGED_ID", Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(
        "FTX_TRANSACTION_ID",
        String,
        ForeignKey("TXN_TRANSACTION.TXN_TRANSACTION_ID"),
        nullable=False,
        index=True,
    )
    fraud_score = Column("FTX_FRAUD_SCORE", Integer, nullable=False)
    risk_tier = Column("FTX_RISK_TIER", String, nullable=False, index=True)
    triggered_rules = Column("FTX_TRIGGERED_RULES", String, nullable=False)
    flagged_at = Column("FTX_FLAGGED_AT", DateTime, nullable=False, default=datetime.now(UTC))
    reviewed = Column("FTX_REVIEWED", Boolean, nullable=False, default=False)
    review_outcome = Column("FTX_REVIEW_OUTCOME", String, nullable=True, default=None)

    transaction = relationship("Transaction", back_populates="flagged_records")

    __table_args__ = (
        Index("ix_ftx_transaction_id", "FTX_TRANSACTION_ID"),
        Index("ix_ftx_risk_tier", "FTX_RISK_TIER"),
    )


class RuleAudit(Base):
    """Audit table tracking frequency of rules triggered across processed transactions."""

    __tablename__ = "RAU_RULEAUDIT"

    rule_name = Column("RAU_RULE_NAME", String, primary_key=True)
    times_triggered = Column("RAU_TIMES_TRIGGERED", Integer, nullable=False, default=0)
    last_updated = Column("RAU_LAST_UPDATED", DateTime, nullable=False, default=datetime.now(UTC))


class CountryRisk(Base):
    """Country risk reference table storing jurisdiction threat tiers."""

    __tablename__ = "CNR_COUNTRYRISK"

    iso_code = Column("CNR_ISO_CD", String(2), primary_key=True)
    country_name = Column("CNR_COUNTRYNAME", String, nullable=False)
    threat_tier = Column("CNR_THREAT_TIER", String, nullable=False)

"""Database manager for SQLite database operations and data loading."""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.database.models import (
    Base,
    CountryRisk,
    FlaggedTransaction,
    RuleAudit,
    Transaction,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "db_config.yaml"
_DEFAULT_COUNTRY_RISK_PATH = (
    Path(__file__).resolve().parents[1] / "transformation" / "country_risk.yaml"
)

COLUMN_MAPPING: dict[str, str] = {
    "transaction_id": "TXN_TRANSACTION_ID",
    "card_id": "TXN_CARD_ID",
    "cardholder_name": "TXN_CARDHOLDER_NAME",
    "timestamp": "TXN_TIMESTAMP",
    "amount": "TXN_AMOUNT",
    "merchant_name": "TXN_MERCHANT_NAME",
    "merchant_category": "TXN_MERCHANT_CATEGORY",
    "merchant_country": "TXN_MERCHANT_COUNTRY",
    "transaction_city": "TXN_TRANSACTION_CITY",
    "transaction_country": "TXN_TRANSACTION_COUNTRY",
    "channel": "TXN_CHANNEL",
    "device_id": "TXN_DEVICE_ID",
    "is_fraud_actual": "TXN_IS_FRAUD_ACTUAL",
    "txn_velocity_1h": "TXN_TXN_VELOCITY_1H",
    "txn_velocity_10min": "TXN_TXN_VELOCITY_10MIN",
    "seconds_since_prev_txn": "TXN_SECONDS_SINCE_PREV_TXN",
    "is_odd_hour": "TXN_IS_ODD_HOUR",
    "merchant_risk_category": "TXN_MERCHANT_RISK_CATEGORY",
    "is_warmup": "TXN_IS_WARMUP",
    "amount_zscore": "TXN_AMOUNT_ZSCORE",
    "is_new_merchant": "TXN_IS_NEW_MERCHANT",
    "geo_jump_flag": "TXN_GEO_JUMP_FLAG",
    "is_foreign_country": "TXN_IS_FOREIGN_COUNTRY",
    "country_risk_tier": "TXN_COUNTRY_RISK_TIER",
    "fraud_score": "TXN_FRAUD_SCORE",
    "risk_tier": "TXN_RISK_TIER",
    "triggered_rules": "TXN_TRIGGERED_RULES",
}

VALID_REVIEW_OUTCOMES = frozenset({"true_positive", "false_positive", "unreviewed"})


class DBManager:
    """Manages database initialization, bulk ingestion, queries, and reporting metrics."""

    def __init__(
        self,
        database_url: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize DBManager with database URL and configuration.

        Args:
            database_url: Database connection string. If None, loaded from config.
            config_path: Path to YAML config file. Defaults to src/database/db_config.yaml.
        """
        self.config_path = Path(config_path or _DEFAULT_CONFIG_PATH)
        cfg: dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        self.database_url = database_url or cfg.get("database_url", "sqlite:///data/fraud.db")
        self.flagged_risk_tiers: list[str] = cfg.get(
            "flagged_risk_tiers", ["Medium", "High", "Critical"]
        )
        self.chunksize: int = cfg.get("bulk_load_chunksize", 1000)

        # If SQLite path, ensure parent directory exists
        if self.database_url.startswith("sqlite:///") and not self.database_url.startswith("sqlite:///:memory:"):
            db_file = self.database_url.replace("sqlite:///", "")
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def initialize_database(self) -> None:
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)

    def reset_database(self) -> None:
        """Drop all tables and recreate them cleanly without migrations."""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def bulk_load_transactions(
        self,
        data: pd.DataFrame | str | Path,
        flagged_tiers: Sequence[str] | None = None,
        chunksize: int | None = None,
    ) -> dict[str, int]:
        """Bulk load scored transactions into TXN_TRANSACTION, FTX_FLAGGEDTRANSACTION, and RAU_RULEAUDIT.

        Args:
            data: DataFrame or path to scored CSV file.
            flagged_tiers: Risk tiers to flag for review. Defaults to configured tiers.
            chunksize: Chunk size for SQL insert. Must satisfy chunksize * num_cols < 30000.

        Returns:
            Dictionary with counts of loaded records across tables.
        """
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        else:
            df = data.copy()

        # 1. Type casting and sanitization
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["is_fraud_actual"] = df["is_fraud_actual"].astype(bool)
        df["is_odd_hour"] = df["is_odd_hour"].astype(bool)
        df["is_warmup"] = df["is_warmup"].astype(bool)
        df["is_new_merchant"] = df["is_new_merchant"].astype(bool)
        df["geo_jump_flag"] = df["geo_jump_flag"].astype(bool)
        df["is_foreign_country"] = df["is_foreign_country"].astype(bool)

        df["amount"] = df["amount"].astype(float)
        df["amount_zscore"] = df["amount_zscore"].astype(float)
        # seconds_since_prev_txn can have NaN (NULL in SQL) for first transaction
        df["seconds_since_prev_txn"] = pd.to_numeric(df["seconds_since_prev_txn"], errors="coerce")

        df["txn_velocity_1h"] = df["txn_velocity_1h"].astype(int)
        df["txn_velocity_10min"] = df["txn_velocity_10min"].astype(int)
        df["fraud_score"] = df["fraud_score"].astype(int)

        df["device_id"] = df["device_id"].astype(str)
        df["triggered_rules"] = df["triggered_rules"].fillna("").astype(str)

        # Rename to database prefixed column names
        df_db = df.rename(columns=COLUMN_MAPPING)

        # 2. Chunksize calculation (Item 7: chunksize * num_cols < 30,000)
        num_cols = len(df_db.columns)
        max_allowed_chunk = 29999 // num_cols
        actual_chunk = min(chunksize or self.chunksize, max_allowed_chunk)

        # Insert transactions
        df_db.to_sql(
            name="TXN_TRANSACTION",
            con=self.engine,
            if_exists="append",
            index=False,
            chunksize=actual_chunk,
        )

        # 3. Populate FlaggedTransaction
        active_flagged_tiers = list(flagged_tiers if flagged_tiers is not None else self.flagged_risk_tiers)
        flagged_mask = df_db["TXN_RISK_TIER"].isin(active_flagged_tiers)
        flagged_source = df_db[flagged_mask]

        if not flagged_source.empty:
            flagged_records = pd.DataFrame(
                {
                    "FTX_TRANSACTION_ID": flagged_source["TXN_TRANSACTION_ID"],
                    "FTX_FRAUD_SCORE": flagged_source["TXN_FRAUD_SCORE"],
                    "FTX_RISK_TIER": flagged_source["TXN_RISK_TIER"],
                    "FTX_TRIGGERED_RULES": flagged_source["TXN_TRIGGERED_RULES"],
                    "FTX_FLAGGED_AT": flagged_source["TXN_TIMESTAMP"],
                    "FTX_REVIEWED": False,
                    "FTX_REVIEW_OUTCOME": None,
                }
            )
            flagged_chunk = min(2000, 29999 // len(flagged_records.columns))
            flagged_records.to_sql(
                name="FTX_FLAGGEDTRANSACTION",
                con=self.engine,
                if_exists="append",
                index=False,
                chunksize=flagged_chunk,
            )
            n_flagged = len(flagged_records)
        else:
            n_flagged = 0

        # 4. Rebuild RuleAudit from triggered_rules
        rule_counts: dict[str, int] = {}
        for rule_str in df_db["TXN_TRIGGERED_RULES"]:
            if rule_str:
                for r in rule_str.split(","):
                    r_clean = r.strip()
                    if r_clean:
                        rule_counts[r_clean] = rule_counts.get(r_clean, 0) + 1

        now = datetime.now(UTC)
        with self.get_session() as session:
            session.query(RuleAudit).delete()
            audit_objects = [
                RuleAudit(rule_name=r_name, times_triggered=cnt, last_updated=now)
                for r_name, cnt in sorted(rule_counts.items())
            ]
            session.add_all(audit_objects)

        return {
            "transactions_loaded": len(df_db),
            "flagged_loaded": n_flagged,
            "rules_audited": len(rule_counts),
        }

    def load_country_risk(self, yaml_path: str | Path | None = None) -> int:
        """Load country threat tiers from YAML into CNR_COUNTRYRISK table."""
        path = Path(yaml_path or _DEFAULT_COUNTRY_RISK_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Country risk YAML not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        countries = data.get("countries", [])
        with self.get_session() as session:
            session.query(CountryRisk).delete()
            objects = [
                CountryRisk(
                    iso_code=c["code"],
                    country_name=c["name"],
                    threat_tier=c["threat_tier"],
                )
                for c in countries
                if "code" in c and "name" in c and "threat_tier" in c
            ]
            session.add_all(objects)
            count = len(objects)

        return count

    def get_flagged(
        self,
        tier: str | Sequence[str] | None = None,
        date_from: datetime | str | None = None,
        date_to: datetime | str | None = None,
    ) -> list[FlaggedTransaction]:
        """Query flagged transactions filtered by risk tier and/or date range.

        Args:
            tier: Single risk tier string or list of risk tiers.
            date_from: Start of date range (inclusive).
            date_to: End of date range (inclusive).

        Returns:
            List of FlaggedTransaction ORM objects.
        """
        with self.get_session() as session:
            query = session.query(FlaggedTransaction)

            if tier is not None:
                if isinstance(tier, str):
                    query = query.filter(FlaggedTransaction.risk_tier == tier)
                else:
                    query = query.filter(FlaggedTransaction.risk_tier.in_(tier))

            if date_from is not None:
                dt_from = pd.to_datetime(date_from).to_pydatetime()
                query = query.filter(FlaggedTransaction.flagged_at >= dt_from)

            if date_to is not None:
                dt_to = pd.to_datetime(date_to).to_pydatetime()
                query = query.filter(FlaggedTransaction.flagged_at <= dt_to)

            return query.all()

    def update_review_outcome(self, transaction_id: str, outcome: str) -> bool:
        """Update review outcome for a flagged transaction and mark as reviewed.

        Args:
            transaction_id: ID of the transaction to review.
            outcome: Outcome string ('true_positive', 'false_positive', 'unreviewed').

        Returns:
            True if transaction was found and updated, False otherwise.
        """
        if outcome not in VALID_REVIEW_OUTCOMES:
            raise ValueError(
                f"Invalid review outcome '{outcome}'. Must be one of {sorted(VALID_REVIEW_OUTCOMES)}"
            )

        with self.get_session() as session:
            record = (
                session.query(FlaggedTransaction)
                .filter(FlaggedTransaction.transaction_id == transaction_id)
                .first()
            )
            if record is None:
                return False

            record.reviewed = cast(Any,True)
            record.review_outcome = cast(Any,outcome)
            return True

    def get_metrics(self) -> dict[str, Any]:
        """Compute aggregate operational metrics from database tables.

        Calculates:
          - total_flagged: Total count of flagged transactions.
          - amount_exposure_by_tier: Total dollar exposure grouped by risk tier.
          - false_positive_rate: false_positive / (true_positive + false_positive) among reviewed rows.
            Returns None if no transactions have been reviewed with a definitive outcome.

        Note: Operational metrics never access is_fraud_actual ground truth.
        """
        with self.get_session() as session:
            # 1. Total flagged
            total_flagged = session.query(func.count(FlaggedTransaction.flagged_id)).scalar() or 0

            # 2. Amount exposure by tier (join FlaggedTransaction with Transaction)
            exposure_query = (
                session.query(
                    FlaggedTransaction.risk_tier,
                    func.sum(Transaction.amount),
                )
                .join(
                    Transaction,
                    FlaggedTransaction.transaction_id == Transaction.transaction_id,
                )
                .group_by(FlaggedTransaction.risk_tier)
                .all()
            )
            amount_exposure: dict[str, float] = {
                tier: round(float(amt), 2) for tier, amt in exposure_query if amt is not None
            }

            # 3. False-positive rate from reviewed transactions
            tp_count = (
                session.query(func.count(FlaggedTransaction.flagged_id))
                .filter(FlaggedTransaction.review_outcome == "true_positive")
                .scalar()
                or 0
            )
            fp_count = (
                session.query(func.count(FlaggedTransaction.flagged_id))
                .filter(FlaggedTransaction.review_outcome == "false_positive")
                .scalar()
                or 0
            )
            total_reviewed = (
                session.query(func.count(FlaggedTransaction.flagged_id))
                .filter(FlaggedTransaction.reviewed == True)  
                .scalar()
                or 0
            )

            reviewed_decisions = tp_count + fp_count
            fp_rate = (fp_count / reviewed_decisions) if reviewed_decisions > 0 else None

            return {
                "total_flagged": total_flagged,
                "amount_exposure_by_tier": amount_exposure,
                "false_positive_rate": fp_rate,
                "total_reviewed": total_reviewed,
                "true_positives": tp_count,
                "false_positives": fp_count,
            }

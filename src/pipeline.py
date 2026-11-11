"""End-to-end fraud detection data pipeline.

Stages:
  1. Ingestion: Generate synthetic raw transactions (data_loader.py)
  2. Feature Engineering: Compute velocity, z-score, odd hours, country risk (feature_engineering.py)
  3. Rule Engine: Evaluate declarative rules, score transactions, assign tiers (engine.py)
  4. Database Load: Reset database and bulk load scored records, flagged transactions, and audits (db_manager.py)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.database.db_manager import DBManager
from src.ingestion.data_loader import (
    DEFAULT_OUTPUT_PATH as RAW_DATA_PATH,
)
from src.ingestion.data_loader import (
    generate_synthetic_transactions,
)
from src.rules_engine.engine import (
    _DEFAULT_SCORED_PATH as SCORED_DATA_PATH,
)
from src.rules_engine.engine import (
    RuleEngine,
    generate_rule_report,
)
from src.transformation.feature_engineering import (
    DEFAULT_PROCESSED_DATA_PATH as ENRICHED_DATA_PATH,
)
from src.transformation.feature_engineering import (
    FeatureEngineer,
)

logger = logging.getLogger("fraud_pipeline")


def run_pipeline() -> dict[str, Any]:
    """Execute the full fraud risk detection pipeline end-to-end.

    Returns:
        Dictionary summarizing row counts, execution times, and DB load metrics.
    """
    total_start = time.perf_counter()
    logger.info("==================================================================")
    logger.info("STARTING END-TO-END FRAUD RISK DETECTION PIPELINE")
    logger.info("==================================================================")

    # ------------------------------------------------------------------
    # Stage 1: Ingestion
    # ------------------------------------------------------------------
    s1_start = time.perf_counter()
    logger.info("[Stage 1/4] Ingestion: Generating raw synthetic transactions...")
    raw_df = generate_synthetic_transactions(output_path=RAW_DATA_PATH)
    s1_duration = time.perf_counter() - s1_start
    logger.info(
        "[Stage 1/4] Ingestion complete: %d rows written to %s in %.2fs",
        len(raw_df),
        RAW_DATA_PATH,
        s1_duration,
    )

    # ------------------------------------------------------------------
    # Stage 2: Feature Engineering
    # ------------------------------------------------------------------
    s2_start = time.perf_counter()
    logger.info("[Stage 2/4] Feature Engineering: Computing features...")
    engineer = FeatureEngineer()
    enriched_df = engineer.run_pipeline(
        raw_filepath=RAW_DATA_PATH,
        output_filepath=ENRICHED_DATA_PATH,
    )
    s2_duration = time.perf_counter() - s2_start
    logger.info(
        "[Stage 2/4] Feature Engineering complete: %d rows in, %d rows out to %s in %.2fs",
        len(raw_df),
        len(enriched_df),
        ENRICHED_DATA_PATH,
        s2_duration,
    )

    # ------------------------------------------------------------------
    # Stage 3: Rule Engine
    # ------------------------------------------------------------------
    s3_start = time.perf_counter()
    logger.info("[Stage 3/4] Rule Engine: Scoring transactions...")
    engine = RuleEngine()
    scored_df = engine.score_transactions(enriched_df)

    SCORED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(SCORED_DATA_PATH, index=False)
    s3_duration = time.perf_counter() - s3_start
    logger.info(
        "[Stage 3/4] Rule Engine complete: %d rows scored, saved to %s in %.2fs",
        len(scored_df),
        SCORED_DATA_PATH,
        s3_duration,
    )

    # Rule evaluation report (ground truth reporting)
    generate_rule_report(scored_df, engine)

    # ------------------------------------------------------------------
    # Stage 4: Database Load
    # ------------------------------------------------------------------
    s4_start = time.perf_counter()
    logger.info("[Stage 4/4] Database Load: Resetting DB and bulk loading records...")
    db_manager = DBManager()
    db_manager.reset_database()

    load_stats = db_manager.bulk_load_transactions(scored_df)
    country_risk_count = db_manager.load_country_risk()
    db_metrics = db_manager.get_metrics()
    s4_duration = time.perf_counter() - s4_start

    logger.info(
        "[Stage 4/4] Database Load complete in %.2fs:\n"
        "  - Transactions loaded: %d\n"
        "  - Flagged transactions: %d\n"
        "  - Rule audits populated: %d\n"
        "  - Country risk reference records: %d\n"
        "  - Total Dollar Exposure by Tier: %s",
        s4_duration,
        load_stats["transactions_loaded"],
        load_stats["flagged_loaded"],
        load_stats["rules_audited"],
        country_risk_count,
        db_metrics["amount_exposure_by_tier"],
    )

    total_duration = time.perf_counter() - total_start
    logger.info("==================================================================")
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN %.2fs", total_duration)
    logger.info("==================================================================")

    return {
        "status": "success",
        "total_duration_seconds": round(total_duration, 2),
        "stage_durations": {
            "ingestion": round(s1_duration, 2),
            "feature_engineering": round(s2_duration, 2),
            "rule_engine": round(s3_duration, 2),
            "database_load": round(s4_duration, 2),
        },
        "row_counts": {
            "raw": len(raw_df),
            "enriched": len(enriched_df),
            "scored": len(scored_df),
            "flagged": load_stats["flagged_loaded"],
        },
        "db_metrics": db_metrics,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_pipeline()


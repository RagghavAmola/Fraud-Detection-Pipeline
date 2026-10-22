"""Feature engineering module for calculating transaction and credit risk features."""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Final, Literal, cast

import pandas as pd
import yaml

# Ensure repository root is on sys.path for direct script execution and package imports
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ingestion.data_loader import (
    HIGH_RISK_MERCHANT_CATEGORIES,
    STANDARD_MERCHANT_CATEGORIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths and configuration
# ---------------------------------------------------------------------------
DEFAULT_RAW_DATA_PATH: Final[Path] = _REPO_ROOT / "data" / "raw" / "transactions.csv"
DEFAULT_PROCESSED_DATA_PATH: Final[Path] = (
    _REPO_ROOT / "data" / "processed" / "transactions_enriched.csv"
)
DEFAULT_CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "feature_config.yaml"
)

DEFAULT_FEATURE_CONFIG: Final[dict[str, Any]] = {
    "velocity_windows": {
        "velocity_1h": "1h",
        "velocity_10min": "10min",
        "closed": "both",
    },
    "odd_hour_window": {
        "start_hour": 1,
        "end_hour": 4,
    },
    "merchant_risk_tiers": {
        "high": list(HIGH_RISK_MERCHANT_CATEGORIES),
        "medium": [
            "Travel",
            "Retail",
            "Entertainment",
            "Restaurant",
        ],
        "low": [
            cat
            for cat in STANDARD_MERCHANT_CATEGORIES
            if cat not in ["Travel", "Retail", "Entertainment", "Restaurant"]
        ],
        "default": "low",
    },
    # Card warm-up and geographic velocity parameters
    "warmup_length_txns": 5,
    "geo_window_hours": 3,
}


# ---------------------------------------------------------------------------
# FeatureEngineer class
# ---------------------------------------------------------------------------
class FeatureEngineer:
    """Feature engineering pipeline for financial transaction risk detection.

    Loads, cleans, and transforms raw transaction logs into enriched features
    including multi-scale velocity counters, lag indicators, odd-hour flags,
    and merchant risk tier classifications.
    """

    def __init__(
        self,
        config: dict[str, Any] | str | Path | None = None,
    ) -> None:
        """Initialize FeatureEngineer with configuration.

        Args:
            config: Optional configuration dictionary, path to a YAML configuration
                file, or None. If None, attempts to load from the default YAML file
                (feature_config.yaml) or falls back to DEFAULT_FEATURE_CONFIG.
        """
        self.config = self._load_config(config)
        self.category_risk_map = self._build_category_risk_map()

    def _load_config(
        self, config: dict[str, Any] | str | Path | None
    ) -> dict[str, Any]:
        """Load and merge configuration with defaults."""
        merged: dict[str, Any] = copy.deepcopy(DEFAULT_FEATURE_CONFIG)

        if isinstance(config, (str, Path)):
            path = Path(config)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                self._deep_merge(merged, loaded)
            else:
                logger.warning(
                    "Config path %s does not exist; falling back to defaults.", path
                )
        elif isinstance(config, dict):
            self._deep_merge(merged, config)
        elif config is None and DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            self._deep_merge(merged, loaded)

        return merged

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
        """Recursively merge override dictionary into base dictionary."""
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                FeatureEngineer._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    def _build_category_risk_map(self) -> dict[str, str]:
        """Construct the category-to-risk-tier lookup mapping.

        Merges definitions from config with imported taxonomy constants from
        src.ingestion.data_loader.
        """
        tiers_cfg = self.config.get("merchant_risk_tiers", {})
        mapping: dict[str, str] = {}

        # Populate configured tiers in order: low, medium, high
        for tier in ("low", "medium", "high"):
            for cat in tiers_cfg.get(tier, []):
                mapping[cat] = tier

        # Ensure all high-risk categories from data_loader default to 'high'
        for cat in HIGH_RISK_MERCHANT_CATEGORIES:
            if cat not in mapping:
                mapping[cat] = "high"

        # Ensure all standard categories from data_loader default to 'low'
        for cat in STANDARD_MERCHANT_CATEGORIES:
            if cat not in mapping:
                mapping[cat] = "low"

        return mapping

    def load_raw_data(
        self, filepath: str | Path | None = None
    ) -> pd.DataFrame:
        """Load raw transaction data from CSV.

        Reads transactions with keep_default_na=False to preserve device_id "N/A"
        strings (in-store/ATM rows) instead of converting them to NaN.
        Parses timestamps as timezone-naive datetimes.
        Prints dtypes once to confirm schema integrity.

        Args:
            filepath: Path to raw transactions CSV. Defaults to DEFAULT_RAW_DATA_PATH.

        Returns:
            DataFrame containing raw transactions with confirmed data types.
        """
        path = Path(filepath or DEFAULT_RAW_DATA_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Raw transaction dataset not found at: {path}")

        # keep_default_na=False keeps "N/A" as string for device_id
        df = pd.read_csv(path, keep_default_na=False)

        # Parse timestamp as naive datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # Cast amount to float
        if "amount" in df.columns:
            df["amount"] = df["amount"].astype(float)

        # Cast is_fraud_actual to bool if present
        if "is_fraud_actual" in df.columns:
            df["is_fraud_actual"] = df["is_fraud_actual"].astype(bool)

        # Print dtypes once to confirm
        print("\n=== Raw Data Schema (dtypes) ===")
        print(df.dtypes)
        print("=================================\n")

        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare transaction data by sorting explicitly and resetting the index.

        The raw dataset is intentionally unsorted; sorting by [card_id, timestamp]
        guarantees chronological sequence per card necessary for vectorized
        trailing rolling windows and lag calculation.

        Args:
            df: Raw or unsorted transactions DataFrame.

        Returns:
            Chronologically sorted DataFrame with reset index.
        """
        df_sorted = df.sort_values(
            by=["card_id", "timestamp"], ascending=[True, True]
        ).reset_index(drop=True)
        return df_sorted

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute enriched fraud detection features on prepared transactions.

        Features computed:
        1. txn_velocity_1h: Trailing 1-hour count of transactions on the same card.
        2. txn_velocity_10min: Trailing 10-minute count of transactions on the same card.
        3. seconds_since_prev_txn: Seconds since the same card's previous transaction.
        4. is_odd_hour: Boolean flag for transactions occurring in odd hours [1, 4].
        5. merchant_risk_category: Risk tier ('low', 'medium', 'high') mapped from merchant_category.
        6. is_warmup: Boolean flag for each card's initial warm-up transactions (first 5).
        7. amount_zscore: Deviation of transaction amount from card's prior historical mean/std.
        8. is_new_merchant: Boolean flag if merchant_name has not been seen before on this card.
        9. geo_jump_flag: Boolean flag if transaction_country differs from prior country in < 3 hours.
        10. is_foreign_country: Boolean flag if transaction_country differs from card's home country.

        Vectorized Pandas only (groupby, rolling, shift). All original columns are preserved.
        is_fraud_actual passes through untouched and is NEVER used as an input to any feature.

        Args:
            df: Prepared DataFrame, sorted by [card_id, timestamp].

        Returns:
            Enriched DataFrame with new feature columns.
        """
        # Ensure chronological sorting per card if not already guaranteed
        df = self.prepare_data(df)

        if len(df) == 0:
            # Handle empty DataFrame gracefully
            df["txn_velocity_1h"] = pd.Series(dtype=int)
            df["txn_velocity_10min"] = pd.Series(dtype=int)
            df["seconds_since_prev_txn"] = pd.Series(dtype=float)
            df["is_odd_hour"] = pd.Series(dtype=bool)
            df["merchant_risk_category"] = pd.Series(dtype=object)
            df["is_warmup"] = pd.Series(dtype=bool)
            df["amount_zscore"] = pd.Series(dtype=float)
            df["is_new_merchant"] = pd.Series(dtype=bool)
            df["geo_jump_flag"] = pd.Series(dtype=bool)
            df["is_foreign_country"] = pd.Series(dtype=bool)
            return df

        # ------------------------------------------------------------------
        # 1 & 2. Transaction Velocities (Trailing 1 hour and Trailing 10 minutes)
        #
        # Time-based rolling window calculation per card:
        # NOTE ON WINDOW INCLUSION:
        # The rolling window boundary is right-closed (closed="both" or closed="right"),
        # meaning the CURRENT transaction row IS counted in the trailing window.
        # An isolated transaction with no prior events in the window evaluates to count = 1.
        #
        # txn_velocity_1h: counts transactions for the card in [t - 1h, t].
        # txn_velocity_10min: counts transactions for the card in [t - 10min, t].
        # The rapid-fire fraud pattern consists of bursts of ~5 transactions
        # within 5 minutes, so txn_velocity_10min captures this burst specifically
        # for downstream rule engine evaluation.
        # ------------------------------------------------------------------
        v_cfg = self.config["velocity_windows"]
        window_1h: str = v_cfg.get("velocity_1h", "1h")
        window_10min: str = v_cfg.get("velocity_10min", "10min")
        closed_type = cast(
            Literal["both", "left", "right", "neither"],
            v_cfg.get("closed", "both"),
        )

        # Vectorized rolling count grouped by card_id
        # Note: sort=False preserves group order matching the pre-sorted card sequence
        v_1h_series = (
            df.groupby("card_id", sort=False)
            .rolling(window_1h, on="timestamp", closed=closed_type)["transaction_id"]
            .count()
        )
        df["txn_velocity_1h"] = v_1h_series.to_numpy(dtype=int)

        v_10min_series = (
            df.groupby("card_id", sort=False)
            .rolling(window_10min, on="timestamp", closed=closed_type)["transaction_id"]
            .count()
        )
        df["txn_velocity_10min"] = v_10min_series.to_numpy(dtype=int)

        # ------------------------------------------------------------------
        # 3. Seconds Since Previous Transaction
        #
        # Vectorized lag calculation using groupby shift(1).
        # For the first transaction of any card, the previous timestamp is NaT,
        # resulting in NaN (float) as expected for an initial transaction.
        # ------------------------------------------------------------------
        prev_timestamp = df.groupby("card_id", sort=False)["timestamp"].shift(1)
        df["seconds_since_prev_txn"] = (
            df["timestamp"] - prev_timestamp
        ).dt.total_seconds()

        # ------------------------------------------------------------------
        # 4. Odd-Hour Indicator
        #
        # Flags transactions during suspicious early-morning hours (e.g. 1 AM to 4 AM).
        # Configured via odd_hour_window (inclusive of start and end hours).
        # ------------------------------------------------------------------
        odd_cfg = self.config.get("odd_hour_window", {})
        start_hour: int = odd_cfg.get("start_hour", 1)
        end_hour: int = odd_cfg.get("end_hour", 4)
        df["is_odd_hour"] = df["timestamp"].dt.hour.between(
            start_hour, end_hour, inclusive="both"
        )

        # ------------------------------------------------------------------
        # 5. Merchant Risk Category Tier
        #
        # Vectorized dictionary mapping merchant_category -> risk tier ('low', 'medium', 'high').
        # Categories not found in the explicit map fall back to the configured default ('low').
        # ------------------------------------------------------------------
        default_tier: str = self.config.get("merchant_risk_tiers", {}).get(
            "default", "low"
        )
        df["merchant_risk_category"] = (
            df["merchant_category"]
            .map(self.category_risk_map)
            .fillna(default_tier)
            .astype(str)
        )

        # ------------------------------------------------------------------
        # 6. Card Warm-up Indicator
        #
        # Treats each card's initial transactions (default: 5) as warm-up.
        # Uses cumcount() per card to determine transaction order sequence.
        # Warm-up transactions have unknown baseline statistics and are flagged here
        # rather than using NaN values for downstream features.
        # ------------------------------------------------------------------
        warmup_txns: int = int(self.config.get("warmup_length_txns", 5))
        card_txn_index = df.groupby("card_id", sort=False).cumcount()
        df["is_warmup"] = card_txn_index < warmup_txns

        # ------------------------------------------------------------------
        # 7. Amount Z-Score (Historical Deviation)
        #
        # Strictly uses prior transactions of the same card (no future leakage).
        # We shift amounts by 1 per card, then compute cumulative expanding mean and std.
        # Guard against warm-up rows, NaNs, and std == 0 by assigning neutral 0.0 z-score.
        # ------------------------------------------------------------------
        shifted_amount = df.groupby("card_id", sort=False)["amount"].shift(1)
        exp_mean = (
            shifted_amount.groupby(df["card_id"], sort=False)
            .expanding()
            .mean()
            .droplevel(0)
        )
        exp_std = (
            shifted_amount.groupby(df["card_id"], sort=False)
            .expanding()
            .std()
            .droplevel(0)
        )

        valid_zscore_mask = (~df["is_warmup"]) & (exp_std > 0) & exp_std.notna()
        df["amount_zscore"] = (
            ((df["amount"] - exp_mean) / exp_std)
            .where(valid_zscore_mask, 0.0)
            .astype(float)
        )

        # ------------------------------------------------------------------
        # 8. New Merchant Indicator
        #
        # Evaluates whether this merchant_name has been seen before for this card_id.
        # Computed via cumcount() on (card_id, merchant_name): cumcount == 0 represents
        # the very first appearance of the merchant for that card.
        # Strictly uses prior transactions (no future leakage).
        # During warm-up (is_warmup == True), is_new_merchant is set to False (neutral)
        # so downstream fraud rules do not fire on initial card activation.
        # ------------------------------------------------------------------
        merchant_seen_count = df.groupby(
            ["card_id", "merchant_name"], sort=False
        ).cumcount()
        df["is_new_merchant"] = (~df["is_warmup"]) & (merchant_seen_count == 0)

        # ------------------------------------------------------------------
        # 9. Geographic Jump Flag (Implausible Travel Speed)
        #
        # Compares transaction_country with the same card's previous transaction country.
        # Flags transactions where transaction_country changed and elapsed time is less than
        # the configured geo travel threshold (geo_window_hours, default: 3 hours).
        #
        # NOTE ON FALSE POSITIVES (Return Trip):
        # We explicitly use transaction_country (NOT merchant_country).
        # When a foreign fraudulent transaction occurs followed by a legitimate domestic
        # transaction within 3 hours, the return transaction will ALSO trigger geo_jump_flag.
        # This is expected behavior for consecutive country differences under tight timeframes,
        # representing a realistic false positive that downstream rules contextualize.
        # ------------------------------------------------------------------
        geo_window_hours: float = float(self.config.get("geo_window_hours", 3))
        geo_window_seconds: float = geo_window_hours * 3600.0

        prev_country = df.groupby("card_id", sort=False)["transaction_country"].shift(1)
        country_changed = (df["transaction_country"] != prev_country) & prev_country.notna()
        time_too_short = df["seconds_since_prev_txn"].notna() & (
            df["seconds_since_prev_txn"] < geo_window_seconds
        )
        df["geo_jump_flag"] = country_changed & time_too_short

        # ------------------------------------------------------------------
        # 10. Foreign Country Indicator
        #
        # Evaluates whether transaction_country differs from the card's inferred home country.
        # Home country is defined as the most frequent transaction_country during that card's
        # warm-up transactions (warmup_length_txns), breaking ties alphabetically.
        # Strict requirement: uses transaction_country (do NOT use merchant_country).
        # False on warm-up rows (is_warmup == True), consistent with other baseline features.
        # ------------------------------------------------------------------
        warmup_df = df[df["is_warmup"]]
        country_counts = (
            warmup_df.groupby(["card_id", "transaction_country"], sort=True)
            .size()
            .reset_index(name="count")
        )
        sorted_counts = country_counts.sort_values(
            by=["card_id", "count", "transaction_country"],
            ascending=[True, False, True],
        )
        home_country_map = (
            sorted_counts.drop_duplicates(subset=["card_id"], keep="first")
            .set_index("card_id")["transaction_country"]
        )

        card_home_country = df["card_id"].map(home_country_map)
        df["is_foreign_country"] = (
            (~df["is_warmup"])
            & (df["transaction_country"] != card_home_country)
            & card_home_country.notna()
        )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the end-to-end data preparation and feature engineering pipeline.

        Args:
            df: Raw transactions DataFrame.

        Returns:
            Enriched DataFrame with all original columns and newly calculated features.
        """
        prepared_df = self.prepare_data(df)
        enriched_df = self.compute_features(prepared_df)
        return enriched_df

    def save_processed_data(
        self, df: pd.DataFrame, output_filepath: str | Path | None = None
    ) -> Path:
        """Save enriched DataFrame to CSV.

        Args:
            df: Enriched transactions DataFrame.
            output_filepath: Output destination path. Defaults to DEFAULT_PROCESSED_DATA_PATH.

        Returns:
            Resolved Path to the saved CSV file.
        """
        out_path = Path(output_filepath or DEFAULT_PROCESSED_DATA_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved %d enriched transactions to %s", len(df), out_path)
        return out_path

    def run_pipeline(
        self,
        raw_filepath: str | Path | None = None,
        output_filepath: str | Path | None = None,
    ) -> pd.DataFrame:
        """Load raw transactions, compute features, and save enriched output.

        Args:
            raw_filepath: Source raw CSV path. Defaults to DEFAULT_RAW_DATA_PATH.
            output_filepath: Output enriched CSV path. Defaults to DEFAULT_PROCESSED_DATA_PATH.

        Returns:
            Enriched DataFrame.
        """
        raw_df = self.load_raw_data(raw_filepath)
        enriched_df = self.transform(raw_df)
        self.save_processed_data(enriched_df, output_filepath)
        return enriched_df


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engineer = FeatureEngineer()
    enriched = engineer.run_pipeline()

    print(f"Total rows processed : {len(enriched):,}")
    print(f"Total columns ({len(enriched.columns)}) : {list(enriched.columns)}")

    display_cols = [
        "card_id",
        "timestamp",
        "amount",
        "merchant_category",
        "txn_velocity_1h",
        "txn_velocity_10min",
        "seconds_since_prev_txn",
        "is_odd_hour",
        "merchant_risk_category",
        "is_warmup",
        "amount_zscore",
        "is_new_merchant",
        "geo_jump_flag",
        "is_foreign_country",
        "is_fraud_actual",
    ]
    print("\nSample enriched rows (first 5):")
    print(enriched[display_cols].head(5).to_string(index=False))

    output_path = DEFAULT_PROCESSED_DATA_PATH
    print(f"\nSuccessfully generated and saved to: {output_path}")

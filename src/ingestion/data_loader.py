"""Data loader module for ingesting raw transaction and credit risk data.

This module generates realistic synthetic credit card transaction data for use
in fraud detection pipeline development and rule engine validation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Reproducibility seed — change to get a different (but still deterministic) dataset.
_RNG_SEED: Final[int] = 42

# Dataset dimensions
NUM_TRANSACTIONS: Final[int] = 50_000
NUM_CARDS: Final[int] = 2_000
WINDOW_DAYS: Final[int] = 90

# Anomaly budget (approximate percentages of total transactions)
ANOMALY_HIGH_VALUE_FRAC: Final[float] = 0.008   # sudden high-value spikes
ANOMALY_RAPID_FIRE_FRAC: Final[float] = 0.006   # burst transactions within minutes
ANOMALY_COUNTRY_MISMATCH_FRAC: Final[float] = 0.005  # foreign-country anomaly
ANOMALY_UNUSUAL_HOUR_FRAC: Final[float] = 0.005  # 2-4 AM transactions
ANOMALY_RISKY_MERCHANT_FRAC: Final[float] = 0.004  # high-risk merchant categories

# Merchant category taxonomy
STANDARD_MERCHANT_CATEGORIES: Final[list[str]] = [
    "Grocery",
    "Restaurant",
    "Fuel",
    "Retail",
    "Travel",
    "Entertainment",
    "Healthcare",
    "Utilities",
    "Subscription",
    "Transportation",
]

HIGH_RISK_MERCHANT_CATEGORIES: Final[list[str]] = [
    "Crypto Exchange",
    "Online Gambling",
    "Pawnshop",
    "Money Transfer",
    "Adult Content",
]

# Channel distribution weights (online, in-store, atm)
CHANNEL_OPTIONS: Final[list[str]] = ["online", "in-store", "atm"]
CHANNEL_WEIGHTS: Final[list[float]] = [0.45, 0.45, 0.10]

# Country pools — "home" countries are common; "abroad" countries trigger anomaly
HOME_COUNTRIES: Final[list[str]] = [
    "US", "GB", "CA", "AU", "DE", "FR", "IN", "SG", "JP", "BR",
]
ABROAD_COUNTRIES: Final[list[str]] = [
    "NG", "RU", "CN", "UA", "RO", "VN", "KP", "IR", "PK", "AO",
]

# Output path (relative to the repo root)
_RAW_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_OUTPUT_PATH: Final[Path] = _RAW_DATA_DIR / "transactions.csv"


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------
def _pool_weights(k: int, alpha: float = 1.5) -> np.ndarray:
    """Zipf-like weights: first merchants are visited far more often."""
    w = 1.0 / np.arange(1, k + 1) ** alpha
    return w / w.sum()


def _draw_pool_merchant(card_row: pd.Series, rng: np.random.Generator) -> tuple[str, str]:
    pool = [item.split("::") for item in card_row["merchant_pool"].split("|")]
    idx = rng.choice(len(pool), p=_pool_weights(len(pool)))
    return pool[idx][0], pool[idx][1]


def _draw_fraud_merchant(
    card_row: pd.Series, fake: Faker, rng: np.random.Generator, p_in_pool: float = 0.12
) -> tuple[str, str]:
    """Mostly a fresh merchant in an unseen category; sometimes a known one."""
    if rng.random() < p_in_pool:
        return _draw_pool_merchant(card_row, rng)
    card_cats = set(card_row["preferred_categories"].split("|"))
    unseen = [c for c in STANDARD_MERCHANT_CATEGORIES if c not in card_cats]
    unseen = unseen or STANDARD_MERCHANT_CATEGORIES
    return fake.company(), unseen[int(rng.integers(0, len(unseen)))]


def _build_card_profiles(
    num_cards: int,
    fake: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create a lookup table of card-level attributes.

    Each card gets a stable home country, typical spending amount, a set of
    preferred merchant categories, and a pre-generated merchant pool drawn from
    those categories.  The pool is stored as a pipe-separated string so that it
    survives pandas serialisation without object-dtype list columns.

    The pool drives the ``is_new_merchant`` signal: normal transactions draw
    ~95 % of merchants from the pool; fraud injectors always draw from outside
    it, making new-merchant frequency a meaningful fraud feature.

    Args:
        num_cards: Number of unique card identifiers to generate.
        fake: A :class:`~faker.Faker` instance for realistic name generation.
        rng: A seeded NumPy random generator for reproducible distributions.

    Returns:
        DataFrame with columns ``card_id``, ``cardholder_name``,
        ``home_country``, ``typical_amount_mean``, ``typical_amount_std``,
        ``preferred_categories`` (pipe-separated str),
        ``merchant_pool`` (pipe-separated str).
    """
    card_ids = [f"CARD-{str(uuid.uuid4()).upper()[:12]}" for _ in range(num_cards)]
    cardholder_names = [fake.name() for _ in range(num_cards)]
    home_countries = rng.choice(HOME_COUNTRIES, size=num_cards)

    # Each cardholder has their own "normal" spending level (log-normal)
    typical_means = rng.lognormal(mean=3.5, sigma=0.8, size=num_cards)  # ~$33 median
    typical_stds = typical_means * rng.uniform(0.2, 0.5, size=num_cards)

    # --- Per-card merchant behaviour ---
    preferred_categories_list: list[str] = []
    merchant_pool_list: list[str] = []

    for _ in range(num_cards):
        # 3-5 preferred categories, drawn without replacement
        n_cats = int(rng.integers(3, 6))  # 3, 4, or 5
        cats = rng.choice(
            STANDARD_MERCHANT_CATEGORIES, size=n_cats, replace=False
        ).tolist()
        preferred_categories_list.append("|".join(cats))

        # 3-5 recurring merchants, each tied to one category.
        # Pool order = popularity rank (first entries are visited most often).
        pool_size = int(rng.integers(3, 6))  # 3..5 inclusive
        pool = [
            f"{fake.company()}::{cats[int(rng.integers(0, len(cats)))]}"
            for _ in range(pool_size)
        ]
        merchant_pool_list.append("|".join(pool))


    return pd.DataFrame(
        {
            "card_id": card_ids,
            "cardholder_name": cardholder_names,
            "home_country": home_countries,
            "typical_amount_mean": typical_means,
            "typical_amount_std": typical_stds,
            "preferred_categories": preferred_categories_list,
            "merchant_pool": merchant_pool_list,
        }
    )


def _sample_normal_transactions(
    n: int,
    cards: pd.DataFrame,
    fake: Faker,
    rng: np.random.Generator,
    window_start: datetime,
    window_seconds: int,
) -> pd.DataFrame:
    """Generate the bulk of legitimate (non-anomalous) transactions.

    Amounts follow each card's individual log-normal distribution. Timestamps
    are uniformly scattered across the 90-day window with a realistic daytime
    bias (most spending between 08:00-22:00).

    Args:
        n: Number of rows to generate.
        cards: Card-profile DataFrame from :func:`_build_card_profiles`.
        fake: Faker instance.
        rng: Seeded NumPy generator.
        window_start: Start of the 90-day observation window.
        window_seconds: Total seconds in the window.

    Returns:
        DataFrame of normal transactions (without ``is_fraud_actual`` yet).
    """
    card_indices = rng.integers(0, len(cards), size=n)
    selected_cards = cards.iloc[card_indices].reset_index(drop=True)

    # --- Timestamps with daytime bias ---
    raw_offsets = rng.uniform(0, window_seconds, size=n)
    # Wrap into a 24-hour clock and apply a Gaussian bias toward midday
    hour_bias = rng.normal(loc=14 * 3600, scale=4 * 3600, size=n) % 86_400
    biased_offsets = (raw_offsets - (raw_offsets % 86_400) + hour_bias).astype(int)
    biased_offsets = np.clip(biased_offsets, 0, window_seconds - 1)
    timestamps = [window_start + timedelta(seconds=int(s)) for s in biased_offsets]

    # --- Amounts sampled per-card ---
    amounts = np.abs(
        rng.normal(
            loc=selected_cards["typical_amount_mean"].to_numpy(),
            scale=selected_cards["typical_amount_std"].to_numpy(),
        )
    ).round(2)
    amounts = np.clip(amounts, 1.0, 5_000.0)

    # --- Merchant info (95 % from card's known pool, 5 % legitimate new merchant) ---
    merchant_names: list[str] = []
    merchant_categories: list[str] = []
    use_pool_flags = rng.random(n) < 0.95  # True -> draw from pool

    for i in range(n):
        card_row = selected_cards.iloc[i]
        if use_pool_flags[i]:
            # Known merchant: name and category come as a pair from the pool
            name, cat = _draw_pool_merchant(card_row, rng)
        else:
            # Legitimate new merchant, in one of the card's usual categories
            cats: list[str] = card_row["preferred_categories"].split("|")
            name = fake.company()
            cat = cats[int(rng.integers(0, len(cats)))]
        merchant_names.append(name)
        merchant_categories.append(cat)


    merchant_countries = selected_cards["home_country"].to_numpy()  # matches home
    transaction_cities = [fake.city() for _ in range(n)]
    transaction_countries = merchant_countries.copy()

    # --- Channels & device IDs ---
    channels = rng.choice(CHANNEL_OPTIONS, size=n, p=CHANNEL_WEIGHTS)
    device_ids = np.where(
        channels == "online",
        [f"DEV-{rng.integers(100000, 999999)}" for _ in range(n)],
        "N/A",
    )

    return pd.DataFrame(
        {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "card_id": selected_cards["card_id"].to_numpy(),
            "cardholder_name": selected_cards["cardholder_name"].to_numpy(),
            "timestamp": timestamps,
            "amount": amounts,
            "merchant_name": merchant_names,
            "merchant_category": merchant_categories,
            "merchant_country": merchant_countries,
            "transaction_city": transaction_cities,
            "transaction_country": transaction_countries,
            "channel": channels,
            "device_id": device_ids,
        }
    )


# ---------------------------------------------------------------------------
# Anomaly injectors — each returns a small DataFrame of fraudulent rows
# ---------------------------------------------------------------------------

def _inject_high_value_spikes(
    n: int,
    cards: pd.DataFrame,
    fake: Faker,
    rng: np.random.Generator,
    window_start: datetime,
    window_seconds: int,
) -> pd.DataFrame:
    """Inject sudden high-value transactions (10-50x the card's normal amount).

    Args:
        n: Number of anomalous rows to create.
        cards: Card-profile DataFrame.
        fake: Faker instance.
        rng: Seeded NumPy generator.
        window_start: Window start datetime.
        window_seconds: Total seconds in the window.

    Returns:
        DataFrame of high-value anomalous transactions.
    """
    card_indices = rng.integers(0, len(cards), size=n)
    selected_cards = cards.iloc[card_indices].reset_index(drop=True)

    multipliers = rng.uniform(10, 50, size=n)
    amounts = (selected_cards["typical_amount_mean"].to_numpy() * multipliers).round(2)

    offsets = rng.integers(0, window_seconds, size=n)
    timestamps = [window_start + timedelta(seconds=int(s)) for s in offsets]

    # Fraud: usually a fresh merchant in a category the card has never used,
    # occasionally (~12%) a known merchant so the signal isn't a perfect separator.
    picks = [_draw_fraud_merchant(selected_cards.iloc[i], fake, rng) for i in range(n)]
    fraud_merchants_hv = [p[0] for p in picks]
    fraud_categories_hv = [p[1] for p in picks]

    return pd.DataFrame(
        {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "card_id": selected_cards["card_id"].to_numpy(),
            "cardholder_name": selected_cards["cardholder_name"].to_numpy(),
            "timestamp": timestamps,
            "amount": np.clip(amounts, 500.0, 50_000.0),
            "merchant_name": fraud_merchants_hv,
            "merchant_category": fraud_categories_hv,
            "merchant_country": selected_cards["home_country"].to_numpy(),
            "transaction_city": [fake.city() for _ in range(n)],
            "transaction_country": selected_cards["home_country"].to_numpy(),
            "channel": rng.choice(CHANNEL_OPTIONS, size=n, p=CHANNEL_WEIGHTS),
            "device_id": [f"DEV-{rng.integers(100000, 999999)}" for _ in range(n)],
            "is_fraud_actual": True,
        }
    )


def _inject_rapid_fire(
    n_bursts: int,
    cards: pd.DataFrame,
    fake: Faker,
    rng: np.random.Generator,
    window_start: datetime,
    window_seconds: int,
    burst_size: int = 5,
) -> pd.DataFrame:
    """Inject clusters of rapid-fire transactions within a few minutes of each other.

    ``n_bursts`` bursts are generated, each containing ``burst_size`` transactions
    on the same card within a 5-minute window, simulating card-testing or
    automated fraud.

    Args:
        n_bursts: Number of distinct burst events.
        cards: Card-profile DataFrame.
        fake: Faker instance.
        rng: Seeded NumPy generator.
        window_start: Window start datetime.
        window_seconds: Total seconds in the window.
        burst_size: Number of transactions per burst event.

    Returns:
        DataFrame of rapid-fire anomalous transactions.
    """
    rows: list[dict] = []
    for _ in range(n_bursts):
        card_idx = int(rng.integers(0, len(cards)))
        card = cards.iloc[card_idx]
        burst_anchor = window_start + timedelta(
            seconds=int(rng.integers(0, window_seconds - 300))
        )
        for _ in range(burst_size):
            offset_seconds = int(rng.integers(0, 300))  # spread within 5 minutes
            ts = burst_anchor + timedelta(seconds=offset_seconds)
            amount = float(
                np.abs(rng.normal(card["typical_amount_mean"], card["typical_amount_std"]))
            )
            # Fraud: fresh merchant outside the card's pool
            merch_name, rf_cat = _draw_fraud_merchant(card, fake, rng)
            rows.append(
                {
                    "transaction_id": str(uuid.uuid4()),
                    "card_id": card["card_id"],
                    "cardholder_name": card["cardholder_name"],
                    "timestamp": ts,
                    "amount": round(max(1.0, amount), 2),
                    "merchant_name": merch_name,  # usually new, sometimes a known merchant
                    "merchant_category": rf_cat,
                    "merchant_country": card["home_country"],
                    "transaction_city": fake.city(),
                    "transaction_country": card["home_country"],
                    "channel": rng.choice(CHANNEL_OPTIONS, p=CHANNEL_WEIGHTS),
                    "device_id": f"DEV-{rng.integers(100000, 999999)}",
                    "is_fraud_actual": True,
                }
            )
    return pd.DataFrame(rows)


def _inject_country_mismatch(
    n: int,
    cards: pd.DataFrame,
    fake: Faker,
    rng: np.random.Generator,
    window_start: datetime,
    window_seconds: int,
) -> pd.DataFrame:
    """Inject transactions originating from high-risk foreign countries.

    The card's home country differs from the transaction country, mimicking
    card-present fraud in a foreign location.

    Args:
        n: Number of anomalous rows to create.
        cards: Card-profile DataFrame.
        fake: Faker instance.
        rng: Seeded NumPy generator.
        window_start: Window start datetime.
        window_seconds: Total seconds in the window.

    Returns:
        DataFrame of country-mismatch anomalous transactions.
    """
    card_indices = rng.integers(0, len(cards), size=n)
    selected_cards = cards.iloc[card_indices].reset_index(drop=True)

    abroad_countries = rng.choice(ABROAD_COUNTRIES, size=n)
    offsets = rng.integers(0, window_seconds, size=n)
    timestamps = [window_start + timedelta(seconds=int(s)) for s in offsets]

    amounts = np.abs(
        rng.normal(
            loc=selected_cards["typical_amount_mean"].to_numpy() * 2,
            scale=selected_cards["typical_amount_std"].to_numpy(),
        )
    ).round(2)

    # Fraud: fresh merchant outside the card's pool
    picks = [_draw_fraud_merchant(selected_cards.iloc[i], fake, rng) for i in range(n)]
    fraud_merchants_cm = [p[0] for p in picks]
    fraud_categories_cm = [p[1] for p in picks]

    return pd.DataFrame(
        {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "card_id": selected_cards["card_id"].to_numpy(),
            "cardholder_name": selected_cards["cardholder_name"].to_numpy(),
            "timestamp": timestamps,
            "amount": np.clip(amounts, 1.0, 10_000.0),
            "merchant_name": fraud_merchants_cm,
            "merchant_category": fraud_categories_cm,
            "merchant_country": abroad_countries,
            "transaction_city": [fake.city() for _ in range(n)],
            "transaction_country": abroad_countries,
            "channel": rng.choice(CHANNEL_OPTIONS, size=n, p=CHANNEL_WEIGHTS),
            "device_id": [f"DEV-{rng.integers(100000, 999999)}" for _ in range(n)],
            "is_fraud_actual": True,
        }
    )


def _inject_unusual_hours(
    n: int,
    cards: pd.DataFrame,
    fake: Faker,
    rng: np.random.Generator,
    window_start: datetime,
    window_seconds: int,
) -> pd.DataFrame:
    """Inject transactions occurring between 02:00 and 04:00 local time.

    Late-night/early-morning purchases (2-4 AM) are a known fraud signal,
    especially for online channels.

    Args:
        n: Number of anomalous rows to create.
        cards: Card-profile DataFrame.
        fake: Faker instance.
        rng: Seeded NumPy generator.
        window_start: Window start datetime.
        window_seconds: Total seconds in the window.

    Returns:
        DataFrame of unusual-hour anomalous transactions.
    """
    card_indices = rng.integers(0, len(cards), size=n)
    selected_cards = cards.iloc[card_indices].reset_index(drop=True)

    # Pick a random day, then pin the hour to 02:00-04:00
    day_offsets = rng.integers(0, WINDOW_DAYS, size=n)
    hour_offsets = rng.integers(2 * 3600, 4 * 3600, size=n)  # 2-4 AM in seconds
    timestamps = [
        window_start + timedelta(days=int(d), seconds=int(h))
        for d, h in zip(day_offsets, hour_offsets)
    ]

    amounts = np.abs(
        rng.normal(
            loc=selected_cards["typical_amount_mean"].to_numpy(),
            scale=selected_cards["typical_amount_std"].to_numpy(),
        )
    ).round(2)

    # Fraud: fresh merchant outside the card's pool
    picks = [_draw_fraud_merchant(selected_cards.iloc[i], fake, rng) for i in range(n)]
    fraud_merchants_uh = [p[0] for p in picks]
    fraud_categories_uh = [p[1] for p in picks]

    return pd.DataFrame(
        {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "card_id": selected_cards["card_id"].to_numpy(),
            "cardholder_name": selected_cards["cardholder_name"].to_numpy(),
            "timestamp": timestamps,
            "amount": np.clip(amounts, 1.0, 5_000.0),
            "merchant_name": fraud_merchants_uh,
            "merchant_category": fraud_categories_uh,
            "merchant_country": selected_cards["home_country"].to_numpy(),
            "transaction_city": [fake.city() for _ in range(n)],
            "transaction_country": selected_cards["home_country"].to_numpy(),
            "channel": rng.choice(["online", "atm"], size=n, p=[0.7, 0.3]),
            "device_id": [f"DEV-{rng.integers(100000, 999999)}" for _ in range(n)],
            "is_fraud_actual": True,
        }
    )


def _inject_risky_merchants(
    n: int,
    cards: pd.DataFrame,
    fake: Faker,
    rng: np.random.Generator,
    window_start: datetime,
    window_seconds: int,
) -> pd.DataFrame:
    """Inject transactions at high-risk merchant categories.

    Categories like "Crypto Exchange" or "Online Gambling" carry elevated fraud
    risk and are flagged by the rule engine.

    Args:
        n: Number of anomalous rows to create.
        cards: Card-profile DataFrame.
        fake: Faker instance.
        rng: Seeded NumPy generator.
        window_start: Window start datetime.
        window_seconds: Total seconds in the window.

    Returns:
        DataFrame of risky-merchant anomalous transactions.
    """
    card_indices = rng.integers(0, len(cards), size=n)
    selected_cards = cards.iloc[card_indices].reset_index(drop=True)

    offsets = rng.integers(0, window_seconds, size=n)
    timestamps = [window_start + timedelta(seconds=int(s)) for s in offsets]

    amounts = np.abs(
        rng.normal(
            loc=selected_cards["typical_amount_mean"].to_numpy() * 1.5,
            scale=selected_cards["typical_amount_std"].to_numpy(),
        )
    ).round(2)

    # Fraud: fresh merchant + high-risk category (never in the card's standard pool)
    fraud_merchants_rm = [fake.company() for _ in range(n)]

    return pd.DataFrame(
        {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "card_id": selected_cards["card_id"].to_numpy(),
            "cardholder_name": selected_cards["cardholder_name"].to_numpy(),
            "timestamp": timestamps,
            "amount": np.clip(amounts, 1.0, 5_000.0),
            "merchant_name": fraud_merchants_rm,
            "merchant_category": rng.choice(HIGH_RISK_MERCHANT_CATEGORIES, size=n),
            "merchant_country": selected_cards["home_country"].to_numpy(),
            "transaction_city": [fake.city() for _ in range(n)],
            "transaction_country": selected_cards["home_country"].to_numpy(),
            "channel": rng.choice(CHANNEL_OPTIONS, size=n, p=CHANNEL_WEIGHTS),
            "device_id": [f"DEV-{rng.integers(100000, 999999)}" for _ in range(n)],
            "is_fraud_actual": True,
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_transactions(
    num_transactions: int = NUM_TRANSACTIONS,
    num_cards: int = NUM_CARDS,
    window_days: int = WINDOW_DAYS,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    seed: int = _RNG_SEED,
) -> pd.DataFrame:
    """Generate a synthetic credit card transaction dataset with injected fraud.

    The dataset simulates realistic spending behaviour across *num_cards* unique
    cards over a *window_days*-day observation window.  Approximately 2-3% of
    records are labelled as fraudulent (``is_fraud_actual=True``) via five
    distinct anomaly injection strategies:

    1. **High-value spikes** - amounts 10-50x a card's normal mean.
    2. **Rapid-fire bursts** - 5 transactions on the same card within 5 minutes.
    3. **Country mismatch** - transaction country differs from the card's home
       country (drawn from a pool of high-risk nations).
    4. **Unusual hours** - transactions between 02:00 and 04:00.
    5. **High-risk merchants** - merchants in categories such as
       "Crypto Exchange" or "Online Gambling".

    Note:
        ``is_fraud_actual`` is a *ground-truth* label intended **only** for
        evaluating rule-engine performance after the fact.  It must **not** be
        passed as a feature into the rule engine itself.

    Args:
        num_transactions: Total rows in the output dataset (default 50 000).
        num_cards: Number of unique card identifiers (default 2 000).
        window_days: Length of the observation window in days (default 90).
        output_path: Destination CSV file path.  Parent directories are created
            automatically.  Defaults to ``data/raw/transactions.csv`` relative
            to the repository root.
        seed: Random seed for full reproducibility (default 42).

    Returns:
        The assembled :class:`~pandas.DataFrame` (also written to *output_path*).

    Raises:
        OSError: If the output directory cannot be created or the file cannot
            be written.

    Example:
        >>> df = generate_synthetic_transactions(num_transactions=1000)
        >>> df.shape[0]
        1000
        >>> set(["transaction_id", "card_id", "is_fraud_actual"]).issubset(df.columns)
        True
    """
    logger.info(
        "Generating %d synthetic transactions across %d cards over %d days (seed=%d).",
        num_transactions,
        num_cards,
        window_days,
        seed,
    )

    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    window_start = datetime.now() - timedelta(days=window_days)
    window_seconds = window_days * 86_400

    # ------------------------------------------------------------------
    # 1. Card profile lookup table
    # ------------------------------------------------------------------
    cards = _build_card_profiles(num_cards, fake, rng)
    logger.debug("Built %d card profiles.", len(cards))

    # ------------------------------------------------------------------
    # 2. Compute anomaly budgets
    # ------------------------------------------------------------------
    n_high_value = int(num_transactions * ANOMALY_HIGH_VALUE_FRAC)
    n_rapid_fire_txns = int(num_transactions * ANOMALY_RAPID_FIRE_FRAC)
    n_country_mismatch = int(num_transactions * ANOMALY_COUNTRY_MISMATCH_FRAC)
    n_unusual_hours = int(num_transactions * ANOMALY_UNUSUAL_HOUR_FRAC)
    n_risky_merchants = int(num_transactions * ANOMALY_RISKY_MERCHANT_FRAC)

    # rapid-fire: each burst injects `burst_size` rows; compute burst count
    burst_size = 5
    n_bursts = max(1, n_rapid_fire_txns // burst_size)
    actual_rapid_fire_txns = n_bursts * burst_size

    total_anomalies = (
        n_high_value
        + actual_rapid_fire_txns
        + n_country_mismatch
        + n_unusual_hours
        + n_risky_merchants
    )
    n_normal = num_transactions - total_anomalies

    logger.info(
        "Anomaly budget -- high-value: %d | rapid-fire: %d | country-mismatch: %d "
        "| unusual-hours: %d | risky-merchants: %d | normal: %d",
        n_high_value,
        actual_rapid_fire_txns,
        n_country_mismatch,
        n_unusual_hours,
        n_risky_merchants,
        n_normal,
    )

    # ------------------------------------------------------------------
    # 3. Build all DataFrames
    # ------------------------------------------------------------------
    normal_df = _sample_normal_transactions(
        n_normal, cards, fake, rng, window_start, window_seconds
    )
    normal_df["is_fraud_actual"] = False

    anomaly_parts: list[pd.DataFrame] = [
        _inject_high_value_spikes(
            n_high_value, cards, fake, rng, window_start, window_seconds
        ),
        _inject_rapid_fire(
            n_bursts, cards, fake, rng, window_start, window_seconds, burst_size
        ),
        _inject_country_mismatch(
            n_country_mismatch, cards, fake, rng, window_start, window_seconds
        ),
        _inject_unusual_hours(
            n_unusual_hours, cards, fake, rng, window_start, window_seconds
        ),
        _inject_risky_merchants(
            n_risky_merchants, cards, fake, rng, window_start, window_seconds
        ),
    ]

    # ------------------------------------------------------------------
    # 4. Concatenate, shuffle, reset index
    # ------------------------------------------------------------------
    all_parts = [normal_df] + anomaly_parts
    df = pd.concat(all_parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Enforce consistent column ordering
    column_order = [
        "transaction_id",
        "card_id",
        "cardholder_name",
        "timestamp",
        "amount",
        "merchant_name",
        "merchant_category",
        "merchant_country",
        "transaction_city",
        "transaction_country",
        "channel",
        "device_id",
        "is_fraud_actual",
    ]
    df = cast(pd.DataFrame, df[column_order])

    # Cast types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)
    df["is_fraud_actual"] = df["is_fraud_actual"].astype(bool)

    # ------------------------------------------------------------------
    # 5. Persist to disk
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    fraud_count = df["is_fraud_actual"].sum()
    fraud_pct = fraud_count / len(df) * 100
    logger.info(
        "Saved %d transactions (%d fraud, %.2f%%) to %s",
        len(df),
        fraud_count,
        fraud_pct,
        output_path,
    )
    return df


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    df = generate_synthetic_transactions()
    print(f"\nDataset shape : {df.shape}")
    print(
        f"Fraud records : {df['is_fraud_actual'].sum()} "
        f"({df['is_fraud_actual'].mean() * 100:.2f}%)"
    )
    print(f"\nSample rows:\n{df.head(5).to_string(index=False)}")

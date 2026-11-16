"""Streamlit dashboard entrypoint for Fraud Detection and Credit Risk Pipeline."""

from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path
from typing import cast

from sqlalchemy.exc import SQLAlchemyError

# Put project root on sys.path so 'from src...' imports resolve reliably
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.database.db_manager import DBManager

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Risk Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Minimal Custom CSS for Container Rounded Corners
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Database Resources and Queries
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db_manager() -> DBManager:
    """Instantiate and cache the DBManager instance."""
    return DBManager()


@st.cache_data
def get_cached_metrics() -> dict:
    """Fetch aggregate operational metrics from the database."""
    db = get_db_manager()
    return db.get_metrics()


@st.cache_data
def get_cached_metadata() -> tuple[list[str], datetime | None, datetime | None]:
    """Fetch distinct merchant categories and date boundaries for filters."""
    db = get_db_manager()
    categories = db.get_merchant_categories()
    min_ts, max_ts = db.get_transaction_date_bounds()
    return categories, min_ts, max_ts


@st.cache_data
def get_cached_flagged_df(
    tiers: tuple[str, ...],
    date_from: str | None,
    date_to: str | None,
    merchant_category: str | None,
) -> pd.DataFrame:
    """Query flagged transactions joined with transaction details respecting filters."""
    db = get_db_manager()
    cat_filter = [merchant_category] if merchant_category and merchant_category != "All Categories" else None
    return db.get_flagged_transactions_df(
        tiers=list(tiers) if tiers else None,
        date_from=date_from,
        date_to=date_to,
        merchant_categories=cat_filter,
    )


@st.cache_data
def get_cached_country_breakdown(
    tiers: tuple[str, ...],
    date_from: str | None,
    date_to: str | None,
    merchant_category: str | None,
) -> pd.DataFrame:
    """Query country breakdown joined with CNR_COUNTRYRISK for threat tier and names."""
    db = get_db_manager()
    cat_filter = [merchant_category] if merchant_category and merchant_category != "All Categories" else None
    return db.get_flagged_country_breakdown(
        tiers=list(tiers) if tiers else None,
        date_from=date_from,
        date_to=date_to,
        merchant_categories=cat_filter,
    )


# ---------------------------------------------------------------------------
# Database Validation
# ---------------------------------------------------------------------------
db_file = _PROJECT_ROOT / "data" / "fraud.db"
if not db_file.exists():
    st.error("Database file `data/fraud.db` is missing. Please run `python -m src.pipeline` first.")
    st.stop()

db = get_db_manager()
try:
    total_transactions_count = db.get_total_transaction_count()
except (SQLAlchemyError, OSError) as exc:
    st.error(f"Failed to access database: {exc}. Please run `python -m src.pipeline` first.")
    st.stop()

if total_transactions_count == 0:
    st.error("Database is empty. Please run `python -m src.pipeline` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Header Row & Top Navigation Control
# ---------------------------------------------------------------------------
col_head_left, col_head_right = st.columns([1.8, 1.2])

with col_head_left:
    st.markdown(
        '### 🛡️ Fraud Risk Detection & Credit Risk Monitoring <span style="background-color: #16a34a; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.6em; font-weight: 700; vertical-align: middle; margin-left: 8px;">LIVE</span>',
        unsafe_allow_html=True,
    )

with col_head_right:
    nav_options = ["Overview", "Review queue", "Rules and countries"]
    if hasattr(st, "segmented_control"):
        selected_view = (
            st.segmented_control(
                "Navigation",
                options=nav_options,
                default="Overview",
                key="nav_view",
                label_visibility="collapsed",
            )
            or "Overview"
        )
    else:
        selected_view = (
            st.radio(
                "Navigation",
                options=nav_options,
                index=0,
                horizontal=True,
                key="nav_view",
                label_visibility="collapsed",
            )
            or "Overview"
        )

# ---------------------------------------------------------------------------
# Top Filter Controls Strip (Bordered Container)
# ---------------------------------------------------------------------------
all_categories, min_date_bound, max_date_bound = get_cached_metadata()

with st.container(border=True):
    st.markdown("**Filter Controls**")
    fcol1, fcol2, fcol3 = st.columns([1.4, 1.3, 1.3])

    # 1. Date Range Picker
    with fcol1:
        if min_date_bound is not None and max_date_bound is not None:
            min_date = min_date_bound.date()
            max_date = max_date_bound.date()
            date_selection = st.date_input(
                "Date Range (Transaction Timestamp)",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Filter transactions by TXN_TIMESTAMP.",
            )
            if isinstance(date_selection, (list, tuple)) and len(date_selection) == 2:
                start_date, end_date = date_selection
                filter_date_from = datetime.combine(start_date, time.min).isoformat()
                filter_date_to = datetime.combine(end_date, time.max).isoformat()
            elif isinstance(date_selection, (list, tuple)) and len(date_selection) == 1:
                start_date = date_selection[0]
                filter_date_from = datetime.combine(start_date, time.min).isoformat()
                filter_date_to = None
            else:
                filter_date_from = None
                filter_date_to = None
        else:
            filter_date_from = None
            filter_date_to = None

    # 2. Risk Tier Multi-Select
    with fcol2:
        available_tiers = ["Medium", "High", "Critical"]
        selected_tiers = st.multiselect(
            "Risk Tiers",
            options=available_tiers,
            default=available_tiers,
            help="Select risk tiers to include in analysis.",
        )

    # 3. Merchant Category Filter
    with fcol3:
        category_options = ["All Categories"] + sorted(all_categories)
        selected_category = st.selectbox(
            "Merchant Category",
            options=category_options,
            index=0,
        )

# Fetch filtered flagged data for charts, preview, and views
flagged_df = get_cached_flagged_df(
    tiers=tuple(selected_tiers),
    date_from=filter_date_from,
    date_to=filter_date_to,
    merchant_category=selected_category,
)

country_df = get_cached_country_breakdown(
    tiers=tuple(selected_tiers),
    date_from=filter_date_from,
    date_to=filter_date_to,
    merchant_category=selected_category,
)

# Fetch system-level metrics (strictly operational, never using is_fraud_actual)
metrics = get_cached_metrics()
total_flagged_count = metrics["total_flagged"]
exposure_by_tier = metrics["amount_exposure_by_tier"]
total_exposure_amount = sum(exposure_by_tier.values())
fp_rate = metrics["false_positive_rate"]
total_reviewed_count = metrics["total_reviewed"]
true_positives = metrics["true_positives"]
false_positives = metrics["false_positives"]

# False positive rate formatting: "N/A" until something is reviewed
fp_rate_str = f"{fp_rate * 100:.1f}%" if fp_rate is not None else "N/A"

# ---------------------------------------------------------------------------
# View 1: Overview
# ---------------------------------------------------------------------------
if selected_view == "Overview":
    col_left, col_center, col_right = st.columns([1, 3, 1.3])

    # --- Left Column: Five Stacked Global KPI Cards ---
    with col_left:
        with st.container(border=True):
            st.metric("Total Transactions", f"{total_transactions_count:,}")
        with st.container(border=True):
            st.metric("Total Flagged", f"{total_flagged_count:,}")
        with st.container(border=True):
            st.metric("Total $ Exposure", f"${total_exposure_amount:,.2f}")
        with st.container(border=True):
            st.metric(
                "False-Positive Rate",
                fp_rate_str,
                help=(
                    f"{false_positives} false positives out of {false_positives + true_positives} definitive reviews."
                    if fp_rate is not None
                    else "No reviews submitted yet."
                ),
            )
        with st.container(border=True):
            st.metric("Reviewed Count", f"{total_reviewed_count:,}")
        st.caption("KPIs show all data; filters apply to charts and tables.")

    # --- Centre Column: Operational Risk Overview & Time Series ---
    with col_center:
        st.subheader("Operational Risk Overview")
        if flagged_df.empty:
            st.warning("No flagged transactions match the active filter criteria.")
        else:
            with st.container(border=True):
                flagged_copy = flagged_df.copy()
                flagged_copy["date"] = pd.Series(
                    pd.to_datetime(flagged_copy["timestamp"]), index=flagged_copy.index
                ).dt.date
                ts_grouped = (
                    flagged_copy.groupby(["date", "risk_tier"])
                    .size()
                    .reset_index(name="flagged_count")
                )
                fig_ts = px.line(
                    ts_grouped,
                    x="date",
                    y="flagged_count",
                    color="risk_tier",
                    title="Flagged Transactions Over Time (by Risk Tier)",
                    markers=True,
                    color_discrete_map={
                        "Medium": "#F59E0B",
                        "High": "#EF4444",
                        "Critical": "#7F1D1D",
                    },
                )
                fig_ts.update_layout(
                    xaxis_title="Transaction Date",
                    yaxis_title="Flagged Count",
                    legend_title="Risk Tier",
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown("#### Filtered Exposure Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                with st.container(border=True):
                    st.metric("Filtered Flagged Count", f"{len(flagged_df):,}")
            with summary_col2:
                with st.container(border=True):
                    st.metric("Filtered Dollar Exposure", f"${flagged_df['amount'].sum():,.2f}")
            with summary_col3:
                with st.container(border=True):
                    avg_score = (
                        f"{flagged_df['fraud_score'].mean():.1f}"
                        if not flagged_df.empty and pd.notna(flagged_df["fraud_score"].mean())
                        else "N/A"
                    )
                    st.metric("Avg Fraud Score", avg_score)

    # --- Right Column: Donut Chart & High-Risk Alerts Preview ---
    with col_right:
        if flagged_df.empty:
            st.warning("No flagged transactions match the active filter criteria.")
        else:
            with st.container(border=True):
                tier_dist = flagged_df["risk_tier"].value_counts().reset_index()
                tier_dist.columns = ["risk_tier", "count"]
                fig_pie = px.pie(
                    tier_dist,
                    names="risk_tier",
                    values="count",
                    hole=0.45,
                    title="Risk Tier Distribution (Filtered)",
                    color="risk_tier",
                    color_discrete_map={
                        "Medium": "#F59E0B",
                        "High": "#EF4444",
                        "Critical": "#7F1D1D",
                    },
                )
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### High-Risk Alerts / Queue Preview")
            high_crit_df = cast(pd.DataFrame, flagged_df[flagged_df["risk_tier"].isin(["High", "Critical"])])
            preview_rows = high_crit_df.nlargest(20, "fraud_score")
            with st.container(height=320, border=True):
                if preview_rows.empty:
                    st.info("No High or Critical tier transactions in filtered view.")
                else:
                    for _, row in preview_rows.iterrows():
                        tier = str(row["risk_tier"])
                        tier_color = "#7F1D1D" if tier == "Critical" else "#EF4444"
                        score = row["fraud_score"]
                        amt = f"${float(row['amount']):,.2f}"
                        cid = str(row["card_id"])
                        ts = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M")
                        st.markdown(
                            f'<div style="padding: 6px 0; border-bottom: 1px solid #e2e8f0; font-size: 0.85rem;">'
                            f'<span style="color: {tier_color}; font-weight: bold;">[{tier}]</span> '
                            f'<b>Score: {score}</b> &bull; <b>{amt}</b><br/>'
                            f'<span style="color: #64748b;">Card: {cid} &bull; {ts}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            st.caption("Review these in the Review queue view.")

# ---------------------------------------------------------------------------
# View 2: Review Queue
# ---------------------------------------------------------------------------
elif selected_view == "Review queue":
    st.subheader("Analyst Review Queue")
    st.markdown(
        "Evaluate flagged transactions, assign review outcomes, and persist feedback to update operational metrics."
    )

    review_filter_col1, review_filter_col2 = st.columns([1, 2])
    with review_filter_col1:
        review_tiers = st.multiselect(
            "Review Queue Tiers",
            options=["Medium", "High", "Critical"],
            default=["High", "Critical"],
            help="Default: High and Critical. Add Medium as needed.",
        )

    # Filter transactions for review queue (max 200 rows, ordered by fraud score desc)
    review_pool = cast(
        pd.DataFrame,
        flagged_df[flagged_df["risk_tier"].isin(review_tiers)],
    )
    review_subset = review_pool.sort_values(by="fraud_score", ascending=False).head(200).copy()

    if review_subset.empty:
        st.info("No transactions available in the review queue for the selected tiers.")
    else:
        # Ensure review_outcome has a clean default string
        review_subset["review_outcome"] = (
            review_subset["review_outcome"].fillna("unreviewed").astype(str)
        )

        display_columns = [
            "transaction_id",
            "card_id",
            "amount",
            "fraud_score",
            "risk_tier",
            "triggered_rules",
            "review_outcome",
        ]
        editor_df = review_subset[display_columns].copy()

        st.caption(f"Displaying top {len(editor_df)} transactions ordered by fraud score (capped at 200).")

        edited_df = st.data_editor(
            editor_df,
            key="analyst_review_editor",
            column_config={
                "transaction_id": st.column_config.TextColumn("Transaction ID", disabled=True),
                "card_id": st.column_config.TextColumn("Card ID", disabled=True),
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f", disabled=True),
                "fraud_score": st.column_config.NumberColumn("Fraud Score", disabled=True),
                "risk_tier": st.column_config.TextColumn("Tier", disabled=True),
                "triggered_rules": st.column_config.TextColumn("Triggered Rules", disabled=True),
                "review_outcome": st.column_config.SelectboxColumn(
                    "Reviewer Outcome",
                    options=["unreviewed", "true_positive", "false_positive"],
                    required=True,
                    help="Select outcome: true_positive, false_positive, or unreviewed.",
                ),
            },
            disabled=[
                "transaction_id",
                "card_id",
                "amount",
                "fraud_score",
                "risk_tier",
                "triggered_rules",
            ],
            use_container_width=True,
            hide_index=True,
        )

        # Save reviews button
        if st.button("Save reviews", type="primary", help="Commit modified review outcomes to SQLite database."):
            changed_mask = edited_df["review_outcome"] != editor_df["review_outcome"]
            changed_rows = edited_df[changed_mask]

            if changed_rows.empty:
                st.info("No review outcomes were changed.")
            else:
                db_inst = get_db_manager()
                saved_count = 0
                failures: list[str] = []

                for _, row in changed_rows.iterrows():
                    txn_id = str(row["transaction_id"])
                    new_outcome = str(row["review_outcome"])
                    success = db_inst.update_review_outcome(txn_id, new_outcome)
                    if success:
                        saved_count += 1
                    else:
                        failures.append(txn_id)

                if failures:
                    st.error(
                        f"Failed to update {len(failures)} record(s): {', '.join(failures[:5])}"
                    )
                if saved_count > 0:
                    st.success(f"Successfully saved {saved_count} review decision(s). Refreshing metrics...")
                    st.cache_data.clear()
                    st.rerun()

# ---------------------------------------------------------------------------
# View 3: Rules and Countries
# ---------------------------------------------------------------------------
elif selected_view == "Rules and countries":
    st.subheader("Rule Triggers & Geographic Breakdown")

    col_rules, col_geo = st.columns(2)

    with col_rules:
        all_rules: list[str] = []
        for rules_str in flagged_df["triggered_rules"].dropna():
            if rules_str:
                for r in str(rules_str).split(","):
                    clean_r = r.strip()
                    if clean_r:
                        all_rules.append(clean_r)

        if not all_rules:
            st.info("No rule triggers recorded in the filtered transactions.")
        else:
            rule_freq = pd.Series(all_rules).value_counts().reset_index()
            rule_freq.columns = ["rule_name", "trigger_count"]
            rule_freq = rule_freq.sort_values(by="trigger_count", ascending=True)

            fig_rules = px.bar(
                rule_freq,
                x="trigger_count",
                y="rule_name",
                orientation="h",
                title="Rule Trigger Frequency (Filtered)",
                text="trigger_count",
                color="trigger_count",
                color_continuous_scale="Reds",
            )
            fig_rules.update_layout(
                xaxis_title="Times Triggered",
                yaxis_title="Rule Name",
                coloraxis_showscale=False,
            )
            fig_rules.update_traces(textposition="outside")
            st.plotly_chart(fig_rules, use_container_width=True)

    with col_geo:
        if country_df.empty:
            st.info("No geographic transaction data matching current filters.")
        else:
            geo_display = country_df.sort_values(by="flagged_count", ascending=True).tail(20)

            fig_geo = px.bar(
                geo_display,
                x="flagged_count",
                y="country_name",
                color="threat_tier",
                orientation="h",
                title="Flagged Transactions by Country (Top 20 by Threat Tier)",
                hover_data={
                    "country_code": True,
                    "threat_tier": True,
                    "flagged_count": True,
                    "total_amount": ":$,.2f",
                },
                color_discrete_map={
                    "low": "#10B981",
                    "medium": "#F59E0B",
                    "high": "#EF4444",
                    "Unknown": "#6B7280",
                },
            )
            fig_geo.update_layout(
                xaxis_title="Flagged Transactions Count",
                yaxis_title="Country",
                legend_title="Threat Tier",
            )
            st.plotly_chart(fig_geo, use_container_width=True)

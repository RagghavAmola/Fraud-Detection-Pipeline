"""Unit tests for Streamlit dashboard application."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


def test_dashboard_app_renders_successfully():
    """Verify that dashboard/app.py executes without exception and renders KPIs and navigation."""
    at = AppTest.from_file("dashboard/app.py")
    at.run(timeout=30)

    # Assert no unhandled exceptions occurred
    assert len(at.exception) == 0, f"App execution raised exceptions: {[e.value for e in at.exception]}"

    # Verify top KPI metrics are present
    labels = [m.label for m in at.metric]
    assert "Total Transactions" in labels
    assert "Total Flagged" in labels
    assert "Total $ Exposure" in labels
    assert "False-Positive Rate" in labels
    assert "Reviewed Count" in labels

    # Verify overview summary metrics are present
    assert "Filtered Flagged Count" in labels
    assert "Filtered Dollar Exposure" in labels
    assert "Avg Fraud Score" in labels

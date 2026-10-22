"""Data transformation and feature engineering module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.transformation.feature_engineering import FeatureEngineer

__all__ = ["FeatureEngineer"]


def __getattr__(name: str):
    if name == "FeatureEngineer":
        from src.transformation.feature_engineering import FeatureEngineer

        return FeatureEngineer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

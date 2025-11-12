"""
Feast repository scaffolding.

This package hosts the objects referenced by `feature_store.yaml` so we can import
them from notebooks, CLI commands, or automated jobs without duplicating logic.
"""

FEAST_PROJECT = "nba_predictor_feast"

__all__ = ["FEAST_PROJECT"]

from __future__ import annotations

from typing import Callable, List

from src.features.cleanup import enforce_feature_whitelist
from src.features.odds import drop_odds_columns
from src.features.plan import DEFAULT_FEATURE_PLAN
from src.features.targets import drop_for_target

# ---------------------------------------------------------------------------
# Default pipelines (built from the feature plan)
# ---------------------------------------------------------------------------

DEFAULT_SILVER_FEATURE_STEPS: List[Callable[[pd.DataFrame], pd.DataFrame]] = DEFAULT_FEATURE_PLAN.feature_functions()

DEFAULT_SILVER_TARGET_STEPS: List[Callable[[pd.DataFrame], pd.DataFrame]] = DEFAULT_FEATURE_PLAN.target_functions()


def gold_steps_for_target(target: str) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
    def _drop_target_columns(df: pd.DataFrame) -> pd.DataFrame:
        return drop_for_target(df, target)

    return [
        drop_odds_columns,
        enforce_feature_whitelist,
        _drop_target_columns,
    ]

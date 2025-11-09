from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .base import BuildResult
from .gold import GoldBuildConfig, build_gold_dataset
from .silver import SilverBuildConfig, build_silver_dataset


def build_silver_and_gold(
    bronze_matches: pd.DataFrame,
    *,
    target: str = "IS_WIN",
    silver_config: Optional[SilverBuildConfig] = None,
    gold_config: Optional[GoldBuildConfig] = None,
) -> Tuple[BuildResult, BuildResult]:
    """
    Convenience helper for notebooks: run silver build, then derive gold for a target.

    Args:
        bronze_matches: Match-level dataframe (two rows per game) prepared from bronze inputs.
        target: Target name for the gold cleaning phase.
        silver_config: Optional overrides for feature/target steps.
        gold_config: Optional overrides for cleaning steps / output format.

    Returns:
        (silver_result, gold_result) BuildResult tuple.
    """

    silver_cfg = silver_config or SilverBuildConfig()
    gold_cfg = gold_config or GoldBuildConfig(target=target)

    silver_result = build_silver_dataset(bronze_matches, config=silver_cfg)
    gold_result = build_gold_dataset(silver_result.dataset, config=gold_cfg)

    return silver_result, gold_result

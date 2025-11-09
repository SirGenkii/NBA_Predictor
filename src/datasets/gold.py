from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from .base import (
    BuildResult,
    DatasetPaths,
    run_pipeline,
    save_dataset,
    timestamped_filename,
)
from .recipes import gold_steps_for_target


@dataclass
class GoldBuildConfig:
    """Configuration holder for the gold dataset cleaning/export."""

    cleaning_steps: Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]] = None
    target: Optional[str] = "IS_WIN"
    output_prefix: str = "gold_dataset"
    output_format: str = "parquet"
    persist_artifact: bool = True
    extra_metadata: dict = field(default_factory=dict)


def build_gold_dataset(
    silver_df: pd.DataFrame,
    config: Optional[GoldBuildConfig] = None,
    paths: Optional[DatasetPaths] = None,
) -> BuildResult:
    """
    Apply leak-safe cleaning on top of the silver dataset and export a gold artifact.

    The `target` field can be set to document which modeling objective the gold file
    is intended for (IS_WIN, POINT_DIFF, POINT_TOTAL, ...).
    """

    cfg = config or GoldBuildConfig()
    dirs = paths or DatasetPaths()

    steps = cfg.cleaning_steps or gold_steps_for_target(cfg.target or "IS_WIN")

    df = silver_df.copy()
    df = run_pipeline(df, steps)

    artifact_path: Optional[Path] = None
    if cfg.persist_artifact:
        suffix = cfg.target.lower() if cfg.target else "all_targets"
        filename = timestamped_filename(f"{cfg.output_prefix}_{suffix}", extension=cfg.output_format)
        artifact_path = dirs.gold / filename
        save_dataset(df, artifact_path, fmt=cfg.output_format)

    metadata = {"rows": len(df), "columns": len(df.columns), "target": cfg.target}
    metadata.update(cfg.extra_metadata)

    return BuildResult(dataset=df, path=artifact_path, metadata=metadata)

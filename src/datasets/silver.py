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
from .recipes import DEFAULT_SILVER_FEATURE_STEPS, DEFAULT_SILVER_TARGET_STEPS


@dataclass
class SilverBuildConfig:
    """Configuration holder for the silver dataset construction."""

    feature_steps: List[Callable[[pd.DataFrame], pd.DataFrame]] = field(
        default_factory=lambda: DEFAULT_SILVER_FEATURE_STEPS.copy()
    )
    target_steps: List[Callable[[pd.DataFrame], pd.DataFrame]] = field(
        default_factory=lambda: DEFAULT_SILVER_TARGET_STEPS.copy()
    )
    output_prefix: str = "silver_dataset"
    output_format: str = "parquet"
    persist_artifact: bool = True
    extra_metadata: dict = field(default_factory=dict)


def build_silver_dataset(
    base_df: pd.DataFrame,
    config: Optional[SilverBuildConfig] = None,
    paths: Optional[DatasetPaths] = None,
) -> BuildResult:
    """
    Run the silver feature/target pipeline on top of the bronze match-level dataset.

    The caller (e.g., notebook 05) is responsible for providing the pre-merged
    match dataframe. Feature and target steps registered in the config are applied
    sequentially so the same code can be reused from scripts or flows.
    """

    cfg = config or SilverBuildConfig()
    dirs = paths or DatasetPaths()

    df = base_df.copy()
    df = run_pipeline(df, cfg.feature_steps)
    df = run_pipeline(df, cfg.target_steps)

    artifact_path: Optional[Path] = None
    if cfg.persist_artifact:
        filename = timestamped_filename(cfg.output_prefix, extension=cfg.output_format)
        artifact_path = dirs.silver / filename
        save_dataset(df, artifact_path, fmt=cfg.output_format)

    metadata = {"rows": len(df), "columns": len(df.columns)}
    metadata.update(cfg.extra_metadata)

    return BuildResult(dataset=df, path=artifact_path, metadata=metadata)

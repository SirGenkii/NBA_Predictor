from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import pandas as pd

from src.config import DATA_BRONZE_DIR, DATA_SILVER_DIR, DATA_GOLD_DIR

Stage = Literal["bronze", "silver", "gold"]
PipelineStep = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class DatasetPaths:
    """Convenience accessor for the bronze/silver/gold directories."""

    bronze: Path = Path(DATA_BRONZE_DIR)
    silver: Path = Path(DATA_SILVER_DIR)
    gold: Path = Path(DATA_GOLD_DIR)

    def for_stage(self, stage: Stage) -> Path:
        mapping = {"bronze": self.bronze, "silver": self.silver, "gold": self.gold}
        try:
            return mapping[stage]
        except KeyError as exc:
            raise ValueError(f"Unknown stage '{stage}'") from exc


@dataclass
class BuildResult:
    dataset: pd.DataFrame
    path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_filename(prefix: str, extension: str = "parquet") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}.{extension}"


def run_pipeline(df: pd.DataFrame, steps: Iterable[PipelineStep]) -> pd.DataFrame:
    result = df
    for step in steps:
        result = step(result)
    return result


def save_dataset(df: pd.DataFrame, path: Path, *, fmt: str = "parquet", **kwargs) -> Path:
    ensure_dir(path.parent)
    if fmt == "parquet":
        df.to_parquet(path, index=False, **kwargs)
    elif fmt == "csv":
        df.to_csv(path, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported format '{fmt}'")
    return path

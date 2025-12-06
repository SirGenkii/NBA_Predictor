from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config import (
    DATA_GOLD_DIR,
    COLS_MATCH_REAL,
    cols_player_stats,
    cols_to_sum,
    cols_to_weighted_avg,
    player_absent_input_cols,
)


def _leak_columns() -> List[str]:
    raw_cols = cols_to_sum + cols_to_weighted_avg + cols_player_stats + list(player_absent_input_cols)
    prefixed = []
    for col in raw_cols:
        prefixed.extend([f"HOME_{col}", f"AWAY_{col}", f"DIFF_{col}"])

    helper = [
        "POINTS_FOR",
        "POINTS_AGAINST",
        "HOME_POINTS_FOR",
        "AWAY_POINTS_FOR",
        "HOME_IS_WIN",
        "AWAY_IS_WIN",
    ]
    return prefixed + helper


def _latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Aucun fichier ne correspond à {pattern} dans {directory}")
    return matches[-1]


@dataclass
class DatasetConfig:
    """Describe how to locate and prepare the modeling dataset."""

    target: str = "IS_WIN"
    gold_dir: Path = Path(DATA_GOLD_DIR)
    gold_pattern: str = "gold_dataset_is_win_*.parquet"
    path: Optional[Path] = None
    drop_columns: List[str] = field(
        default_factory=lambda: [
            "GAME_ID",
            "GAME_DATE",
            "SEASON",
            "HOME_TEAM_ID",
            "AWAY_TEAM_ID",
        ]
        + COLS_MATCH_REAL
        + _leak_columns()
    )
    keep_numeric_only: bool = True
    dropna: bool = False
    use_feast: bool = False
    feast_repo_path: Path = Path("src/feast")
    feast_feature_service: Optional[str] = None
    feast_timestamp_column: str = "GAME_DATE"
    feast_entity_mapping: Dict[str, str] = field(default_factory=lambda: {"match_id": "GAME_ID"})

    def resolve_path(self) -> Path:
        if self.path:
            resolved = Path(self.path)
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            return resolved
        return _latest_file(self.gold_dir, self.gold_pattern)


@dataclass
class TrainingConfig:
    """Generic training hyper-parameters and MLflow settings."""

    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    split_strategy: str = "walk_forward"  # or "random"
    season_column: str = "SEASON"
    train_seasons: int = 6
    calibration_seasons: int = 2
    test_seasons: int = 2
    walk_forward_folds: int = 1
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "is_win_modeling"
    model_output_dir: Path = Path("data/models")
    task_type: str = "regression"  # or "classification"
    enable_learning_curve: bool = True
    pivot_value: Optional[float] = None
    pivot_values: Optional[List[float]] = None
    enable_sigma_model: bool = True
    min_sigma: float = 5.0
    registry_model_name: Optional[str] = None
    registry_stage: str = "Production"
    registry_archive_existing: bool = True
    production_model_key: Optional[str] = None


@dataclass
class ModelRunConfig:
    """Describe which models to train and optional stacking behaviour."""

    models: List[str] = field(default_factory=lambda: ["lgbm", "xgb", "stacking"])
    enable_stacking: bool = True

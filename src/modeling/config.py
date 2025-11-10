from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.config import (
    DATA_GOLD_DIR,
    COLS_MATCH_REAL,
    cols_to_sum,
    cols_to_weighted_avg,
    cols_player_stats,
    player_absent_input_cols,
)


def _leak_columns() -> List[str]:
    base_stats = cols_to_sum + cols_to_weighted_avg
    opp_stats = [f"OPP_{col}" for col in base_stats]
    player_stats = cols_player_stats + [f"OPP_{col}" for col in cols_player_stats]
    availability_cols = [
        "has_absent",
        "has_top_absent",
        "top_player_absent",
        "top_player_absent_rate",
        "top_player_injury_rate",
        "top_player_resting_rate",
        "top_player_suspension_rate",
        "top_player_personal_rate",
        "top_player_absent_other_rate",
        "top_player_count",
        "num_absent",
        "num_injured",
        "num_resting",
        "num_suspended",
        "num_personal",
        "num_absent_other",
        "num_present",
    ] + list(player_absent_input_cols)
    availability_cols = list(dict.fromkeys(availability_cols))
    availability_cols += [f"OPP_{col}" for col in availability_cols]
    explicit_targets = [
        "POINTS_FOR",
        "POINTS_AGAINST",
        "POINT_DIFF",
        "PTS",
        "OPP_PTS",
        "points_traditional",
        "OPP_points_traditional",
    ]
    return base_stats + opp_stats + player_stats + availability_cols + explicit_targets


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
            "TEAM_ID",
            "OPP_TEAM_ID",
            "GAME_DATE",
            "OPP_GAME_DATE",
            "MATCHUP",
            "SEASON",
        ]
        + COLS_MATCH_REAL
        + _leak_columns()
    )
    keep_numeric_only: bool = True
    dropna: bool = False

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
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "is_win_modeling"
    model_output_dir: Path = Path("data/models")
    task_type: str = "classification"  # or "regression"
    enable_learning_curve: bool = True
    pivot_value: Optional[float] = None
    pivot_values: Optional[List[float]] = None
    enable_sigma_model: bool = True
    min_sigma: float = 5.0


@dataclass
class ModelRunConfig:
    """Describe which models to train and optional stacking behaviour."""

    models: List[str] = field(default_factory=lambda: ["lgbm", "xgb", "stacking"])
    enable_stacking: bool = True

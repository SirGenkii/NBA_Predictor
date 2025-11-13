from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.config import DATA_SILVER_DIR
from .config import DatasetConfig, TrainingConfig
from .trainer import ModelTrainer


@dataclass(frozen=True)
class TrainerBundle:
    dataset: DatasetConfig
    training: TrainingConfig


def point_total_bundle() -> TrainerBundle:
    dataset_cfg = DatasetConfig(
        target="POINT_TOTAL",
        gold_dir=DATA_SILVER_DIR,
        gold_pattern="silver_dataset_*.parquet",
        use_feast=True,
        feast_feature_service="point_total_service",
    )
    training_cfg = TrainingConfig(
        experiment_name="point_total_regression",
        tracking_uri="file:./mlruns",
        test_size=0.2,
        random_state=42,
        stratify=False,
        task_type="regression",
        pivot_value=220.0,
        pivot_values=[v + 0.5 for v in range(218, 245)],
        enable_learning_curve=False,
        enable_sigma_model=True,
        min_sigma=6.0,
    )
    return TrainerBundle(dataset=dataset_cfg, training=training_cfg)


def build_point_total_trainer() -> ModelTrainer:
    bundle = point_total_bundle()
    return ModelTrainer(bundle.dataset, bundle.training)

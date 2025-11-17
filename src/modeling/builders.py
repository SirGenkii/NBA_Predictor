from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from src.config import (
    DATA_SILVER_DIR,
    MLFLOW_POINT_TOTAL_MODEL_NAME,
    MLFLOW_POINT_TOTAL_MODEL_STAGE,
    POINT_TOTAL_DEFAULT_MODELS,
    POINT_TOTAL_PRODUCTION_MODEL_KEY,
)
from .config import DatasetConfig, TrainingConfig
from .trainer import ModelTrainer


@dataclass(frozen=True)
class TrainerBundle:
    dataset: DatasetConfig
    training: TrainingConfig


def point_total_bundle(*, enable_registry: bool = False) -> TrainerBundle:
    dataset_cfg = DatasetConfig(
        target="POINT_TOTAL",
        gold_dir=DATA_SILVER_DIR,
        gold_pattern="silver_dataset_*.parquet",
        # Keep all Feast features (numeric) and let Feast/schema_loader drop only true leaks/IDs.
        drop_columns=[],
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
        registry_model_name=MLFLOW_POINT_TOTAL_MODEL_NAME if enable_registry else None,
        registry_stage=MLFLOW_POINT_TOTAL_MODEL_STAGE,
        production_model_key=POINT_TOTAL_PRODUCTION_MODEL_KEY if enable_registry else None,
    )
    return TrainerBundle(dataset=dataset_cfg, training=training_cfg)


def build_point_total_trainer(*, enable_registry: bool = False) -> ModelTrainer:
    bundle = point_total_bundle(enable_registry=enable_registry)
    return ModelTrainer(bundle.dataset, bundle.training)


def is_win_bundle() -> TrainerBundle:
    dataset_cfg = DatasetConfig(
        target="IS_WIN",
        gold_dir=DATA_SILVER_DIR,
        gold_pattern="silver_dataset_*.parquet",
        use_feast=True,
        feast_feature_service="is_win_service",
    )
    training_cfg = TrainingConfig(
        experiment_name="is_win_classification",
        tracking_uri="file:./mlruns",
        stratify=True,
        task_type="classification",
        enable_learning_curve=False,
    )
    return TrainerBundle(dataset=dataset_cfg, training=training_cfg)


def build_is_win_trainer() -> ModelTrainer:
    bundle = is_win_bundle()
    return ModelTrainer(bundle.dataset, bundle.training)

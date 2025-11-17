from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import optuna

from src.config import POINT_TOTAL_DEFAULT_MODELS
from .config import DatasetConfig, TrainingConfig
from .trainer import ModelTrainer


@dataclass
class TuningResult:
    model: str
    best_value: float
    best_params: Dict[str, float]
    study: optuna.Study


def _param_space(trial: optuna.trial.Trial, model_key: str, task_type: str) -> Dict[str, float]:
    if model_key == "lgbm":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 400, 1400, step=200),
            "model__num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_key == "xgb":
        return {
            "model__n_estimators": trial.suggest_int("n_estimators", 400, 1600, step=200),
            "model__max_depth": trial.suggest_int("max_depth", 3, 9),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1e-1, log=True),
        }
    # if model_key == "ngboost":
    #     return {
    #         "model__n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=200),
    #         "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
    #         "model__minibatch_frac": trial.suggest_float("minibatch_frac", 0.5, 1.0),
    #     }
    if model_key == "catboost":
        return {
            "model__depth": trial.suggest_int("depth", 6, 10),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "model__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if model_key in {"stacking", "stacking_full"}:
        if task_type == "classification":
            return {
                "final_estimator__C": trial.suggest_float("stack_C", 1e-3, 10.0, log=True),
            }
        return {
            "final_estimator__alpha": trial.suggest_float("stack_alpha", 1e-3, 10.0, log=True),
        }
    if model_key == "stacking_linear":
        return {}
    if model_key == "stacking_xgbmeta":
        return {
            "final_estimator__learning_rate": trial.suggest_float("stack_lr", 0.01, 0.2, log=True),
            "final_estimator__max_depth": trial.suggest_int("stack_depth", 2, 5),
        }
    raise ValueError(f"No Optuna search space configured for model '{model_key}'.")


def tune_point_total(
    dataset_cfg: DatasetConfig,
    training_cfg: TrainingConfig,
    *,
    models: Optional[List[str]] = None,
    n_trials: int = 25,
    metric: str = "rmse",
) -> List[TuningResult]:
    """
    Run Optuna to tune the regression models used for POINT_TOTAL.

    Example
    -------
    >>> dataset_cfg = DatasetConfig(target="POINT_TOTAL", path="data/03_gold/point_total.parquet")
    >>> training_cfg = TrainingConfig(task_type="regression", pivot_values=[225, 230])
    >>> results = tune_point_total(dataset_cfg, training_cfg, models=["lgbm", "xgb"], n_trials=20)
    >>> for res in results:
    ...     print(res.model, res.best_value, res.best_params)
    """

    trainer = ModelTrainer(dataset_cfg, training_cfg)
    available = trainer.available_models()
    desired = models or POINT_TOTAL_DEFAULT_MODELS
    target_models = [model for model in desired if model in available]
    return _run_tuning(trainer, target_models, n_trials=n_trials, metric=metric)


def tune_is_win(
    dataset_cfg: DatasetConfig,
    training_cfg: TrainingConfig,
    *,
    models: Optional[List[str]] = None,
    n_trials: int = 25,
    metric: str = "log_loss",
) -> List[TuningResult]:
    """Optuna tuning helper for IS_WIN classification."""

    trainer = ModelTrainer(dataset_cfg, training_cfg)
    target_models = models or trainer.available_models()
    return _run_tuning(trainer, target_models, n_trials=n_trials, metric=metric)


def _run_tuning(
    trainer: ModelTrainer,
    models: List[str],
    *,
    n_trials: int,
    metric: str,
) -> List[TuningResult]:
    results: List[TuningResult] = []

    direction = "maximize" if metric.lower() == "r2" else "minimize"

    for model_key in models:
        if model_key not in trainer.available_models():
            continue

        def objective(trial: optuna.trial.Trial) -> float:
            overrides = _param_space(trial, model_key, trainer.training_cfg.task_type)
            metrics = trainer.train_with_params(
                model_key,
                param_overrides=overrides,
                log_run=False,
                run_name=f"tuning_{model_key}",
            )
            value = metrics.get(metric)
            if value is None:
                raise ValueError(f"Metric '{metric}' introuvable dans {metrics.keys()}")
            return value

        study = optuna.create_study(
            study_name=f"{model_key}_point_total_tuning",
            direction=direction,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_value = study.best_value
        results.append(
            TuningResult(
                model=model_key,
                best_value=best_value,
                best_params=study.best_params,
                study=study,
            )
        )

    return results

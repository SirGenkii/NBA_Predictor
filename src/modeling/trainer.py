from __future__ import annotations

import json
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import is_dataclass
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.utils._tags import default_tags
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import CRPScore
from catboost import CatBoostRegressor

from .config import DatasetConfig, TrainingConfig
from .data import load_dataset, prepare_features, train_test_split_data
from .metrics import classification_metrics, confusion_matrix_values, regression_metrics
from .plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_pred_vs_actual,
    plot_residual_hist,
    plot_gaussian_prediction,
)
from .constants import CALIBRATION_ARTIFACT_PATH


class RegressorPipeline(Pipeline, RegressorMixin):
    """Pipeline that advertises estimator_type=regressor for sklearn meta-estimators."""


def _make_regressor_pipeline(steps):
    return RegressorPipeline(steps=steps)


def _ensure_regressor_tags(tags):
    """Normalize sklearn tags objects/dicts and mark estimator as regressor."""
    if is_dataclass(tags):
        tags.estimator_type = "regressor"
        return tags

    if hasattr(tags, "todict"):
        tags = tags.todict()
    elif isinstance(tags, Mapping):
        tags = dict(tags)
    else:
        tags = dict(tags)
    tags["estimator_type"] = "regressor"
    return tags


class SklearnCompatibleCatBoost(CatBoostRegressor, RegressorMixin, BaseEstimator):
    """CatBoost wrapper exposing sklearn tags."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __sklearn_tags__(self):
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            tags = default_tags(self)
        return _ensure_regressor_tags(tags)


class SklearnCompatibleNGB(NGBRegressor, RegressorMixin, BaseEstimator):
    """Ensure ngboost plays nicely with sklearn stacking/tag checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return _ensure_regressor_tags(tags)


class ModelTrainer:
    def __init__(self, dataset_cfg: DatasetConfig, training_cfg: TrainingConfig):
        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg

        mlflow.set_tracking_uri(training_cfg.tracking_uri)
        mlflow.set_experiment(training_cfg.experiment_name)

        self._dataset_path = None
        self._feature_names: List[str] = []
        self._sigma_models: Dict[str, Pipeline] = {}
        self._load_data()

    def _load_data(self):
        df = load_dataset(self.dataset_cfg)
        self._dataset_path = df.attrs.get("source_path")
        X, y, feature_names = prepare_features(df, self.dataset_cfg)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split_data(X, y, self.training_cfg)
        self._feature_names = feature_names

    def available_models(self) -> List[str]:
        base = ["lgbm", "xgb", "stacking"]
        if self.training_cfg.task_type == "regression":
            base.extend(
                [
                    "ngboost",  # désactivé temporairement
                    "catboost",
                    "stacking_full",
                    "stacking_linear",
                    "stacking_xgbmeta",
                ]
            )
            base.append("xgb_calibrated")
        return base

    def train_models(self, models: List[str]) -> pd.DataFrame:
        results = []
        for model_key in models:
            if model_key not in self.available_models():
                raise ValueError(f"Modèle {model_key} non supporté.")
            metrics = self._train_single_model(model_key)
            results.append({"model": model_key, **metrics})
        return pd.DataFrame(results)

    def train_with_params(
        self,
        model_key: str,
        *,
        param_overrides: Optional[dict] = None,
        log_run: bool = True,
        run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        if model_key not in self.available_models():
            raise ValueError(f"Modèle {model_key} non supporté.")
        return self._train_single_model(
            model_key,
            param_overrides=param_overrides,
            log_run=log_run,
            run_name=run_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_single_model(
        self,
        model_key: str,
        *,
        param_overrides: Optional[dict] = None,
        log_run: bool = True,
        run_name: Optional[str] = None,
    ) -> Dict[str, float]:

        pipeline = self._build_model(model_key, overrides=param_overrides)
        run_name = run_name or f"{model_key}"

        context = mlflow.start_run(run_name=run_name) if log_run else nullcontext()
        with context:
            if log_run:
                mlflow.log_param("dataset_path", self._dataset_path)
                mlflow.log_param("target", self.dataset_cfg.target)
                mlflow.log_params(
                    {
                        "test_size": self.training_cfg.test_size,
                        "random_state": self.training_cfg.random_state,
                        "model_key": model_key,
                        "task_type": self.training_cfg.task_type,
                    }
                )

            if self.training_cfg.enable_learning_curve and log_run:
                self._log_learning_curve(pipeline, model_key)

            pipeline.fit(self.X_train, self.y_train)
            y_pred = pd.Series(pipeline.predict(self.X_test), index=self.X_test.index)

            if self.training_cfg.task_type == "classification":
                y_proba = pd.Series(
                    pipeline.predict_proba(self.X_test)[:, 1],
                    index=self.X_test.index,
                )
                metrics = classification_metrics(self.y_test, y_pred, y_proba)
                if log_run:
                    for k, v in metrics.items():
                        mlflow.log_metric(k, float(v))

                    cm_vals = confusion_matrix_values(self.y_test, y_pred)
                    for k, v in cm_vals.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"cm_{k}", v)

                preds_df = pd.DataFrame(
                    {
                        "y_true": self.y_test,
                        "y_pred": y_pred,
                        "proba": y_proba,
                    },
                    index=self.X_test.index,
                )
                if log_run:
                    self._log_classification_plots(model_key, y_pred, y_proba)
            else:
                metrics = regression_metrics(self.y_test, y_pred)
                if log_run:
                    for k, v in metrics.items():
                        mlflow.log_metric(k, float(v))
                sigma_series = self._get_sigma_predictions(model_key, pipeline)
                if log_run:
                    mlflow.log_metric("residual_std", float(sigma_series.mean()))
                calibration_payload = None
                calibration_frame = None
                if model_key == "xgb_calibrated":
                    calibration_payload, calibration_frame = self._fit_isotonic_calibrators(
                        y_pred,
                        sigma_series,
                        self._pivot_list(),
                    )
                    if calibration_payload and log_run:
                        self._log_calibration_artifact(calibration_payload, calibration_frame)
                preds_df = pd.DataFrame(
                    {
                        "y_true": self.y_test,
                        "y_pred": y_pred,
                        "sigma": sigma_series,
                        "residual": self.y_test - y_pred,
                    },
                    index=self.X_test.index,
                )
                if log_run:
                    self._log_regression_plots(
                        model_key,
                        y_pred,
                        sigma_series,
                        self._pivot_list(),
                    )

            if log_run:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    preds_df.to_csv(tmp.name, index=False)
                    mlflow.log_artifact(tmp.name, artifact_path="predictions")

                importances = self._extract_feature_importances(pipeline)
                if importances is not None:
                    fi_names, fi_values = self._align_feature_importances(
                        self._feature_names, importances
                    )
                    fig = plot_feature_importance(fi_names, fi_values)
                    if fig:
                        mlflow.log_figure(
                            fig, f"plots/{model_key}_feature_importance.png"
                        )
                        plt.close(fig)
                    fi_df = pd.DataFrame(
                        {
                            "feature": np.ravel(fi_names),
                            "importance": np.ravel(fi_values),
                        }
                    ).sort_values("importance", ascending=False)
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        fi_df.to_csv(tmp.name, index=False)
                        mlflow.log_artifact(tmp.name, artifact_path="feature_importance")

                model_artifact_logged = False
                active_run = mlflow.active_run()
                model_uri = None
                if active_run:
                    model_uri = f"runs:/{active_run.info.run_id}/model"
                try:
                    sample_X = (
                        self.X_train.head(5)
                        if hasattr(self.X_train, "head")
                        else self.X_train[:5]
                    )
                    sample_y = pipeline.predict(sample_X)
                    signature = infer_signature(sample_X, sample_y)
                    mlflow.sklearn.log_model(
                        pipeline,
                        artifact_path="model",
                        signature=signature,
                        input_example=sample_X,
                    )
                    model_artifact_logged = True
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[trainer] Warning: mlflow.sklearn.log_model failed ({exc}); "
                        "logging raw artifact instead."
                    )

                if not model_artifact_logged:
                    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                        joblib.dump(pipeline, tmp.name)
                        mlflow.log_artifact(tmp.name, artifact_path="model")

                if (
                    model_artifact_logged
                    and self.training_cfg.registry_model_name
                    and self.training_cfg.production_model_key
                    and model_key == self.training_cfg.production_model_key
                    and model_uri
                ):
                    self._register_model_version(model_uri)

            return metrics

    def _build_model(self, key: str, overrides: Optional[dict] = None) -> Pipeline:
        if self.training_cfg.task_type == "classification":
            pipeline = self._build_classification_model(key)
        else:
            pipeline = self._build_regression_model(key)
        if overrides:
            pipeline = pipeline.set_params(**self._normalize_overrides(pipeline, overrides))
        return pipeline

    def _register_model_version(self, model_uri: str) -> None:
        if not model_uri:
            return
        name = self.training_cfg.registry_model_name
        if not name:
            return
        try:
            result = mlflow.register_model(model_uri, name)
            stage = self.training_cfg.registry_stage
            if stage:
                client = MlflowClient()
                client.transition_model_version_stage(
                    name=name,
                    version=result.version,
                    stage=stage,
                    archive_existing_versions=self.training_cfg.registry_archive_existing,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[trainer] Warning: MLflow registration failed ({exc}).")

    @staticmethod
    def _normalize_overrides(pipeline: Pipeline, overrides: dict) -> dict:
        """Accept bare parameter names (e.g., `n_estimators`) and route them to the underlying model."""

        normalized = {}
        has_model_step = hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps
        alias = {
            "stack_alpha": "final_estimator__alpha",
            "stack_depth": "final_estimator__max_depth",
            "stack_lr": "final_estimator__learning_rate",
        }

        for raw_key, value in overrides.items():
            key = alias.get(raw_key, raw_key)

            if has_model_step:
                # Avoid double-prefixing if already targeted at the model step.
                if key.startswith("model__"):
                    normalized[key] = value
                    continue
                if "__" in key:
                    normalized[f"model__{key}"] = value
                    continue
                normalized[f"model__{key}"] = value
                continue

            normalized[key] = value
        return normalized

    def _build_classification_model(self, key: str) -> Pipeline:
        if key == "lgbm":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=400,
                            learning_rate=0.03,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=self.training_cfg.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        if key == "xgb":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=500,
                            learning_rate=0.03,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            eval_metric="logloss",
                            tree_method="hist",
                            random_state=self.training_cfg.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        if key == "stacking":
            estimators = [
                ("lgbm", self._build_classification_model("lgbm")),
                ("xgb", self._build_classification_model("xgb")),
            ]
            final_estimator = LogisticRegression(
                max_iter=1000, random_state=self.training_cfg.random_state
            )
            return StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                stack_method="predict_proba",
                n_jobs=-1,
                passthrough=False,
            )
        raise ValueError(f"Modèle {key} non supporté.")

    def _build_regression_model(self, key: str) -> Pipeline:
        if key == "lgbm":
            return _make_regressor_pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=600,
                            learning_rate=0.035,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=self.training_cfg.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        if key in {"xgb", "xgb_calibrated"}:
            return _make_regressor_pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=800,
                            learning_rate=0.035,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method="hist",
                            random_state=self.training_cfg.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        if key == "ngboost":
            ngb_model = SklearnCompatibleNGB(
                Dist=Normal,
                Score=CRPScore,
                n_estimators=800,
                learning_rate=0.03,
                natural_gradient=True,
                verbose=False,
                random_state=self.training_cfg.random_state,
            )
            return _make_regressor_pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                    (
                        "model",
                        ngb_model,
                    ),
                ]
            )
        if key == "catboost":
            cat_model = SklearnCompatibleCatBoost(
                depth=8,
                iterations=1000,
                learning_rate=0.03,
                subsample=0.8,
                loss_function="RMSE",
                random_seed=self.training_cfg.random_state,
                verbose=False,
            )
            return _make_regressor_pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        cat_model,
                    ),
                ]
            )
        if key == "stacking":
            estimators = [
                ("lgbm", self._build_regression_model("lgbm")),
                ("xgb", self._build_regression_model("xgb")),
                # ("ngboost", self._build_regression_model("ngboost")),
                ("catboost", self._build_regression_model("catboost")),
            ]
            final_estimator = Ridge()
            return StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                n_jobs=-1,
                passthrough=False,
            )
        if key == "stacking_full":
            estimators = [
                ("lgbm", self._build_regression_model("lgbm")),
                ("xgb", self._build_regression_model("xgb")),
                # ("ngboost", self._build_regression_model("ngboost")),
                ("catboost", self._build_regression_model("catboost")),
            ]
            final_estimator = Ridge(alpha=0.1)
            stacker = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                n_jobs=-1,
                passthrough=True,
            )
            return _make_regressor_pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", stacker),
                ]
            )
        if key == "stacking_linear":
            estimators = [
                ("lgbm", self._build_regression_model("lgbm")),
                ("xgb", self._build_regression_model("xgb")),
                # ("ngboost", self._build_regression_model("ngboost")),
                ("catboost", self._build_regression_model("catboost")),
            ]
            final_estimator = LinearRegression(fit_intercept=False)
            return StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                n_jobs=-1,
                passthrough=False,
            )
        if key == "stacking_xgbmeta":
            estimators = [
                ("lgbm", self._build_regression_model("lgbm")),
                ("xgb", self._build_regression_model("xgb")),
                # ("ngboost", self._build_regression_model("ngboost")),
                ("catboost", self._build_regression_model("catboost")),
            ]
            final_estimator = XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=self.training_cfg.random_state,
                n_jobs=-1,
            )
            return StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                n_jobs=-1,
                passthrough=False,
            )
        raise ValueError(f"Modèle {key} non supporté.")

    def _extract_feature_importances(self, pipeline):
        model = None
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            model = pipeline.named_steps["model"]
        elif hasattr(pipeline, "estimators_"):
            return None
        else:
            model = pipeline

        if hasattr(model, "feature_importances_"):
            return np.array(model.feature_importances_)
        if hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = coef[0]
            return np.abs(coef)
        return None

    def _align_feature_importances(self, feature_names, importances):
        names = np.asarray(feature_names).ravel()
        imps = np.asarray(importances).ravel()
        min_len = min(len(names), len(imps))
        return names[:min_len], imps[:min_len]

    def _get_sigma_predictions(self, model_key: str, pipeline) -> pd.Series:
        if not self.training_cfg.enable_sigma_model:
            residuals = self.y_train - pipeline.predict(self.X_train)
            sigma = max(float(residuals.std()), self.training_cfg.min_sigma)
            return pd.Series(
                np.full(len(self.X_test), sigma),
                index=self.X_test.index,
            )

        sigma_model = self._sigma_models.get(model_key)
        if sigma_model is None:
            residuals = np.abs(self.y_train - pipeline.predict(self.X_train))
            target = np.maximum(residuals, self.training_cfg.min_sigma)
            sigma_model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=300,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=self.training_cfg.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            sigma_model.fit(self.X_train, target)
            self._sigma_models[model_key] = sigma_model

        sigma_pred = sigma_model.predict(self.X_test)
        sigma_pred = np.maximum(sigma_pred, self.training_cfg.min_sigma)
        return pd.Series(sigma_pred, index=self.X_test.index)

    def _pivot_list(self) -> List[float]:
        if self.training_cfg.pivot_values:
            return self.training_cfg.pivot_values
        if self.training_cfg.pivot_value is not None:
            return [self.training_cfg.pivot_value]
        return []

    def _log_learning_curve(self, pipeline, model_key: str):
        scoring = (
            "roc_auc"
            if self.training_cfg.task_type == "classification"
            else "neg_root_mean_squared_error"
        )
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            self.X_train,
            self.y_train,
            cv=3,
            scoring=scoring,
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)
        ylabel = "ROC AUC"
        if self.training_cfg.task_type == "regression":
            train_mean = -train_mean
            test_mean = -test_mean
            ylabel = "RMSE"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(train_sizes, train_mean, label="train")
        ax.plot(train_sizes, test_mean, label="cv")
        ax.set_title(f"Learning curve - {model_key}")
        ax.set_xlabel("Training examples")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, f"plots/{model_key}_learning_curve.png")
        plt.close(fig)

    def _log_classification_plots(self, model_key: str, y_pred, y_proba):
        figs = {
            "confusion_matrix.png": plot_confusion_matrix(self.y_test, y_pred),
            "roc_curve.png": plot_roc_curve(self.y_test, y_proba),
            "calibration.png": plot_calibration_curve(self.y_test, y_proba),
        }
        for name, fig in figs.items():
            mlflow.log_figure(fig, f"plots/{model_key}_{name}")
            plt.close(fig)

    def _log_regression_plots(self, model_key: str, y_pred, sigma_series: pd.Series, pivots: List[float]):
        figs = {
            "pred_vs_actual.png": plot_pred_vs_actual(self.y_test, y_pred),
            "residual_hist.png": plot_residual_hist(self.y_test, y_pred),
        }
        for name, fig in figs.items():
            mlflow.log_figure(fig, f"plots/{model_key}_{name}")
            plt.close(fig)
        if not sigma_series.empty:
            self._log_gaussian_examples(model_key, y_pred, sigma_series, pivots)
            if pivots:
                self._log_pivot_probabilities(model_key, y_pred, sigma_series, pivots)

    def _log_gaussian_examples(self, model_key: str, y_pred, sigma_series: pd.Series, pivots: List[float]):
        if sigma_series.empty:
            return
        n_samples = min(3, len(y_pred))
        rng = np.random.default_rng(self.training_cfg.random_state)
        sample_idx = rng.choice(len(y_pred), size=n_samples, replace=False)
        pivot = pivots[0] if pivots else (self.training_cfg.pivot_value or float(self.y_train.mean()))
        for idx in sample_idx:
            fig = plot_gaussian_prediction(
                mean=float(y_pred.iloc[idx]),
                sigma=float(max(sigma_series.iloc[idx], self.training_cfg.min_sigma)),
                actual=float(self.y_test.iloc[idx]),
                pivot=pivot,
            )
            if fig:
                mlflow.log_figure(fig, f"plots/{model_key}_gaussian_{idx}.png")
                plt.close(fig)

    def _log_pivot_probabilities(self, model_key: str, y_pred, sigma_series: pd.Series, pivots: List[float]):
        probs_entries = []
        for pivot in pivots:
            prob_under = norm.cdf(
                pivot,
                loc=y_pred,
                scale=sigma_series.clip(lower=self.training_cfg.min_sigma),
            )
            tmp = pd.DataFrame(
                {
                    "prediction": y_pred,
                    "actual": self.y_test,
                    "sigma": sigma_series,
                    "pivot": pivot,
                    "prob_under": prob_under,
                }
            )
            tmp["prob_over"] = 1 - tmp["prob_under"]
            probs_entries.append(tmp)
        probs_df = pd.concat(probs_entries)
        summary = probs_df.groupby("pivot")[["prob_under", "prob_over"]].mean()
        tmp_dir = Path(tempfile.mkdtemp())
        summary_path = tmp_dir / f"{model_key}_pivot_summary.csv"
        summary.to_csv(summary_path)
        mlflow.log_artifact(summary_path, artifact_path=f"probabilities/{model_key}")

        sample = probs_df.sample(
            n=min(2000, len(probs_df)), random_state=self.training_cfg.random_state
        )
        sample_path = tmp_dir / f"{model_key}_pivot_samples.csv"
        sample.to_csv(sample_path, index=False)
        mlflow.log_artifact(
            sample_path, artifact_path=f"probabilities/{model_key}/samples"
        )

    def _fit_isotonic_calibrators(
        self,
        y_pred: pd.Series,
        sigma_series: pd.Series,
        pivots: List[float],
    ) -> tuple[Dict[str, Dict[str, List[float]]], Optional[pd.DataFrame]]:
        if not pivots:
            return {}, None
        calibrations: Dict[str, Dict[str, List[float]]] = {}
        analysis_rows: List[pd.DataFrame] = []
        sigma = sigma_series.clip(lower=self.training_cfg.min_sigma).to_numpy()
        preds = y_pred.to_numpy()
        actual = self.y_test.to_numpy()
        for pivot in pivots:
            actual_over = (actual > pivot).astype(int)
            if actual_over.min() == actual_over.max():
                continue
            prob_over = 1 - norm.cdf(pivot, loc=preds, scale=sigma)
            ir = IsotonicRegression(out_of_bounds="clip")
            try:
                ir.fit(prob_over, actual_over)
            except ValueError:
                continue
            calibrated = ir.predict(prob_over)
            calibrations[str(float(pivot))] = {
                "x": ir.X_thresholds_.tolist(),
                "y": ir.y_thresholds_.tolist(),
            }
            analysis_rows.append(
                pd.DataFrame(
                    {
                        "pivot": float(pivot),
                        "prob_raw": prob_over,
                        "prob_calibrated": calibrated,
                        "actual_over": actual_over,
                    }
                )
            )
        frame = None
        if analysis_rows:
            frame = pd.concat(analysis_rows, ignore_index=True)
        return calibrations, frame

    def _log_calibration_artifact(
        self,
        payload: Dict[str, Dict[str, List[float]]],
        frame: Optional[pd.DataFrame],
    ) -> None:
        if not payload:
            return
        mlflow.log_dict(payload, CALIBRATION_ARTIFACT_PATH)
        if frame is None or frame.empty:
            return
        summary = (
            frame.groupby("pivot")
            .agg(
                prob_raw_mean=("prob_raw", "mean"),
                prob_calibrated_mean=("prob_calibrated", "mean"),
                actual_mean=("actual_over", "mean"),
                count=("actual_over", "count"),
            )
            .reset_index()
        )
        tmp_dir = Path(tempfile.mkdtemp())
        raw_path = tmp_dir / "calibration_samples.csv"
        frame.to_csv(raw_path, index=False)
        mlflow.log_artifact(raw_path, artifact_path="calibration")
        summary_path = tmp_dir / "calibration_summary.csv"
        summary.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path, artifact_path="calibration")

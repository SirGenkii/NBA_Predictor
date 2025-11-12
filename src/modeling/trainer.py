from __future__ import annotations

import tempfile
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import nullcontext
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import CRPScore

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
            base.append("ngboost")
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

                mlflow.sklearn.log_model(pipeline, artifact_path="model")

            return metrics

    def _build_model(self, key: str, overrides: Optional[dict] = None) -> Pipeline:
        if self.training_cfg.task_type == "classification":
            pipeline = self._build_classification_model(key)
        else:
            pipeline = self._build_regression_model(key)
        if overrides:
            pipeline = pipeline.set_params(**self._normalize_overrides(pipeline, overrides))
        return pipeline

    @staticmethod
    def _normalize_overrides(pipeline: Pipeline, overrides: dict) -> dict:
        """Accept bare parameter names (e.g., `n_estimators`) and route them to the underlying model."""

        normalized = {}
        for key, value in overrides.items():
            if "__" in key:
                normalized[key] = value
                continue
            if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
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
            return Pipeline(
                steps=[
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
        if key == "xgb":
            return Pipeline(
                steps=[
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
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                    (
                        "model",
                        NGBRegressor(
                            Dist=Normal,
                            Score=CRPScore,
                            n_estimators=800,
                            learning_rate=0.03,
                            natural_gradient=True,
                            verbose=False,
                            random_state=self.training_cfg.random_state,
                        ),
                    ),
                ]
            )
        if key == "stacking":
            estimators = [
                ("lgbm", self._build_regression_model("lgbm")),
                ("xgb", self._build_regression_model("xgb")),
            ]
            final_estimator = Ridge()
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
            n=min(1000, len(probs_df)), random_state=self.training_cfg.random_state
        )
        sample_path = tmp_dir / f"{model_key}_pivot_samples.csv"
        sample.to_csv(sample_path, index=False)
        mlflow.log_artifact(
            sample_path, artifact_path=f"probabilities/{model_key}/samples"
        )

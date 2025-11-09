from __future__ import annotations

import tempfile
from typing import Dict, List

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import DatasetConfig, TrainingConfig
from .data import load_dataset, prepare_features, train_test_split_data
from .metrics import classification_metrics, confusion_matrix_values
from .plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)


class ModelTrainer:
    def __init__(self, dataset_cfg: DatasetConfig, training_cfg: TrainingConfig):
        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg

        mlflow.set_tracking_uri(training_cfg.tracking_uri)
        mlflow.set_experiment(training_cfg.experiment_name)

        self._dataset_path = None
        self._feature_names: List[str] = []
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
        return ["lgbm", "xgb", "stacking"]

    def train_models(self, models: List[str]) -> pd.DataFrame:
        results = []
        for model_key in models:
            if model_key not in self.available_models():
                raise ValueError(f"Modèle {model_key} non supporté.")
            metrics = self._train_single_model(model_key)
            results.append({"model": model_key, **metrics})
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_single_model(self, model_key: str) -> Dict[str, float]:

        pipeline = self._build_model(model_key)
        run_name = f"{model_key}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("dataset_path", self._dataset_path)
            mlflow.log_param("target", self.dataset_cfg.target)
            mlflow.log_params(
                {
                    "test_size": self.training_cfg.test_size,
                    "random_state": self.training_cfg.random_state,
                    "model_key": model_key,
                }
            )

            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            y_proba = pipeline.predict_proba(self.X_test)[:, 1]

            metrics = classification_metrics(self.y_test, y_pred, y_proba)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            cm_vals = confusion_matrix_values(self.y_test, y_pred)
            for k, v in cm_vals.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"cm_{k}", v)

            # Log prediction samples
            preds_df = pd.DataFrame(
                {
                    "y_true": self.y_test,
                    "y_pred": y_pred,
                    "proba": y_proba,
                },
                index=self.X_test.index,
            )
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                preds_df.to_csv(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, artifact_path="predictions")

            # Log plots
            figs = {
                "confusion_matrix.png": plot_confusion_matrix(self.y_test, y_pred),
                "roc_curve.png": plot_roc_curve(self.y_test, y_proba),
                "calibration.png": plot_calibration_curve(self.y_test, y_proba),
            }
            for name, fig in figs.items():
                mlflow.log_figure(fig, f"plots/{model_key}_{name}")
                plt.close(fig)

            # Feature importance
            importances = self._extract_feature_importances(pipeline)
            if importances is not None:
                fi_names, fi_values = self._align_feature_importances(
                    self._feature_names, importances
                )
                fig = plot_feature_importance(fi_names, fi_values)
                if fig:
                    mlflow.log_figure(fig, f"plots/{model_key}_feature_importance.png")
                    plt.close(fig)
                fi_df = pd.DataFrame(
                    {"feature": fi_names, "importance": fi_values}
                ).sort_values("importance", ascending=False)
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    fi_df.to_csv(tmp.name, index=False)
                    mlflow.log_artifact(tmp.name, artifact_path="feature_importance")

            # Log model artifact
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            return metrics

    def _build_model(self, key: str) -> Pipeline:
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
                ("lgbm", self._build_model("lgbm")),
                ("xgb", self._build_model("xgb")),
            ]
            final_estimator = LogisticRegression(
                max_iter=1000, random_state=self.training_cfg.random_state
            )
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                stack_method="predict_proba",
                n_jobs=-1,
                passthrough=False,
            )
            return stacking
        raise ValueError(f"Modèle {key} non supporté.")

    def _extract_feature_importances(self, pipeline):
        model = None
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            model = pipeline.named_steps["model"]
        elif hasattr(pipeline, "estimators_"):
            # Stacking: pas de feature importance globale
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
        if len(importances) == len(feature_names):
            return feature_names, importances
        min_len = min(len(importances), len(feature_names))
        aligned_names = feature_names[:min_len]
        aligned_importances = importances[:min_len]
        return aligned_names, aligned_importances

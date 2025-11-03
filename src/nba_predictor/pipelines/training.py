from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import json
import tempfile

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import structlog
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    RocCurveDisplay,
)

from nba_predictor import settings
from nba_predictor.datasets.training import build_training_dataset

logger = structlog.get_logger("nba_predictor.pipelines.training")

TARGET_COLUMN = "target_home_win"
DEFAULT_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "nba_predictor")
DEFAULT_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "nba_predictor")


@dataclass
class TrainConfig:
    ingest_ts: str | None = None
    dataset_path: Path | None = None
    experiment_name: str = DEFAULT_EXPERIMENT
    model_name: str = DEFAULT_MODEL_NAME
    feature_prefix_diff: str = "_diff"
    val_ratio: float = 0.15
    test_ratio: float = 0.15


def _set_mlflow_tracking() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = f"file://{Path.cwd() / 'mlruns'}"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _time_split(
    df: pd.DataFrame,
    *,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("game_date").reset_index(drop=True)
    n_rows = len(df_sorted)
    test_size = max(int(n_rows * test_ratio), 1)
    val_size = max(int(n_rows * val_ratio), 1)
    train_end = max(n_rows - test_size - val_size, 1)
    val_end = min(train_end + val_size, n_rows - test_size)

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    return train_df, val_df, test_df


def _select_feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.endswith(prefix)]


def train_model(config: TrainConfig) -> str:
    """
    Train a LightGBM classifier using the prepared training dataset and log artifacts to MLflow.
    Returns the MLflow run_id.
    """

    ingest_ts = config.ingest_ts
    dataset_path = config.dataset_path

    if dataset_path is None:
        dataset_path = build_training_dataset(ingest_ts=ingest_ts)
    dataset_path = Path(dataset_path)
    if ingest_ts is None:
        ingest_ts = dataset_path.stem.split("_")[-1]

    df = _load_dataset(dataset_path)
    feature_cols = _select_feature_columns(df, config.feature_prefix_diff)
    if not feature_cols:
        raise ValueError("No feature columns found with suffix '_diff'.")

    y = df[TARGET_COLUMN].astype(int)
    X = df[feature_cols]

    train_df, val_df, test_df = _time_split(
        df.assign(**{TARGET_COLUMN: y}),
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]

    _set_mlflow_tracking()
    mlflow.set_experiment(config.experiment_name)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    with mlflow.start_run(run_name=f"lightgbm_{ingest_ts}") as run:
        run_id = run.info.run_id
        mlflow.log_param("dataset_path", str(dataset_path))
        mlflow.log_param("features_count", len(feature_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("val_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="logloss",
            verbose=False,
        )

        def _log_split_metrics(split: str, labels: pd.Series, probs: np.ndarray) -> dict[str, float]:
            preds = (probs >= 0.5).astype(int)
            auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
            ll = log_loss(labels, probs, eps=1e-7)
            acc = accuracy_score(labels, preds)
            mlflow.log_metric(f"{split}_auc", auc)
            mlflow.log_metric(f"{split}_logloss", ll)
            mlflow.log_metric(f"{split}_accuracy", acc)
            return {"auc": auc, "logloss": ll, "accuracy": acc}

        prob_train = model.predict_proba(X_train)[:, 1]
        prob_val = model.predict_proba(X_val)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]

        metrics_train = _log_split_metrics("train", y_train, prob_train)
        metrics_val = _log_split_metrics("val", y_val, prob_val)
        metrics_test = _log_split_metrics("test", y_test, prob_test)

        feature_importances = pd.Series(model.booster_.feature_importance(), index=feature_cols).sort_values(
            ascending=False
        )
        importance_path = settings.data_paths.gold_training_sets / f"feature_importance_{run_id}.csv"
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        feature_importances.to_csv(importance_path, header=["importance"])
        mlflow.log_artifact(str(importance_path), artifact_path="artifacts")

        # Visual artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            def _save_plot(filename: str) -> Path:
                path = tmpdir_path / filename
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
                return path

            # Feature importances (top 30)
            plt.figure(figsize=(8, 10))
            feature_importances.head(30).sort_values().plot.barh()
            imp_plot = _save_plot("feature_importance_top30.png")
            mlflow.log_artifact(str(imp_plot), artifact_path="plots")

            # ROC curves
            plt.figure(figsize=(6, 6))
            RocCurveDisplay.from_predictions(y_train, prob_train, name="train")
            RocCurveDisplay.from_predictions(y_val, prob_val, name="val")
            RocCurveDisplay.from_predictions(y_test, prob_test, name="test")
            roc_path = _save_plot("roc_curves.png")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

            # Precision-Recall curves
            plt.figure(figsize=(6, 6))
            for split, labels, probs in [
                ("train", y_train, prob_train),
                ("val", y_val, prob_val),
                ("test", y_test, prob_test),
            ]:
                prec, rec, _ = precision_recall_curve(labels, probs)
                plt.plot(rec, prec, label=split)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            pr_path = _save_plot("precision_recall.png")
            mlflow.log_artifact(str(pr_path), artifact_path="plots")

            # Calibration curve
            plt.figure(figsize=(6, 6))
            CalibrationDisplay.from_predictions(y_val, prob_val, name="val")
            calib_path = _save_plot("calibration_val.png")
            mlflow.log_artifact(str(calib_path), artifact_path="plots")

            # Confusion matrix sur test
            cm = confusion_matrix(y_test, (prob_test >= 0.5).astype(int))
            plt.figure(figsize=(5, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, v, ha="center", va="center")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            cm_path = _save_plot("confusion_matrix_test.png")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            # Learning curves (LightGBM evals)
            evals = model.booster_.evals_result()
            if evals:
                plt.figure(figsize=(8, 5))
                for key, values in evals.items():
                    for metric_name, metric_values in values.items():
                        plt.plot(metric_values, label=f"{key}_{metric_name}")
                plt.legend()
                plt.xlabel("Iteration")
                plt.ylabel("Metric")
                learning_path = _save_plot("learning_curves.png")
                mlflow.log_artifact(str(learning_path), artifact_path="plots")

            # SHAP summary plot (optionnel, sur un échantillon)
            try:
                background = shap.sample(X_train, 200)
                explainer = shap.TreeExplainer(model.booster_)
                shap_values = explainer.shap_values(background)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, background, show=False, plot_type="bar")
                shap_bar_path = _save_plot("shap_summary_bar.png")
                mlflow.log_artifact(str(shap_bar_path), artifact_path="plots")

                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values, background, show=False)
                shap_summary_path = _save_plot("shap_summary.png")
                mlflow.log_artifact(str(shap_summary_path), artifact_path="plots")
            except Exception as exc:  # pragma: no cover
                logger.warning("shap_failed", error=str(exc))

            # Predictions export (val/test)
            val_preds = val_df.assign(
                prediction_proba=prob_val,
                prediction_label=(prob_val >= 0.5).astype(int),
            )
            test_preds = test_df.assign(
                prediction_proba=prob_test,
                prediction_label=(prob_test >= 0.5).astype(int),
            )
            val_path = tmpdir_path / "predictions_val.parquet"
            test_path = tmpdir_path / "predictions_test.parquet"
            val_preds.to_parquet(val_path, index=False)
            test_preds.to_parquet(test_path, index=False)
            mlflow.log_artifact(str(val_path), artifact_path="predictions")
            mlflow.log_artifact(str(test_path), artifact_path="predictions")

            # Metrics summary JSON
            metrics_summary = {
                "ingest_ts": ingest_ts,
                "dataset_path": str(dataset_path),
                "metrics": {
                    "train": metrics_train,
                    "val": metrics_val,
                    "test": metrics_test,
                },
            }
            json_path = tmpdir_path / "metrics_summary.json"
            json_path.write_text(json.dumps(metrics_summary, indent=2))
            mlflow.log_artifact(str(json_path), artifact_path="artifacts")

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.model_name,
        )

        logger.info(
            "train_model_complete",
            run_id=run_id,
            dataset=str(dataset_path),
            features=len(feature_cols),
        )
        return run_id


__all__ = ["TrainConfig", "train_model"]

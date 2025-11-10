from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    return fig


def plot_calibration_curve(y_true, y_proba, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(5, 4))
    CalibrationDisplay.from_predictions(y_true, y_proba, n_bins=n_bins, ax=ax)
    ax.set_title("Calibration curve")
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names: Sequence[str], importances, top_n: int = 30):
    if importances is None or feature_names is None:
        return None
    feature_array = np.asarray(feature_names)
    importance_array = np.asarray(importances)
    top_n = min(top_n, len(feature_array), len(importance_array))
    if top_n == 0:
        return None
    order = np.argsort(importance_array)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_array[order][::-1], importance_array[order][::-1])
    ax.set_title(f"Top {top_n} feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    return fig


def plot_residual_hist(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=30, alpha=0.7)
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual (actual - pred)")
    fig.tight_layout()
    return fig


def plot_gaussian_prediction(mean, sigma, actual, pivot=None):
    if sigma <= 0:
        return None
    span = max(30, 4 * sigma)
    xs = np.linspace(mean - span, mean + span, 400)
    pred_pdf = norm.pdf(xs, loc=mean, scale=sigma)
    actual_pdf = norm.pdf(xs, loc=actual, scale=2.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, pred_pdf, label="Predicted")
    ax.plot(xs, actual_pdf, label="Actual")
    ax.fill_between(xs, pred_pdf, alpha=0.2)
    if pivot is not None:
        ax.axvline(pivot, color="red", linestyle="--", label=f"Pivot {pivot:.1f}")
    ax.set_title("Predicted vs Actual distribution")
    ax.set_xlabel("Total points")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig

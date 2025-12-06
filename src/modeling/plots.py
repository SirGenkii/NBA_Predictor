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
    importance_array = np.asarray(importances).ravel()

    min_len = min(len(feature_array), len(importance_array))
    feature_array = feature_array[:min_len]
    importance_array = importance_array[:min_len]

    mask = np.isfinite(importance_array)
    feature_array = feature_array[mask]
    importance_array = importance_array[mask]
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


def plot_market_vs_model_curve(
    labels,
    market_probs,
    pred_mean: float,
    pred_sigma: float,
    actual: float,
    *,
    min_sigma: float = 1.0,
    title: str | None = None,
):
    """Overlay market over-prob curve, model-implied curve, and actual total."""
    labels_arr = np.asarray(labels, dtype=float)
    market_arr = np.asarray(market_probs, dtype=float)
    mask = np.isfinite(labels_arr) & np.isfinite(market_arr)
    labels_arr = labels_arr[mask]
    market_arr = market_arr[mask]
    if labels_arr.size == 0:
        return None
    order = np.argsort(labels_arr)
    labels_arr = labels_arr[order]
    market_arr = np.clip(market_arr[order], 0.0, 1.0)

    x_low = float(np.min(labels_arr))
    x_high = float(np.max(labels_arr))
    x_low = min(x_low, pred_mean, actual)
    x_high = max(x_high, pred_mean, actual)
    span = max(10.0, x_high - x_low)
    pad = span * 0.2
    xs = np.linspace(x_low - pad, x_high + pad, 400)

    sigma = max(pred_sigma, min_sigma)
    model_probs = 1.0 - norm.cdf(xs, loc=pred_mean, scale=sigma)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(labels_arr, market_arr, label="Market over prob", color="C0", marker="o")
    ax.plot(xs, model_probs, label="Model over prob", color="C1")
    ax.axvline(actual, color="k", linestyle="--", label=f"Actual {actual:.1f}")
    ax.axvline(pred_mean, color="C1", linestyle=":", label=f"Pred mean {pred_mean:.1f}")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Total points pivot")
    ax.set_ylabel("P(Over)")
    ax.set_title(title or "Market vs model curve")
    ax.legend()
    fig.tight_layout()
    return fig

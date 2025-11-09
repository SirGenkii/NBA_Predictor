from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
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


def plot_feature_importance(feature_names: Sequence[str], importances: np.ndarray, top_n: int = 30):
    if importances is None or feature_names is None:
        return None
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1])
    ax.set_title(f"Top {top_n} feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig

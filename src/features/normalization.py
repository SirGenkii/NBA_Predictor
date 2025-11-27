from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MATCHUP_WINDOWS


def apply_normalized_matchup_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Derive relative/normalized matchup metrics to help modeling stability."""

    df = match_df.copy()

    for window in MATCHUP_WINDOWS:
        _add_top_usage_ratios(df, window)
        _add_bench_relative_metrics(df, window)
        _add_seasonal_zscores(df, window, base_col=f"MATCH_IMPORTANCE_SCORE_{window}")
        _add_seasonal_zscores(df, window, base_col=f"MATCH_POINTS_VARIANCE_{window}")
        _add_seasonal_zscores(df, window, base_col=f"MATCH_PACE_VARIANCE_{window}")

    return df


def _add_top_usage_ratios(df: pd.DataFrame, window: int) -> None:
    loss_col = f"MATCH_TOP_USAGE_MISSING_{window}"
    expected_col = f"TOTAL_POINTS_EXPECTED_{window}"
    if loss_col not in df or expected_col not in df:
        return

    denom = df[expected_col].abs().clip(lower=1e-3)
    ratio = df[loss_col] / denom
    df[f"MATCH_TOP_USAGE_LOSS_RATIO_{window}"] = ratio
    df[f"MATCH_TOP_USAGE_LOSS_TANH_{window}"] = np.tanh(ratio)


def _add_bench_relative_metrics(df: pd.DataFrame, window: int) -> None:
    bench_col = f"MATCH_BENCH_USAGE_GAP_{window}"
    perf_col = f"MATCH_PLAYER_PERF_GAP_{window}"
    pace_col = f"MATCH_PACE_{window}"

    if bench_col in df and perf_col in df:
        denom = (df[bench_col].abs() + df[perf_col].abs()).clip(lower=1e-6)
        df[f"MATCH_BENCH_USAGE_RATIO_{window}"] = df[bench_col] / denom

    if bench_col in df and pace_col in df:
        df[f"MATCH_BENCH_PACE_ADJ_{window}"] = df[bench_col] * df[pace_col]


def _add_seasonal_zscores(df: pd.DataFrame, window: int, *, base_col: str) -> None:
    if "SEASON" not in df or base_col not in df:
        return
    grouped = df.groupby("SEASON")[base_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, 1.0)
    df[f"{base_col}_Z"] = (df[base_col] - mean) / std

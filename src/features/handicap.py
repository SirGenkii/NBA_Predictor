from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from src.config import (
    HANDICAP_APPLY_ISOTONE,
    HANDICAP_APPLY_OPTIMIZED_SIGMA,
    HANDICAP_ENABLE_SMOOTHING,
    HANDICAP_GRID_MAX,
    HANDICAP_GRID_MIN,
    HANDICAP_GRID_REL_RANGE,
    HANDICAP_GRID_STEP,
    HANDICAP_SIGMA_GRID_DEFAULT,
    HANDICAP_SIGMA_GRID_OPTIMIZED,
    HANDICAP_SLOPE_PRIOR_SCALE,
    HANDICAP_TOP_K,
)


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_label(value: float) -> str:
    rounded = round(float(value), 1)
    text = f"{rounded:.1f}"
    text = text.replace("-", "m").replace(".", "_")
    return text


@dataclass
class HandicapConfig:
    grid_min: float = HANDICAP_GRID_MIN
    grid_max: float = HANDICAP_GRID_MAX
    grid_step: float = HANDICAP_GRID_STEP
    grid_rel_range: float = HANDICAP_GRID_REL_RANGE
    sigma_grid_default: float = HANDICAP_SIGMA_GRID_DEFAULT
    sigma_grid_optimized: float = HANDICAP_SIGMA_GRID_OPTIMIZED
    enable_smoothing: bool = HANDICAP_ENABLE_SMOOTHING
    apply_isotone: bool = HANDICAP_APPLY_ISOTONE
    apply_optimized_sigma: bool = HANDICAP_APPLY_OPTIMIZED_SIGMA
    top_k: int = HANDICAP_TOP_K
    slope_prior_scale: float = HANDICAP_SLOPE_PRIOR_SCALE

    @property
    def sigma_grid(self) -> float:
        if self.apply_optimized_sigma:
            return float(self.sigma_grid_optimized)
        return float(self.sigma_grid_default)


DEFAULT_HANDICAP_CONFIG = HandicapConfig()


def apply_handicap_features(
    match_df: pd.DataFrame,
    cfg: Optional[HandicapConfig] = None,
) -> pd.DataFrame:
    """Parse handicap JSON odds and emit curve/grid features."""

    config = cfg or DEFAULT_HANDICAP_CONFIG
    df = match_df.copy()

    grid_abs = np.arange(config.grid_min, config.grid_max + 0.5 * config.grid_step, config.grid_step)
    grid_rel = np.arange(
        -config.grid_rel_range,
        config.grid_rel_range + 0.5 * config.grid_step,
        config.grid_step,
    )

    season_prior = _season_priors(df)
    features: List[Dict[str, float]] = []

    for idx, row in df.iterrows():
        raw = _select_handicap_value(row)
        season_key = str(row.get("SEASON", ""))
        prior = season_prior.get(season_key, season_prior.get("__global__"))
        row_feats = _compute_handicap_features(
            raw,
            grid_abs=grid_abs,
            grid_rel=grid_rel,
            prior_total=prior,
            config=config,
        )
        features.append(row_feats)

    feat_df = pd.DataFrame(features, index=df.index)
    df = pd.concat([df, feat_df], axis=1)
    df = df.drop(columns=["HOME_handicap", "AWAY_handicap", "handicap"], errors="ignore")
    return df


def _season_priors(df: pd.DataFrame) -> Dict[str, float]:
    if "POINT_TOTAL" in df.columns:
        global_mean = float(df["POINT_TOTAL"].mean())
        priors = df.groupby("SEASON")["POINT_TOTAL"].mean().to_dict()
        return {str(k): float(v) for k, v in priors.items() if pd.notna(v)} | {"__global__": global_mean}
    default = 0.5 * (HANDICAP_GRID_MIN + HANDICAP_GRID_MAX)
    return {"__global__": default}


def _select_handicap_value(row: pd.Series):
    for col in ("HOME_handicap", "AWAY_handicap", "handicap"):
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def _compute_handicap_features(
    raw_value,
    *,
    grid_abs: Sequence[float],
    grid_rel: Sequence[float],
    prior_total: Optional[float],
    config: HandicapConfig,
) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    base_missing = {
        "handicap_central_label": np.nan,
        "handicap_slope_at_central": np.nan,
        "handicap_range": np.nan,
        "handicap_pivot_count": 0,
        "handicap_avg_spacing": np.nan,
        "handicap_avg_vig": np.nan,
        "handicap_max_vig": np.nan,
        "handicap_entropy": np.nan,
        "handicap_iqr_width": np.nan,
        "handicap_asymmetry_near_central": np.nan,
        "handicap_pivots_min": np.nan,
        "handicap_pivots_max": np.nan,
    }
    feats.update(base_missing)
    flags = {
        "handicap_missing": 0,
        "handicap_parse_error": 0,
        "handicap_is_default_odds": 0,
        "handicap_is_sparse": 0,
        "handicap_is_dense": 0,
        "handicap_has_asymmetry": 0,
    }
    feats.update(flags)

    grid_values = {}
    for g in grid_abs:
        col = f"handicap_over_{_format_label(g)}"
        grid_values[col] = np.nan
        grid_values[f"{col}_missing"] = 1
    for offset in grid_rel:
        col = f"handicap_over_rel_{_format_label(offset)}"
        grid_values[col] = np.nan
        grid_values[f"{col}_missing"] = 1
    feats.update(grid_values)

    top_k = max(0, int(config.top_k))
    for k in range(top_k):
        feats[f"handicap_over_p{k}"] = np.nan
        feats[f"handicap_label_p{k}"] = np.nan
        feats[f"handicap_over_p{k}_missing"] = 1

    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        feats["handicap_missing"] = 1
        return feats

    parsed, parse_error = _parse_handicap_json(raw_value)
    if parse_error:
        feats["handicap_parse_error"] = 1
    if not parsed:
        feats["handicap_missing"] = 1
        return feats

    labels, probs_over, vigs, is_default = _extract_probs(parsed)
    feats["handicap_is_default_odds"] = int(is_default)
    if len(labels) == 0:
        feats["handicap_missing"] = 1
        return feats

    feats["handicap_pivot_count"] = len(labels)
    feats["handicap_is_sparse"] = int(len(labels) <= 2)
    feats["handicap_is_dense"] = int(len(labels) >= 5)
    feats["handicap_pivots_min"] = float(np.min(labels))
    feats["handicap_pivots_max"] = float(np.max(labels))
    feats["handicap_range"] = float(np.max(labels) - np.min(labels)) if len(labels) else np.nan
    feats["handicap_avg_spacing"] = float(np.diff(np.sort(labels)).mean()) if len(labels) > 1 else np.nan
    feats["handicap_avg_vig"] = float(np.mean(vigs)) if len(vigs) else np.nan
    feats["handicap_max_vig"] = float(np.max(vigs)) if len(vigs) else np.nan

    # Fit logistic curve to get central label and slope
    center, scale = _fit_logistic(labels, probs_over, slope_prior_scale=config.slope_prior_scale)
    feats["handicap_central_label"] = center
    if scale is not None and scale > 0:
        feats["handicap_slope_at_central"] = float(1.0 / (4.0 * scale))
        try:
            q75 = center + scale * logit(0.25)
            q25 = center + scale * logit(0.75)
            feats["handicap_iqr_width"] = float(q75 - q25)
        except Exception:
            feats["handicap_iqr_width"] = np.nan

    if probs_over:
        probs_clipped = np.clip(probs_over, 1e-6, 1 - 1e-6)
        entropy = -np.mean(probs_clipped * np.log(probs_clipped) + (1 - probs_clipped) * np.log(1 - probs_clipped))
        feats["handicap_entropy"] = float(entropy)

    asym = _nearest_asymmetry(labels, probs_over, center)
    feats["handicap_asymmetry_near_central"] = asym
    feats["handicap_has_asymmetry"] = int(abs(asym) > 0.02)

    _fill_top_k(features=feats, labels=labels, probs=probs_over, center=center, top_k=top_k)

    sigma_grid = config.sigma_grid
    abs_values, abs_missing = _smooth_to_grid(
        labels,
        probs_over,
        np.asarray(grid_abs),
        sigma_grid=sigma_grid,
        enable=config.enable_smoothing,
        apply_isotone=config.apply_isotone,
    )
    for val, miss, g in zip(abs_values, abs_missing, grid_abs):
        col = f"handicap_over_{_format_label(g)}"
        feats[col] = val
        feats[f"{col}_missing"] = int(miss)

    prior = prior_total if prior_total is not None and not pd.isna(prior_total) else feats["handicap_central_label"]
    if np.isnan(prior):
        prior = 0.5 * (config.grid_min + config.grid_max)
    rel_grid_actuals = np.asarray(grid_rel) + float(prior)
    rel_values, rel_missing = _smooth_to_grid(
        labels,
        probs_over,
        rel_grid_actuals,
        sigma_grid=sigma_grid,
        enable=config.enable_smoothing,
        apply_isotone=config.apply_isotone,
    )
    for val, miss, offset in zip(rel_values, rel_missing, grid_rel):
        col = f"handicap_over_rel_{_format_label(offset)}"
        feats[col] = val
        feats[f"{col}_missing"] = int(miss)

    return feats


def _parse_handicap_json(raw_value) -> Tuple[List[dict], bool]:
    parse_error = False
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except Exception:
            parse_error = True
            return [], parse_error
    elif isinstance(raw_value, dict):
        parsed = raw_value
    else:
        return [], True

    if not isinstance(parsed, dict):
        parse_error = True
        return [], parse_error
    entries = []
    for key, val in parsed.items():
        if isinstance(val, dict):
            label = _safe_float(val.get("label", key))
            odds_over = _safe_float(val.get("odds_over"))
            odds_under = _safe_float(val.get("odds_under"))
        else:
            label = _safe_float(key)
            odds_over = None
            odds_under = None
        if label is None:
            continue
        entries.append({"label": label, "odds_over": odds_over, "odds_under": odds_under})
    return entries, parse_error


def _extract_probs(entries: List[dict]) -> Tuple[List[float], List[float], List[float], bool]:
    labels: List[float] = []
    probs_over: List[float] = []
    vigs: List[float] = []

    default_odds = True
    for item in entries:
        label = _safe_float(item.get("label"))
        odds_over = _safe_float(item.get("odds_over"))
        odds_under = _safe_float(item.get("odds_under"))
        if label is None:
            continue
        if odds_over is None or odds_under is None:
            odds_over = odds_under = 1.73
        if not np.isfinite(odds_over) or not np.isfinite(odds_under):
            continue
        if abs(odds_over - 1.73) > 1e-6 or abs(odds_under - 1.73) > 1e-6:
            default_odds = False
        raw_over = 1.0 / odds_over
        raw_under = 1.0 / odds_under
        overround = raw_over + raw_under
        if overround <= 0:
            continue
        prob_over = raw_over / overround
        labels.append(label)
        probs_over.append(prob_over)
        vigs.append(overround - 1.0)

    order = np.argsort(labels)
    labels = [labels[i] for i in order]
    probs_over = [probs_over[i] for i in order]
    vigs = [vigs[i] for i in order]
    return labels, probs_over, vigs, default_odds


def _logistic_over(x, center, scale):
    z = (np.asarray(x) - center) / max(scale, 1e-6)
    return 1.0 - expit(z)


def _fit_logistic(labels: Sequence[float], probs: Sequence[float], *, slope_prior_scale: float) -> Tuple[float, Optional[float]]:
    if not labels:
        return np.nan, None
    x = np.asarray(labels, dtype=float)
    y = np.asarray(probs, dtype=float)
    if len(x) == 1:
        return float(x[0]), slope_prior_scale
    p0 = [float(np.median(x)), float(slope_prior_scale)]
    try:
        bounds = ([float(np.min(x)) - 10, 0.1], [float(np.max(x)) + 10, 50.0])
        params, _ = curve_fit(_logistic_over, x, y, p0=p0, bounds=bounds, maxfev=2000)
        center, scale = params
        return float(center), float(scale)
    except Exception:
        return float(p0[0]), float(p0[1])


def _nearest_asymmetry(labels: Sequence[float], probs: Sequence[float], center: float) -> float:
    if not labels or np.isnan(center):
        return np.nan
    idx = int(np.argmin(np.abs(np.asarray(labels) - center)))
    return float(probs[idx] - 0.5)


def _fill_top_k(
    *,
    features: Dict[str, float],
    labels: Sequence[float],
    probs: Sequence[float],
    center: float,
    top_k: int,
) -> None:
    if top_k <= 0 or not labels:
        return
    order = np.argsort(np.abs(np.asarray(labels) - center))
    for rank, idx in enumerate(order[:top_k]):
        features[f"handicap_over_p{rank}"] = float(probs[idx])
        features[f"handicap_label_p{rank}"] = float(labels[idx])
        features[f"handicap_over_p{rank}_missing"] = 0


def _smooth_to_grid(
    labels: Sequence[float],
    probs: Sequence[float],
    grid: np.ndarray,
    *,
    sigma_grid: float,
    enable: bool,
    apply_isotone: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.full(len(grid), np.nan)
    missing = np.ones(len(grid), dtype=int)
    if not labels:
        return values, missing
    labels_arr = np.asarray(labels, dtype=float)
    probs_arr = np.asarray(probs, dtype=float)

    if enable and sigma_grid > 0:
        for i, g in enumerate(grid):
            diff = labels_arr - g
            mask = np.abs(diff) <= 4.0
            if not np.any(mask):
                continue
            weights = np.exp(-(diff[mask] ** 2) / (2 * sigma_grid * sigma_grid))
            if weights.sum() <= 0:
                continue
            values[i] = float(np.average(probs_arr[mask], weights=weights))
            missing[i] = 0
        if apply_isotone:
            finite_mask = np.isfinite(values)
            if finite_mask.sum() >= 2:
                iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
                values[finite_mask] = iso.fit_transform(grid[finite_mask], values[finite_mask])
    else:
        for i, g in enumerate(grid):
            exact_mask = np.isclose(labels_arr, g, atol=1e-6)
            if np.any(exact_mask):
                values[i] = float(np.mean(probs_arr[exact_mask]))
                missing[i] = 0

    return values, missing

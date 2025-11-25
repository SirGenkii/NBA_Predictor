from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Iterable, List, Optional, Sequence

from src.config import (
    NBA_MASS_EDGE_THRESHOLD,
    NBA_MASS_KELLY_SCALING,
    NBA_MASS_SAFE_COVERAGE,
    NBA_MASS_UNCERTAINTY_SCALE,
)
from src.nba_mass_prediction.betting import kelly_fraction
from src.nba_mass_prediction.schemas import (
    KellyBreakdown,
    MatchStrategyResult,
    PivotEvaluation,
    PivotSideEvaluation,
    PredictionBundle,
)


def _normal_cdf(x: float, mean: float, sigma: float) -> float:
    if sigma <= 0:
        sigma = 1e-6
    z = (x - mean) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def _lookup_calibrated_probability(prob_map: dict, pivot: float) -> Optional[float]:
    if not prob_map:
        return None
    if pivot in prob_map:
        return float(prob_map[pivot])
    pivot_key = str(float(pivot))
    if pivot_key in prob_map:
        return float(prob_map[pivot_key])
    # try rounding issues
    for key in (f"{pivot:.1f}", f"{pivot:.2f}"):
        if key in prob_map:
            return float(prob_map[key])
    return None


@dataclass(frozen=True)
class StrategyConfig:
    edge_threshold: float = NBA_MASS_EDGE_THRESHOLD
    watch_threshold: float = NBA_MASS_EDGE_THRESHOLD / 2
    safe_edge_threshold: float = NBA_MASS_EDGE_THRESHOLD / 2
    safe_coverage_threshold: float = NBA_MASS_SAFE_COVERAGE
    min_probability: float = 0.5
    safe_confidence_threshold: float = 0.2
    kelly_scales: Sequence[float] = (NBA_MASS_KELLY_SCALING,)


def _validate_decimal_odds(value: float) -> float:
    if value <= 1.0:
        raise ValueError(f"Cote décimale invalide: {value}")
    return value


def _format_side_label(label: Optional[str], side: str, pivot: float) -> str:
    """Return a side-consistent label (avoid 'Moins' with an over pick)."""
    if label:
        lower = label.lower()
        if side == "over" and ("over" in lower or "plus" in lower):
            return label
        if side == "under" and ("under" in lower or "moins" in lower):
            return label
    prefix = "Over" if side == "over" else "Under"
    if pivot.is_integer():
        return f"{prefix} {int(pivot)}"
    return f"{prefix} {pivot:.1f}"


def _build_side_evaluation(
    *,
    side: str,
    probability: float,
    odds: float,
    kelly_scales: Sequence[float],
) -> PivotSideEvaluation:
    _validate_decimal_odds(odds)
    probability = max(0.0, min(1.0, probability))
    fair_odds = 1.0 / probability if probability > 0 else float("inf")
    edge = probability * odds - 1.0
    kelly_full = kelly_fraction(probability, odds)
    scaled = {f"{scale:.2f}x": max(0.0, kelly_full * scale) for scale in kelly_scales}
    return PivotSideEvaluation(
        side=side,
        probability=probability,
        odds=odds,
        fair_odds=fair_odds,
        edge=edge,
        kelly=KellyBreakdown(full=kelly_full, scaled=scaled),
    )


def evaluate_match_predictions(
    prediction: PredictionBundle,
    *,
    config: Optional[StrategyConfig] = None,
) -> MatchStrategyResult:
    cfg = config or StrategyConfig()
    raw_entries: List[dict] = []

    probability_map = prediction.pivot_probabilities or {}

    for market in prediction.match.markets:
        mean = prediction.mean_total + prediction.model_bias
        sigma = prediction.sigma * (1 + prediction.model_uncertainty / NBA_MASS_UNCERTAINTY_SCALE)
        prob_over = _lookup_calibrated_probability(probability_map, market.pivot)
        if prob_over is None:
            prob_over = 1.0 - _normal_cdf(market.pivot, mean, sigma)
        prob_under = 1.0 - prob_over

        over_eval = _build_side_evaluation(
            side="over",
            probability=prob_over,
            odds=market.over_odds,
            kelly_scales=cfg.kelly_scales,
        )
        under_eval = _build_side_evaluation(
            side="under",
            probability=prob_under,
            odds=market.under_odds,
            kelly_scales=cfg.kelly_scales,
        )

        raw_entries.append(
            {
                "pivot": market.pivot,
                "label": market.label,
                "bookmaker": market.bookmaker,
                "mean": mean,
                "sigma": sigma,
                "pivot_diff": abs(market.pivot - mean),
                "over": over_eval,
                "under": under_eval,
                "max_edge": max(over_eval.edge, under_eval.edge),
            }
        )

    best_over_idx: Optional[int] = None
    best_under_idx: Optional[int] = None
    best_over_edge = cfg.edge_threshold
    best_under_edge = cfg.edge_threshold

    for idx, entry in enumerate(raw_entries):
        if entry["over"].edge >= best_over_edge and entry["over"].probability >= cfg.min_probability:
            best_over_edge = entry["over"].edge
            best_over_idx = idx
        if entry["under"].edge >= best_under_edge and entry["under"].probability >= cfg.min_probability:
            best_under_edge = entry["under"].edge
            best_under_idx = idx

    evaluations: List[PivotEvaluation] = []
    for idx, entry in enumerate(raw_entries):
        recommendation = "skip"
        if idx == best_over_idx:
            recommendation = "bet_over"
        elif idx == best_under_idx:
            recommendation = "bet_under"
        elif entry["max_edge"] >= cfg.watch_threshold:
            recommendation = "watch"

        evaluations.append(
            PivotEvaluation(
                pivot=entry["pivot"],
                label=entry["label"],
                bookmaker=entry["bookmaker"],
                mean_total=entry["mean"],
                sigma=entry["sigma"],
                over=entry["over"],
                under=entry["under"],
                recommendation=recommendation,
            )
        )

    def _summary_entry(eval_: PivotEvaluation, side: str) -> dict:
        pick = eval_.over if side == "over" else eval_.under
        return {
            "pivot": eval_.pivot,
            "label": _format_side_label(eval_.label, side, eval_.pivot),
            "bookmaker": eval_.bookmaker,
            "side": side,
            "edge": pick.edge,
            "probability": pick.probability,
            "kelly": pick.kelly.full,
            "odds": pick.odds,
        }

    recommended_summary = [
        _summary_entry(eval_, eval_.recommendation.replace("bet_", ""))
        for eval_ in evaluations
        if eval_.recommendation.startswith("bet_")
    ]

    safe_candidate: Optional[tuple] = None  # (idx, side, confidence)
    safe_rank: Optional[tuple] = None
    for idx, entry in enumerate(raw_entries):
        for side in ("over", "under"):
            pick_eval = entry[side]
            confidence = abs(pick_eval.probability - 0.5)
            if pick_eval.edge < cfg.safe_edge_threshold or confidence < cfg.safe_confidence_threshold:
                continue
            score = (-confidence, -pick_eval.edge, entry["pivot_diff"])
            if safe_rank is None or score < safe_rank:
                safe_rank = score
                safe_candidate = (idx, side, confidence)

    if safe_candidate:
        idx, side, confidence = safe_candidate
        safe_entry = _summary_entry(evaluations[idx], side)
        safe_entry["confidence"] = confidence
        safe_summary = {"pick": safe_entry}
    else:
        safe_summary = {
            "pick": None,
            "reason": (
                "Aucun pari ne dépasse les seuils "
                f"edge ≥ {cfg.safe_edge_threshold:.3f} et confiance ≥ {cfg.safe_confidence_threshold:.2f}."
            ),
        }

    summary = {
        "recommended": recommended_summary,
        "safe": safe_summary,
    }

    return MatchStrategyResult(
        prediction=prediction,
        evaluations=tuple(evaluations),
        summary=summary,
    )


__all__ = ["StrategyConfig", "evaluate_match_predictions"]

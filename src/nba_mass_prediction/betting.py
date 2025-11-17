from __future__ import annotations

import math
from typing import NamedTuple, Optional


class KellyRecommendation(NamedTuple):
    side: str
    fraction: float
    probability: float
    odds: float
    edge: float


def _safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def kelly_fraction(probability: float, odds: float) -> float:
    if odds <= 1:
        return 0.0
    if probability <= 0:
        return 0.0
    if probability >= 1:
        probability = 1.0

    net_odds = odds - 1.0
    if net_odds <= 0:
        return 0.0

    fraction = (probability * odds - 1.0) / net_odds
    return max(0.0, fraction)


def recommend_bet(
    prob_player1: object,
    prob_player2: object,
    odds_player1: object,
    odds_player2: object,
) -> Optional[KellyRecommendation]:
    candidates: list[KellyRecommendation] = []

    prob1 = _safe_float(prob_player1)
    odds1 = _safe_float(odds_player1)
    if prob1 is not None and odds1 is not None:
        fraction1 = kelly_fraction(prob1, odds1)
        if fraction1 > 0:
            edge1 = prob1 * odds1 - 1.0
            candidates.append(
                KellyRecommendation(
                    side="player1",
                    fraction=fraction1,
                    probability=prob1,
                    odds=odds1,
                    edge=edge1,
                )
            )

    prob2 = _safe_float(prob_player2)
    odds2 = _safe_float(odds_player2)
    if prob2 is not None and odds2 is not None:
        fraction2 = kelly_fraction(prob2, odds2)
        if fraction2 > 0:
            edge2 = prob2 * odds2 - 1.0
            candidates.append(
                KellyRecommendation(
                    side="player2",
                    fraction=fraction2,
                    probability=prob2,
                    odds=odds2,
                    edge=edge2,
                )
            )

    if not candidates:
        return None

    return max(candidates, key=lambda item: item.fraction)


__all__ = ["kelly_fraction", "recommend_bet", "KellyRecommendation"]


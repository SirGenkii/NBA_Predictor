from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PivotMarket:
    """Describe a bookmaker pivot line (total points over/under)."""

    pivot: float
    over_odds: float
    under_odds: float
    label: Optional[str] = None
    bookmaker: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchPayload:
    """Raw payload emitted by adapters before ID resolution."""

    home_team: str
    away_team: str
    match_date: date
    markets: Tuple[PivotMarket, ...]
    match_time: Optional[str] = None
    screenshot_path: Optional[Path] = None
    source: Optional[str] = None  # e.g., "ocr", "api"
    raw_payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TeamIdentity:
    team_id: int
    canonical_name: str
    aliases: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResolvedMatch:
    """Match after resolving team names to IDs."""

    home_team: TeamIdentity
    away_team: TeamIdentity
    input_home_team: str
    input_away_team: str
    match_date: date
    markets: Tuple[PivotMarket, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    screenshot_path: Optional[Path] = None
    raw_payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionBundle:
    """Prediction result coming from the engine for a given match."""

    match: ResolvedMatch
    mean_total: float
    sigma: float
    pivot_probabilities: Mapping[float, float]  # probability of over for each pivot
    model_path: Optional[str]
    model_bias: float
    model_uncertainty: float


@dataclass(frozen=True)
class KellyBreakdown:
    full: float
    scaled: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PivotSideEvaluation:
    side: str  # "over" or "under"
    probability: float
    odds: float
    fair_odds: float
    edge: float
    kelly: KellyBreakdown


@dataclass(frozen=True)
class PivotEvaluation:
    pivot: float
    label: Optional[str]
    bookmaker: Optional[str]
    mean_total: float
    sigma: float
    over: PivotSideEvaluation
    under: PivotSideEvaluation
    recommendation: str  # "bet_over", "bet_under", "watch", "skip"


@dataclass(frozen=True)
class MatchStrategyResult:
    prediction: PredictionBundle
    evaluations: Tuple[PivotEvaluation, ...]
    summary: Mapping[str, object] = field(default_factory=dict)

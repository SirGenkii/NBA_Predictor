from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PredictionRequest:
    """Describe a future match to score."""

    home_team_id: int
    away_team_id: int
    match_date: str  # YYYY-MM-DD

    def datetime(self) -> datetime:
        return datetime.fromisoformat(self.match_date)

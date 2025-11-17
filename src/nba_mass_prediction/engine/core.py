from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.feast.pipeline import (
    FEATURE_STORE_PATH,
    refresh_and_build,
    run_feast_apply,
    run_feast_materialize,
)
from src.nba_mass_prediction.schemas import PredictionBundle, ResolvedMatch
from src.prediction.point_total import PredictionOutput, infer_season, run_prediction_pipeline
from src.prediction.schemas import PredictionRequest


class PredictionEngineError(RuntimeError):
    """Raised when the prediction pipeline fails."""


@dataclass
class PredictionEngine:
    auto_refresh: bool = True
    feast_targets: Sequence[str] = ("POINT_TOTAL",)
    materialize_days: int = 30
    logger: logging.Logger = logging.getLogger("nba_mass_prediction.engine")
    _last_refresh_ts: Optional[datetime] = None

    def refresh(self, seasons: Iterable[str]) -> None:
        season_list = sorted({season for season in seasons if season})
        if not season_list:
            return
        try:
            refresh_and_build(
                seasons=season_list,
                targets=self.feast_targets,
                build_gold=True,
            )
            run_feast_apply(feature_store=FEATURE_STORE_PATH)
            since = datetime.now(timezone.utc) - timedelta(days=self.materialize_days)
            run_feast_materialize(
                since=since,
                feature_store=FEATURE_STORE_PATH,
                incremental=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise PredictionEngineError(f"Feast refresh failed: {exc}") from exc
        self._last_refresh_ts = datetime.now()
        self.logger.info("Prediction engine refreshed for seasons: %s", ", ".join(sorted(seasons)))

    def score_batch(
        self,
        matches: Sequence[ResolvedMatch],
        *,
        extra_pivots: Optional[Iterable[float]] = None,
        refresh_data: bool = False,
    ) -> List[PredictionBundle]:
        if not matches:
            return []

        requests: List[PredictionRequest] = []
        request_keys: List[Tuple[int, int, str]] = []
        seasons: set[str] = set()
        all_pivots: set[float] = set(extra_pivots or [])

        for match in matches:
            match_date_str = match.match_date.isoformat()
            req = PredictionRequest(
                home_team_id=match.home_team.team_id,
                away_team_id=match.away_team.team_id,
                match_date=match_date_str,
            )
            requests.append(req)
            request_keys.append((req.home_team_id, req.away_team_id, req.match_date))
            seasons.add(infer_season(match_date_str))
            for market in match.markets:
                all_pivots.add(float(market.pivot))

        if self.auto_refresh and self._last_refresh_ts is None:
            self.refresh(seasons)
        elif refresh_data:
            self.refresh(seasons)

        pivot_list = sorted(all_pivots)

        try:
            outputs = run_prediction_pipeline(
                requests,
                pivots=pivot_list,
                refresh_data=False,  # refresh is handled explicitly via self.refresh
            )
        except Exception as exc:  # noqa: BLE001
            raise PredictionEngineError(f"Erreur pipeline point_total: {exc}") from exc

        mapping: Dict[Tuple[int, int, str], PredictionOutput] = {}
        for req, out in zip(requests, outputs):
            mapping[(req.home_team_id, req.away_team_id, req.match_date)] = out

        bundles: List[PredictionBundle] = []
        for key, match in zip(request_keys, matches):
            output = mapping.get(key)
            if output is None:
                raise PredictionEngineError(f"Aucune prédiction retournée pour {key}")
            bundle = PredictionBundle(
                match=match,
                mean_total=float(output.prediction),
                sigma=float(output.sigma),
                pivot_probabilities=dict(output.probabilities),
                model_path=output.model_path,
                model_bias=output.model_bias,
                model_uncertainty=output.model_uncertainty,
            )
            bundles.append(bundle)
        return bundles


__all__ = ["PredictionEngine", "PredictionEngineError"]

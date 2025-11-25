from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from src.config import (
    NBA_MASS_EDGE_THRESHOLD,
    NBA_MASS_KELLY_SCALING,
    NBA_MASS_PAYLOAD_DIR,
    NBA_MASS_PREDICTIONS_CSV,
    NBA_MASS_RUNS_DIR,
)
from src.nba_mass_prediction.adapters.ocr_adapter import (
    OCRExtractionError,
    extract_matches_from_image,
)
from src.nba_mass_prediction.engine import PredictionEngine
from src.nba_mass_prediction.resolver import MatchResolver, TeamResolver, TeamResolverError
from src.nba_mass_prediction.schemas import MatchPayload, MatchStrategyResult, PivotMarket
from src.nba_mass_prediction.strategy import StrategyConfig, evaluate_match_predictions

LOGGER = logging.getLogger("nba_mass_prediction.runner")


def _make_run_id() -> str:
    return datetime.now().strftime("nba-run-%Y%m%d-%H%M%S-%f")


def _market_to_dict(market: PivotMarket) -> dict:
    return {
        "pivot": market.pivot,
        "over_odds": market.over_odds,
        "under_odds": market.under_odds,
        "label": market.label,
        "bookmaker": market.bookmaker,
        "metadata": dict(market.metadata),
    }


def _evaluation_to_dict(result: MatchStrategyResult) -> dict:
    evaluations = []
    for evaluation in result.evaluations:
        evaluations.append(
            {
                "pivot": evaluation.pivot,
                "label": evaluation.label,
                "bookmaker": evaluation.bookmaker,
                "mean_total": evaluation.mean_total,
                "sigma": evaluation.sigma,
                "recommendation": evaluation.recommendation,
                "over": {
                    "probability": evaluation.over.probability,
                    "odds": evaluation.over.odds,
                    "fair_odds": evaluation.over.fair_odds,
                    "edge": evaluation.over.edge,
                    "kelly": {
                        "full": evaluation.over.kelly.full,
                        "scaled": dict(evaluation.over.kelly.scaled),
                    },
                },
                "under": {
                    "probability": evaluation.under.probability,
                    "odds": evaluation.under.odds,
                    "fair_odds": evaluation.under.fair_odds,
                    "edge": evaluation.under.edge,
                    "kelly": {
                        "full": evaluation.under.kelly.full,
                        "scaled": dict(evaluation.under.kelly.scaled),
                    },
                },
            }
        )
    return {
        "match": {
            "home": {
                "team_id": result.prediction.match.home_team.team_id,
                "name": result.prediction.match.home_team.canonical_name,
                "aliases": list(result.prediction.match.home_team.aliases),
                "input_name": result.prediction.match.input_home_team,
            },
            "away": {
                "team_id": result.prediction.match.away_team.team_id,
                "name": result.prediction.match.away_team.canonical_name,
                "aliases": list(result.prediction.match.away_team.aliases),
                "input_name": result.prediction.match.input_away_team,
            },
            "match_date": result.prediction.match.match_date.isoformat(),
            "metadata": dict(result.prediction.match.metadata),
            "markets": [_market_to_dict(market) for market in result.prediction.match.markets],
            "screenshot_path": str(result.prediction.match.screenshot_path) if result.prediction.match.screenshot_path else None,
            "raw_payload": dict(result.prediction.match.raw_payload),
        },
        "prediction": {
            "mean_total": result.prediction.mean_total,
            "sigma": result.prediction.sigma,
            "pivot_probabilities": dict(result.prediction.pivot_probabilities),
            "model_path": result.prediction.model_path,
            "model_bias": result.prediction.model_bias,
            "model_uncertainty": result.prediction.model_uncertainty,
        },
        "evaluations": evaluations,
        "summary": dict(result.summary),
    }


def _ensure_match_resolver(resolver: Optional[MatchResolver]) -> MatchResolver:
    if resolver is not None:
        return resolver
    return MatchResolver(TeamResolver())


def process_payloads(
    engine: PredictionEngine,
    payloads: Sequence[MatchPayload],
    *,
    resolver: Optional[MatchResolver] = None,
    output_dir: Path | str = NBA_MASS_RUNS_DIR,
    strategy_config: Optional[StrategyConfig] = None,
    run_id: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> dict:
    """Resolve payloads, run predictions, compute strategy, and persist artifacts."""
    if not payloads:
        raise ValueError("Aucun payload fourni.")
    match_resolver = _ensure_match_resolver(resolver)

    resolved_matches = []
    for payload in payloads:
        resolved_matches.append(match_resolver.resolve(payload))

    bundles = engine.score_batch(resolved_matches)
    cfg = strategy_config or StrategyConfig(edge_threshold=NBA_MASS_EDGE_THRESHOLD, kelly_scales=(NBA_MASS_KELLY_SCALING,))

    strategy_results = [evaluate_match_predictions(bundle, config=cfg) for bundle in bundles]
    model_path = next((bundle.model_path for bundle in bundles if bundle.model_path), None)
    model_label = _format_model_label(model_path)
    LOGGER.info("Modèle point_total utilisé pour ce run: %s", model_label)

    artifact = {
        "run_id": run_id or _make_run_id(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "matches": [_evaluation_to_dict(result) for result in strategy_results],
        "metadata": dict(extra_metadata or {}),
        "model_path": model_path,
        "model_label": model_label,
    }

    output_path = _write_artifact(artifact, output_dir)
    _append_prediction_log(artifact["run_id"], strategy_results, NBA_MASS_PREDICTIONS_CSV)
    return {
        "run_id": artifact["run_id"],
        "artifact": artifact,
        "output_path": output_path,
        "results": strategy_results,
        "model_label": model_label,
    }


def _write_artifact(artifact: dict, output_dir: Path | str) -> Path:
    target_dir = Path(output_dir)
    dated_dir = target_dir / datetime.now().date().isoformat()
    dated_dir.mkdir(parents=True, exist_ok=True)
    path = dated_dir / f"{artifact['run_id']}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    return path


def _format_model_label(model_path: str | None) -> str:
    if not model_path:
        return "stacking local (fallback)"
    if str(model_path).startswith("models:/"):
        return f"registry {model_path}"
    return f"mlflow run {model_path}"


def _append_prediction_log(run_id: str, results: List[MatchStrategyResult], csv_path: Path | str) -> None:
    if not results:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    rows: List[dict] = []

    for result in results:
        match = result.prediction.match
        safe_entries = result.summary.get("safe") or {}
        safe_pairs: set[tuple] = set()
        safe_pick_entry = safe_entries.get("pick") if isinstance(safe_entries, dict) else None
        safe_confidence = None
        if isinstance(safe_pick_entry, dict):
            safe_pairs.add((safe_pick_entry.get("side"), safe_pick_entry.get("pivot")))
            safe_confidence = safe_pick_entry.get("confidence")
        safe_reason = safe_entries.get("reason") if isinstance(safe_entries, dict) else None
        pair_coverage = None
        pair_edge = None

        screenshot = match.screenshot_path.name if match.screenshot_path else None
        for evaluation in result.evaluations:
            for side_name, side_eval in (("over", evaluation.over), ("under", evaluation.under)):
                rows.append(
                    {
                        "run_id": run_id,
                        "screenshot": screenshot,
                        "match_date": match.match_date.isoformat(),
                        "home_team_id": match.home_team.team_id,
                        "home_team_name": match.home_team.canonical_name,
                        "away_team_id": match.away_team.team_id,
                        "away_team_name": match.away_team.canonical_name,
                        "input_home_name": match.input_home_team,
                        "input_away_name": match.input_away_team,
                        "bookmaker": evaluation.bookmaker,
                        "pivot": evaluation.pivot,
                        "mean_total": evaluation.mean_total,
                        "sigma": evaluation.sigma,
                        "side": side_name,
                        "probability": side_eval.probability,
                        "odds": side_eval.odds,
                        "fair_odds": side_eval.fair_odds,
                        "edge": side_eval.edge,
                        "kelly_full": side_eval.kelly.full,
                        "recommendation": evaluation.recommendation,
                        "model_path": result.prediction.model_path,
                        "safe_pick": int((side_name, evaluation.pivot) in safe_pairs),
                        "safe_pick_confidence": safe_confidence if (side_name, evaluation.pivot) in safe_pairs else None,
                        "safe_pair_coverage": pair_coverage,
                        "safe_pair_edge_sum": pair_edge,
                        "safe_reason": safe_reason,
                        "model_bias": result.prediction.model_bias,
                        "model_uncertainty": result.prediction.model_uncertainty,
                    }
                )

    if not rows:
        return

    fieldnames = [
        "run_id",
        "screenshot",
        "match_date",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "input_home_name",
        "input_away_name",
        "bookmaker",
        "pivot",
        "mean_total",
        "sigma",
        "side",
        "probability",
        "odds",
        "fair_odds",
        "edge",
        "kelly_full",
        "recommendation",
        "model_path",
        "safe_pick",
        "safe_pick_confidence",
        "safe_pair_coverage",
        "safe_pair_edge_sum",
        "safe_reason",
        "model_bias",
        "model_uncertainty",
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def process_screenshot(
    engine: PredictionEngine,
    screenshot_path: Path | str,
    *,
    resolver: Optional[MatchResolver] = None,
    output_dir: Path | str = NBA_MASS_RUNS_DIR,
) -> dict:
    """Run OCR on a screenshot then feed the payloads into the prediction pipeline."""
    path = Path(screenshot_path)
    try:
        payloads, ocr_payload = extract_matches_from_image(path)
    except OCRExtractionError as exc:
        _write_error_payload(path, str(exc))
        raise

    return process_payloads(
        engine,
        payloads,
        resolver=resolver,
        output_dir=output_dir,
        extra_metadata={"ocr_payload": ocr_payload, "screenshot": str(path)},
    )


def _write_error_payload(path: Path, error: str) -> None:
    payload_dir = Path(NBA_MASS_PAYLOAD_DIR) / "errors" / datetime.now().date().isoformat()
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "screenshot": str(path),
        "error": error,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    filename = payload_dir / f"{path.stem}_error.json"
    filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


__all__ = ["process_payloads", "process_screenshot"]

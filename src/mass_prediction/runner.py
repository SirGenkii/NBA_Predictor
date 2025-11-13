from __future__ import annotations

import json
import multiprocessing as mp
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import (
    MASS_PREDICTION_EXPORT_PATTERN,
    MASS_PREDICTION_FEATURE_EXPORT_DIR,
    WEBSITE_MASS_PREDICTION_FEATURE_EXPORT_DIR,
    DISCORD_SUBSCRIBER_PREDICTION_BOT_TOKEN,
    DISCORD_SUBSCRIBER_PREDICTION_CHANNEL_ID,
    DISCORD_SUBSCRIBER_ROLE_MENTION,
)
from src.mass_prediction.betting import recommend_bet
from src.mass_prediction.notifications import (
    send_discord_debug_summary,
    send_discord_free_pick,
    send_discord_pending_matches,
    send_discord_subscriber_summary,
    send_discord_mention,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.mass_prediction.ocr import OCRExtractionError, OCRMatch, extract_matches_from_image
from src.mass_prediction.predictor import MassPredictionEngine, MatchInput, MatchPrediction, PredictionError

from src.odds_api.db import SessionLocal
from src.odds_api.models import Event, OddsSnapshot, PredictionJob
from src.odds_ingest import persist
from src.odds_ingest.mass_predictions import PredictionRecord


@dataclass
class ProcessedMatch:
    match_index: int
    ocr: OCRMatch
    prediction: MatchPrediction


@dataclass
class ScreenshotProcessingResult:
    run_id: str
    screenshot_path: Path
    ocr_payload: Dict[str, object]
    matches: List[ProcessedMatch]
    errors: List[str]


@contextmanager
def _session_scope() -> Iterator[Session]:
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _iter_screenshots(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        yield path


def _move_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination = dst_dir / src.name
    counter = 1
    while destination.exists():
        destination = dst_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), str(destination))
    return destination


def _store_payload(result: ScreenshotProcessingResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / f"{result.run_id}.json"
    data = {
        "run_id": result.run_id,
        "screenshot": str(result.screenshot_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ocr_payload": result.ocr_payload,
        "errors": result.errors,
    }
    with payload_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _should_process(path: Path, settle_seconds: float) -> bool:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return (time.time() - mtime) >= settle_seconds


def _process_screenshots(
    *,
    engine: MassPredictionEngine,
    screenshots_dir: Path,
    processed_dir: Path,
    error_dir: Path,
    payload_dir: Optional[Path],
    settle_seconds: float,
    max_workers: int,
    logger: Optional[logging.Logger] = None,
) -> bool:
    log = logger or logging.getLogger(__name__)

    screenshots_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    if payload_dir is not None:
        payload_dir.mkdir(parents=True, exist_ok=True)

    any_processed = False
    for path in _iter_screenshots(screenshots_dir):
        if not _should_process(path, settle_seconds):
            continue
        log.info("Détection nouvelle image: %s", path.name)
        try:
            result = process_screenshot(path, engine, max_workers=max_workers, logger=log)
        except Exception as exc:  # noqa: BLE001
            log.exception("Erreur inattendue pendant le traitement de %s: %s", path, exc)
            _move_file(path, error_dir)
            continue

        if payload_dir is not None and (result.ocr_payload or result.errors):
            day_dir = payload_dir / datetime.now().date().isoformat()
            _store_payload(result, day_dir)

        destination_dir = processed_dir if result.matches else error_dir
        moved_path = _move_file(path, destination_dir)
        log.info("Screenshot déplacé vers %s", moved_path)

        any_processed = True
    return any_processed


def _process_database_jobs(
    *,
    engine: MassPredictionEngine,
    logger: Optional[logging.Logger] = None,
    mention_admin: bool = True,
) -> bool:
    log = logger or logging.getLogger(__name__)

    job_id: Optional[int] = None
    event_id: Optional[int] = None
    event_external_id: Optional[str] = None

    with _session_scope() as session:
        job = (
            session.execute(
                select(PredictionJob)
                .where(PredictionJob.status == "new")
                .order_by(PredictionJob.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .first()
        )
        if not job:
            return False

        event = session.get(Event, job.event_id)
        if not event or not event.external_id:
            job.status = "error"
            job.logs = "Événement introuvable pour cette demande."
            job.finished_at = datetime.now(timezone.utc)
            log.warning("Job %s impossible à traiter: événement manquant.", job.id)
            return True

        snapshot = (
            session.execute(
                select(OddsSnapshot)
                .where(OddsSnapshot.event_id == event.id)
                .order_by(OddsSnapshot.captured_at.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        if not snapshot:
            job.status = "error"
            job.logs = "Aucun snapshot de cotes disponible."
            job.finished_at = datetime.now(timezone.utc)
            log.warning("Job %s sans snapshot pour %s.", job.id, event.external_id)
            return True

        job_id = job.id
        event_id = event.id
        event_external_id = event.external_id
        job.status = "processing"
        job.logs = None
        job.started_at = datetime.now(timezone.utc)
        log.info("Job %s récupéré (%s)", job_id, event.event_name or event.competition or event.external_id)

        session.flush()
        session.expunge(job)
        session.expunge(event)
        session.expunge(snapshot)

    assert job_id is not None and event_id is not None and event_external_id is not None

    result = process_website_prediction(
        engine=engine,
        event=event,
        snapshot=snapshot,
        logger=log,
        mention_admin=mention_admin,
    )

    if result.errors:
        message = result.errors[0]
        with _session_scope() as session:
            job = session.get(PredictionJob, job_id)
            if job:
                job.status = "error"
                job.logs = message
                job.finished_at = datetime.now(timezone.utc)
        log.warning("Job %s terminé en erreur: %s", job_id, message)
        return True

    summary: Optional[str] = None

    with _session_scope() as session:
        job = session.get(PredictionJob, job_id)
        if not job:
            log.warning("Job %s introuvable en base au moment de la finalisation.", job_id)
            return True
        event = session.get(Event, event_id)
        if not event:
            job.status = "error"
            job.logs = "Événement introuvable lors de la finalisation."
            job.finished_at = datetime.now(timezone.utc)
            log.warning("Job %s impossible à finaliser: événement manquant.", job_id)
            return True

        stored_predictions = persist.persist_predictions(session, result.records)
        for stored_prediction in stored_predictions:
            stored_prediction.event = event
        session.flush()

        if result.records:
            first_record = result.records[0]
            summary = f"{first_record.player1_name} vs {first_record.player2_name} | modèle {first_record.model_key}"
        else:
            summary = "Aucune prédiction générée"

        job.status = "success"
        job.logs = summary
        job.finished_at = datetime.now(timezone.utc)

    log.info("Job %s terminé avec succès (%s): %s", job_id, result.run_id, summary)
    return True


def process_once(
    *,
    engine: MassPredictionEngine,
    source: Literal["database", "screenshots"],
    logger: Optional[logging.Logger] = None,
    screenshots_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    error_dir: Optional[Path] = None,
    payload_dir: Optional[Path] = None,
    settle_seconds: float = 1.5,
    max_workers: int = 4,
) -> bool:
    if source == "database":
        return _process_database_jobs(engine=engine, logger=logger)
    if source == "screenshots":
        if screenshots_dir is None or processed_dir is None or error_dir is None:
            raise ValueError("screenshots_dir, processed_dir et error_dir sont requis pour le mode screenshot")
        return _process_screenshots(
            engine=engine,
            screenshots_dir=screenshots_dir,
            processed_dir=processed_dir,
            error_dir=error_dir,
            payload_dir=payload_dir,
            settle_seconds=settle_seconds,
            max_workers=max_workers,
            logger=logger,
        )
    raise ValueError(f"Source inconnue pour process_once: {source}")


CSV_BASE_COLUMNS: List[str] = [
    "processed_at",
    "run_id",
    "match_index",
    "prediction_uid",
    "match_signature",
    "screenshot_path",
    "player1_name",
    "player1_id",
    "player2_name",
    "player2_id",
    "tourney_id",
    "tournament",
    "tournament_identifier",
    "tournament_location",
    "match_date",
    "match_time",
    "model_key",
    "prob_player1",
    "prob_player2",
    "fair_odds_player1",
    "fair_odds_player2",
    "odds_player1",
    "odds_player2",
    "edge_player1",
    "edge_player2",
]

CSV_PROBABILITY_COLUMNS: List[str] = [
    "prob_p_xgb",
    "prob_p_cat",
    "prob_p_lgb",
    "prob_p_hgb",
    "prob_p_weighted",
    "prob_p_blend",
    "prob_p_blend_platt",
    "prob_p_blend_iso",
]

CSV_KELLY_COLUMNS: List[str] = [
    "kelly_target",
    "kelly_fraction",
    "kelly_probability",
    "kelly_odds",
    "kelly_edge",
]

FEATURE_BASE_COLUMNS: List[str] = [
    "processed_at",
    "run_id",
    "match_index",
    "prediction_uid",
    "match_signature",
    "feature_direction",
    "model_key",
    "player1_id",
    "player2_id",
    "player1_name",
    "player2_name",
    "tourney_id",
    "tournament_identifier",
    "tournament",
    "match_date",
    "match_time",
]

_ENGINE_SINGLETON: Optional[MassPredictionEngine] = None
_ALLOW_RANK_FALLBACK = bool(int(os.getenv("MASS_PRED_ALLOW_RANK_FALLBACK", "0")))


def _set_shared_engine(engine: MassPredictionEngine) -> None:
    global _ENGINE_SINGLETON
    _ENGINE_SINGLETON = engine


def _init_worker_engine(run_dir: Optional[str], max_history_rows: Optional[int]) -> None:
    global _ENGINE_SINGLETON
    _ENGINE_SINGLETON = MassPredictionEngine(
        run_dir=run_dir,
        max_history_rows=max_history_rows,
        allow_rank_fallback=_ALLOW_RANK_FALLBACK,
    )


def _predict_with_shared_engine(match_input: MatchInput) -> MatchPrediction:
    if _ENGINE_SINGLETON is None:
        raise RuntimeError("Moteur de prédiction non initialisé dans le worker")
    return _ENGINE_SINGLETON.predict(match_input)


def _resolve_process_context() -> Tuple[mp.context.BaseContext, str]:
    try:
        ctx = mp.get_context("fork")
        return ctx, "fork"
    except ValueError:
        ctx = mp.get_context()
        return ctx, ctx.get_start_method()


def _ordered_csv_columns(
    df_columns: Iterable[str],
    existing_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    if existing_columns:
        order = list(existing_columns)
        seen = set(order)
        canonical = CSV_BASE_COLUMNS + CSV_PROBABILITY_COLUMNS + CSV_KELLY_COLUMNS
        for column in canonical:
            if column not in seen:
                order.append(column)
                seen.add(column)
        for column in sorted(df_columns):
            if column not in seen:
                order.append(column)
                seen.add(column)
        return order

    order: List[str] = []
    seen: Set[str] = set()

    def push(column: str) -> None:
        if column not in seen:
            order.append(column)
            seen.add(column)

    for column in CSV_BASE_COLUMNS:
        push(column)
    for column in CSV_PROBABILITY_COLUMNS:
        push(column)

    dynamic_prob_columns = sorted(
        column
        for column in df_columns
        if column.startswith("prob_") and column not in seen
    )
    for column in dynamic_prob_columns:
        push(column)

    for column in CSV_KELLY_COLUMNS:
        push(column)

    for column in sorted(df_columns):
        push(column)

    return order


def _ordered_feature_columns(
    df_columns: Iterable[str],
    existing_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    if existing_columns:
        order = list(existing_columns)
        seen = set(order)
        for column in FEATURE_BASE_COLUMNS:
            if column not in seen and column in df_columns:
                order.append(column)
                seen.add(column)
        for column in df_columns:
            if column not in seen:
                order.append(column)
                seen.add(column)
        return order

    order: List[str] = []
    seen: Set[str] = set()

    for column in FEATURE_BASE_COLUMNS:
        if column in df_columns and column not in seen:
            order.append(column)
            seen.add(column)
    for column in df_columns:
        if column not in seen:
            order.append(column)
            seen.add(column)
    return order


def _make_run_id(timestamp: Optional[datetime] = None) -> str:
    ts = timestamp or datetime.now()
    return ts.strftime("run-%Y%m%d-%H%M%S-%f")


def _build_match_input(ocr_match: OCRMatch, screenshot_path: Path, raw_entry: Dict[str, object]) -> MatchInput:
    return MatchInput(
        player1_name=ocr_match.player1,
        player2_name=ocr_match.player2,
        tournament_name=ocr_match.tournament,
        match_date=ocr_match.match_date,
        match_time=ocr_match.match_time,
        odds_player1=ocr_match.odds_player1,
        odds_player2=ocr_match.odds_player2,
        source_image=str(screenshot_path),
        raw_match=raw_entry,
    )


def _serialize_prediction(run_id: str, match: ProcessedMatch) -> Tuple[Dict[str, object], List[pd.DataFrame]]:
    req = match.prediction.request
    pred = match.prediction
    ocr = match.ocr
    primary = pred.model_key

    sorted_ids = sorted([
        (pred.player1.player_id or ""),
        (pred.player2.player_id or ""),
    ])

    tournament_identifier = pred.tournament.get("identifier") if pred.tournament else None
    tournament_atp_id = pred.tournament.get("atp_id") if pred.tournament else None
    tournament_year = pred.tournament.get("year") if pred.tournament else None

    tourney_id: Optional[str] = None
    if tournament_atp_id is not None:
        try:
            suffix_raw = str(tournament_atp_id).strip()
            suffix = suffix_raw
            try:
                numeric_suffix = float(suffix_raw)
            except (TypeError, ValueError):
                pass
            else:
                if numeric_suffix.is_integer():
                    suffix = str(int(numeric_suffix))
            inferred_year: Optional[int] = None
            if tournament_year is not None and not pd.isna(tournament_year):
                inferred_year = int(tournament_year)
            elif req.match_date is not None:
                inferred_year = req.match_date.year
            if inferred_year is not None:
                tourney_id = f"{inferred_year}-{suffix}"
        except (TypeError, ValueError):
            tourney_id = None

    recommendation = recommend_bet(
        pred.prob_player1,
        pred.prob_player2,
        ocr.odds_player1,
        ocr.odds_player2,
    )

    processed_at_naive = datetime.now()
    processed_at_local = datetime.now(ZoneInfo("Europe/Copenhagen"))

    match_date_obj = None
    raw_match_date = req.match_date
    if raw_match_date is not None:
        if isinstance(raw_match_date, pd.Timestamp):
            if not pd.isna(raw_match_date):
                match_date_obj = raw_match_date.date()
        elif isinstance(raw_match_date, datetime):
            match_date_obj = raw_match_date.date()
        elif hasattr(raw_match_date, "isoformat") and hasattr(raw_match_date, "year"):
            match_date_obj = raw_match_date
        else:
            parsed = pd.to_datetime(raw_match_date, errors="coerce")
            if parsed is not None and not pd.isna(parsed):
                match_date_obj = parsed.date()

    if match_date_obj is not None and processed_at_local.hour < 4:
        processed_local_date = processed_at_local.date()
        if match_date_obj == processed_local_date + timedelta(days=1):
            match_date_obj = processed_local_date

    forward_primary = None
    reverse_primary = None
    if primary and primary in pred.raw_forward.columns:
        value = pred.raw_forward.iloc[0].get(primary)
        if value is not None and not pd.isna(value):
            forward_primary = float(value)
    if primary and primary in pred.raw_reverse.columns:
        value = pred.raw_reverse.iloc[0].get(primary)
        if value is not None and not pd.isna(value):
            reverse_primary = float(value)
    reverse_primary_player1 = None
    if reverse_primary is not None:
        reverse_primary_player1 = float(max(0.0, min(1.0, 1.0 - reverse_primary)))
    primary_gap = None
    if forward_primary is not None and reverse_primary_player1 is not None:
        primary_gap = float(forward_primary - reverse_primary_player1)

    processed_at_str = processed_at_naive.isoformat(timespec="seconds")
    prediction_uid = f"{run_id}_{match.match_index}"
    match_signature = "|".join(
        sorted_ids
        + [
            str(tourney_id)
            if tourney_id
            else str(tournament_identifier)
            if tournament_identifier
            else str(req.tournament_name or "")
        ]
    )

    row = {
        "processed_at": processed_at_str,
        "run_id": run_id,
        "match_index": match.match_index,
        "prediction_uid": prediction_uid,
        "match_signature": match_signature,
        "screenshot_path": str(req.source_image or ""),
        "player1_name": pred.player1.display_name,
        "player1_id": pred.player1.player_id,
        "player2_name": pred.player2.display_name,
        "player2_id": pred.player2.player_id,
        "tourney_id": tourney_id,
        "tournament": pred.tournament.get("name") if pred.tournament else req.tournament_name,
        "tournament_identifier": tournament_identifier,
        "tournament_location": (
            pred.tournament.get("city")
            or pred.tournament.get("tourney_location")
            if pred.tournament
            else None
        ),
        "match_date": match_date_obj.isoformat() if match_date_obj else None,
        "match_time": req.match_time,
        "model_key": primary,
        "prob_player1": pred.prob_player1,
        "prob_player2": pred.prob_player2,
        "fair_odds_player1": pred.fair_odds_player1,
        "fair_odds_player2": pred.fair_odds_player2,
        "odds_player1": ocr.odds_player1,
        "odds_player2": ocr.odds_player2,
        "edge_player1": (
            (ocr.odds_player1 - pred.fair_odds_player1)
            if ocr.odds_player1 is not None and pred.fair_odds_player1 is not None
            else None
        ),
        "edge_player2": (
            (ocr.odds_player2 - pred.fair_odds_player2)
            if ocr.odds_player2 is not None and pred.fair_odds_player2 is not None
            else None
        ),
        "forward_prob_primary": forward_primary,
        "reverse_prob_primary": reverse_primary,
        "reverse_prob_primary_player1": reverse_primary_player1,
        "primary_prob_gap": primary_gap,
    }

    # Add raw probabilities for debugging/comparison
    for key in sorted(pred.combined_probabilities):
        row[f"prob_{key.lower()}"] = pred.combined_probabilities[key]

    # Ensure Kelly columns always exist and appear after probability columns
    kelly_fields = (
        "kelly_target",
        "kelly_fraction",
        "kelly_probability",
        "kelly_odds",
        "kelly_edge",
    )
    for field in kelly_fields:
        row[field] = None

    if recommendation is not None:
        row["kelly_target"] = recommendation["side"]
        row["kelly_fraction"] = recommendation["fraction"]
        row["kelly_probability"] = recommendation["probability"]
        row["kelly_odds"] = recommendation["odds"]
        row["kelly_edge"] = recommendation["edge"]

    feature_metadata = {
        "processed_at": processed_at_str,
        "run_id": run_id,
        "match_index": match.match_index,
        "prediction_uid": prediction_uid,
        "match_signature": match_signature,
        "model_key": primary,
        "player1_id": pred.player1.player_id,
        "player2_id": pred.player2.player_id,
        "player1_name": pred.player1.display_name,
        "player2_name": pred.player2.display_name,
        "tourney_id": tourney_id,
        "tournament_identifier": tournament_identifier,
        "tournament": row["tournament"],
        "match_date": row["match_date"],
        "match_time": row["match_time"],
    }
    feature_frames = _build_feature_frames(prediction=pred, metadata=feature_metadata)

    return row, feature_frames


def _build_feature_frames(*, prediction: MatchPrediction, metadata: Dict[str, object]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for direction, features_df in (
        ("forward", getattr(prediction, "features_forward", pd.DataFrame())),
        ("reverse", getattr(prediction, "features_reverse", pd.DataFrame())),
    ):
        if features_df is None or features_df.empty:
            continue
        enriched = features_df.copy()
        meta_with_direction = dict(metadata)
        meta_with_direction["feature_direction"] = direction
        for key, value in meta_with_direction.items():
            enriched[key] = value
        ordered_meta = [col for col in FEATURE_BASE_COLUMNS if col in meta_with_direction]
        remaining_cols = [col for col in enriched.columns if col not in ordered_meta]
        enriched = enriched[ordered_meta + remaining_cols]
        frames.append(enriched)
    return frames


def _append_to_daily_csv(rows: List[Dict[str, object]]) -> Path:
    today = datetime.now().date().isoformat()
    output_path = Path(MASS_PREDICTION_EXPORT_PATTERN.format(date=today))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    existing_columns: Optional[List[str]] = None
    output_exists = output_path.exists()
    if output_exists:
        existing_columns = pd.read_csv(output_path, nrows=0).columns.tolist()

    column_order = _ordered_csv_columns(df.columns, existing_columns)
    df = df.reindex(columns=column_order, fill_value=None)

    header = not output_exists
    df.to_csv(output_path, mode="a", header=header, index=False)
    return output_path


def _append_features_csv(frames: List[pd.DataFrame]) -> Optional[Path]:
    return _append_feature_frames_to_dir(
        frames,
        output_dir=MASS_PREDICTION_FEATURE_EXPORT_DIR,
        prefix="mass_prediction_features",
    )


def _append_feature_frames_to_dir(
    frames: List[pd.DataFrame],
    *,
    output_dir: Path | str,
    prefix: str,
) -> Optional[Path]:
    if not frames:
        return None

    today = datetime.now().date().isoformat()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"{prefix}_{today}.csv"

    combined = pd.concat(frames, ignore_index=True, sort=False)

    existing_columns: Optional[List[str]] = None
    output_exists = output_path.exists()
    if output_exists:
        existing_columns = pd.read_csv(output_path, nrows=0).columns.tolist()

    column_order = _ordered_feature_columns(combined.columns, existing_columns)
    combined = combined.reindex(columns=column_order, fill_value=None)

    header = not output_exists
    combined.to_csv(output_path, mode="a", header=header, index=False)
    return output_path


def _append_website_features_csv(frames: List[pd.DataFrame]) -> Optional[Path]:
    return _append_feature_frames_to_dir(
        frames,
        output_dir=WEBSITE_MASS_PREDICTION_FEATURE_EXPORT_DIR,
        prefix="website_mass_prediction_features",
    )


def _players_from_event_snapshot(event, snapshot) -> list[str]:
    players: list[str] = []
    if getattr(event, "players", None):
        players = [str(player) for player in event.players if isinstance(player, str)]
    if len(players) < 2 and getattr(snapshot, "selections", None):
        fallback: list[str] = []
        for sel in snapshot.selections[:2]:
            if isinstance(sel, dict):
                name = sel.get("name") or sel.get("runner_name")
                if name:
                    fallback.append(str(name))
        while len(fallback) < 2:
            fallback.append(f"Selection {len(fallback) + 1}")
        players = fallback
    if len(players) < 2:
        raise ValueError("Impossible de déterminer les joueurs pour l'événement.")
    return players[:2]


def _odds_from_snapshot(snapshot) -> tuple[float, float]:
    selections = getattr(snapshot, "selections", None)
    if not selections or len(selections) < 2:
        raise ValueError("Snapshot incomplet : deux sélections minimum attendues.")
    odds: list[float] = []
    for sel in selections[:2]:
        value = None
        if isinstance(sel, dict):
            raw = sel.get("odds_decimal") or sel.get("odds")
            if raw is not None:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None
        if value is None or value <= 0:
            raise ValueError("Cote invalide dans le snapshot.")
        odds.append(value)
    return odds[0], odds[1]


def _build_match_input_from_event(event, snapshot) -> MatchInput:
    players = _players_from_event_snapshot(event, snapshot)
    odds_player1, odds_player2 = _odds_from_snapshot(snapshot)
    match_date = None
    match_time = None
    if getattr(event, "start_utc", None):
        match_date = event.start_utc.date()
        match_time = event.start_utc.strftime("%H:%M")
    tournament = getattr(event, "competition", None) or getattr(event, "event_name", None) or "Custom Tournament"
    return MatchInput(
        player1_name=players[0],
        player2_name=players[1],
        tournament_name=tournament,
        match_date=match_date,
        match_time=match_time,
        odds_player1=odds_player1,
        odds_player2=odds_player2,
        source_image=None,
        raw_match={
            "event_external_id": getattr(event, "external_id", None),
            "odds_snapshot_id": getattr(snapshot, "id", None),
        },
    )


def _prediction_record_from_processed(
    *,
    run_id: str,
    match_index: int,
    processed_match: ProcessedMatch,
    event,
    snapshot,
) -> PredictionRecord:
    pred = processed_match.prediction
    req = getattr(pred, "request", None)
    ocr = processed_match.ocr

    processed_at = datetime.now(timezone.utc)
    odds1, odds2 = _odds_from_snapshot(snapshot)
    recommendation = recommend_bet(
        pred.prob_player1,
        pred.prob_player2,
        odds1,
        odds2,
    )

    player1_name = pred.player1.display_name if getattr(pred, "player1", None) else (req.player1_name if req else ocr.player1)
    player2_name = pred.player2.display_name if getattr(pred, "player2", None) else (req.player2_name if req else ocr.player2)
    player1_id = pred.player1.player_id if getattr(pred, "player1", None) else None
    player2_id = pred.player2.player_id if getattr(pred, "player2", None) else None

    sorted_ids = sorted(str(pid).strip() for pid in (player1_id, player2_id) if pid)
    if not sorted_ids:
        sorted_ids = sorted(str(name).strip() for name in (player1_name, player2_name) if name)

    tournament = getattr(pred, "tournament", {}) or {}
    tournament_identifier = tournament.get("identifier")
    tournament_atp_id = tournament.get("atp_id")
    tournament_year = tournament.get("year")

    match_date_value = getattr(req, "match_date", None)
    match_year = None
    if match_date_value is not None:
        if hasattr(match_date_value, "year"):
            match_year = match_date_value.year
        else:
            try:
                match_year = datetime.fromisoformat(str(match_date_value)).year
            except ValueError:
                match_year = None

    inferred_year = (
        int(tournament_year)
        if tournament_year not in (None, "")
        else match_year
    )

    tourney_suffix = None
    if tournament_atp_id not in (None, ""):
        suffix = str(tournament_atp_id).strip()
        try:
            numeric_suffix = float(suffix)
        except ValueError:
            tourney_suffix = suffix
        else:
            tourney_suffix = str(int(numeric_suffix)) if numeric_suffix.is_integer() else suffix

    tourney_id = None
    if tourney_suffix and inferred_year:
        tourney_id = f"{int(inferred_year)}-{tourney_suffix}"

    signature_parts = list(sorted_ids)
    if tourney_id:
        signature_parts.append(tourney_id)
    elif tournament_identifier:
        signature_parts.append(str(tournament_identifier))
    else:
        tournament_name = req.tournament_name if req else ocr.tournament
        signature_parts.append(str(tournament_name or ""))

    match_signature = "|".join(signature_parts)

    match_start = getattr(event, "start_utc", None)
    record = PredictionRecord(
        processed_at=processed_at,
        prediction_uid=f"{run_id}_{match_index}",
        match_signature=match_signature,
        model_key=pred.model_key,
        match_start=match_start,
        player1_name=player1_name or "",
        player2_name=player2_name or "",
        prob_player1=pred.prob_player1,
        prob_player2=pred.prob_player2,
        odds_player1=odds1,
        odds_player2=odds2,
        fair_odds_player1=pred.fair_odds_player1,
        fair_odds_player2=pred.fair_odds_player2,
        kelly_target=recommendation.get("side") if recommendation else None,
        kelly_fraction=recommendation.get("fraction") if recommendation else None,
        kelly_edge=recommendation.get("edge") if recommendation else None,
        raw={
            "source": "admin-ui",
            "event_external_id": getattr(event, "external_id", None),
            "event_internal_id": getattr(event, "id", None),
            "snapshot_id": getattr(snapshot, "id", None),
            "combined_probabilities": getattr(pred, "combined_probabilities", {}),
        },
    )
    return record


@dataclass
class WebsitePredictionResult:
    run_id: str
    matches: List[ProcessedMatch]
    records: List[PredictionRecord]
    errors: List[str]



def process_screenshot(
    screenshot_path: Path | str,
    engine: MassPredictionEngine,
    *,
    max_workers: int = 4,
    logger: Optional[logging.Logger] = None,
) -> ScreenshotProcessingResult:
    path = Path(screenshot_path)
    log = logger or logging.getLogger(__name__)
    run_id = _make_run_id()

    log.info("[%s] Début traitement screenshot %s", run_id, path.name)

    try:
        matches, payload = extract_matches_from_image(path)
    except OCRExtractionError as exc:
        log.error("[%s] OCR échoué: %s", run_id, exc)
        return ScreenshotProcessingResult(
            run_id=run_id,
            screenshot_path=path,
            ocr_payload={},
            matches=[],
            errors=[str(exc)],
        )

    log.info("[%s] OCR réussi: %d match(s) détecté(s)", run_id, len(matches))
    send_discord_pending_matches(run_id, matches)

    processed_matches: List[ProcessedMatch] = []
    errors: List[str] = []

    jobs: List[Tuple[int, OCRMatch, MatchInput]] = []
    for idx, (ocr_match, raw_entry) in enumerate(zip(matches, payload.get("matches", []), strict=False), start=1):
        log.info(
            "[%s] Lancement prédiction match #%d : %s vs %s for tournament %s",
            run_id,
            idx,
            ocr_match.player1,
            ocr_match.player2,
            ocr_match.tournament,
        )
        match_input = _build_match_input(ocr_match, path, raw_entry if isinstance(raw_entry, dict) else {})
        jobs.append((idx, ocr_match, match_input))

    def record_prediction(idx: int, ocr_match: OCRMatch, prediction: MatchPrediction) -> None:
        processed_matches.append(
            ProcessedMatch(
                match_index=idx,
                ocr=ocr_match,
                prediction=prediction,
            )
        )

        primary_key = prediction.model_key
        forward_primary = None
        reverse_primary = None
        if primary_key and primary_key in prediction.raw_forward.columns:
            value = prediction.raw_forward.iloc[0].get(primary_key)
            if value is not None and not pd.isna(value):
                forward_primary = float(value)
        if primary_key and primary_key in prediction.raw_reverse.columns:
            value = prediction.raw_reverse.iloc[0].get(primary_key)
            if value is not None and not pd.isna(value):
                reverse_primary = float(value)
        if forward_primary is not None and reverse_primary is not None:
            reverse_as_player1 = float(max(0.0, min(1.0, 1.0 - reverse_primary)))
            diff_primary = forward_primary - reverse_as_player1
            log.info(
                "[%s] Match #%d diff primary=%s forward=%.4f reverse=%.4f reverse(p1)=%.4f gap=%.4f",
                run_id,
                idx,
                primary_key,
                forward_primary,
                reverse_primary,
                reverse_as_player1,
                diff_primary,
            )
        log.info(
            "[%s] Match #%d terminé : %s vs %s (modèle=%s)",
            run_id,
            idx,
            ocr_match.player1,
            ocr_match.player2,
            _label_for_model_key(prediction.model_key),
        )

    if not jobs:
        pass
    elif max_workers <= 1:
        for idx, ocr_match, match_input in jobs:
            try:
                prediction = engine.predict(match_input)
            except PredictionError as exc:
                message = f"Match #{idx} échoué ({ocr_match.player1} vs {ocr_match.player2}): {exc}"
                log.error("[%s] %s", run_id, message)
                errors.append(message)
                continue
            except Exception as exc:  # noqa: BLE001
                message = f"Match #{idx} erreur inattendue ({ocr_match.player1} vs {ocr_match.player2}): {exc}"
                log.exception("[%s] %s", run_id, message)
                errors.append(message)
                continue
            record_prediction(idx, ocr_match, prediction)
    else:
        mode_env = os.getenv("MASS_PREDICTION_PARALLEL_MODE", "auto").strip().lower()
        use_threads = False
        ctx: Optional[mp.context.BaseContext] = None
        start_method: Optional[str] = None

        if mode_env in {"thread", "threads"}:
            use_threads = True
        elif mode_env in {"process", "processes", "proc"}:
            use_threads = False
        else:
            ctx, start_method = _resolve_process_context()
            if start_method == "fork":
                use_threads = True
            else:
                use_threads = False

        if use_threads:
            log.info("[%s] Parallélisation par threads (workers=%d)", run_id, max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures: List[Tuple[int, OCRMatch, MatchInput, object]] = []
                for idx, ocr_match, match_input in jobs:
                    future = executor.submit(engine.predict, match_input)
                    futures.append((idx, ocr_match, match_input, future))

                for idx, ocr_match, match_input, future in futures:
                    try:
                        prediction = future.result()
                    except PredictionError as exc:
                        message = f"Match #{idx} échoué ({ocr_match.player1} vs {ocr_match.player2}): {exc}"
                        log.error("[%s] %s", run_id, message)
                        errors.append(message)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        message = f"Match #{idx} erreur inattendue ({ocr_match.player1} vs {ocr_match.player2}): {exc}"
                        log.exception("[%s] %s", run_id, message)
                        errors.append(message)
                        continue
                    record_prediction(idx, ocr_match, prediction)
        else:
            if ctx is None or start_method is None:
                ctx, start_method = _resolve_process_context()
            pool_kwargs = {
                "max_workers": max_workers,
                "mp_context": ctx,
            }
            if start_method == "fork":
                _set_shared_engine(engine)
            else:
                pool_kwargs["initializer"] = _init_worker_engine
                pool_kwargs["initargs"] = (
                    str(engine.run_path) if engine.run_path else None,
                    getattr(engine, "max_history_rows", None),
                )

            log.info(
                "[%s] Parallélisation par processus (mode=%s, workers=%d)",
                run_id,
                start_method,
                max_workers,
            )
            with ProcessPoolExecutor(**pool_kwargs) as executor:
                futures = []
                for idx, ocr_match, match_input in jobs:
                    future = executor.submit(_predict_with_shared_engine, match_input)
                    futures.append((idx, ocr_match, match_input, future))

                for idx, ocr_match, match_input, future in futures:
                    try:
                        prediction = future.result()
                    except PredictionError as exc:
                        message = f"Match #{idx} échoué ({ocr_match.player1} vs {ocr_match.player2}): {exc}"
                        log.error("[%s] %s", run_id, message)
                        errors.append(message)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        message = f"Match #{idx} erreur inattendue ({ocr_match.player1} vs {ocr_match.player2}): {exc}"
                        log.exception("[%s] %s", run_id, message)
                        errors.append(message)
                        continue
                    record_prediction(idx, ocr_match, prediction)

    if processed_matches:
        serialized = [_serialize_prediction(run_id, match) for match in processed_matches]
        rows = [row for row, _ in serialized]
        feature_frames: List[pd.DataFrame] = []
        for _, frames in serialized:
            feature_frames.extend(frames)

        output_path = _append_to_daily_csv(rows)
        log.info("[%s] Résultats enregistrés dans %s", run_id, output_path)

        feature_output_path = _append_features_csv(feature_frames)
        if feature_output_path:
            log.info("[%s] Features enregistrées dans %s", run_id, feature_output_path)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.odds_ingest.cli",
                    "mass-csv",
                    "--path",
                    str(output_path),
                    "--persist",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            log.info("[%s] CLI ingestion exécutée (%s)", run_id, result.stdout.strip())
        except subprocess.CalledProcessError as exc:
            log.error(
                "[%s] CLI ingestion échec: %s",
                run_id,
                exc.stderr.strip() if exc.stderr else exc.stdout.strip(),
            )

        send_discord_debug_summary(run_id, path, processed_matches, mention_admin=True)


        current_datetime_str = datetime.now().strftime("%d/%m/%Y %H:%M")

        send_discord_mention(
            DISCORD_SUBSCRIBER_PREDICTION_BOT_TOKEN, 
            DISCORD_SUBSCRIBER_PREDICTION_CHANNEL_ID, 
            [DISCORD_SUBSCRIBER_ROLE_MENTION],
            content=f"🔮 Prédictions du jour mes ptits choux : {current_datetime_str} 🔮"
        )
        send_discord_subscriber_summary(run_id, path, processed_matches, mention_subscribers=False)
        send_discord_free_pick(run_id, path, processed_matches)

    else:
        log.info("[%s] Aucun match exploitable pour %s", run_id, path.name)

    log.info(
        "[%s] Fin traitement screenshot %s (succès=%d, erreurs=%d)",
        run_id,
        path.name,
        len(processed_matches),
        len(errors),
    )

    return ScreenshotProcessingResult(
        run_id=run_id,
        screenshot_path=path,
        ocr_payload=payload,
        matches=processed_matches,
        errors=errors,
    )


__all__ = [
    "process_screenshot",
    "process_website_prediction",
    "ProcessedMatch",
    "ScreenshotProcessingResult",
    "WebsitePredictionResult",
]

MODEL_KEY_LABELS = {
    "P_WEIGHTED": "Weighted Blend (raw)",
    "P_BLEND_PLATT": "Meta Blend (Platt)",
    "P_BLEND": "Meta Blend (logit)",
    "P_BLEND_ISO": "Meta Blend (Isotonic)",
    "P_XGB": "XGBoost",
    "P_CAT": "CatBoost",
    "P_LGB": "LightGBM",
    "P_HGB": "HistGradientBoosting",
}


def _label_for_model_key(key: str) -> str:
    return MODEL_KEY_LABELS.get(key, key)
def process_website_prediction(
    *,
    engine: MassPredictionEngine,
    event,
    snapshot,
    logger: Optional[logging.Logger] = None,
    mention_admin: bool = True,
) -> WebsitePredictionResult:
    log = logger or logging.getLogger(__name__)
    run_id = datetime.now(timezone.utc).strftime("admin-%Y%m%d-%H%M%S-%f")

    try:
        match_input = _build_match_input_from_event(event, snapshot)
        players = [match_input.player1_name, match_input.player2_name]
        odds1, odds2 = _odds_from_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001
        message = f"Préparation impossible pour l'événement {getattr(event, 'external_id', '?')}: {exc}"
        log.error("[%s] %s", run_id, message)
        return WebsitePredictionResult(run_id=run_id, matches=[], records=[], errors=[message])

    ocr_match = OCRMatch(
        tournament=match_input.tournament_name,
        match_date=match_input.match_date,
        match_time=match_input.match_time,
        player1=players[0],
        player2=players[1],
        odds_player1=odds1,
        odds_player2=odds2,
        raw={
            "event_external_id": getattr(event, "external_id", None),
            "event_internal_id": getattr(event, "id", None),
            "snapshot_id": getattr(snapshot, "id", None),
        },
    )

    try:
        prediction = engine.predict(match_input)
    except PredictionError as exc:
        message = f"Échec de la prédiction: {exc}"
        log.error("[%s] %s", run_id, message)
        return WebsitePredictionResult(run_id=run_id, matches=[], records=[], errors=[str(exc)])
    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] Erreur inattendue pendant la prédiction", run_id)
        return WebsitePredictionResult(run_id=run_id, matches=[], records=[], errors=[str(exc)])

    processed_match = ProcessedMatch(match_index=1, ocr=ocr_match, prediction=prediction)
    record = _prediction_record_from_processed(
        run_id=run_id,
        match_index=1,
        processed_match=processed_match,
        event=event,
        snapshot=snapshot,
    )

    log.info(
        "[%s] Prédiction admin générée: %s vs %s (modèle %s)",
        run_id,
        record.player1_name,
        record.player2_name,
        record.model_key,
    )

    _, feature_frames = _serialize_prediction(run_id, processed_match)
    feature_output_path = _append_website_features_csv(feature_frames)
    if feature_output_path:
        log.info("[%s] Features admin enregistrées dans %s", run_id, feature_output_path)

    pseudo_path = Path(f"admin-{getattr(event, 'external_id', 'unknown')}.txt")
    send_discord_debug_summary(
        run_id,
        pseudo_path,
        [processed_match],
        mention_admin=mention_admin,
        mention_subscriber=False,
    )

    return WebsitePredictionResult(
        run_id=run_id,
        matches=[processed_match],
        records=[record],
        errors=[],
    )

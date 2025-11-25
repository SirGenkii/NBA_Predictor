from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    NBA_MASS_LOG_DIR,
    NBA_MASS_RUNS_DIR,
    NBA_MASS_SCREENSHOT_DIR,
    NBA_MASS_SCREENSHOT_ERROR_DIR,
    NBA_MASS_SCREENSHOT_PROCESSED_DIR,
)
from src.nba_mass_prediction import PredictionEngine, process_screenshot
from src.nba_mass_prediction.adapters.json_api import payloads_from_json
from src.nba_mass_prediction.adapters.ocr_adapter import OCRExtractionError
from src.nba_mass_prediction.engine import PredictionEngineError
from src.nba_mass_prediction.resolver import MatchResolver, TeamResolver, TeamResolverError
from src.nba_mass_prediction.runner import process_payloads
try:
    from src.watcher_state import write_state  # type: ignore
except ImportError:  # pragma: no cover
    def write_state(state: str, **kwargs) -> None:
        logging.getLogger("nba_mass_prediction").warning(
            "watcher_state module introuvable. Etat %s non persistant.", state
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_payload(exc: Exception, source: str) -> dict[str, str]:
    return {
        "message": str(exc),
        "at": _utcnow(),
        "source": source,
    }


@dataclass
class ActivityStats:
    processed: int = 0
    last_run_id: Optional[str] = None
    last_output_path: Optional[str] = None
    last_processed_at: Optional[str] = None
    last_source: Optional[str] = None
    model_label: Optional[str] = None
    last_error: Optional[dict] = None

    def absorb(self, other: "ActivityStats") -> None:
        if other.processed:
            self.processed += other.processed
            self.last_run_id = other.last_run_id or self.last_run_id
            self.last_output_path = other.last_output_path or self.last_output_path
            self.last_processed_at = other.last_processed_at or self.last_processed_at
            self.last_source = other.last_source or self.last_source
            self.model_label = other.model_label or self.model_label

        if other.last_error is not None:
            self.last_error = other.last_error
        elif other.processed:
            # clear previous error on successful run
            self.last_error = None


def configure_logger() -> logging.Logger:
    log_dir = Path(NBA_MASS_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date().isoformat()
    log_path = log_dir / f"nba_mass_prediction_{today}.log"

    logger = logging.getLogger("nba_mass_prediction")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _iter_screenshots(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime):
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            yield path


def _should_process(path: Path, settle_seconds: float) -> bool:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return (time.time() - mtime) >= settle_seconds


def _move_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination = dst_dir / src.name
    counter = 1
    while destination.exists():
        destination = dst_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    src.replace(destination)
    return destination


def _process_directory(
    *,
    engine: PredictionEngine,
    resolver: MatchResolver,
    screenshots_dir: Path,
    processed_dir: Path,
    error_dir: Path,
    settle_seconds: float,
    logger: logging.Logger,
) -> ActivityStats:
    stats = ActivityStats()
    for path in _iter_screenshots(screenshots_dir):
        if not _should_process(path, settle_seconds):
            continue
        logger.info("Détection nouvelle image: %s", path.name)
        try:
            result = process_screenshot(engine, path, resolver=resolver, output_dir=NBA_MASS_RUNS_DIR)
        except (OCRExtractionError, TeamResolverError, PredictionEngineError) as exc:
            logger.error("Échec traitement %s: %s", path.name, exc)
            _move_file(path, error_dir)
            stats.last_error = _error_payload(exc, f"screenshot:{path.name}")
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur inattendue pendant le traitement de %s: %s", path, exc)
            _move_file(path, error_dir)
            stats.last_error = _error_payload(exc, f"screenshot:{path.name}")
            continue

        _move_file(path, processed_dir)
        model_label = result.get("model_label") or result.get("artifact", {}).get("model_label") or "modèle inconnu"
        logger.info(
            "Screenshot %s traité (run_id=%s, output=%s, modèle=%s)",
            path.name,
            result["run_id"],
            result["output_path"],
            model_label,
        )
        stats.processed += 1
        stats.last_run_id = result["run_id"]
        stats.last_output_path = str(result["output_path"])
        stats.last_processed_at = _utcnow()
        stats.last_source = f"screenshot:{path.name}"
        stats.model_label = model_label
        stats.last_error = None
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Watchdog NBA mass prediction (screenshots + API).")
    parser.add_argument("--screenshots-dir", type=Path, default=Path(NBA_MASS_SCREENSHOT_DIR))
    parser.add_argument("--processed-dir", type=Path, default=Path(NBA_MASS_SCREENSHOT_PROCESSED_DIR))
    parser.add_argument("--error-dir", type=Path, default=Path(NBA_MASS_SCREENSHOT_ERROR_DIR))
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--input-json", type=Path, help="Fichier JSON contenant des matchs à injecter (bypass OCR).")
    args = parser.parse_args()

    logger = configure_logger()
    pid = os.getpid()

    write_state("loading", pid=pid, message="Initialisation du moteur NBA mass prediction")

    try:
        engine = PredictionEngine()
        resolver = MatchResolver(TeamResolver())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Impossible d'initialiser le moteur NBA")
        write_state(
            "error",
            pid=pid,
            stopped_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise

    started_at = datetime.now(timezone.utc).isoformat()
    base_state = {
        "pid": pid,
        "started_at": started_at,
        "engine": "nba_mass_prediction",
    }
    state_context: dict[str, Optional[object]] = {
        "heartbeat": started_at,
        "processed_total": 0,
        "last_run_id": None,
        "last_output_path": None,
        "last_processed_at": None,
        "last_source": None,
        "model_label": None,
        "last_error": None,
    }
    write_state("running", **base_state, heartbeat=started_at, processed_total=0)

    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.error_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_interval = max(30.0, float(args.poll_interval))
    last_heartbeat = time.monotonic()

    def _process_all() -> ActivityStats:
        aggregate = ActivityStats()
        if args.input_json and args.input_json.exists():
            logger.info("Ingestion du JSON %s", args.input_json)
            payloads = payloads_from_json(args.input_json, screenshot_path=None)
            result = process_payloads(engine, payloads, resolver=resolver, output_dir=NBA_MASS_RUNS_DIR)
            logger.info("JSON traité (run_id=%s, output=%s)", result["run_id"], result["output_path"])
            args.input_json.unlink()
            aggregate.processed += 1
            aggregate.last_run_id = result["run_id"]
            aggregate.last_output_path = str(result["output_path"])
            aggregate.last_processed_at = _utcnow()
            aggregate.last_source = f"payload:{args.input_json.name}"
            aggregate.model_label = result.get("model_label") or result.get("artifact", {}).get("model_label")
            aggregate.last_error = None
        directory_stats = _process_directory(
            engine=engine,
            resolver=resolver,
            screenshots_dir=args.screenshots_dir,
            processed_dir=args.processed_dir,
            error_dir=args.error_dir,
            settle_seconds=args.settle_seconds,
            logger=logger,
        )
        aggregate.absorb(directory_stats)
        return aggregate

    def _flush_state(activity: Optional[ActivityStats] = None, *, force: bool = False) -> None:
        nonlocal last_heartbeat
        state_updated = False

        if activity:
            if activity.processed:
                state_context["processed_total"] = int(state_context["processed_total"] or 0) + activity.processed
                state_updated = True
            for attr in ("last_run_id", "last_output_path", "last_processed_at", "last_source", "model_label"):
                value = getattr(activity, attr)
                if value is not None:
                    state_context[attr] = value
                    state_updated = True
            if activity.last_error is not None or activity.processed:
                state_context["last_error"] = activity.last_error
                state_updated = True

        now = time.monotonic()
        if not (force or state_updated or (now - last_heartbeat) >= heartbeat_interval):
            return

        state_context["heartbeat"] = _utcnow()
        payload = dict(base_state)
        payload.update(state_context)
        write_state("running", **payload)
        last_heartbeat = now

    def _finalize_state(label: str, **extra: object) -> None:
        payload = dict(base_state)
        payload.update(state_context)
        payload.update(extra)
        write_state(label, **payload)

    stop_requested = False
    shutdown_reason: Optional[str] = None

    def _handle_signal(signum: int, _frame: Optional[object]) -> None:
        nonlocal stop_requested, shutdown_reason
        signal_name = signal.Signals(signum).name
        logger.info("Signal %s reçu, arrêt en cours...", signal_name)
        stop_requested = True
        shutdown_reason = f"Signal {signal_name}"

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(sig, _handle_signal)
        except (AttributeError, ValueError):
            pass

    if args.run_once:
        stats = _process_all()
        if not stats.processed and stats.last_error is None:
            logger.info("Aucun screenshot ni payload à traiter.")
        _flush_state(stats, force=True)
        _finalize_state(
            "stopped",
            stopped_at=_utcnow(),
            reason="Exécution run-once terminée",
        )
        return

    logger.info("Watchdog démarré sur %s", args.screenshots_dir)

    try:
        while True:
            stats = _process_all()
            force_write = bool(stats.processed or stats.last_error)
            _flush_state(stats, force=force_write)

            if stop_requested:
                logger.info("Arrêt demandé (%s).", shutdown_reason)
                _finalize_state(
                    "stopped",
                    stopped_at=_utcnow(),
                    reason=shutdown_reason or "Signal externe",
                )
                return

            if not stats.processed:
                _flush_state(force=False)
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        _finalize_state(
            "stopped",
            stopped_at=_utcnow(),
            reason="Arrêt manuel",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Watcher arrêté suite à une erreur")
        _finalize_state(
            "error",
            stopped_at=_utcnow(),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()

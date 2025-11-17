from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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
) -> bool:
    any_processed = False
    for path in _iter_screenshots(screenshots_dir):
        if not _should_process(path, settle_seconds):
            continue
        logger.info("Détection nouvelle image: %s", path.name)
        try:
            result = process_screenshot(engine, path, resolver=resolver, output_dir=NBA_MASS_RUNS_DIR)
        except (OCRExtractionError, TeamResolverError, PredictionEngineError) as exc:
            logger.error("Échec traitement %s: %s", path.name, exc)
            _move_file(path, error_dir)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur inattendue pendant le traitement de %s: %s", path, exc)
            _move_file(path, error_dir)
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
        any_processed = True
    return any_processed


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
    write_state("running", **base_state, heartbeat=started_at)

    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.error_dir.mkdir(parents=True, exist_ok=True)

    def _process_all() -> bool:
        processed = False
        if args.input_json and args.input_json.exists():
            logger.info("Ingestion du JSON %s", args.input_json)
            payloads = payloads_from_json(args.input_json, screenshot_path=None)
            result = process_payloads(engine, payloads, resolver=resolver, output_dir=NBA_MASS_RUNS_DIR)
            logger.info("JSON traité (run_id=%s, output=%s)", result["run_id"], result["output_path"])
            args.input_json.unlink()
            processed = True
        processed = (
            _process_directory(
                engine=engine,
                resolver=resolver,
                screenshots_dir=args.screenshots_dir,
                processed_dir=args.processed_dir,
                error_dir=args.error_dir,
                settle_seconds=args.settle_seconds,
                logger=logger,
            )
            or processed
        )
        return processed

    if args.run_once:
        if not _process_all():
            logger.info("Aucun screenshot ni payload à traiter.")
        write_state(
            "stopped",
            pid=pid,
            stopped_at=datetime.now(timezone.utc).isoformat(),
            reason="Exécution run-once terminée",
        )
        return

    logger.info("Watchdog démarré sur %s", args.screenshots_dir)
    heartbeat_interval = max(30.0, float(args.poll_interval))
    last_heartbeat = time.monotonic()

    try:
        while True:
            processed = _process_all()

            now = time.monotonic()
            if (now - last_heartbeat) >= heartbeat_interval:
                write_state(
                    "running",
                    **base_state,
                    heartbeat=datetime.now(timezone.utc).isoformat(),
                )
                last_heartbeat = now

            if not processed:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        write_state(
            "stopped",
            pid=pid,
            stopped_at=datetime.now(timezone.utc).isoformat(),
            reason="Arrêt manuel",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Watcher arrêté suite à une erreur")
        write_state(
            "error",
            pid=pid,
            stopped_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MASS_PREDICTION_LOG_DIR,
    MASS_PREDICTION_SCREENSHOT_DIR,
    MASS_PREDICTION_SCREENSHOT_ERROR_DIR,
    MASS_PREDICTION_SCREENSHOT_PROCESSED_DIR,
    DEFAULT_MAX_WORKERS,
)
from src.mass_prediction import MassPredictionEngine
from src.mass_prediction.notifications import send_discord_engine_startup
from src.mass_prediction.runner import process_once
from src.watcher_state import write_state

ALLOW_RANK_FALLBACK = bool(int(os.getenv("MASS_PRED_ALLOW_RANK_FALLBACK", "0")))


def configure_logger() -> logging.Logger:
    log_dir = Path(MASS_PREDICTION_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date().isoformat()
    log_path = log_dir / f"mass_prediction_{today}.log"

    logger = logging.getLogger("mass_prediction")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Surveille la base et les screenshots pour lancer les prédictions.")
    parser.add_argument("--screenshots-dir", type=Path, default=Path(MASS_PREDICTION_SCREENSHOT_DIR))
    parser.add_argument("--processed-dir", type=Path, default=Path(MASS_PREDICTION_SCREENSHOT_PROCESSED_DIR))
    parser.add_argument("--error-dir", type=Path, default=Path(MASS_PREDICTION_SCREENSHOT_ERROR_DIR))
    parser.add_argument("--payload-dir", type=Path, default=Path(MASS_PREDICTION_LOG_DIR) / "ocr_payloads")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--run-once", action="store_true", help="Traite les données disponibles puis quitte")
    args = parser.parse_args()

    logger = configure_logger()
    pid = os.getpid()

    write_state(
        "loading",
        pid=pid,
        message="Initialisation du moteur de prédiction",
    )
    
    logger.info("Initialisation du moteur de prédiction...")
    send_discord_engine_startup()
    try:
        engine = MassPredictionEngine(allow_rank_fallback=ALLOW_RANK_FALLBACK)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Impossible d'initialiser le moteur de prédiction")
        write_state(
            "error",
            pid=pid,
            stopped_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise

    started_at = datetime.now(timezone.utc).isoformat()
    engine_run_path = getattr(engine, "run_path", None)
    if engine_run_path is not None:
        engine_run_path = str(engine_run_path)
    base_state = {
        "pid": pid,
        "started_at": started_at,
        "engine_run_path": engine_run_path,
        "allow_rank_fallback": ALLOW_RANK_FALLBACK,
    }
    write_state("running", **base_state, heartbeat=started_at)

    logger.info("Modèle chargé depuis %s", engine.run_path)
    logger.info("Nombre de workers par défaut: %s", DEFAULT_MAX_WORKERS)

    args.screenshots_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.error_dir.mkdir(parents=True, exist_ok=True)

    db_kwargs = {
        "engine": engine,
        "source": "database",
        "logger": logger,
    }

    screenshot_kwargs = {
        "engine": engine,
        "source": "screenshots",
        "logger": logger,
        "screenshots_dir": args.screenshots_dir,
        "processed_dir": args.processed_dir,
        "error_dir": args.error_dir,
        "payload_dir": args.payload_dir,
        "settle_seconds": args.settle_seconds,
        "max_workers": args.max_workers,
    }

    if args.run_once:
        processed = False
        processed = process_once(**db_kwargs) or processed
        processed = process_once(**screenshot_kwargs) or processed
        if not processed:
            logger.info("Aucun job ni screenshot à traiter")
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
    final_state = None
    try:
        while True:
            processed = process_once(**db_kwargs)
            processed = process_once(**screenshot_kwargs) or processed

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
        final_state = {
            "status": "stopped",
            "pid": pid,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "reason": "Arrêt manuel",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Watcher arrêté suite à une erreur")
        write_state(
            "error",
            pid=pid,
            stopped_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )
        raise
    finally:
        if final_state is not None:
            status = final_state.pop("status")
            write_state(status, **final_state)


if __name__ == "__main__":
    main()

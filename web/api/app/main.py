"""FastAPI backend exposing NBA mass prediction runs and watcher status."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

APP_TITLE = "NBA Mass Prediction Viewer"
DESCRIPTION = "Read run artifacts and watcher state directly from the filesystem."

def _default_data_dir() -> Path:
    path = Path(__file__).resolve()
    parents = path.parents
    if len(parents) >= 4:
        candidate = parents[3] / "data" / "mass_prediction_nba"
        if candidate.exists():
            return candidate
    return Path("/data/mass_prediction_nba")


DATA_DIR = Path(os.environ.get("NBA_MASS_DATA_DIR", _default_data_dir())).resolve()
RUNS_DIR = DATA_DIR / "runs"
WATCHER_STATE_PATH = DATA_DIR / "watcher_state.json"
PREDICTIONS_LOG = DATA_DIR / "predictions_log.csv"

app = FastAPI(title=APP_TITLE, description=DESCRIPTION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Fichier introuvable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"JSON illisible dans {path}: {exc}") from exc


def _sorted_run_files() -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    dated_dirs = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    ordered: List[Path] = []
    for directory in dated_dirs:
        day_runs = sorted(directory.glob("*.json"), key=lambda p: p.name)
        ordered.extend(day_runs)
    return ordered


def _best_bet(run: dict) -> Optional[dict]:
    best: Optional[dict] = None
    for match in run.get("matches", []):
        for evaluation in match.get("evaluations", []):
            for side in ("over", "under"):
                offer = evaluation.get(side)
                if not isinstance(offer, dict):
                    continue
                edge = offer.get("edge")
                if edge is None:
                    continue
                if best is None or edge > best["edge"]:
                    best = {
                        "edge": edge,
                        "label": evaluation.get("label"),
                        "pivot": evaluation.get("pivot"),
                        "bet": side,
                        "probability": offer.get("probability"),
                        "odds": offer.get("odds"),
                        "matchup": _matchup_label(match.get("match")),
                        "kelly": offer.get("kelly", {}).get("scaled", {}),
                    }
    return best


def _matchup_label(match: Optional[dict]) -> Optional[str]:
    if not isinstance(match, dict):
        return None
    home = match.get("home", {}).get("name")
    away = match.get("away", {}).get("name")
    if home and away:
        return f"{away} @ {home}"
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "data_dir": str(DATA_DIR)}


@app.get("/runs")
def list_runs(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    files = _sorted_run_files()
    selected = files[offset : offset + limit]
    summaries: List[dict] = []
    for path in selected:
        run = _load_json(path)
        best = _best_bet(run)
        summaries.append(
            {
                "run_id": run.get("run_id") or path.stem,
                "created_at": run.get("created_at"),
                "matches_count": len(run.get("matches", [])),
                "model_label": run.get("model_label"),
                "path": str(path.relative_to(DATA_DIR)) if DATA_DIR in path.parents else str(path),
                "best_bet": best,
            }
        )
    return {
        "runs": summaries,
        "total": len(files),
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < len(files),
    }


def _find_run_path(run_id: str) -> Path:
    candidates = list(RUNS_DIR.glob(f"*/{run_id}.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"Run {run_id} introuvable")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    path = _find_run_path(run_id)
    run = _load_json(path)
    run["path"] = str(path.relative_to(DATA_DIR)) if DATA_DIR in path.parents else str(path)
    return run


@app.get("/runs/{run_id}/matches/{match_index}")
def get_match(run_id: str, match_index: int) -> dict:
    run = get_run(run_id)
    matches = run.get("matches") or []
    if match_index >= len(matches):
        raise HTTPException(status_code=404, detail=f"Match {match_index} introuvable pour {run_id}")
    match = matches[match_index]
    return {
        "run_id": run_id,
        "match_index": match_index,
        "match": match,
    }


@app.get("/watcher-state")
def watcher_state() -> dict:
    if not WATCHER_STATE_PATH.exists():
        raise HTTPException(status_code=404, detail="watcher_state.json introuvable")
    return _load_json(WATCHER_STATE_PATH)


@app.get("/predictions-log")
def predictions_log(limit: int = Query(100, ge=1, le=1000)) -> dict:
    if not PREDICTIONS_LOG.exists():
        raise HTTPException(status_code=404, detail="predictions_log.csv introuvable")
    lines = PREDICTIONS_LOG.read_text().strip().splitlines()
    header, *rows = lines
    latest = rows[-limit:][::-1]
    return {
        "header": header.split(","),
        "rows": [row.split(",") for row in latest],
    }

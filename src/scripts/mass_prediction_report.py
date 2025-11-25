from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NBA_MASS_PREDICTIONS_CSV, NBA_MASS_RUNS_DIR  # noqa: E402


@dataclass
class MatchInfo:
    run_id: str
    match_date: str
    home: str
    away: str
    screenshot: Optional[str]
    mean_total: Optional[float]
    sigma: Optional[float]
    match_key: tuple[int, int, str]


@dataclass
class BetRow:
    match: MatchInfo
    recommendation: str
    side: str
    pivot: Optional[float]
    bookmaker: Optional[str]
    probability: Optional[float]
    odds: Optional[float]
    fair_odds: Optional[float]
    edge: Optional[float]
    kelly: Optional[float]
    safe_pick: bool
    safe_pair_coverage: Optional[float]
    safe_pair_edge_sum: Optional[float]
    safe_pick_confidence: Optional[float]
    safe_reason: Optional[str]


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _latest_run_path(runs_dir: Path) -> Path:
    candidates = sorted(runs_dir.glob("*/*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Aucun run trouvé dans {runs_dir}")
    return candidates[-1]


def _find_run_path(runs_dir: Path, *, run_id: Optional[str], match_date: Optional[str]) -> Path:
    if run_id:
        matches = list(runs_dir.glob(f"*/{run_id}.json"))
        if not matches:
            raise FileNotFoundError(f"Run {run_id} introuvable dans {runs_dir}")
        return matches[0]

    if match_date:
        date_dir = runs_dir / match_date
        candidates = sorted(date_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(f"Aucun run pour la date {match_date} dans {runs_dir}")

    return _latest_run_path(runs_dir)


def _load_match_index(artifact: dict) -> dict[tuple[int, int, str], MatchInfo]:
    matches: dict[tuple[int, int, str], MatchInfo] = {}
    for match in artifact.get("matches", []):
        meta = match.get("match") or {}
        home = meta.get("home") or {}
        away = meta.get("away") or {}
        match_date = meta.get("match_date")
        prediction = match.get("prediction") or {}
        key = (home.get("team_id"), away.get("team_id"), match_date)
        matches[key] = MatchInfo(
            run_id=artifact.get("run_id", ""),
            match_date=match_date,
            home=home.get("name") or "Home",
            away=away.get("name") or "Away",
            screenshot=meta.get("screenshot_path"),
            mean_total=_safe_float(prediction.get("mean_total")),
            sigma=_safe_float(prediction.get("sigma")),
            match_key=key,
        )
    return matches


def _load_predictions_for_run(run_id: str, match_index: dict[tuple[int, int, str], MatchInfo]) -> list[BetRow]:
    bets: list[BetRow] = []
    csv_path = Path(NBA_MASS_PREDICTIONS_CSV)
    if not csv_path.exists():
        return bets

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("run_id") != run_id:
                continue

            match_key = (
                int(row["home_team_id"]),
                int(row["away_team_id"]),
                row.get("match_date"),
            )
            match_info = match_index.get(match_key)
            if not match_info:
                continue

            bets.append(
                BetRow(
                    match=match_info,
                    recommendation=row.get("recommendation", ""),
                    side=row.get("side", ""),
                    pivot=_safe_float(row.get("pivot")),
                    bookmaker=row.get("bookmaker") or None,
                    probability=_safe_float(row.get("probability")),
                    odds=_safe_float(row.get("odds")),
                    fair_odds=_safe_float(row.get("fair_odds")),
                    edge=_safe_float(row.get("edge")),
                    kelly=_safe_float(row.get("kelly_full")),
                    safe_pick=row.get("safe_pick") == "1",
                    safe_pair_coverage=_safe_float(row.get("safe_pair_coverage")),
                    safe_pair_edge_sum=_safe_float(row.get("safe_pair_edge_sum")),
                    safe_pick_confidence=_safe_float(row.get("safe_pick_confidence")),
                    safe_reason=row.get("safe_reason") or None,
                )
            )
    return bets


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_odds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _format_edge(edge: Optional[float]) -> str:
    if edge is None:
        return "-"
    return f"{edge * 100:+.1f}%"


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return "?"
    return f"{value:.1f}"


def _format_pivot(value: Optional[float]) -> str:
    if value is None:
        return "pivot ?"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def _title_side(side: str) -> str:
    if side.lower() == "over":
        return "Over"
    if side.lower() == "under":
        return "Under"
    return side


def _group_by_match(bets: Iterable[BetRow]) -> dict[tuple[int, int, str], list[BetRow]]:
    grouped: dict[tuple[int, int, str], list[BetRow]] = {}
    for bet in bets:
        grouped.setdefault(bet.match.match_key, []).append(bet)
    return grouped


def _print_match_block(match: MatchInfo, bets: list[BetRow], top_n: int) -> None:
    header = f"{match.away} @ {match.home} — {match.match_date}"
    print(header)
    if match.screenshot:
        print(f"  screenshot: {match.screenshot}")
    if match.mean_total is not None:
        sigma_part = f" (σ {_format_number(match.sigma)})" if match.sigma is not None else ""
        print(f"  modèle: mean_total {_format_number(match.mean_total)}{sigma_part}")

    recommended = [b for b in bets if b.recommendation.startswith("bet_")]
    recommended = [
        b
        for b in recommended
        if b.recommendation.replace("bet_", "", 1) == b.side
    ]

    if not recommended:
        positive = [b for b in bets if (b.edge or 0) > 0]
        fallback = sorted(positive or bets, key=lambda b: b.edge or -999)[:top_n]
        print("  (aucun bet recommandé par la stratégie, top edges pour info)")
        to_display = fallback
    else:
        to_display = sorted(recommended, key=lambda b: b.edge or 0, reverse=True)

    for bet in to_display:
        label = f"{_title_side(bet.side)} { _format_pivot(bet.pivot) }"
        details = [
            f"cote { _format_odds(bet.odds) }",
            f"fair { _format_odds(bet.fair_odds) }",
            f"proba { _format_pct(bet.probability) }",
            f"edge { _format_edge(bet.edge) }",
            f"kelly { _format_pct(bet.kelly) }",
        ]
        tags = []
        rec_side = bet.recommendation.replace("bet_", "", 1) if bet.recommendation.startswith("bet_") else None
        if rec_side and rec_side == bet.side:
            tags.append("bet")
        elif rec_side:
            tags.append(f"pick={rec_side}")
        if bet.safe_pick:
            label_safe = "SAFE"
            if bet.safe_pick_confidence is not None:
                label_safe = f"{label_safe} (conf {bet.safe_pick_confidence:.2f})"
            tags.append(label_safe)
        elif bet.safe_reason:
            tags.append(f"safe_reason={bet.safe_reason}")
        if bet.bookmaker:
            tags.append(f"book {bet.bookmaker}")

        suffix = f" [{' | '.join(tags)}]" if tags else ""
        print(f"  - {label}: {', '.join(details)}{suffix}")
    print()


def run_report(*, run_path: Path, top_n: int) -> None:
    artifact = json.loads(run_path.read_text())
    run_id = artifact.get("run_id") or run_path.stem
    created_at = artifact.get("created_at")
    model_label = artifact.get("model_label") or artifact.get("model_path") or "modèle inconnu"
    match_index = _load_match_index(artifact)
    bets = _load_predictions_for_run(run_id, match_index)

    if not bets:
        print(f"Aucune entrée dans le CSV pour le run {run_id}.")
        return

    grouped = _group_by_match(bets)

    print(f"Run {run_id} ({created_at or 'date inconnue'}) — {model_label}")
    print(f"Fichier: {run_path}")
    total_recommended = sum(1 for b in bets if b.recommendation.startswith("bet_"))
    total_safe = sum(1 for b in bets if b.safe_pick)
    print(f"Matches: {len(grouped)} | Paris recommandés: {total_recommended} | Picks safe: {total_safe}")
    print()

    for match_key, match_bets in grouped.items():
        match_info = match_index.get(match_key)
        if match_info:
            _print_match_block(match_info, match_bets, top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Résumé lisible des paris issus des runs mass_prediction_nba."
    )
    parser.add_argument("--run-id", help="Identifiant de run précis (sinon dernier run).")
    parser.add_argument(
        "--date",
        dest="match_date",
        help="Date du run (YYYY-MM-DD) pour choisir le dernier run du jour.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Nombre de lignes à afficher si aucun bet recommandé.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(NBA_MASS_RUNS_DIR)
    try:
        run_path = _find_run_path(runs_dir, run_id=args.run_id, match_date=args.match_date)
    except FileNotFoundError as exc:
        print(f"Erreur: {exc}")
        raise SystemExit(1)

    run_report(run_path=run_path, top_n=args.top)


if __name__ == "__main__":
    main()

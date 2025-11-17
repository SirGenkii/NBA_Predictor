from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.nba_mass_prediction.schemas import MatchPayload, PivotMarket


def _parse_date(value: str) -> date:
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _parse_markets(markets: Iterable[Dict[str, Any]]) -> List[PivotMarket]:
    parsed: List[PivotMarket] = []
    for entry in markets:
        pivot = float(entry["pivot"])
        over_odds = float(entry["over_odds"])
        under_odds = float(entry["under_odds"])
        parsed.append(
            PivotMarket(
                pivot=pivot,
                over_odds=over_odds,
                under_odds=under_odds,
                label=entry.get("label"),
                bookmaker=entry.get("bookmaker"),
                metadata={k: v for k, v in entry.items() if k not in {"pivot", "over_odds", "under_odds", "label", "bookmaker"}},
            )
        )
    return parsed


def _to_payload(entry: Dict[str, Any], *, source: str, screenshot_path: Path | None) -> MatchPayload:
    markets_data = entry.get("markets")
    if not markets_data:
        raise ValueError("Payload absent de marchés.")
    markets = _parse_markets(markets_data)
    return MatchPayload(
        home_team=str(entry["home_team"]),
        away_team=str(entry["away_team"]),
        match_date=_parse_date(entry["match_date"]),
        match_time=entry.get("match_time"),
        markets=tuple(markets),
        screenshot_path=screenshot_path,
        source=source,
        raw_payload=entry,
    )


def payloads_from_json(
    data: str | Path | Dict[str, Any] | Sequence[Dict[str, Any]],
    *,
    source: str = "api",
    screenshot_path: Path | None = None,
) -> List[MatchPayload]:
    """Convert JSON input (dict, list, or file path) into MatchPayload objects."""
    if isinstance(data, (str, Path)) and Path(data).exists():
        text = Path(data).read_text()
        parsed = json.loads(text)
    else:
        parsed = data

    if isinstance(parsed, dict):
        entries = parsed.get("matches") or parsed.get("data") or [parsed]
        if isinstance(entries, dict):
            entries = [entries]
    else:
        entries = parsed

    if not isinstance(entries, list):
        raise ValueError("Payload JSON inattendu.")

    return [_to_payload(entry, source=source, screenshot_path=screenshot_path) for entry in entries]


__all__ = ["payloads_from_json"]

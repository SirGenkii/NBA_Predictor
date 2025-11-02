from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import polars as pl


@dataclass(frozen=True)
class FieldRule:
    strategy: str
    alias: Optional[str] = None
    weight_col: Optional[str] = "minutes_float"


DEFAULT_FIELD_RULES: Dict[str, FieldRule] = {
    "team_minutes_total": FieldRule(strategy="sum"),
    "player_count": FieldRule(strategy="count"),
}


def build_rules_from_schema(columns: Iterable[str]) -> Dict[str, FieldRule]:
    rules: Dict[str, FieldRule] = {}
    for name in columns:
        if name in DEFAULT_FIELD_RULES:
            continue
        lower_name = name.lower()
        if any(keyword in lower_name for keyword in ("made", "attempted", "points", "rebounds", "assists", "steals", "blocks", "turnovers", "fouls", "minutes", "possessions", "wins", "losses", "pts", "reb", "ast", "blk", "stl", "tov", "pf")):
            rules[name] = FieldRule(strategy="sum")
        elif any(keyword in lower_name for keyword in ("percentage", "pct", "ratio", "rate", "usage")):
            rules[name] = FieldRule(strategy="weighted_minutes")
        elif any(keyword in lower_name for keyword in ("rating", "net", "pace", "estimated", "pie")):
            rules[name] = FieldRule(strategy="weighted_minutes")
        else:
            rules[name] = FieldRule(strategy="first")
    return rules

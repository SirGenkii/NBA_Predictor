from __future__ import annotations

import difflib
import os
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from src.config import DATA_TEAMS_DIR
from src.nba_mass_prediction.schemas import MatchPayload, ResolvedMatch, TeamIdentity
from src.utils import get_latest_file


def _normalize(value: str) -> str:
    """Lowercase ASCII helper used for alias matching."""
    text = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in text if not unicodedata.combining(ch))
    collapsed = "".join(ch for ch in stripped if ch.isalnum() or ch.isspace())
    return " ".join(collapsed.split()).lower()


def _alias_variants(full_name: str, city: str, abbreviation: str) -> List[str]:
    tokens = full_name.split()
    nickname = tokens[-1] if tokens else full_name
    variants = {
        full_name,
        nickname,
        city,
        f"{city} {nickname}".strip(),
        f"{city} {nickname}s".strip(),
        abbreviation,
    }
    if " " in city:
        parts = city.split()
        initials = "".join(word[0] for word in parts if word)
        variants.add(f"{initials} {nickname}")
        variants.add(initials)
    return sorted(set(filter(None, variants)))


class TeamResolverError(RuntimeError):
    """Raised when a team cannot be resolved."""


@dataclass
class TeamResolver:
    data_dir: Path = Path(DATA_TEAMS_DIR)
    min_ratio: float = 0.6
    _by_alias: Dict[str, TeamIdentity] = field(init=False, default_factory=dict)
    _by_id: Dict[int, TeamIdentity] = field(init=False, default_factory=dict)
    _source_path: Optional[Path] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.refresh()

    @property
    def source_path(self) -> Optional[Path]:
        return self._source_path

    def refresh(self) -> None:
        directory = Path(self.data_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Répertoire DATA_TEAMS_DIR introuvable: {directory}")
        latest_file = Path(get_latest_file(directory))
        df = pd.read_csv(latest_file)
        alias_map: Dict[str, TeamIdentity] = {}
        id_map: Dict[int, TeamIdentity] = {}
        for row in df.to_dict("records"):
            try:
                team_id = int(row["id"])
            except (TypeError, ValueError):
                continue
            full_name = str(row.get("full_name") or row.get("name") or "").strip()
            if not full_name:
                continue
            city = str(row.get("city") or "").strip()
            abbreviation = str(row.get("abbreviation") or "").strip()

            aliases = tuple(sorted(set(_alias_variants(full_name, city, abbreviation))))
            normalized_aliases = {_normalize(alias) for alias in aliases if alias}
            identity = TeamIdentity(team_id=team_id, canonical_name=full_name, aliases=aliases)

            id_map[team_id] = identity
            for alias in normalized_aliases:
                alias_map[alias] = identity

        if not alias_map:
            raise TeamResolverError("Impossible de charger les équipes NBA (dataset vide).")

        self._by_alias = alias_map
        self._by_id = id_map
        self._source_path = latest_file

    def resolve_name(self, name: str) -> TeamIdentity:
        normalized = _normalize(name)
        if not normalized:
            raise TeamResolverError("Nom d'équipe vide.")
        if normalized in self._by_alias:
            return self._by_alias[normalized]
        candidates = difflib.get_close_matches(normalized, self._by_alias.keys(), n=1, cutoff=self.min_ratio)
        if candidates:
            return self._by_alias[candidates[0]]
        raise TeamResolverError(f"Impossible de résoudre l'équipe: {name!r}")

    def resolve_id(self, team_id: int) -> TeamIdentity:
        identity = self._by_id.get(int(team_id))
        if not identity:
            raise TeamResolverError(f"TEAM_ID inconnu: {team_id}")
        return identity


@dataclass
class MatchResolver:
    team_resolver: TeamResolver
    default_match_date: Optional[date] = None

    def resolve(
        self,
        payload: MatchPayload,
        *,
        match_date_override: Optional[date] = None,
        home_team_id_override: Optional[int] = None,
        away_team_id_override: Optional[int] = None,
    ) -> ResolvedMatch:
        home_team = (
            self.team_resolver.resolve_id(home_team_id_override)
            if home_team_id_override is not None
            else self.team_resolver.resolve_name(payload.home_team)
        )
        away_team = (
            self.team_resolver.resolve_id(away_team_id_override)
            if away_team_id_override is not None
            else self.team_resolver.resolve_name(payload.away_team)
        )
        match_date = match_date_override or self.default_match_date or payload.match_date
        metadata = {
            "match_time": payload.match_time,
            "source": payload.source,
            "overrides": {
                "match_date": bool(match_date_override or self.default_match_date),
                "home_team": home_team_id_override is not None,
                "away_team": away_team_id_override is not None,
            },
            "team_dataset": str(self.team_resolver.source_path) if self.team_resolver.source_path else None,
        }
        return ResolvedMatch(
            home_team=home_team,
            away_team=away_team,
            input_home_team=payload.home_team,
            input_away_team=payload.away_team,
            match_date=match_date,
            markets=payload.markets,
            metadata=metadata,
            screenshot_path=payload.screenshot_path,
            raw_payload=payload.raw_payload,
        )


__all__ = ["TeamResolver", "MatchResolver", "TeamResolverError"]

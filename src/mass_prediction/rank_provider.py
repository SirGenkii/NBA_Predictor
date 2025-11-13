from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from src.config import ATP_PLAYER_RANK_POINTS_DIR
from src.mass_prediction.text import normalize_text


@dataclass(frozen=True)
class RankEntry:
    rank: int
    points: float
    snapshot_date: datetime
    source_path: Path


class CsvRankProvider:
    """Provide ATP ranks/points from the latest snapshot CSV.

    The expected CSV columns are ``rank`` (int), ``points`` (float/int), and name
    columns such as ``first_name`` / ``last_name`` or ``player_name``.
    Optionally a ``player_id`` column can be supplied to allow direct lookups.
    """

    def __init__(self, directory: Optional[Path | str] = None) -> None:
        self.directory = Path(directory or ATP_PLAYER_RANK_POINTS_DIR)
        self._loaded_path: Optional[Path] = None
        self._snapshot_date: Optional[datetime] = None
        self._entries_by_player_id: Dict[str, RankEntry] = {}
        self._entries_by_alias: Dict[str, RankEntry] = {}

    def refresh(self) -> None:
        latest_file = self._find_latest_file()
        if latest_file is None:
            raise FileNotFoundError(
                f"Aucun fichier CSV trouvé dans {self.directory}. "
                "Générez un snapshot via le script local_scrap_atp_ranking."
            )
        if latest_file == self._loaded_path:
            return

        df = pd.read_csv(latest_file)
        required_columns = {"rank", "points"}
        if not required_columns <= set(df.columns):
            raise ValueError(
                f"Le fichier {latest_file} doit contenir les colonnes {sorted(required_columns)}."
            )

        snapshot_date = self._infer_snapshot_date(latest_file)
        entries_by_player: Dict[str, RankEntry] = {}
        entries_by_alias: Dict[str, RankEntry] = {}

        def make_entry(row: pd.Series) -> RankEntry:
            return RankEntry(
                rank=int(row["rank"]),
                points=float(row["points"]),
                snapshot_date=snapshot_date,
                source_path=latest_file,
            )

        for _, row in df.iterrows():
            try:
                entry = make_entry(row)
            except (TypeError, ValueError):
                continue

            player_id = row.get("player_id")
            if isinstance(player_id, str) and player_id.strip():
                entries_by_player[player_id.strip()] = entry

            # Build alias keys from column combinations
            aliases = set()
            if "player_name" in row and isinstance(row["player_name"], str):
                aliases.add(normalize_text(row["player_name"]))
            first_name = row.get("first_name")
            last_name = row.get("last_name")
            if isinstance(first_name, str) and isinstance(last_name, str):
                aliases.add(normalize_text(f"{first_name} {last_name}"))
                aliases.add(normalize_text(f"{last_name} {first_name}"))
            elif isinstance(first_name, str):
                aliases.add(normalize_text(first_name))
            elif isinstance(last_name, str):
                aliases.add(normalize_text(last_name))

            for alias in filter(None, aliases):
                entries_by_alias.setdefault(alias, entry)

        self._loaded_path = latest_file
        self._snapshot_date = snapshot_date
        self._entries_by_player_id = entries_by_player
        self._entries_by_alias = entries_by_alias

    def get(self, player_id: str, aliases: Iterable[str]) -> Optional[RankEntry]:
        self.refresh()
        if player_id:
            entry = self._entries_by_player_id.get(player_id)
            if entry:
                return entry
        for alias in aliases:
            normalized = normalize_text(alias)
            entry = self._entries_by_alias.get(normalized)
            if entry:
                return entry
        return None

    def _find_latest_file(self) -> Optional[Path]:
        if not self.directory.exists():
            return None
        candidates = [p for p in self.directory.iterdir() if p.suffix.lower() == ".csv" and p.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _infer_snapshot_date(path: Path) -> datetime:
        """Try to derive a snapshot datetime from the filename or fallback to mtime."""
        stem = path.stem
        for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d"):
            for token in stem.split("_"):
                try:
                    return datetime.strptime(token, fmt)
                except ValueError:
                    continue
        return datetime.fromtimestamp(path.stat().st_mtime)

    @property
    def snapshot_info(self) -> Tuple[Optional[datetime], Optional[Path]]:
        return self._snapshot_date, self._loaded_path

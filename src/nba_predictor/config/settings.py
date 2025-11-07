from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import TomlConfigSettingsSource

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_under_project(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


@dataclass(frozen=True)
class DataPaths:
    """Container for the main folders used by the project."""

    root: Path
    legacy_root: Path
    legacy_raw: Path
    legacy_raw_last: Path
    bronze_root: Path
    bronze_games: Path
    bronze_boxscores: Path
    bronze_player_availability: Path
    silver_root: Path
    silver_team_game_facts: Path
    silver_team_boxscores_agg: Path
    silver_player_availability: Path
    silver_team_form_windowed: Path
    silver_matchups_h2h_base: Path
    silver_matchups_h2h_features: Path
    silver_feast_offline: Path
    gold_root: Path
    gold_training_sets: Path
    gold_scoring_payloads: Path
    gold_monitoring_snapshots: Path
    logs_root: Path


class _TomlSettingsSource(TomlConfigSettingsSource):
    """Custom settings source that loads values from the first matching TOML file."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        self._candidate_paths: List[Path] = self._discover_candidate_paths()
        super().__init__(settings_cls, toml_file=self._candidate_paths)

    @staticmethod
    def _discover_candidate_paths() -> List[Path]:
        explicit = os.getenv("NBA_SETTINGS_FILE")
        candidates: List[Path] = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        candidates.append(Path("settings.toml"))
        candidates.append(Path.cwd() / "settings.toml")
        return candidates


class Settings(BaseSettings):
    """Application configuration exposed via Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_prefix="NBA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(
        default="dev",
        description="Name of the active environment (dev/staging/prod).",
    )
    data_root: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data",
        description="Root folder containing the data lake layout.",
    )
    logs_root: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "logs",
        description="Root folder where log files are persisted.",
    )
    config_name: str | None = Field(
        default=None,
        description="Optional named configuration profile.",
    )
    boxscore_batch_size: int = Field(
        default=25,
        ge=1,
        description="Batch size used when scraping boxscores.",
    )
    feature_windows: Tuple[int, ...] = Field(
        default=(3, 5, 10, 25),  # Reduced from (3, 5, 10, 25, 50, 100, 200) to avoid memory explosion
        description="Rolling window lengths used for feature aggregation.",
    )
    feature_windows_top_players: Tuple[int, ...] = Field(
        default=(1, 2, 3, 5, 10),
        description="Rolling windows used for player availability features.",
    )

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        object.__setattr__(self, "data_root", _resolve_under_project(self.data_root))
        object.__setattr__(self, "logs_root", _resolve_under_project(self.logs_root))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            _TomlSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            init_settings,
            file_secret_settings,
        )

    @property
    def data_paths(self) -> DataPaths:
        root = self.data_root
        legacy_root = root / "legacy"
        bronze_root = root / "bronze"
        bronze_api_root = bronze_root / "nba_api"
        silver_root = root / "silver"
        gold_root = root / "gold"

        return DataPaths(
            root=root,
            legacy_root=legacy_root,
            legacy_raw=legacy_root / "raw",
            legacy_raw_last=legacy_root / "raw_last",
            bronze_root=bronze_root,
            bronze_games=bronze_api_root / "games",
            bronze_boxscores=bronze_api_root / "boxscores",
            bronze_player_availability=bronze_api_root / "player_availability",
            silver_root=silver_root,
            silver_team_game_facts=silver_root / "team_game_facts",
            silver_team_boxscores_agg=silver_root / "team_boxscores_agg",
            silver_player_availability=silver_root / "player_availability",
            silver_team_form_windowed=silver_root / "team_form_windowed",
            silver_matchups_h2h_base=silver_root / "matchups_h2h_base",
            silver_matchups_h2h_features=silver_root / "matchups_h2h_features",
            silver_feast_offline=silver_root / "feast_offline",
            gold_root=gold_root,
            gold_training_sets=gold_root / "training_sets",
            gold_scoring_payloads=gold_root / "scoring_payloads",
            gold_monitoring_snapshots=gold_root / "monitoring_snapshots",
            logs_root=self.logs_root,
        )

    def ensure_directories(self) -> None:
        """Ensure core directories exist."""
        paths = [
            self.data_paths.legacy_root,
            self.data_paths.legacy_raw,
            self.data_paths.legacy_raw_last,
            self.data_paths.bronze_root,
            self.data_paths.silver_root,
            self.data_paths.gold_root,
            self.logs_root,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""

    cfg = Settings()
    cfg.ensure_directories()
    return cfg


settings = get_settings()

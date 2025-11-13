from __future__ import annotations

import difflib
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.config import ATP_PLAYER_RANK_POINTS_DIR, MASS_PREDICTION_DIR
from src.features.steps.elo import EloFeatureState, EloGradientState
from src.features.steps.form import MatchCountState, PStatsLastKState, WinLastKState
from src.features.steps.head_to_head import H2HFeatureState
from src.features.steps.match_stats import MatchLengthState
from src.features.steps.tourney import TourneyStartGapState
from src.mass_prediction.data_loader import (
    ArtifactsBundle,
    compute_latest_player_stats,
    load_latest_artifacts,
    load_latest_history,
    load_players_catalog,
    load_tournaments_catalog,
)
from src.mass_prediction.rank_provider import CsvRankProvider
from src.mass_prediction.text import normalize_text
from src.prediction_pipeline_result import (
    NEUTRAL_TO_PIPELINE_MATCH_COLUMNS,
    build_result_feature_dataset,
    build_features_for_matches,
    predict_from_features,
)
from src.enrichment.tournament_identity import (
    resolve_seed_entry_for_players,
)


@dataclass
class PlayerProfile:
    player_id: str
    display_name: str
    aliases: Tuple[str, ...]
    country: Optional[str]
    hand: Optional[str]
    height_cm: Optional[float]
    birthdate: Optional[pd.Timestamp]
    latest_rank: Optional[float]
    latest_rank_points: Optional[float]
    age_latest: Optional[float]


@dataclass
class MatchInput:
    player1_name: str
    player2_name: str
    tournament_name: Optional[str]
    match_date: Optional[date]
    match_time: Optional[str]
    odds_player1: float
    odds_player2: float
    source_image: Optional[str] = None
    raw_match: Optional[Dict[str, Any]] = None


@dataclass
class MatchPrediction:
    request: MatchInput
    player1: PlayerProfile
    player2: PlayerProfile
    tournament: Dict[str, Any]
    model_key: str
    prob_player1: float
    prob_player2: float
    fair_odds_player1: float
    fair_odds_player2: float
    combined_probabilities: Dict[str, float]
    raw_forward: pd.DataFrame
    raw_reverse: pd.DataFrame
    features_forward: pd.DataFrame
    features_reverse: pd.DataFrame


class PredictionError(RuntimeError):
    """Raised when a match cannot be processed."""


def _safe_float(value: Optional[float], default: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return float(value)


def _compute_age(birthdate: Optional[pd.Timestamp], target_date: pd.Timestamp) -> Optional[float]:
    if birthdate is None or pd.isna(birthdate) or pd.isna(target_date):
        return None
    delta = target_date - birthdate
    if pd.isna(delta):
        return None
    days = getattr(delta, "days", None)
    if days is None:
        return None
    if days < 0:
        return None
    return days / 365.25


DEFAULT_HISTORY_TAIL = 50000


class MassPredictionEngine:
    def __init__(
        self,
        *,
        run_dir: Optional[Path | str] = None,
        max_history_rows: Optional[int] = None,
        history_df: Optional[pd.DataFrame] = None,
        history_tail: Optional[int] = None,
        rank_provider: Optional[CsvRankProvider] = None,
        allow_rank_fallback: bool = False,
    ) -> None:
        self._bundle: ArtifactsBundle = load_latest_artifacts(run_dir)
        self.artifacts = self._bundle.artifacts
        self.run_path = self._bundle.run_path
        self.primary_prob_key = self.artifacts.primary_prob_key
        self.max_history_rows = max_history_rows
        self.allow_rank_fallback = allow_rank_fallback
        self.rank_provider: Optional[CsvRankProvider] = rank_provider or CsvRankProvider()
        self.rank_snapshot_date: Optional[datetime] = None
        self.rank_snapshot_path: Optional[Path] = None

        try:
            self.rank_provider.refresh()
            snapshot_date, snapshot_path = self.rank_provider.snapshot_info
            self.rank_snapshot_date = snapshot_date
            self.rank_snapshot_path = snapshot_path
        except (FileNotFoundError, ValueError) as exc:
            if allow_rank_fallback:
                self.rank_provider = None
            else:
                raise PredictionError(
                    "Snapshot ATP rank/points introuvable. "
                    f"Ajoutez un fichier CSV dans {ATP_PLAYER_RANK_POINTS_DIR} "
                    "ou instanciez MassPredictionEngine avec allow_rank_fallback=True."
                ) from exc

        if history_df is None:
            self.history_df = load_latest_history()
        else:
            self.history_df = history_df

        if max_history_rows is not None and len(self.history_df) > max_history_rows:
            self.history_df = self.history_df.tail(max_history_rows).reset_index(drop=True)


        self.history_tail = history_tail if history_tail is not None else len(self.history_df)


        self.players_catalog = load_players_catalog()
        self.tournaments_df = load_tournaments_catalog()
        self.latest_stats = compute_latest_player_stats(self.history_df)
        self.player_profiles = self._build_player_profiles()
        self.tournaments_index = self._build_tournaments_index()
        Path(MASS_PREDICTION_DIR).mkdir(parents=True, exist_ok=True)

        self._initialize_feature_artifacts(self.history_df)

    # -- Player resolution -------------------------------------------------

    def _build_player_profiles(self) -> Dict[str, PlayerProfile]:
        stats_lookup: Dict[str, Dict[str, Any]] = (
            self.latest_stats.to_dict("index") if not self.latest_stats.empty else {}
        )
        profiles: Dict[str, PlayerProfile] = {}
        alias_map: Dict[str, str] = {}

        for row in self.players_catalog.to_dict("records"):
            player_id = str(row.get("player_id") or row.get("id") or "").strip()
            if not player_id:
                continue
            stats_row = stats_lookup.get(player_id, {})
            display_name = (
                row.get("atpname")
                or row.get("player")
                or stats_row.get("name")
                or player_id
            )
            birthdate = row.get("birthdate")
            if isinstance(birthdate, str):
                birthdate = pd.to_datetime(birthdate, errors="coerce")
            height_cm = row.get("height")
            if isinstance(height_cm, str):
                height_cm = pd.to_numeric(height_cm, errors="coerce")

            aliases: List[str] = []
            for candidate in {
                display_name,
                row.get("player"),
                row.get("atpname"),
                stats_row.get("name"),
            }:
                if candidate and isinstance(candidate, str):
                    aliases.append(candidate)
            aliases = [alias for alias in aliases if alias]
            if not aliases:
                aliases.append(player_id)

            profile = PlayerProfile(
                player_id=player_id,
                display_name=str(display_name),
                aliases=tuple({normalize_text(alias) for alias in aliases}),
                country=(row.get("ioc") or stats_row.get("ioc")),
                hand=(row.get("hand") or stats_row.get("hand")),
                height_cm=float(height_cm) if height_cm is not None and not pd.isna(height_cm) else None,
                birthdate=birthdate if birthdate is not None and not pd.isna(birthdate) else None,
                latest_rank=float(stats_row.get("rank")) if stats_row.get("rank") is not None else None,
                latest_rank_points=float(stats_row.get("rank_points")) if stats_row.get("rank_points") is not None else None,
                age_latest=float(stats_row.get("age")) if stats_row.get("age") is not None else None,
            )
            profiles[player_id] = profile
            for alias in profile.aliases:
                alias_map.setdefault(alias, player_id)

        self._player_alias_map = alias_map
        self._player_profiles = profiles
        return profiles

    def _resolve_player(self, name: str) -> PlayerProfile:
        if not name:
            raise PredictionError("Nom de joueur manquant")
        normalized = normalize_text(name)
        player_id = self._player_alias_map.get(normalized)
        if player_id:
            return self._player_profiles[player_id]

        candidates = list(self._player_alias_map.keys())
        matches = difflib.get_close_matches(normalized, candidates, n=1, cutoff=0.75)
        if matches:
            player_id = self._player_alias_map[matches[0]]
            return self._player_profiles[player_id]
        raise PredictionError(f"Joueur introuvable dans le catalogue : {name}")

    # -- Tournament resolution --------------------------------------------

    def _build_tournaments_index(self) -> pd.DataFrame:
        df = self.tournaments_df.copy()
        df["_norm_identifier"] = df["identifier"].fillna("").map(lambda x: normalize_text(str(x)))
        df["_norm_name"] = df["name"].fillna("").map(lambda x: normalize_text(str(x)))
        df["_norm_city"] = df.get("city", "").fillna("").map(lambda x: normalize_text(str(x)))
        df["_norm_name_tokens"] = df["_norm_name"].map(lambda x: tuple(sorted(set(x.split()))))
        df["_norm_city_tokens"] = df["_norm_city"].map(lambda x: tuple(sorted(set(x.split()))))
        return df

    def _tournament_variants(self, name: str) -> List[str]:
        base = normalize_text(name)
        tokens = base.split()
        variants: Set[str] = {base}

        if tokens and tokens[0] in {"atp", "wta", "itf"}:
            variants.add(" ".join(tokens[1:]))

        no_digits = [token for token in tokens if not token.isdigit()]
        if no_digits:
            variants.add(" ".join(no_digits))
        if tokens and tokens[0] in {"atp", "wta", "itf"}:
            no_digits = [token for token in tokens[1:] if not token.isdigit()]
            if no_digits:
                variants.add(" ".join(no_digits))

        # Remove generic descriptors that often appear in betting sites
        stopwords = {"tour", "series", "masters", "open"}
        stripped = [token for token in tokens if token not in stopwords]
        if stripped:
            variants.add(" ".join(stripped))

        return [variant for variant in variants if variant]

    def _resolve_tournament(self, name: Optional[str], match_date: Optional[pd.Timestamp]) -> Dict[str, Any]:
        if not name:
            raise PredictionError("Tournoi non spécifié dans le screenshot")

        df = self.tournaments_index
        variants = self._tournament_variants(name)

        # Exact match shortcuts
        for variant in variants:
            for col in ("_norm_identifier", "_norm_name", "_norm_city"):
                mask = df[col] == variant
                if mask.any():
                    return df[mask].iloc[0].to_dict()

        best_score = 0.0
        best_row: Optional[pd.Series] = None

        for variant in variants:
            variant_tokens = set(variant.split())
            for _, row in df.iterrows():
                base_strings = [row.get("_norm_identifier"), row.get("_norm_name"), row.get("_norm_city")]
                base_strings = [s for s in base_strings if s]
                if not base_strings:
                    continue
                ratio = max(difflib.SequenceMatcher(None, variant, s).ratio() for s in base_strings)
                score = ratio

                # Boost if tokens align
                name_tokens = set(row.get("_norm_name_tokens", ()))
                city_tokens = set(row.get("_norm_city_tokens", ()))
                if variant_tokens and variant_tokens.issubset(name_tokens | city_tokens):
                    score += 0.2

                # Boost if date overlaps tournament window
                if match_date is not None:
                    start = row.get("start_date")
                    end = row.get("end_date")
                    if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                        if pd.notna(start) and pd.notna(end) and start <= match_date <= end:
                            score += 0.1

                if score > best_score:
                    best_score = score
                    best_row = row

        if best_row is None or best_score < 0.55:
            raise PredictionError(f"Tournoi introuvable dans le catalogue : {name}")

        return best_row.to_dict()

    # -- Prediction -------------------------------------------------------

    def _build_stub(
        self,
        player1: PlayerProfile,
        player2: PlayerProfile,
        tournament: Dict[str, Any],
        tournament_start: pd.Timestamp,
        match_date: pd.Timestamp,
    ) -> pd.DataFrame:
        tourney_level = tournament.get("tourney_level") or tournament.get("level")
        if not tourney_level or (isinstance(tourney_level, float) and np.isnan(tourney_level)):
            tourney_level = "A"
        tourney_level = str(tourney_level).strip() or "A"

        indoor_value = tournament.get("indoor_flag")
        if indoor_value is None:
            indoor_value = tournament.get("indoor") or tournament.get("indoor_outdoor")

        rank_a, points_a = self._resolve_rank_points(player1)
        rank_b, points_b = self._resolve_rank_points(player2)
        birthdate_a = player1.birthdate if isinstance(player1.birthdate, pd.Timestamp) else None
        birthdate_b = player2.birthdate if isinstance(player2.birthdate, pd.Timestamp) else None
        birthdate_a_str = birthdate_a.date().isoformat() if birthdate_a is not None and pd.notna(birthdate_a) else None
        birthdate_b_str = birthdate_b.date().isoformat() if birthdate_b is not None and pd.notna(birthdate_b) else None
        fallback_age_a = _safe_float(player1.age_latest, np.nan)
        fallback_age_b = _safe_float(player2.age_latest, np.nan)
        age_a_value = np.nan if birthdate_a_str else fallback_age_a
        age_b_value = np.nan if birthdate_b_str else fallback_age_b
        # Default stub (will be enriched with seed/entry if available)
        stub = {
            "tourney_id": tournament.get("identifier", "custom"),
            "tourney_name": tournament.get("name", tournament.get("identifier", "Custom Tournament")),
            "surface": tournament.get("surface", "Hard"),
            "draw_size": tournament.get("draw_size", 32),
            "tourney_level": tourney_level,
            "tourney_date": tournament_start,
            "round": tournament.get("round", "R32"),
            "best_of": tournament.get("best_of", 3),
            "match_num": 1,
            "player1_id": player1.player_id,
            "player2_id": player2.player_id,
            "player1_name": player1.display_name,
            "player2_name": player2.display_name,
            "player1_rank": rank_a,
            "player2_rank": rank_b,
            "player1_rank_points": points_a,
            "player2_rank_points": points_b,
            "player1_age": age_a_value,
            "player2_age": age_b_value,
            "player1_birthdate": birthdate_a_str,
            "player2_birthdate": birthdate_b_str,
            "player1_height": _safe_float(player1.height_cm, np.nan),
            "player2_height": _safe_float(player2.height_cm, np.nan),
            "player1_hand": player1.hand,
            "player2_hand": player2.hand,
            "player1_country": player1.country,
            "player2_country": player2.country,
            "player1_seed": np.nan,
            "player2_seed": np.nan,
            # Keep dataset semantics: empty string means Direct Acceptance (DA)
            "player1_entry": "",
            "player2_entry": "",
            "minutes": np.nan,
            "score": None,
            "match_date": match_date,
        }
        if indoor_value is not None and not (isinstance(indoor_value, float) and np.isnan(indoor_value)):
            try:
                indoor_numeric = float(indoor_value)
            except (TypeError, ValueError):
                indoor_normalized = str(indoor_value).strip().lower()
                if indoor_normalized in {"1", "true", "t", "y", "yes", "indoor"}:
                    indoor_numeric = 1.0
                elif indoor_normalized in {"0", "false", "f", "n", "no", "outdoor", "out"}:
                    indoor_numeric = 0.0
                else:
                    indoor_numeric = None
            else:
                indoor_numeric = 1.0 if indoor_numeric >= 1 else 0.0
            if indoor_numeric is not None:
                stub["indoor_flag"] = indoor_numeric
                stub["indoor"] = indoor_numeric

        # Ensure all neutral serve stats exist
        for neutral_col in NEUTRAL_TO_PIPELINE_MATCH_COLUMNS.keys():
            if neutral_col.startswith("player1_") or neutral_col.startswith("player2_"):
                stub.setdefault(neutral_col, np.nan)

        # Try to enrich with seeds/entries via tournament PDF (OpenAI) using our prepared catalog
        try:
            # Build a minimal catalog record list for the resolver
            catalog_records = self.tournaments_df.to_dict("records") if hasattr(self, "tournaments_df") else []
            t_name = stub.get("tourney_name") or tournament.get("identifier")
            players = [player1.display_name, player2.display_name]
            resolved_tourney_id, resolved_map, _ = resolve_seed_entry_for_players(
                t_name,
                match_date.to_pydatetime(),
                players,
                catalog_records,
                use_cache=True,
                force_refresh=False,
            )
            # Set a consistent touney_id if found
            if resolved_tourney_id:
                stub["tourney_id"] = resolved_tourney_id
            # Populate seeds and entries if available
            p1_info = resolved_map.get(player1.display_name)
            p2_info = resolved_map.get(player2.display_name)
            if p1_info is not None:
                if p1_info.seed is not None:
                    stub["player1_seed"] = int(p1_info.seed)
                # Map None (DA) to empty string to preserve bronze semantics
                stub["player1_entry"] = (p1_info.entry or "")
            if p2_info is not None:
                if p2_info.seed is not None:
                    stub["player2_seed"] = int(p2_info.seed)
                stub["player2_entry"] = (p2_info.entry or "")
        except Exception:
            # Non-fatal: keep defaults and proceed
            pass

        return pd.DataFrame([stub])

    def _combine_probabilities(self, preds_forward: pd.DataFrame, preds_reverse: pd.DataFrame) -> Dict[str, float]:
        prob_cols = [col for col in preds_forward.columns if col.startswith("P_")]
        combined: Dict[str, float] = {}
        for col in prob_cols:
            pf = preds_forward.iloc[0].get(col)
            pr = preds_reverse.iloc[0].get(col)
            if pf is None or pd.isna(pf) or pr is None or pd.isna(pr):
                continue
            combined[col] = float(0.5 * (float(pf) + (1.0 - float(pr))))
        return combined

    def _resolve_rank_points(self, profile: PlayerProfile) -> Tuple[float, float]:
        fallback_rank = _safe_float(profile.latest_rank, 0.0)
        fallback_points = _safe_float(profile.latest_rank_points, 0.0)
        provider = self.rank_provider
        if provider is None:
            return fallback_rank, fallback_points

        aliases = list(profile.aliases) if profile.aliases else []
        aliases.append(profile.display_name)
        entry = provider.get(profile.player_id, aliases)
        if entry is not None:
            return float(entry.rank), float(entry.points)

        if self.allow_rank_fallback:
            return fallback_rank, fallback_points

        raise PredictionError(
            f"Rank/points introuvables pour {profile.display_name} ({profile.player_id}). "
            "Complétez le snapshot ATP ou autorisez le fallback."
        )

    def predict(self, request: MatchInput) -> MatchPrediction:
        player1 = self._resolve_player(request.player1_name)
        player2 = self._resolve_player(request.player2_name)

        match_date_value = request.match_date or datetime.utcnow().date()
        match_timestamp = pd.to_datetime(match_date_value)

        tournament_info = self._resolve_tournament(request.tournament_name, match_timestamp)

        start_date_raw = tournament_info.get("start_date")
        tourney_timestamp: pd.Timestamp
        if isinstance(start_date_raw, pd.Timestamp):
            tourney_timestamp = start_date_raw if pd.notna(start_date_raw) else match_timestamp
        elif isinstance(start_date_raw, (datetime, date)):
            tourney_timestamp = pd.to_datetime(start_date_raw)
        elif start_date_raw is not None:
            parsed = pd.to_datetime(start_date_raw, errors="coerce")
            tourney_timestamp = parsed if pd.notna(parsed) else match_timestamp
        else:
            tourney_timestamp = match_timestamp

        stub_forward = self._build_stub(
            player1,
            player2,
            tournament_info,
            tourney_timestamp,
            match_timestamp,
        )
        stub_reverse = self._build_stub(
            player2,
            player1,
            tournament_info,
            tourney_timestamp,
            match_timestamp,
        )

        artifacts_forward = self._clone_feature_artifacts()
        artifacts_reverse = self._clone_feature_artifacts()

        features_forward = build_features_for_matches(
            history_raw=self.history_df,
            matches_raw=stub_forward,
            history_tail=self.history_tail,
            verbose=False,
            artifacts=artifacts_forward,
        )
        features_reverse = build_features_for_matches(
            history_raw=self.history_df,
            matches_raw=stub_reverse,
            history_tail=self.history_tail,
            verbose=False,
            artifacts=artifacts_reverse,
        )

        if features_forward.empty or features_reverse.empty:
            raise PredictionError("Impossible de construire les features pour ce match")

        features_forward = features_forward.assign(PLAYER_1=player1.player_id, PLAYER_2=player2.player_id)
        features_reverse = features_reverse.assign(PLAYER_1=player2.player_id, PLAYER_2=player1.player_id)

        preds_forward = predict_from_features(features_forward, self.artifacts)
        preds_reverse = predict_from_features(features_reverse, self.artifacts)

        if preds_forward.empty or preds_reverse.empty:
            raise PredictionError("Le modèle n'a pas renvoyé de probabilités")

        combined_probabilities = self._combine_probabilities(preds_forward, preds_reverse)
        if not combined_probabilities:
            raise PredictionError("Aucune probabilité exploitable")

        priority: List[str] = []
        if self.primary_prob_key:
            priority.append(self.primary_prob_key)
        priority.extend(["P_BLEND_PLATT", "P_WEIGHTED", "P_BLEND", "P_BLEND_ISO", "P_LGB", "P_XGB", "P_CAT", "P_HGB"])

        primary_key: Optional[str] = None
        seen: set[str] = set()
        for key in priority:
            if key in seen:
                continue
            seen.add(key)
            if key in combined_probabilities:
                primary_key = key
                break
        if primary_key is None:
            raise PredictionError("Aucune probabilité principale disponible")

        prob_player1 = combined_probabilities[primary_key]
        prob_player2 = max(0.0, min(1.0, 1.0 - prob_player1))
        fair_odds_player1 = float(1.0 / np.clip(prob_player1, 1e-6, 1 - 1e-6))
        fair_odds_player2 = float(1.0 / np.clip(prob_player2, 1e-6, 1 - 1e-6))

        return MatchPrediction(
            request=request,
            player1=player1,
            player2=player2,
            tournament=tournament_info,
            model_key=primary_key,
            prob_player1=float(prob_player1),
            prob_player2=float(prob_player2),
            fair_odds_player1=fair_odds_player1,
            fair_odds_player2=fair_odds_player2,
            combined_probabilities=combined_probabilities,
            raw_forward=preds_forward,
            raw_reverse=preds_reverse,
            features_forward=features_forward,
            features_reverse=features_reverse,
        )

    def _initialize_feature_artifacts(self, history_df: pd.DataFrame) -> None:
        self.history_df = history_df.reset_index(drop=True)
        self._feature_artifacts: Dict[str, object] = {}
        build_result_feature_dataset(self.history_df, verbose=False, artifacts=self._feature_artifacts)

    def reset_history(self, history_df: pd.DataFrame) -> None:
        self._initialize_feature_artifacts(history_df)
        self.latest_stats = compute_latest_player_stats(self.history_df)
        self.player_profiles = self._build_player_profiles()

    def _match_row_to_pipeline(self, match_row: pd.Series) -> pd.DataFrame:
        data = match_row.to_frame().T.copy()
        rename_map = {
            neutral: pipeline
            for neutral, pipeline in NEUTRAL_TO_PIPELINE_MATCH_COLUMNS.items()
            if neutral in data.columns and pipeline not in data.columns
        }
        if rename_map:
            data = data.rename(columns=rename_map)
        for neutral, pipeline in NEUTRAL_TO_PIPELINE_MATCH_COLUMNS.items():
            if pipeline not in data.columns:
                data[pipeline] = match_row.get(neutral, None)
        mirror_cols = [
            'best_of',
            'round',
            'draw_size',
            'tourney_level',
            'tourney_date',
            'tourney_name',
            'tourney_id',
            'surface',
            'match_num',
            'indoor',
            'indoor_flag',
            'score',
            'minutes',
        ]
        for col in mirror_cols:
            if col not in data.columns and col in match_row.index:
                data[col] = match_row[col]
        return data

    def update_state_with_match(self, match_row: pd.Series) -> None:
        match_df = self._match_row_to_pipeline(match_row)
        build_features_for_matches(
            history_raw=self.history_df,
            matches_raw=match_df,
            history_tail=self.history_tail,
            verbose=False,
            artifacts=self._feature_artifacts,
        )
        combined_row = match_row.to_frame().T.copy()
        for col in match_df.columns:
            combined_row[col] = match_df[col].iloc[0]
        self.history_df = pd.concat([self.history_df, combined_row], ignore_index=True, sort=False)
        self.latest_stats = compute_latest_player_stats(self.history_df)
        self.player_profiles = self._build_player_profiles()
    def _clone_feature_artifacts(self) -> Dict[str, object]:
        cloned: Dict[str, object] = {}
        for key, value in self._feature_artifacts.items():
            if hasattr(value, "clone"):
                cloned[key] = value.clone()
            elif isinstance(value, (dict, list, set, tuple)):
                cloned[key] = deepcopy(value)
            else:
                cloned[key] = value
        return cloned


__all__ = ["MassPredictionEngine", "MatchInput", "MatchPrediction", "PredictionError", "PlayerProfile"]

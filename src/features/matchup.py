from __future__ import annotations

from typing import List

import pandas as pd

from src.config import MATCHUP_WINDOWS


def add_matchup_scoring_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute matchup-level aggregates combining HOME_/AWAY_ rolling stats."""

    result = df.copy()
    _add_elo_matchup(result)

    for window in MATCHUP_WINDOWS:
        _add_pace_features(result, window)
        _add_scoring_features(result, window)
        _add_rating_gaps(result, window)
        _add_availability_gaps(result, window)
        _add_possession_pressure_features(result, window)
        _add_fastbreak_features(result, window)
        _add_shot_profile_gaps(result, window)
        _add_ft_pressure_features(result, window)
        _add_playmaking_features(result, window)
        _add_defense_pressure_features(result, window)
        _add_shot_variance_features(result, window)
        _add_player_personnel_features(result, window)
        _add_match_context_features(result, window)
        _add_game_state_features(result, window)

    return result


def _add_pace_features(df: pd.DataFrame, window: int) -> None:
    home_col = f"HOME_ROLL_pace_advanced_{window}"
    away_col = f"AWAY_ROLL_pace_advanced_{window}"
    if home_col in df.columns and away_col in df.columns:
        df[f"MATCH_PACE_{window}"] = (df[home_col] + df[away_col]) / 2
        df[f"PACE_DIFF_{window}"] = df[home_col] - df[away_col]


def _add_scoring_features(df: pd.DataFrame, window: int) -> None:
    home_pts = f"HOME_ROLL_points_traditional_{window}"
    away_pts = f"AWAY_ROLL_points_traditional_{window}"

    if home_pts in df.columns and away_pts in df.columns:
        df[f"TOTAL_POINTS_EXPECTED_{window}"] = df[home_pts] + df[away_pts]
        df[f"POINTS_DIFF_EXPECTED_{window}"] = df[home_pts] - df[away_pts]


def _add_rating_gaps(df: pd.DataFrame, window: int) -> None:
    home_off = f"HOME_ROLL_offensiveRating_advanced_{window}"
    away_def = f"AWAY_ROLL_defensiveRating_advanced_{window}"
    away_off = f"AWAY_ROLL_offensiveRating_advanced_{window}"
    home_def = f"HOME_ROLL_defensiveRating_advanced_{window}"

    if home_off in df.columns and away_def in df.columns:
        df[f"MATCH_OFF_DEF_GAP_{window}"] = df[home_off] - df[away_def]

    if away_off in df.columns and home_def in df.columns:
        df[f"MATCH_DEF_VS_OPP_OFF_{window}"] = df[home_def] - df[away_off]


def _add_elo_matchup(df: pd.DataFrame) -> None:
    if "HOME_ELO_DIFF" in df.columns:
        df["MATCH_ELO_GAP"] = df["HOME_ELO_DIFF"]

    if "HOME_ELO_DIFF_SEASON" in df.columns:
        df["MATCH_ELO_GAP_SEASON"] = df["HOME_ELO_DIFF_SEASON"]

    if "HOME_ELO_MEAN" in df.columns and "AWAY_ELO_MEAN" in df.columns:
        df["MATCH_ELO_LEVEL_AVG"] = (df["HOME_ELO_MEAN"] + df["AWAY_ELO_MEAN"]) / 2


def _add_availability_gaps(df: pd.DataFrame, window: int) -> None:
    rate = f"ROLL_top_player_absent_rate_{window}"
    home_col = f"HOME_{rate}"
    away_col = f"AWAY_{rate}"
    if home_col in df.columns and away_col in df.columns:
        df[f"MATCH_AVAILABILITY_GAP_{window}"] = df[home_col] - df[away_col]


def _add_possession_pressure_features(df: pd.DataFrame, window: int) -> None:
    orb_home = f"HOME_ROLL_percentageReboundsOffensive_usage_{window}"
    orb_away = f"AWAY_ROLL_percentageReboundsOffensive_usage_{window}"
    tov_home = f"HOME_ROLL_percentageTurnovers_usage_{window}"
    tov_away = f"AWAY_ROLL_percentageTurnovers_usage_{window}"

    orb_edge_col = f"MATCH_ORB_EDGE_{window}"
    tov_edge_col = f"MATCH_TOV_EDGE_{window}"

    orb_edge = None
    if orb_home in df.columns and orb_away in df.columns:
        orb_edge = df[orb_home] - df[orb_away]
        df[orb_edge_col] = orb_edge

    tov_edge = None
    if tov_home in df.columns and tov_away in df.columns:
        # Positive edge means home équipe protège mieux la balle (moins de turnovers).
        tov_edge = df[tov_away] - df[tov_home]
        df[tov_edge_col] = tov_edge

    components = []
    if orb_edge is not None:
        components.append(orb_edge.fillna(0))
    if tov_edge is not None:
        components.append(tov_edge.fillna(0))
    fastbreak_diff_col = f"MATCH_FASTBREAK_DIFF_{window}"
    if fastbreak_diff_col in df.columns:
        components.append(df[fastbreak_diff_col].fillna(0))

    if components:
        total = components[0]
        for comp in components[1:]:
            total = total + comp
        df[f"MATCH_EXTRA_POSSESSIONS_{window}"] = total


def _add_fastbreak_features(df: pd.DataFrame, window: int) -> None:
    fb_home = f"HOME_ROLL_pointsFastBreak_misc_{window}"
    fb_away = f"AWAY_ROLL_pointsFastBreak_misc_{window}"
    poss_home = f"HOME_ROLL_possessions_advanced_{window}"
    poss_away = f"AWAY_ROLL_possessions_advanced_{window}"

    if not all(col in df.columns for col in (fb_home, fb_away, poss_home, poss_away)):
        return

    home_rate = (df[fb_home] / df[poss_home].replace(0, pd.NA)).fillna(0)
    away_rate = (df[fb_away] / df[poss_away].replace(0, pd.NA)).fillna(0)

    df[f"MATCH_FASTBREAK_RATE_{window}"] = (home_rate + away_rate) / 2
    df[f"MATCH_FASTBREAK_DIFF_{window}"] = home_rate - away_rate


def _add_shot_profile_gaps(df: pd.DataFrame, window: int) -> None:
    specs = [
        ("percentageFieldGoalsAttempted3pt_scoring", "3PT_PROFILE"),
        ("percentagePointsPaint_scoring", "RIM_PROFILE"),
        ("percentagePointsMidrange2pt_scoring", "MIDRANGE_PROFILE"),
    ]
    location_components = []
    for base_col, label in specs:
        home_off = f"HOME_ROLL_{base_col}_{window}"
        away_def = f"AWAY_ROLL_OPP_{base_col}_{window}"
        away_off = f"AWAY_ROLL_{base_col}_{window}"
        home_def = f"HOME_ROLL_OPP_{base_col}_{window}"
        if all(col in df.columns for col in (home_off, away_def, away_off, home_def)):
            home_match = df[home_off] - df[away_def]
            away_match = df[away_off] - df[home_def]
            gap_col = f"MATCH_{label}_{window}"
            df[gap_col] = home_match - away_match
            location_components.append(df[gap_col].abs())

    if location_components:
        total = location_components[0]
        for comp in location_components[1:]:
            total = total + comp
        df[f"MATCH_LOCATION_RISK_{window}"] = total


def _add_ft_pressure_features(df: pd.DataFrame, window: int) -> None:
    ft_home = f"HOME_ROLL_freeThrowAttemptRate_fourfactors_{window}"
    ft_away = f"AWAY_ROLL_freeThrowAttemptRate_fourfactors_{window}"
    foul_home = f"HOME_ROLL_OPP_percentagePersonalFouls_usage_{window}"
    foul_away = f"AWAY_ROLL_OPP_percentagePersonalFouls_usage_{window}"

    if not all(col in df.columns for col in (ft_home, ft_away, foul_home, foul_away)):
        return

    home_pressure = (df[ft_home] + df[foul_away]) / 2
    away_pressure = (df[ft_away] + df[foul_home]) / 2
    df[f"MATCH_FT_PRESSURE_{window}"] = home_pressure - away_pressure
    avg_foul_rate = (df[foul_home] + df[foul_away]) / 2
    df[f"MATCH_FOUL_PRONE_{window}"] = avg_foul_rate
    df[f"MATCH_FOUL_RATE_DIFF_{window}"] = df[foul_home] - df[foul_away]
    df[f"MATCH_BONUS_RISK_{window}"] = (avg_foul_rate > 0.23).astype(int)


def _add_playmaking_features(df: pd.DataFrame, window: int) -> None:
    assist_home = f"HOME_ROLL_assistPercentage_advanced_{window}"
    assist_away = f"AWAY_ROLL_assistPercentage_advanced_{window}"
    assist_allowed_home = f"HOME_ROLL_OPP_percentageAssists_usage_{window}"
    assist_allowed_away = f"AWAY_ROLL_OPP_percentageAssists_usage_{window}"

    if all(col in df.columns for col in (assist_home, assist_away, assist_allowed_home, assist_allowed_away)):
        home_creation = df[assist_home] - df[assist_allowed_away]
        away_creation = df[assist_away] - df[assist_allowed_home]
        df[f"MATCH_CREATION_GAP_{window}"] = home_creation - away_creation


def _add_defense_pressure_features(df: pd.DataFrame, window: int) -> None:
    steals_home = f"HOME_ROLL_OPP_percentageSteals_usage_{window}"
    steals_away = f"AWAY_ROLL_OPP_percentageSteals_usage_{window}"
    force_home = f"HOME_ROLL_OPP_teamTurnoverPercentage_fourfactors_{window}"
    force_away = f"AWAY_ROLL_OPP_teamTurnoverPercentage_fourfactors_{window}"
    tov_home = f"HOME_ROLL_percentageTurnovers_usage_{window}"
    tov_away = f"AWAY_ROLL_percentageTurnovers_usage_{window}"

    blocks_home = f"HOME_ROLL_OPP_percentageBlocks_usage_{window}"
    blocks_away = f"AWAY_ROLL_OPP_percentageBlocks_usage_{window}"
    block_allowed_home = f"HOME_ROLL_percentageBlocksAllowed_usage_{window}"
    block_allowed_away = f"AWAY_ROLL_percentageBlocksAllowed_usage_{window}"

    if all(col in df.columns for col in (steals_home, steals_away, force_home, force_away, tov_home, tov_away)):
        home_pressure = df[steals_home] + df[force_home] - df[tov_away]
        away_pressure = df[steals_away] + df[force_away] - df[tov_home]
        df[f"MATCH_TURNOVER_PRESSURE_{window}"] = home_pressure - away_pressure

    if all(col in df.columns for col in (blocks_home, blocks_away, block_allowed_home, block_allowed_away)):
        home_rim = df[blocks_home] - df[block_allowed_away]
        away_rim = df[blocks_away] - df[block_allowed_home]
        df[f"MATCH_PROTECTION_EDGE_{window}"] = home_rim - away_rim


def _add_shot_variance_features(df: pd.DataFrame, window: int) -> None:
    home_std = f"HOME_ROLL_OPP_percentageFieldGoalsAttempted3pt_scoring_{window}_STD"
    away_std = f"AWAY_ROLL_OPP_percentageFieldGoalsAttempted3pt_scoring_{window}_STD"
    if home_std in df.columns and away_std in df.columns:
        df[f"MATCH_3PT_VARIANCE_{window}"] = df[home_std] + df[away_std]


def _add_player_personnel_features(df: pd.DataFrame, window: int) -> None:
    perf_home = f"HOME_ROLL_player_perf_score_sum_{window}"
    perf_away = f"AWAY_ROLL_player_perf_score_sum_{window}"
    if perf_home in df.columns and perf_away in df.columns:
        df[f"MATCH_PLAYER_PERF_GAP_{window}"] = df[perf_home] - df[perf_away]

    depth_home = f"HOME_ROLL_num_present_{window}"
    depth_away = f"AWAY_ROLL_num_present_{window}"
    if depth_home in df.columns and depth_away in df.columns:
        df[f"MATCH_ROTATION_DEPTH_{window}"] = df[depth_home] - df[depth_away]

    bench_ratio_home = f"HOME_ROLL_BENCH_USAGE_RATIO_{window}"
    bench_ratio_away = f"AWAY_ROLL_BENCH_USAGE_RATIO_{window}"
    if bench_ratio_home in df.columns and bench_ratio_away in df.columns:
        df[f"MATCH_BENCH_USAGE_GAP_{window}"] = df[bench_ratio_home] - df[bench_ratio_away]

    abs_home = f"HOME_ROLL_top_player_absent_rate_{window}"
    abs_away = f"AWAY_ROLL_top_player_absent_rate_{window}"
    if all(col in df.columns for col in (perf_home, perf_away, abs_home, abs_away)):
        home_loss = df[perf_home] * df[abs_home]
        away_loss = df[perf_away] * df[abs_away]
        df[f"MATCH_TOP_USAGE_MISSING_{window}"] = home_loss - away_loss


def _add_match_context_features(df: pd.DataFrame, window: int) -> None:
    if "HOME_IS_PLAYOFF" in df.columns and "AWAY_IS_PLAYOFF" in df.columns:
        df["MATCH_IS_PLAYOFF"] = ((df["HOME_IS_PLAYOFF"] == 1) | (df["AWAY_IS_PLAYOFF"] == 1)).astype(int)
    if "HOME_IS_IN_SEASON_TOURNAMENT" in df.columns and "AWAY_IS_IN_SEASON_TOURNAMENT" in df.columns:
        df["MATCH_IS_IN_SEASON_TOURNAMENT"] = (
            (df["HOME_IS_IN_SEASON_TOURNAMENT"] == 1) | (df["AWAY_IS_IN_SEASON_TOURNAMENT"] == 1)
        ).astype(int)
    if "HOME_IS_FINAL_WEEK" in df.columns and "AWAY_IS_FINAL_WEEK" in df.columns:
        df["MATCH_IS_FINAL_WEEK"] = (
            (df["HOME_IS_FINAL_WEEK"] == 1) | (df["AWAY_IS_FINAL_WEEK"] == 1)
        ).astype(int)

    playoff_home = f"HOME_ROLL_IS_PLAYOFF_{window}"
    playoff_away = f"AWAY_ROLL_IS_PLAYOFF_{window}"
    ist_home = f"HOME_ROLL_IS_IN_SEASON_TOURNAMENT_{window}"
    ist_away = f"AWAY_ROLL_IS_IN_SEASON_TOURNAMENT_{window}"
    final_home = f"HOME_ROLL_IS_FINAL_WEEK_{window}"
    final_away = f"AWAY_ROLL_IS_FINAL_WEEK_{window}"

    components = []
    for col in (playoff_home, playoff_away):
        if col in df.columns:
            components.append(df[col].fillna(0))
    for col in (ist_home, ist_away):
        if col in df.columns:
            components.append(0.5 * df[col].fillna(0))
    for col in (final_home, final_away):
        if col in df.columns:
            components.append(0.3 * df[col].fillna(0))

    if components:
        total = components[0]
        for comp in components[1:]:
            total = total + comp
        df[f"MATCH_IMPORTANCE_SCORE_{window}"] = total


def _add_game_state_features(df: pd.DataFrame, window: int) -> None:
    points_home = f"HOME_ROLL_points_traditional_{window}_STD"
    points_away = f"AWAY_ROLL_points_traditional_{window}_STD"
    pace_home = f"HOME_ROLL_pace_advanced_{window}_STD"
    pace_away = f"AWAY_ROLL_pace_advanced_{window}_STD"

    if points_home in df.columns and points_away in df.columns:
        var_points = (df[points_home].fillna(0) + df[points_away].fillna(0)) / 2
        df[f"MATCH_POINTS_VARIANCE_{window}"] = var_points
        if "MATCH_ELO_GAP" in df.columns:
            gap = df["MATCH_ELO_GAP"].abs().fillna(0)
            df[f"MATCH_BLOWOUT_RISK_{window}"] = (gap / 200).astype(float) * (1 / (1 + var_points))
            df[f"MATCH_CLUTCH_FLAG_{window}"] = ((gap < 30) & (var_points < 5)).astype(int)

    if pace_home in df.columns and pace_away in df.columns:
        var_pace = (df[pace_home].fillna(0) + df[pace_away].fillna(0)) / 2
        df[f"MATCH_PACE_VARIANCE_{window}"] = var_pace

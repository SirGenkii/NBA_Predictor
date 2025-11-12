from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import pandas as pd

from .availability import apply_availability_features
from .cleanup import drop_helper_columns, drop_raw_team_columns
from .matchup import add_matchup_scoring_features
from .targets import add_point_targets
from .team import apply_team_history_features


@dataclass(frozen=True)
class FeatureStep:
    name: str
    func: Callable[[pd.DataFrame], pd.DataFrame]
    stage: str
    description: str = ""
    targets: Sequence[str] = field(default_factory=lambda: ["IS_WIN", "POINT_DIFF", "POINT_TOTAL"])

    def summary(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage,
            "description": self.description,
            "targets": list(self.targets),
        }


@dataclass
class FeaturePlan:
    feature_steps: List[FeatureStep]
    target_steps: List[FeatureStep]

    def feature_functions(self) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
        return [step.func for step in self.feature_steps]

    def target_functions(self) -> List[Callable[[pd.DataFrame], pd.DataFrame]]:
        return [step.func for step in self.target_steps]

    def summary(self) -> dict:
        return {
            "feature_steps": [step.summary() for step in self.feature_steps],
            "target_steps": [step.summary() for step in self.target_steps],
        }


DEFAULT_FEATURE_PLAN = FeaturePlan(
    feature_steps=[
        FeatureStep(
            name="team_history",
            func=apply_team_history_features,
            stage="team_features",
            description="Convert match rows to team view, compute rolling stats (rest, win, H2H, Elo).",
        ),
        FeatureStep(
            name="availability_rollups",
            func=apply_availability_features,
            stage="availability_features",
            description="Leak-safe rolling absence/injury rates for key players.",
        ),
        FeatureStep(
            name="matchup_scoring",
            func=add_matchup_scoring_features,
            stage="matchup_features",
            description="Combine home/away rolling stats into matchup-level pace, total, and gap metrics.",
        ),
        FeatureStep(
            name="drop_raw_team_columns",
            func=drop_raw_team_columns,
            stage="cleanup",
            description="Remove raw per-game stat columns once rollups exist.",
        ),
        FeatureStep(
            name="drop_helper_columns",
            func=drop_helper_columns,
            stage="cleanup",
            description="Remove helper columns used only during feature construction.",
        ),
    ],
    target_steps=[
        FeatureStep(
            name="point_targets",
            func=add_point_targets,
            stage="targets",
            description="Derive POINTS_FOR/AGAINST, POINT_DIFF, POINT_TOTAL, and IS_WIN for match rows.",
        ),
    ],
)

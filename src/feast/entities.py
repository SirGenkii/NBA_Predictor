from __future__ import annotations

from feast import Entity, ValueType


match_entity = Entity(
    name="match_id",
    join_keys=["match_id"],
    description="Unique identifier for a game (mirrors GAME_ID).",
    value_type=ValueType.STRING,
)


team_entity = Entity(
    name="team_id",
    join_keys=["team_id"],
    description="NBA team identifier used for rolling/team features.",
    value_type=ValueType.STRING,
)

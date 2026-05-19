"""Model for the mc_sighting_events table — sighting records updated each tick."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class SightingEvent:
    """A single sighting record: one team spots an enemy player at a given tick.

    Attributes:
        demo_id:           Integer surrogate key (FK → demo_matches).
        round_num:         Round number (1-based).
        tick:              Game tick of the sighting.
        spotter_team:      ``'A'`` or ``'B'`` — the team doing the spotting.
        spotter_player_id: Integer surrogate key of the individual player who spotted
                           the enemy, or ``None`` if attribution is unavailable.
        spotted_player_id: Integer surrogate key of the spotted enemy.
        area_id:          Nav-mesh area where the spotted player was last seen;
                          ``None`` if unmapped.
    """

    demo_id: int = 0
    round_num: int = 0
    tick: int = 0
    spotter_team: str = ""
    spotter_player_id: Optional[int] = None
    spotted_player_id: int = 0
    area_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "demo_id": self.demo_id,
            "round_num": self.round_num,
            "tick": self.tick,
            "spotter_team": self.spotter_team,
            "spotter_player_id": self.spotter_player_id,
            "spotted_player_id": self.spotted_player_id,
            "area_id": self.area_id,
        }

    @staticmethod
    def from_dict(data: dict) -> "SightingEvent":
        return SightingEvent(
            demo_id=data.get("demo_id", 0),
            round_num=data.get("round_num", 0),
            tick=data.get("tick", 0),
            spotter_team=data.get("spotter_team", ""),
            spotter_player_id=data.get("spotter_player_id"),
            spotted_player_id=data.get("spotted_player_id", 0),
            area_id=data.get("area_id"),
        )

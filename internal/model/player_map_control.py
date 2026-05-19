"""Per-player map control attribution stats for a single round or game average."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PlayerMapControlStats:
    """Map control contribution metrics for one player in one round.

    All *_pct fields are percentages of total navigable map area (0–100).

    Attributes:
        player_id:             Integer surrogate key (FK → players).
        round_num:             Round number (1-based); 0 for game-wide aggregates.
        avg_active_pct:        Mean % of map inside this player's vision cone per
                               alive sample tick.
        avg_unique_pct:        Mean % only this player's cone covers (no teammate
                               overlap) per alive sample tick.  Always ≤ avg_active_pct.
        avg_denial_pct:        Mean % of the enemy mobility bubble that falls inside
                               this player's vision cone per alive sample tick.  Measures
                               how many routes the player is locking down.
        total_clearance_pct:   % of map first opened by this player this round
                               (i.e. areas that entered cleared_by_team for the first
                               time because of this player's cone).
        passive_attributed_pct: % of map that was passively held by the team (at the
                                moment of attribution) whose first clearer was this
                                player.  Reflects how durable the player's clearances
                                are.
        death_impact_pct:      Unique coverage (avg_unique_pct at that tick scaled to
                               map %) at the last alive sample tick before the player
                               died.  Zero when the player survived the round.
        survived:              True when the player was alive at round end.
        round_alive_pct:       Percentage of the round's sample ticks the player was alive (0–100).
    """

    player_id: int = 0
    round_num: int = 0
    team: str = ""  # "A" or "B"

    avg_active_pct: float = 0.0
    avg_unique_pct: float = 0.0
    avg_denial_pct: float = 0.0
    total_clearance_pct: float = 0.0
    passive_attributed_pct: float = 0.0
    death_impact_pct: float = 0.0

    survived: bool = False
    round_alive_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "round_num": self.round_num,
            "team": self.team,
            "avg_active_pct": self.avg_active_pct,
            "avg_unique_pct": self.avg_unique_pct,
            "avg_denial_pct": self.avg_denial_pct,
            "total_clearance_pct": self.total_clearance_pct,
            "passive_attributed_pct": self.passive_attributed_pct,
            "death_impact_pct": self.death_impact_pct,
            "survived": self.survived,
            "round_alive_pct": self.round_alive_pct,
        }

    @staticmethod
    def from_dict(data: dict) -> "PlayerMapControlStats":
        return PlayerMapControlStats(
            player_id=data.get("player_id", 0),
            round_num=data.get("round_num", 0),
            team=data.get("team", ""),
            avg_active_pct=data.get("avg_active_pct", 0.0),
            avg_unique_pct=data.get("avg_unique_pct", 0.0),
            avg_denial_pct=data.get("avg_denial_pct", 0.0),
            total_clearance_pct=data.get("total_clearance_pct", 0.0),
            passive_attributed_pct=data.get("passive_attributed_pct", 0.0),
            death_impact_pct=data.get("death_impact_pct", 0.0),
            survived=data.get("survived", False),
            round_alive_pct=data.get("round_alive_pct", 0.0),
        )

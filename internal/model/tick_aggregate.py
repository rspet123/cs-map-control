"""Model for the mc_tick_aggregates table — per-sample-tick team percentages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TickAggregate:
    """Team-level map control percentages at a single sample tick within a round.

    All *_pct fields are percentages of total navigable map area (0–100).

    Attributes:
        demo_id:            Integer surrogate key (FK → demo_matches).
        round_num:          Round number (1-based).
        tick:               Game tick of this sample.
        team_a_active_pct:  % of map actively controlled by team A.
        team_a_passive_pct: % of map passively held by team A.
        team_b_active_pct:  % of map actively controlled by team B.
        team_b_passive_pct: % of map passively held by team B.
        contested_pct:      % of map contested between both teams.
        neutral_pct:        % of map with no control attribution.
    """

    demo_id: int = 0
    round_num: int = 0
    tick: int = 0
    team_a_active_pct: float = 0.0
    team_a_passive_pct: float = 0.0
    team_b_active_pct: float = 0.0
    team_b_passive_pct: float = 0.0
    contested_pct: float = 0.0
    neutral_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "demo_id": self.demo_id,
            "round_num": self.round_num,
            "tick": self.tick,
            "team_a_active_pct": self.team_a_active_pct,
            "team_a_passive_pct": self.team_a_passive_pct,
            "team_b_active_pct": self.team_b_active_pct,
            "team_b_passive_pct": self.team_b_passive_pct,
            "contested_pct": self.contested_pct,
            "neutral_pct": self.neutral_pct,
        }

    @staticmethod
    def from_dict(data: dict) -> "TickAggregate":
        return TickAggregate(
            demo_id=data.get("demo_id", 0),
            round_num=data.get("round_num", 0),
            tick=data.get("tick", 0),
            team_a_active_pct=data.get("team_a_active_pct", 0.0),
            team_a_passive_pct=data.get("team_a_passive_pct", 0.0),
            team_b_active_pct=data.get("team_b_active_pct", 0.0),
            team_b_passive_pct=data.get("team_b_passive_pct", 0.0),
            contested_pct=data.get("contested_pct", 0.0),
            neutral_pct=data.get("neutral_pct", 0.0),
        )

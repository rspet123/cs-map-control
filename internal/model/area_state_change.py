"""Model for the mc_area_state_changes table — written when an area's control state changes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AreaStateChange:
    """A single control-state transition for one nav area in one round.

    ``from_state`` / ``to_state`` are one of:
    ``'neutral'``, ``'contested'``, ``'a_active'``, ``'b_active'``,
    ``'a_passive'``, ``'b_passive'``.

    To reconstruct area state at tick T query all rows with
    ``area_id=X AND tick <= T`` ordered by tick descending and take the first.
    Areas with no rows were ``'neutral'`` from round start.

    Attributes:
        demo_id:               Integer surrogate key (FK → demo_matches).
        round_num:             Round number (1-based).
        area_id:               Nav-mesh area identifier.
        tick:                  Game tick at which the transition occurred.
        from_state:            Control state before the transition.
        to_state:              Control state after the transition.
        in_a_mobility:         True if a team-A player could reach this area at transition time.
        in_b_mobility:         True if a team-B player could reach this area at transition time.
        responsible_steamid:   Steam-ID64 of the player most responsible for the transition,
                               or empty string when attribution is ambiguous.
    """

    demo_id: int = 0
    round_num: int = 0
    area_id: int = 0
    tick: int = 0
    from_state: str = "neutral"
    to_state: str = "neutral"
    in_a_mobility: bool = False
    in_b_mobility: bool = False
    responsible_steamid: str = ""

    def to_dict(self) -> dict:
        return {
            "demo_id": self.demo_id,
            "round_num": self.round_num,
            "area_id": self.area_id,
            "tick": self.tick,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "in_a_mobility": self.in_a_mobility,
            "in_b_mobility": self.in_b_mobility,
            "responsible_steamid": self.responsible_steamid,
        }

    @staticmethod
    def from_dict(data: dict) -> "AreaStateChange":
        return AreaStateChange(
            demo_id=data.get("demo_id", 0),
            round_num=data.get("round_num", 0),
            area_id=data.get("area_id", 0),
            tick=data.get("tick", 0),
            from_state=data.get("from_state", "neutral"),
            to_state=data.get("to_state", "neutral"),
            in_a_mobility=data.get("in_a_mobility", False),
            in_b_mobility=data.get("in_b_mobility", False),
            responsible_steamid=data.get("responsible_steamid", ""),
        )

"""Model for the mc_tick_players table — per-player position + attribution snapshot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TickPlayer:
    """Player position and map-control attribution at a single sample tick.

    Attributes:
        demo_id:     Integer surrogate key (FK → demo_matches).
        round_num:   Round number (1-based).
        tick:        Game tick of this sample.
        player_id:   Integer surrogate key (FK → players).
        team:        ``'A'`` or ``'B'``.
        x:           Player world X coordinate.
        y:           Player world Y coordinate.
        z:           Player world Z coordinate.
        yaw:         View angle yaw (degrees).
        pitch:       View angle pitch (degrees).
        area_id:     Nav-mesh area the player occupies; ``None`` if unmapped.
        health:      Player health (0–100).
        active_size: Raw nav-area units inside this player's vision cone.
        unique_size: Nav-area units only this player watches (no teammate overlap).
        denial_size: Enemy-mobility units inside this player's vision cone.
    """

    demo_id: int = 0
    round_num: int = 0
    tick: int = 0
    player_id: int = 0
    team: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    area_id: int | None = None
    health: int = 0
    active_size: float = 0.0
    unique_size: float = 0.0
    denial_size: float = 0.0

    def to_dict(self) -> dict:
        return {
            "demo_id": self.demo_id,
            "round_num": self.round_num,
            "tick": self.tick,
            "player_id": self.player_id,
            "team": self.team,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "area_id": self.area_id,
            "health": self.health,
            "active_size": self.active_size,
            "unique_size": self.unique_size,
            "denial_size": self.denial_size,
        }

    @staticmethod
    def from_dict(data: dict) -> "TickPlayer":
        return TickPlayer(
            demo_id=data.get("demo_id", 0),
            round_num=data.get("round_num", 0),
            tick=data.get("tick", 0),
            player_id=data.get("player_id", 0),
            team=data.get("team", ""),
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0),
            yaw=data.get("yaw", 0.0),
            pitch=data.get("pitch", 0.0),
            area_id=data.get("area_id"),
            health=data.get("health", 0),
            active_size=data.get("active_size", 0.0),
            unique_size=data.get("unique_size", 0.0),
            denial_size=data.get("denial_size", 0.0),
        )

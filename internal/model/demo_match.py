"""Model for the demo_matches table — one row per parsed demo file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class DemoMatch:
    """Metadata for a single parsed CS demo file.

    Attributes:
        demo_id:       Auto-increment integer primary key.
        demo_hash:     SHA-256 hex digest of the demo file (unique key).
        demo_filename: Original filename of the demo.
        map_name:      Map played (e.g. ``de_dust2``).
        created_at:    Timestamp of insertion; ``None`` when not yet persisted.
    """

    demo_id: int = 0
    demo_hash: str = ""
    demo_filename: str = ""
    map_name: str = ""
    created_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "demo_id": self.demo_id,
            "demo_hash": self.demo_hash,
            "demo_filename": self.demo_filename,
            "map_name": self.map_name,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "DemoMatch":
        return DemoMatch(
            demo_id=data.get("demo_id", 0),
            demo_hash=data.get("demo_hash", ""),
            demo_filename=data.get("demo_filename", ""),
            map_name=data.get("map_name", ""),
            created_at=data.get("created_at"),
        )

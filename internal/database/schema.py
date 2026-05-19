"""SQLAlchemy Core table definitions for the map control DB schema.

Call ``create_all(engine)`` once at application startup to create any missing
tables and apply pending column migrations.  Safe to call repeatedly.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    TIMESTAMP,
    text,
)
from sqlalchemy.dialects.mysql import SMALLINT as MySQLSmallInt
from sqlalchemy.engine import Engine

metadata = MetaData()

# ---------------------------------------------------------------------------
# demo_matches — one row per parsed demo file, keyed by auto-increment ID.
# ---------------------------------------------------------------------------
demo_matches = Table(
    "demo_matches",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), primary_key=True, autoincrement=True),
    Column("demo_hash", String(64), nullable=False, unique=True),
    Column("demo_filename", String(512), nullable=False),
    Column("map_name", String(64), nullable=False, server_default=""),
    Column(
        "status",
        Enum("processing", "done", "failed", name="demo_status"),
        nullable=False,
        server_default="done",
    ),
    Column("created_at", TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")),
    Column("sample_interval", MySQLSmallInt(unsigned=True), nullable=True),
)

# ---------------------------------------------------------------------------
# players — one row per unique Steam account seen across all parsed demos.
# ---------------------------------------------------------------------------
players = Table(
    "players",
    metadata,
    Column("player_id", MySQLSmallInt(unsigned=True), primary_key=True, autoincrement=True),
    Column("steamid", String(32), nullable=False, unique=True),
    Column("name", String(128), nullable=False),
)

# ---------------------------------------------------------------------------
# mc_rounds — per-round team-level aggregate percentages.
# ---------------------------------------------------------------------------
mc_rounds = Table(
    "mc_rounds",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("team_a_active_pct", Float),
    Column("team_a_passive_pct", Float),
    Column("team_b_active_pct", Float),
    Column("team_b_passive_pct", Float),
    Column("contested_pct", Float),
    Column("neutral_pct", Float),
    Column("winner_team", String(1), nullable=True),
)

# ---------------------------------------------------------------------------
# mc_demo_teams — one row per player per demo, recording team assignment and
# the side (T/CT) that player's team started on in round 1.  Written once per
# demo immediately before round analysis begins.
# ---------------------------------------------------------------------------
mc_demo_teams = Table(
    "mc_demo_teams",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("player_id", MySQLSmallInt(unsigned=True), ForeignKey("players.player_id", ondelete="CASCADE"), primary_key=True),
    Column("team", String(1), nullable=False),
    Column("starting_side", String(2), nullable=False),
)

# ---------------------------------------------------------------------------
# mc_round_sides — one row per round recording which side team A was on.
# team_b_side is always the complement (T↔CT).
# ---------------------------------------------------------------------------
mc_round_sides = Table(
    "mc_round_sides",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("team_a_side", String(2), nullable=False),
)

# ---------------------------------------------------------------------------
# mc_tick_aggregates — per-sample-tick team percentages (time series).
# ---------------------------------------------------------------------------
mc_tick_aggregates = Table(
    "mc_tick_aggregates",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("tick", Integer, primary_key=True),
    Column("team_a_active_pct", Float),
    Column("team_a_passive_pct", Float),
    Column("team_b_active_pct", Float),
    Column("team_b_passive_pct", Float),
    Column("contested_pct", Float),
    Column("neutral_pct", Float),
)

# ---------------------------------------------------------------------------
# mc_area_state_changes — written only when an area's control state changes.
# ---------------------------------------------------------------------------
_area_state_enum = Enum(
    "neutral", "contested", "a_active", "b_active", "a_passive", "b_passive",
    name="area_control_state",
)

# ---------------------------------------------------------------------------
# mc_area_round_stats — per-area, per-round aggregate tick counts.
# Replaces mc_area_state_changes for area-importance analysis at ~50x fewer rows.
# Only areas that were ever non-neutral are stored; ticks_neutral can be derived
# as: ticks_sampled - ticks_a_ctrl - ticks_b_ctrl - ticks_contested.
# ---------------------------------------------------------------------------
mc_area_round_stats = Table(
    "mc_area_round_stats",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("area_id", Integer, primary_key=True),
    Column("ticks_a_ctrl", SmallInteger, nullable=False, server_default="0"),
    Column("ticks_b_ctrl", SmallInteger, nullable=False, server_default="0"),
    Column("ticks_contested", SmallInteger, nullable=False, server_default="0"),
    Column("ticks_sampled", SmallInteger, nullable=False, server_default="0"),
)

# ---------------------------------------------------------------------------
# mc_tick_players — per-player position + attribution snapshot each sample tick.
# ---------------------------------------------------------------------------
mc_tick_players = Table(
    "mc_tick_players",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("tick", Integer, primary_key=True),
    Column("player_id", MySQLSmallInt(unsigned=True), primary_key=True),
    Column("team", String(1), nullable=False),
    Column("x", Float),
    Column("y", Float),
    Column("z", Float),
    Column("yaw", Float),
    Column("pitch", Float),
    Column("area_id", Integer),
    Column("health", Integer),
    Column("active_size", Float),
    Column("unique_size", Float),
    Column("denial_size", Float),
)

# ---------------------------------------------------------------------------
# mc_player_rounds — end-of-round attribution stats per player.
# ---------------------------------------------------------------------------
mc_player_rounds = Table(
    "mc_player_rounds",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("player_id", MySQLSmallInt(unsigned=True), primary_key=True),
    Column("team", String(1), nullable=False),
    Column("avg_active_pct", Float),
    Column("avg_unique_pct", Float),
    Column("avg_denial_pct", Float),
    Column("total_clearance_pct", Float),
    Column("passive_attributed_pct", Float),
    Column("death_impact_pct", Float),
    Column("survived", Boolean),
    Column("round_alive_pct", Float),
)

# ---------------------------------------------------------------------------
# mc_sighting_events — last-seen record updates written each sample tick.
# ---------------------------------------------------------------------------
mc_sighting_events = Table(
    "mc_sighting_events",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("tick", Integer, primary_key=True),
    Column("spotter_team", String(1), primary_key=True),
    Column("spotted_player_id", MySQLSmallInt(unsigned=True), primary_key=True),
    Column("spotter_player_id", MySQLSmallInt(unsigned=True), nullable=True),
    Column("area_id", Integer),
)

# ---------------------------------------------------------------------------
# bomb_plants — one row per round in which the T-side planted the bomb.
# ---------------------------------------------------------------------------
bomb_plants = Table(
    "bomb_plants",
    metadata,
    Column("demo_id", MySQLSmallInt(unsigned=True), ForeignKey("demo_matches.demo_id", ondelete="CASCADE"), primary_key=True),
    Column("round_num", SmallInteger, primary_key=True),
    Column("player_id", MySQLSmallInt(unsigned=True), ForeignKey("players.player_id", ondelete="SET NULL"), nullable=True),
    Column("site", String(1), nullable=True),
    Column("x", Float, nullable=True),
    Column("y", Float, nullable=True),
    Column("z", Float, nullable=True),
    Column("detonated", Boolean, nullable=True),
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def create_all(engine: Engine) -> None:
    """Create all missing tables.

    Safe to call on every startup — ``checkfirst=True`` skips tables that
    already exist.
    """
    # MySQL's has_table() asserts schema is not None.  In some SQLAlchemy
    # versions _can_create_table reads dialect.default_schema_name directly
    # (not via schema_translate_map), so we set it explicitly from the URL
    # before issuing any DDL.  This is safe: dialect.initialize() would set
    # it to the same value; we're just guaranteeing it's present.
    engine.dialect.default_schema_name = engine.url.database
    with engine.connect() as conn:
        metadata.create_all(conn, checkfirst=True)

if __name__ == "__main__":
    import os
    import sys
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL
    from dotenv import load_dotenv

    load_dotenv()

    url = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "statistics"),
    )
    engine = create_engine(url)
    create_all(engine)
    print("Database schema is up to date.")
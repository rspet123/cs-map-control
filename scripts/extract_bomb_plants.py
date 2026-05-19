"""Back-fill bomb plant locations for all already-processed demos.

For every demo with status='done', parses the demo (without running any
map-control analysis), extracts the per-round bomb plant event (x, y, z,
site, planting player, detonated), and writes one row per planted round into
the bomb_plants table.

Safe to re-run: INSERT IGNORE skips rows that already exist.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root on sys.path so internal imports resolve
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import polars as pl
from awpy import Demo

from internal.database.connections import DBConnections
from internal.database.map_control_writer import ensure_players, upsert_players
from internal.database.schema import create_all

# ---------------------------------------------------------------------------
# Configuration — must match main.py
# ---------------------------------------------------------------------------

DEMO_DIR = Path(r"Z:\AnalysisDemos")
_PLAYER_PROPS = ["health", "team_num", "X", "Y", "Z", "yaw", "pitch", "name"]

# CS2 bomb_planted event sends site as an integer (0=A, 1=B).
# AWPY may or may not convert this; handle both forms.
_SITE_MAP: dict[str, str] = {"0": "A", "1": "B"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_site(raw) -> Optional[str]:
    """Return a single-character site string ('A' or 'B'), or None.

    AWPY exposes bombsite as the CS2 place name (e.g. 'BombsiteA', 'BombsiteB')
    so we scan from the end of the string for the first 'A' or 'B' character.
    Integer forms ('0'/'1') are also handled via _SITE_MAP.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Integer form: '0' -> 'A', '1' -> 'B'
    mapped = _SITE_MAP.get(s)
    if mapped:
        return mapped
    # String form: scan from the end for 'A' or 'B' (handles 'BombsiteA', 'BombsiteB', etc.)
    for ch in reversed(s.upper()):
        if ch in ("A", "B"):
            return ch
    return None


def _extract_plant_records(
    demo_id: int,
    bomb_df: pl.DataFrame,
    player_id_map: dict[str, int],
) -> list[dict]:
    """Return one record per planted round derived from *bomb_df*."""
    if bomb_df.is_empty():
        return []

    plant_rows = bomb_df.filter(pl.col("event") == "plant")
    if plant_rows.is_empty():
        return []

    detonated_rounds: set = set(
        bomb_df.filter(pl.col("event") == "detonate")["round_num"].to_list()
    )

    records = []
    for row in plant_rows.iter_rows(named=True):
        steamid = str(row.get("steamid") or "")
        records.append({
            "demo_id": demo_id,
            "round_num": row["round_num"],
            "player_id": player_id_map.get(steamid),
            "site": _normalise_site(row.get("bombsite")),
            "x": row.get("X"),
            "y": row.get("Y"),
            "z": row.get("Z"),
            "detonated": int(row["round_num"] in detonated_rounds),
        })
    return records


def _insert_plants(db_pool, records: list[dict]) -> None:
    if not records:
        return
    sql = (
        "INSERT IGNORE INTO bomb_plants "
        "(demo_id, round_num, player_id, site, x, y, z, detonated) "
        "VALUES (%(demo_id)s, %(round_num)s, %(player_id)s, %(site)s, "
        "%(x)s, %(y)s, %(z)s, %(detonated)s)"
    )
    cursor = db_pool.cursor()
    try:
        cursor.executemany(sql, records)
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Per-demo processing
# ---------------------------------------------------------------------------

def process_demo(demo_id: int, filename: str, db_pool) -> int:
    """Parse one demo and insert its bomb plant rows.

    Returns the number of rows written (may be 0 if no plant occurred or the
    file is missing).
    """
    path = DEMO_DIR / filename
    if not path.exists():
        logger.warning("[SKIP] File not found: %s", filename)
        return 0

    logger.info("[PARSE] %s (demo_id=%d)", filename, demo_id)
    demo = Demo(path=path)
    demo.parse(player_props=_PLAYER_PROPS)

    ticks_df: Optional[pl.DataFrame] = getattr(demo, "ticks", None)
    if ticks_df is None or ticks_df.is_empty():
        logger.warning("[SKIP] No tick data: %s", filename)
        return 0

    all_steamids = (
        ticks_df
        .select(pl.col("steamid").cast(pl.Utf8))
        .unique()["steamid"]
        .to_list()
    )
    player_id_map = ensure_players(db_pool, all_steamids)

    if "name" in ticks_df.columns:
        player_name_records = (
            ticks_df
            .select([
                pl.col("steamid").cast(pl.Utf8),
                pl.col("name").cast(pl.Utf8),
                pl.col("tick"),
            ])
            .sort("tick", descending=True)
            .unique(subset=["steamid"], keep="first")
            .select(["steamid", "name"])
            .to_dicts()
        )
        upsert_players(db_pool, player_name_records)

    bomb_df: pl.DataFrame = demo.bomb
    records = _extract_plant_records(demo_id, bomb_df, player_id_map)
    _insert_plants(db_pool, records)
    logger.info("[DONE] %s — %d plant(s) written", filename, len(records))
    return len(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    db = DBConnections()
    if not db.connect_prefire_db():
        logger.error("Database connection failed — aborting.")
        sys.exit(1)
    db_pool = db.prefire_db
    assert db_pool is not None  # guaranteed by connect_prefire_db() returning True

    # Create bomb_plants (and any other missing tables) if not yet present.
    create_all(db_pool._engine)

    cursor = db_pool.cursor()
    try:
        cursor.execute(
            "SELECT demo_id, demo_filename FROM demo_matches "
            "WHERE status = 'done' ORDER BY demo_id"
        )
        demos = cursor.fetchall()
    finally:
        cursor.close()

    if not demos:
        logger.info("No processed demos found.")
        return

    logger.info("Found %d processed demo(s) to backfill.", len(demos))
    total_plants = 0
    for demo_id, filename in demos:
        try:
            total_plants += process_demo(int(demo_id), filename, db_pool)
        except Exception:
            logger.exception("[FAILED] demo_id=%d  file=%s", demo_id, filename)

    logger.info("Backfill complete. Total plant rows written: %d", total_plants)


if __name__ == "__main__":
    main()

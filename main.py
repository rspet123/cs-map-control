"""Demo scanner and map-control analysis pipeline.

Scans ``DEMO_DIR`` for ``.dem`` files, skips any already stored in the
database (status 'processing' or 'done'), and runs a multithreaded analysis
on the remainder using ``MAX_WORKERS`` concurrent threads.

Each demo is processed in isolation:
  1. Marked as 'processing' in ``demo_matches``.
  2. Parsed with AWPY.
  3. Geometry loaded / retrieved from a per-map shared cache.
  4. ``MapControlService.compute()`` run, writing analytics to the DB
     asynchronously via a shared ``MapControlDBWriter``.
  5. Player names upserted into the ``players`` table.
  6. Status updated to 'done' (or 'failed' on any unhandled exception).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import logfire
import polars as pl
from awpy import Demo
from dotenv import load_dotenv

load_dotenv()

from internal.database.connections import DBConnections
from internal.database.map_control_writer import MapControlDBWriter, ensure_players, upsert_players
from internal.database.schema import create_all
from internal.statistics.geometry import GeometryService
from internal.statistics.mapcontrol import MapControlService, SAMPLE_INTERVAL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_DIR = Path(os.getenv("DEMO_DIR", r"Z:\AnalysisDemos"))
_PLAYER_PROPS = ["health", "team_num", "X", "Y", "Z", "yaw", "pitch", "name"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared geometry cache (thread-safe via double-checked locking)
# ---------------------------------------------------------------------------

_geo_cache: Dict[str, GeometryService] = {}
_geo_lock = threading.Lock()


def _get_geometry(map_name: str) -> GeometryService:
    """Return the cached ``GeometryService`` for *map_name*, creating it if needed.

    The first call for a given map name initialises the nav mesh and
    pre-computes the visibility matrix — both are expensive operations that
    must only happen once per map across all worker threads.
    """
    geo = _geo_cache.get(map_name)
    if geo is not None:
        return geo
    with _geo_lock:
        geo = _geo_cache.get(map_name)
        if geo is not None:
            return geo
        logger.info("Loading geometry for map '%s'...", map_name)
        t0 = time.perf_counter()
        geo = GeometryService(map_name)
        geo.precompute_visibility()
        _geo_cache[map_name] = geo
        logger.info(
            "Geometry ready for '%s' (%.1fs) - nav=%s vis=%s",
            map_name, time.perf_counter() - t0,
            geo.is_nav_available, geo.is_vis_matrix_available,
        )
    return geo


# ---------------------------------------------------------------------------
# SHA-256 hashing
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*, reading in 8 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Demo scanner
# ---------------------------------------------------------------------------

def _already_processed(db_pool) -> Set[str]:
    """Return the set of demo hashes that are 'processing' or 'done'."""
    cursor = db_pool.cursor()
    try:
        cursor.execute(
            "SELECT demo_hash FROM demo_matches WHERE status IN ('processing', 'done')"
        )
        return {row[0] for row in cursor.fetchall()}
    finally:
        cursor.close()


def scan_new_demos(db_pool, directory: Path) -> List[Tuple[Path, str]]:
    """Return ``(path, sha256_hash)`` pairs for demos not yet in the database."""
    done = _already_processed(db_pool)
    results: List[Tuple[Path, str]] = []
    for path in sorted(directory.glob("*.dem")):
        print(f"Checking {path.name}...")
        demo_hash = _sha256(path)
        if demo_hash not in done:
            results.append((path, demo_hash))
        else:
            logger.info("Skipping already-processed demo: %s", path.name)
    return results


# ---------------------------------------------------------------------------
# DB status helpers (synchronous, direct pool calls)
# ---------------------------------------------------------------------------

def _mark_processing(db_pool, demo_hash: str, filename: str) -> int:
    """Insert or mark processing and return the demo_id."""
    cursor = db_pool.cursor()
    try:
        cursor.execute(
            "INSERT INTO demo_matches (demo_hash, demo_filename, map_name, status) "
            "VALUES (%s, %s, '', 'processing') "
            "ON DUPLICATE KEY UPDATE status = 'processing', "
            "demo_id = LAST_INSERT_ID(demo_id)",
            (demo_hash, filename),
        )
        cursor.execute("SELECT LAST_INSERT_ID()")
        row = cursor.fetchone()
        return int(row[0])
    finally:
        cursor.close()


def _update_map_name(db_pool, demo_id: int, map_name: str) -> None:
    cursor = db_pool.cursor()
    try:
        cursor.execute(
            "UPDATE demo_matches SET map_name = %s WHERE demo_id = %s",
            (map_name, demo_id),
        )
    finally:
        cursor.close()


def _update_status(db_pool, demo_id: int, status: str) -> None:
    cursor = db_pool.cursor()
    try:
        cursor.execute(
            "UPDATE demo_matches SET status = %s WHERE demo_id = %s",
            (status, demo_id),
        )
    finally:
        cursor.close()


def _update_sample_interval(db_pool, demo_id: int, interval: int) -> None:
    cursor = db_pool.cursor()
    try:
        cursor.execute(
            "UPDATE demo_matches SET sample_interval = %s WHERE demo_id = %s",
            (interval, demo_id),
        )
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Team splitting (identical to map_control_test.py)
# ---------------------------------------------------------------------------

def _split_teams(ticks_df: pl.DataFrame) -> Tuple[List[str], List[str]]:
    modal = (
        ticks_df
        .with_columns(pl.col("steamid").cast(pl.Utf8))
        .group_by(["steamid", "team_num"])
        .agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True)
        .unique(subset=["steamid"], keep="first")
    )
    team_a = modal.filter(pl.col("team_num") == 2)["steamid"].to_list()
    team_b = modal.filter(pl.col("team_num") == 3)["steamid"].to_list()
    return team_a, team_b


# ---------------------------------------------------------------------------
# Player name extraction
# ---------------------------------------------------------------------------

def _extract_player_names(ticks_df: pl.DataFrame) -> List[Dict]:
    """Return a deduplicated list of ``{steamid, name}`` dicts from tick data.

    Uses the last-seen name per steamid (highest tick wins) so renames
    within a demo are captured correctly.
    """
    if "name" not in ticks_df.columns:
        logger.warning("'name' column absent from ticks - player names not stored")
        return []
    return (
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


# ---------------------------------------------------------------------------
# Per-demo worker
# ---------------------------------------------------------------------------

def process_demo(
    path: Path,
    demo_hash: str,
    db_pool,
    writer: MapControlDBWriter,
) -> None:
    filename = path.name
    logger.info("[START] %s  hash=%s", filename, demo_hash[:12])
    t_start = time.perf_counter()

    demo_id: int = 0
    try:
        demo_id = _mark_processing(db_pool, demo_hash, filename)

        # --- Parse ---
        demo = Demo(path=str(path))
        map_name: str = demo.header.get("map_name", "")
        logger.info("[PARSE] %s - map: %s", filename, map_name or "(unknown)")
        t0 = time.perf_counter()
        demo.parse(player_props=_PLAYER_PROPS)
        logger.info("[PARSE] %s done in %.1fs", filename, time.perf_counter() - t0)

        _update_map_name(db_pool, demo_id, map_name)

        ticks_df: Optional[pl.DataFrame] = getattr(demo, "ticks", None)
        rounds_df: Optional[pl.DataFrame] = getattr(demo, "rounds", None)

        if ticks_df is None or ticks_df.is_empty():
            raise ValueError("No tick data in demo")
        if rounds_df is None or rounds_df.is_empty():
            raise ValueError("No round data in demo")

        team_a, team_b = _split_teams(ticks_df)
        player_name_records = _extract_player_names(ticks_df)
        logger.info(
            "[PARSE] %s - Team A: %d  Team B: %d  players: %d",
            filename, len(team_a), len(team_b), len(player_name_records),
        )

        # --- Geometry ---
        if not map_name:
            raise ValueError("map_name is empty - cannot load geometry")
        geometry = _get_geometry(map_name)
        if not geometry.is_nav_available:
            raise RuntimeError(
                f"Nav mesh unavailable for '{map_name}' - run 'awpy get navs'"
            )

        # --- Ensure all players exist in DB before analysis ---
        all_steamids = (
            ticks_df
            .select(pl.col("steamid").cast(pl.Utf8))
            .unique()["steamid"]
            .to_list()
        )
        player_id_map = ensure_players(db_pool, all_steamids)

        # --- Analysis ---
        service = MapControlService(geometry)
        logger.info("[ANALYZE] %s - starting round analysis...", filename)
        t0 = time.perf_counter()
        service.compute(
            ticks_df, rounds_df, team_a, team_b,
            demo_id=demo_id, player_id_map=player_id_map, writer=writer,
        )
        logger.info("[ANALYZE] %s done in %.1fs", filename, time.perf_counter() - t0)

        # --- Flush DB writes for this demo before marking done ---
        writer.flush()

        # --- Player names ---
        upsert_players(db_pool, player_name_records)

        _update_sample_interval(db_pool, demo_id, SAMPLE_INTERVAL)
        _update_status(db_pool, demo_id, "done")
        logger.info(
            "[DONE] %s  (total %.1fs)", filename, time.perf_counter() - t_start
        )

    except Exception:
        logger.exception("[FAILED] %s", filename)
        if demo_id:
            try:
                _update_status(db_pool, demo_id, "failed")
            except Exception:
                logger.exception("[FAILED] Could not update status for %s", filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logfire.configure(send_to_logfire=False)

    # --- DB ---
    db = DBConnections()
    if not db.connect_prefire_db():
        logger.error("Database connection failed - aborting.")
        sys.exit(1)
    db_pool = db.prefire_db
    assert db_pool is not None  # guaranteed by connect_prefire_db() returning True

    # --- Schema (create missing tables + migrate demo_matches.status) ---
    create_all(db_pool._engine)

    # --- Scan ---
    if not DEMO_DIR.exists():
        logger.error("Demo directory does not exist: %s", DEMO_DIR)
        sys.exit(1)

    logger.info("Scanning %s...", DEMO_DIR)
    new_demos = scan_new_demos(db_pool, DEMO_DIR)
    if not new_demos:
        logger.info("No new demos to process.")
        return

    logger.info("Found %d new demo(s) to process.", len(new_demos))

    # --- Shared async writer ---
    writer = MapControlDBWriter(db_pool)

    # --- Sequential processing ---
    t_total = time.perf_counter()
    for path, demo_hash in new_demos:
        process_demo(path, demo_hash, db_pool, writer)

    writer.close()
    logger.info(
        "All demos processed in %.1fs.", time.perf_counter() - t_total
    )


if __name__ == "__main__":
    main()

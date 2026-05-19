"""Non-blocking database writer for map control analytics.

``MapControlDBWriter`` runs a daemon thread that drains a ``queue.Queue``,
issuing bulk ``INSERT IGNORE`` statements against the database pool.  The
analysis thread (``MapControlService._compute_round``) enqueues batches via
``put()`` without waiting for confirmation — the actual insert happens
asynchronously in the background.

Usage::

    writer = MapControlDBWriter(db_connections.prefire_db)
    round_df, player_df = service.compute(
        ticks_df, rounds_df, team_a, team_b,
        demo_id=demo_id, player_id_map=player_id_map, writer=writer,
    )
    writer.flush()   # wait for all queued inserts to finish
    writer.close()   # stop the daemon thread
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Dict, List, Optional

import logfire

logger = logging.getLogger(__name__)

_SENTINEL = object()


class MapControlDBWriter:
    """Enqueue DB writes from the map-control analysis thread; insert in background.

    All public methods are safe to call from any thread.  The internal worker
    thread is a daemon so it will not block interpreter shutdown if ``close()``
    is never called — though calling it is recommended for clean teardown.
    """

    def __init__(self, db_pool) -> None:
        """
        Args:
            db_pool: A ``DatabasePool`` instance (``internal.database.connections``).
        """
        self._pool = db_pool
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="mc-db-writer"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API (analysis-thread side)
    # ------------------------------------------------------------------

    def put(self, table: str, records: List[Dict]) -> None:
        """Enqueue *records* for bulk ``INSERT IGNORE`` into *table*.

        Non-blocking: returns immediately.  An empty *records* list is silently
        ignored.
        """
        if not records:
            return
        self._queue.put((table, records))

    def flush(self, timeout: float = 30.0) -> None:
        """Block until every enqueued batch has been inserted.

        Call this after finishing all ``put()`` calls for a match so the
        process does not exit before writes complete.  *timeout* is not
        enforced by this method — ``queue.join()`` blocks indefinitely, but
        the worker thread logs and skips on any DB error so the queue will
        always drain.
        """
        self._queue.join()

    def close(self) -> None:
        """Drain remaining items, stop the worker thread, and release resources."""
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=60.0)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            # Block until at least one item is available.
            first = self._queue.get()

            if first is _SENTINEL:
                self._queue.task_done()
                return

            # Drain every additional item that is already in the queue so we
            # can coalesce them into one executemany per table rather than
            # sending thousands of individual round-trips to the database.
            pending = [first]
            while True:
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            # Group rows by table, collecting sentinel presence.
            coalesced: Dict[str, List[Dict]] = {}
            sentinel_found = False
            for item in pending:
                if item is _SENTINEL:
                    sentinel_found = True
                else:
                    tbl, recs = item
                    if tbl not in coalesced:
                        coalesced[tbl] = []
                    coalesced[tbl].extend(recs)

            # One executemany per table.
            for tbl, recs in coalesced.items():
                try:
                    self._insert_batch(tbl, recs)
                except Exception:
                    logfire.exception("mc-db-writer insert failed", table=tbl)

            # Each queue.get() requires exactly one task_done().
            for _ in pending:
                self._queue.task_done()

            if sentinel_found:
                return

    def _insert_batch(self, table: str, records: List[Dict]) -> None:
        """Issue a single ``executemany`` INSERT IGNORE for all *records*.

        Column order is derived from the first record's keys; all subsequent
        records must have the same key set (guaranteed by internal callers).
        Table and column names originate from internal code only, never from
        user input.
        """
        if not records:
            return
        cols = list(records[0].keys())
        col_list = ", ".join(f"`{c}`" for c in cols)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"INSERT IGNORE INTO `{table}` ({col_list}) VALUES ({placeholders})"
        rows = [tuple(r[c] for c in cols) for r in records]
        cursor = self._pool.cursor()
        try:
            cursor.executemany(sql, rows)
        finally:
            cursor.close()


def ensure_players(pool, steamids: List[str]) -> Dict[str, int]:
    """Insert placeholder rows for *steamids* and return a ``{steamid: player_id}`` map.

    Uses ``INSERT IGNORE`` so existing rows are untouched.  The name column is
    left as an empty string; call ``upsert_players()`` afterwards to fill in
    actual names once tick data is available.
    """
    if not steamids:
        return {}
    unique_ids = list(dict.fromkeys(steamids))  # deduplicate, preserve order
    cursor = pool.cursor()
    try:
        cursor.executemany(
            "INSERT IGNORE INTO `players` (steamid, name) VALUES (%s, '')",
            [(sid,) for sid in unique_ids],
        )
        placeholders = ", ".join(["%s"] * len(unique_ids))
        cursor.execute(
            f"SELECT steamid, player_id FROM `players` WHERE steamid IN ({placeholders})",
            unique_ids,
        )
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        cursor.close()


def upsert_players(pool, player_records: List[Dict]) -> None:
    """Synchronously update the ``name`` column for known players.

    Rows are guaranteed to exist (``ensure_players`` inserted them first), so
    a plain ``UPDATE`` is used instead of ``INSERT ... ON DUPLICATE KEY UPDATE``
    to avoid burning auto-increment IDs on no-op inserts.
    """
    if not player_records:
        return
    sql = "UPDATE `players` SET name = %s WHERE steamid = %s"
    rows = [(str(r["name"]), str(r["steamid"])) for r in player_records]
    cursor = pool.cursor()
    try:
        cursor.executemany(sql, rows)
    finally:
        cursor.close()

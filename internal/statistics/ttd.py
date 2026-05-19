"""Service for computing Time-to-Damage (TTD) statistics using AWPY VisibilityChecker."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import logfire
import polars as pl
from awpy.constants import DEFAULT_SERVER_TICKRATE

from internal.service.statistics.geometry import GeometryService

# Z offset from foot origin to eye position (CS2 standing eye height above foot origin)
_EYE_HEIGHT: float = 64.0


class TTDService:
    """Computes per-kill Time-to-Damage (TTD) using AWPY VisibilityChecker BVH ray-casts.

    TTD is the number of ticks between the first tick the attacker has unobstructed
    line-of-sight to the victim's eye position and the first tick the attacker deals
    damage to that victim in the same round.  Results are returned as a Polars
    DataFrame; nothing is persisted to the database.

    Args:
        geometry_service: Shared geometry instance for the match map.  Visibility
            checks are skipped (returning None) when
            ``geometry_service.is_visibility_available`` is False.
    """

    def __init__(self, geometry_service: GeometryService) -> None:
        self._geometry = geometry_service

    def compute(
        self,
        kills_df: pl.DataFrame,
        damages_df: Optional[pl.DataFrame],
        ticks_df: Optional[pl.DataFrame],
        rounds_df: pl.DataFrame,
    ) -> Optional[pl.DataFrame]:
        """Compute per-kill TTD metrics.

        For every kill in *kills_df* this method:

        1. Iterates ticks forward from freeze-end to the kill tick, ray-casting
           the attacker↔victim line via ``VisibilityChecker.is_visible`` until the
           first unobstructed tick (``first_visible_tick``).
        2. Finds the earliest damage event in *damages_df* from that attacker to
           that victim in the same round (``first_damage_tick``).
        3. Returns ``ttd_ticks = max(0, first_damage_tick − first_visible_tick)``
           and ``ttd_seconds = ttd_ticks / 128``.

        Args:
            kills_df:   AWPY kills DataFrame with round_num, attacker_steamid,
                        victim_steamid, tick columns.
            damages_df: AWPY damages DataFrame; may be None.
            ticks_df:   AWPY ticks DataFrame with tick, steamid, X, Y, Z columns;
                        may be None or empty.
            rounds_df:  AWPY rounds DataFrame with round_num, freeze_end columns.

        Returns:
            DataFrame with columns:
                round_num (int), attacker_steamid (str), victim_steamid (str),
                kill_tick (int), first_visible_tick (int | null),
                first_damage_tick (int | null), ttd_ticks (int | null),
                ttd_seconds (float | null)
            or None if computation is not possible.
        """
        if not self._geometry.is_visibility_available:
            logfire.warning(
                "ttd skipped: .tri geometry file not available",
                hint="run 'awpy get tris' to download geometry files",
            )
            return None

        if ticks_df is None or ticks_df.is_empty():
            logfire.warning("ttd skipped: ticks dataframe unavailable")
            return None

        required_tick_cols = {"tick", "steamid", "X", "Y", "Z"}
        missing_cols = required_tick_cols - set(ticks_df.columns)
        if missing_cols:
            logfire.warning(
                "ttd skipped: missing columns in ticks dataframe",
                missing_columns=list(missing_cols),
            )
            return None

        # Build round_num → freeze_end_tick lookup
        freeze_map: Dict[int, int] = {
            row["round_num"]: int(row["freeze_end"])
            for row in rounds_df.filter(pl.col("round_num") >= 1).iter_rows(named=True)
            if row.get("freeze_end") is not None
        }

        # Pre-compute first-damage-tick per (round_num, attacker_sid, victim_sid)
        first_damage_map: Dict[Tuple[int, str, str], int] = self._build_first_damage_map(damages_df)

        # Normalise steamid to Utf8 once so per-kill filters are cheap
        ticks_norm = ticks_df.with_columns(pl.col("steamid").cast(pl.Utf8).alias("_sid"))

        records = []
        for kill_row in kills_df.filter(pl.col("round_num") >= 1).iter_rows(named=True):
            rnum: int = kill_row.get("round_num", 0)
            kill_tick: int = kill_row.get("tick", 0) or 0
            attacker_sid = _norm_sid(kill_row.get("attacker_steamid"))
            victim_sid = _norm_sid(kill_row.get("victim_steamid"))

            if not attacker_sid or not victim_sid:
                continue

            freeze_end = freeze_map.get(rnum)
            if freeze_end is None:
                continue

            first_visible = self._first_visible_tick(
                ticks_norm, attacker_sid, victim_sid, freeze_end, kill_tick
            )
            first_damage = first_damage_map.get((rnum, attacker_sid, victim_sid))

            if first_visible is None or first_damage is None:
                ttd_ticks: Optional[int] = None
                ttd_seconds: Optional[float] = None
            else:
                ttd_ticks = max(0, first_damage - first_visible)
                ttd_seconds = ttd_ticks / DEFAULT_SERVER_TICKRATE

            records.append(
                {
                    "round_num": rnum,
                    "attacker_steamid": attacker_sid,
                    "victim_steamid": victim_sid,
                    "kill_tick": kill_tick,
                    "first_visible_tick": first_visible,
                    "first_damage_tick": first_damage,
                    "ttd_ticks": ttd_ticks,
                    "ttd_seconds": ttd_seconds,
                }
            )

        if not records:
            return None

        return pl.DataFrame(records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _first_visible_tick(
        self,
        ticks_norm: pl.DataFrame,
        attacker_sid: str,
        victim_sid: str,
        freeze_end_tick: int,
        kill_tick: int,
    ) -> Optional[int]:
        """Walk forward from freeze-end to kill_tick and return the first tick
        at which *checker* reports an unobstructed ray between the two players'
        eye positions.  Returns None if no such tick is found.
        """
        window = (
            ticks_norm.filter(
                pl.col("tick").is_between(freeze_end_tick, kill_tick)
                & pl.col("_sid").is_in([attacker_sid, victim_sid])
            )
            .select(["tick", "_sid", "X", "Y", "Z"])
        )
        if window.is_empty():
            return None

        a_ticks = (
            window.filter(pl.col("_sid") == attacker_sid)
            .select(["tick", "X", "Y", "Z"])
            .rename({"X": "aX", "Y": "aY", "Z": "aZ"})
        )
        v_ticks = (
            window.filter(pl.col("_sid") == victim_sid)
            .select(["tick", "X", "Y", "Z"])
            .rename({"X": "vX", "Y": "vY", "Z": "vZ"})
        )

        paired = a_ticks.join(v_ticks, on="tick", how="inner").sort("tick")
        if paired.is_empty():
            return None

        for row in paired.iter_rows(named=True):
            a_eye = (row["aX"], row["aY"], (row["aZ"] or 0.0) + _EYE_HEIGHT)
            v_eye = (row["vX"], row["vY"], (row["vZ"] or 0.0) + _EYE_HEIGHT)
            try:
                if self._geometry.is_visible(a_eye, v_eye):
                    return int(row["tick"])
            except Exception:
                logfire.exception(
                    "visibility check raised unexpectedly",
                    tick=row["tick"],
                    attacker_steamid=attacker_sid,
                )
                return None

        return None

    def _build_first_damage_map(
        self,
        damages_df: Optional[pl.DataFrame],
    ) -> Dict[Tuple[int, str, str], int]:
        """Return {(round_num, attacker_steamid, victim_steamid): first_damage_tick}
        for every attacker→victim pair in *damages_df*.
        """
        if damages_df is None or damages_df.is_empty():
            return {}

        required = {"round_num", "attacker_steamid", "victim_steamid", "tick", "dmg_health_real"}
        if not required.issubset(set(damages_df.columns)):
            return {}

        try:
            agg = (
                damages_df.filter(
                    pl.col("attacker_steamid").is_not_null()
                    & pl.col("victim_steamid").is_not_null()
                    & (pl.col("dmg_health_real") > 0)
                )
                .group_by(["round_num", "attacker_steamid", "victim_steamid"])
                .agg(pl.col("tick").min().alias("first_damage_tick"))
            )
            return {
                (
                    r["round_num"],
                    str(r["attacker_steamid"]),
                    str(r["victim_steamid"]),
                ): int(r["first_damage_tick"])
                for r in agg.iter_rows(named=True)
            }
        except Exception:
            logfire.exception("failed to build first-damage map for TTD")
            return {}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _norm_sid(value: object) -> Optional[str]:
    """Return str(value) if truthy, else None."""
    return str(value) if value else None

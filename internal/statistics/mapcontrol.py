"""Service for computing per-round map control percentages using vision cones and nav mesh."""

from __future__ import annotations

import heapq
import math
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple

import logfire
import polars as pl

from internal.model.player_map_control import PlayerMapControlStats
from internal.statistics.geometry import EYE_HEIGHT, GeometryService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sample one tick out of every N when walking the round tick-by-tick
SAMPLE_INTERVAL: int = 16  

# Half the horizontal CS2 FOV (full 106°) used for vision cone classification.
FOV_HALF_DEG: float = 53.0

# Maximum distance (CS units) beyond which areas are not considered visible.
# ~3 000 units ≈ 12 seconds of walking at 250 u/s.
_MAX_VISION_DISTANCE: float = 3_000.0

# CS2 walk speed (units per second) and derived per-tick budget.
_WALK_SPEED_UPS: float = 250.0
_TICKRATE: float = 128.0
WALK_SPEED_UNITS_PER_TICK: float = _WALK_SPEED_UPS / _TICKRATE  # ≈ 1.953 u/tick


def _detect_round_team_side(
    tick_rows: Dict[int, List[dict]], freeze_end: int, round_end: int, team_set: Set[str]
) -> str:
    """Return 'T' or 'CT' for *team_set* in the given round tick range.

    Scans from *freeze_end* forward until a team member's team_num is found.
    Falls back to 'T' if no data is available.
    """
    for tick_num in range(freeze_end, round_end + 1):
        rows = tick_rows.get(tick_num)
        if not rows:
            continue
        for row in rows:
            if row.get("_sid") in team_set:
                tn = row.get("team_num")
                if tn is not None:
                    return "T" if int(tn) == 2 else "CT"
    return "T"


class MapControlService:
    """Computes per-round map control percentages using vision cones and the nav mesh.

    The algorithm samples player positions every ``SAMPLE_INTERVAL`` ticks within
    each round.  For every sample tick it:

    1. Projects each alive player's vision cone onto nav areas (FOV angle + LOS
       ray-cast via ``GeometryService``).
    2. Updates per-enemy-player *last-seen* records whenever an enemy occurs inside
       a friendly vision cone.
    3. Propagates each alive enemy's *mobility bubble* — the set of nav areas they
       could currently occupy — via a budget-bounded Dijkstra from the last-seen
       position.  Vision cones act as impassable barriers: the bubble cannot cross
       into areas currently being watched by the opposing team.  Players with no
       last-seen record have an unbounded bubble (position unknown).
    4. Classifies every nav area into one of six mutually exclusive states
       (priority order):

       * **CONTESTED**      — area is inside both teams' vision cones simultaneously.
       * **TEAM_A_ACTIVE**  — inside Team A's vision cone only.
       * **TEAM_B_ACTIVE**  — inside Team B's vision cone only.
       * **TEAM_A_PASSIVE** — not actively watched; Team B players provably cannot
                               reach it (outside their mobility bubbles).
       * **TEAM_B_PASSIVE** — symmetric.
       * **NEUTRAL**        — everything else (enemy could be there, nobody watching).

    Per-area sizes (2D polygon area from the nav mesh) are summed per state, divided
    by total navigable map area, and averaged across sampled ticks within each round.

    Results are returned as a Polars DataFrame; nothing is persisted to the database.
    """

    def __init__(
        self,
        geometry_service: GeometryService,
        debug_output_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self._geometry = geometry_service
        self._debug_output_dir: Optional[pathlib.Path] = debug_output_dir
        self._debug_map_img: Any = None  # None = not yet loaded; False = not available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ticks_df: Optional[pl.DataFrame],
        rounds_df: pl.DataFrame,
        team_a_steamids: List[str],
        team_b_steamids: List[str],
        demo_id: int = 0,
        player_id_map: Optional[Dict[str, int]] = None,
        writer: Optional[Any] = None,
    ) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """Compute per-round map control percentages and per-player attribution.

        Args:
            ticks_df:          AWPY ticks DataFrame; must contain columns
                               tick, steamid, X, Y, Z, health, yaw, pitch.
            rounds_df:         AWPY rounds DataFrame; must contain columns
                               round_num, freeze_end, end.
            team_a_steamids:   Steam-ID64 strings for team A players.
            team_b_steamids:   Steam-ID64 strings for team B players.

        Returns:
            Tuple of:
            - Round DataFrame with columns: round_num, team_a_active_pct,
              team_a_passive_pct, team_b_active_pct, team_b_passive_pct,
              contested_pct, neutral_pct.
            - Player DataFrame with columns: steamid, round_num, avg_active_pct,
              avg_unique_pct, avg_denial_pct, total_clearance_pct,
              passive_attributed_pct, death_impact_pct, survived,
              sample_ticks_alive.
            Both elements are None when computation is not possible.
        """
        if not self._geometry.is_nav_available:
            logfire.warning(
                "map control skipped: nav mesh unavailable",
                hint="run 'awpy get navs' to download navigation meshes",
            )
            return None, None

        if ticks_df is None or ticks_df.is_empty():
            logfire.warning("map control skipped: ticks dataframe unavailable")
            return None, None

        required_cols = {"tick", "steamid", "X", "Y", "Z", "health", "yaw", "pitch"}
        missing = required_cols - set(ticks_df.columns)
        if missing:
            logfire.warning(
                "map control skipped: missing columns in ticks dataframe",
                missing_columns=list(missing),
            )
            return None, None

        areas = self._geometry.all_areas()
        if not areas:
            return None, None

        total_map_size: float = sum(float(a.size) for a in areas.values())
        if total_map_size <= 0:
            return None, None

        set_a: Set[str] = set(team_a_steamids)
        set_b: Set[str] = set(team_b_steamids)

        # Normalise steamid column once, then convert to a plain Python dict grouped by
        # tick value so per-round sampling can look up rows in O(1) instead of running
        # a full-DataFrame filter scan for every sampled tick.
        ticks_norm = ticks_df.with_columns(pl.col("steamid").cast(pl.Utf8).alias("_sid"))
        tick_rows: Dict[int, List[dict]] = {}
        for _row in ticks_norm.iter_rows(named=True):
            _t = int(_row["tick"])
            if _t not in tick_rows:
                tick_rows[_t] = []
            tick_rows[_t].append(_row)

        # Precompute float area sizes once to avoid repeated attribute access and
        # float() conversion inside the hot per-tick attribution loops.
        area_sizes: Dict[int, float] = {
            aid: float(a.size) for aid, a in areas.items() if float(a.size) > 0
        }

        # Determine team A's starting side from the earliest tick with team_num data.
        team_a_starting_side = "T"
        if "team_num" in ticks_df.columns:
            _first_side = (
                ticks_df
                .with_columns(pl.col("steamid").cast(pl.Utf8).alias("_sid"))
                .filter(pl.col("_sid").is_in(list(set_a)) & (pl.col("team_num") > 0))
                .sort("tick")
                .head(1)
                .select("team_num")
            )
            if not _first_side.is_empty():
                team_a_starting_side = "T" if int(_first_side["team_num"][0]) == 2 else "CT"
        team_b_starting_side = "CT" if team_a_starting_side == "T" else "T"

        # Write mc_demo_teams once per demo (all players, one batch).
        if writer is not None and demo_id and player_id_map:
            _demo_team_records = []
            for _sid in team_a_steamids:
                _pid = player_id_map.get(_sid)
                if _pid is not None:
                    _demo_team_records.append(
                        {"demo_id": demo_id, "player_id": _pid, "team": "A", "starting_side": team_a_starting_side}
                    )
            for _sid in team_b_steamids:
                _pid = player_id_map.get(_sid)
                if _pid is not None:
                    _demo_team_records.append(
                        {"demo_id": demo_id, "player_id": _pid, "team": "B", "starting_side": team_b_starting_side}
                    )
            if _demo_team_records:
                writer.put("mc_demo_teams", _demo_team_records)

        round_records = []
        player_records: List[Dict] = []
        for round_row in rounds_df.filter(pl.col("round_num") >= 1).iter_rows(named=True):
            rnum: int = round_row["round_num"]
            freeze_end: int = round_row.get("freeze_end") or 0
            round_end: int = round_row.get("end") or 0

            if round_end <= freeze_end:
                continue

            # Determine which side team A is on this round from tick data.
            team_a_side = _detect_round_team_side(tick_rows, freeze_end, round_end, set_a)

            # Compute winner relative to team A/B.
            _winner_raw = round_row.get("winner")
            winner_team: Optional[str] = None
            if _winner_raw is not None:
                _winner_str = str(_winner_raw).lower()
                if _winner_str == "t":
                    winner_team = "A" if team_a_side == "T" else "B"
                elif _winner_str == "ct":
                    winner_team = "A" if team_a_side == "CT" else "B"

            if writer is not None and demo_id:
                writer.put("mc_round_sides", [{"demo_id": demo_id, "round_num": rnum, "team_a_side": team_a_side}])

            result = self._compute_round(
                tick_rows, rnum, freeze_end, round_end,
                set_a, set_b, areas, area_sizes, total_map_size,
                demo_id=demo_id, player_id_map=player_id_map, writer=writer,
                winner_team=winner_team,
            )
            if result is not None:
                round_record, player_stats_dict = result
                round_records.append(round_record)
                for ps in player_stats_dict.values():
                    player_records.append(ps.to_dict())

        if not round_records:
            return None, None

        logfire.info("map control computed", rounds=len(round_records))
        round_df = pl.DataFrame(round_records)
        player_df = pl.DataFrame(player_records) if player_records else None
        return round_df, player_df

    # ------------------------------------------------------------------
    # Per-round computation
    # ------------------------------------------------------------------

    def _compute_round(
        self,
        tick_rows: Dict[int, List[dict]],
        rnum: int,
        freeze_end: int,
        round_end: int,
        set_a: Set[str],
        set_b: Set[str],
        areas: dict,
        area_sizes: Dict[int, float],
        total_map_size: float,
        demo_id: int = 0,
        player_id_map: Optional[Dict[str, int]] = None,
        writer: Optional[Any] = None,
        winner_team: Optional[str] = None,
    ) -> Optional[Tuple[Dict, Dict[str, PlayerMapControlStats]]]:
        """Compute averaged control percentages and per-player attribution for a single round."""
        # Per-player nav area at the start of the round (freeze_end tick).
        spawn_area_a: Dict[str, int] = {}  # Team A player → area at spawn
        spawn_area_b: Dict[str, int] = {}  # Team B player → area at spawn
        for _sr in tick_rows.get(freeze_end, []):
            if (_sr.get("health") or 0) <= 0:
                continue
            _sid = str(_sr["_sid"])
            _area = self._geometry.get_area_at(
                float(_sr["X"] or 0), float(_sr["Y"] or 0), float(_sr["Z"] or 0)
            )
            if _area is None:
                continue
            if _sid in set_a:
                spawn_area_a[_sid] = _area
            elif _sid in set_b:
                spawn_area_b[_sid] = _area

        # Per-round intel state, persisted across sample ticks.
        # last_seen_by_a[b_steamid] = (area_id, tick)  — where Team A last saw this Team B player.
        # last_seen_by_b[a_steamid] = (area_id, tick)  — vice versa.
        # Pre-seeded with spawn positions: both teams know where the enemy starts.
        last_seen_by_a: Dict[str, Tuple[int, int]] = {
            sid: (area_id, freeze_end) for sid, area_id in spawn_area_b.items()
        }
        last_seen_by_b: Dict[str, Tuple[int, int]] = {
            sid: (area_id, freeze_end) for sid, area_id in spawn_area_a.items()
        }
        # Per-tracked-enemy incremental mobility frontier: dist map {area_id → walk distance
        # from last-seen origin} maintained across sample ticks.  On each tick the frontier
        # is first *cleared* of areas the opposing team is currently watching (T cannot be
        # there without being spotted), then *expanded* by one sample-interval's worth of
        # budget via Dijkstra using the current vision as barriers.  Re-spot resets the
        # map to a single seed at the new last-seen area.
        #
        # This correctly models "could the enemy have reached this area?": if a corridor
        # was blocked when the budget would first have allowed crossing it, the frontier
        # cannot grow past it; if the corridor was open at that moment, areas beyond it
        # enter the frontier and stay (the T might have crossed before it was guarded).
        b_frontier_dist: Dict[str, Dict[int, float]] = {
            sid: {area_id: 0.0} for sid, area_id in spawn_area_b.items()
        }  # B players tracked by Team A
        a_frontier_dist: Dict[str, Dict[int, float]] = {
            sid: {area_id: 0.0} for sid, area_id in spawn_area_a.items()
        }  # A players tracked by Team B
        # Cumulative union of all areas ever covered by each team's vision cones this round.
        # Used to gate passive control: an area is only passively held if it was cleared
        # at some point AND no spotted enemy could have walked back into it.
        cleared_by_a: Set[int] = set()
        cleared_by_b: Set[int] = set()

        state_sums: Dict[str, float] = {
            "a_active": 0.0, "a_passive": 0.0,
            "b_active": 0.0, "b_passive": 0.0,
            "contested": 0.0, "neutral": 0.0,
        }
        sample_count: int = 0
        # Passive-area sets from the previous sample tick, used to detect staleness
        # (passive → neutral transition) for debug frames.  Initialised empty so the
        # first tick produces no decay arrows (there was no prior passive state).
        _last_tick_passive_a: Set[int] = set()
        _last_tick_passive_b: Set[int] = set()

        # --- Per-player attribution state ---
        # first_cleared_by_{team}[area_id] = steamid of the first player whose vision/
        # physical presence added this area to cleared_by_{team} this round.
        first_cleared_by_a: Dict[int, str] = {}
        first_cleared_by_b: Dict[int, str] = {}

        # Per-player running sums (keyed by steamid); accumulate across sample ticks.
        _pmc_active_a:      Dict[str, float] = {}  # total active area size while alive
        _pmc_unique_a:      Dict[str, float] = {}  # total uniquely-watched area size
        _pmc_denial_a:      Dict[str, float] = {}  # total enemy-mobility area locked down
        _pmc_passive_a:     Dict[str, float] = {}  # total passive area attributed (first-clearer)
        _pmc_ticks_a:       Dict[str, int]   = {}  # sample ticks alive
        _pmc_last_unique_a: Dict[str, float] = {}  # unique size at last alive tick
        _pmc_death_impact_a: Dict[str, float] = {} # unique size at tick of death
        _pmc_active_b:      Dict[str, float] = {}
        _pmc_unique_b:      Dict[str, float] = {}
        _pmc_denial_b:      Dict[str, float] = {}
        _pmc_passive_b:     Dict[str, float] = {}
        _pmc_ticks_b:       Dict[str, int]   = {}
        _pmc_last_unique_b: Dict[str, float] = {}
        _pmc_death_impact_b: Dict[str, float] = {}
        # Track alive sets from the previous tick for death detection.
        _prev_alive_a_ids: Set[str] = set()
        _prev_alive_b_ids: Set[str] = set()
        # Per-area tick count accumulators for mc_area_round_stats.
        # Only non-neutral states are tracked; ticks_neutral is derived on write.
        area_a_ctrl_ticks: Dict[int, int] = {}   # a_active + a_passive ticks
        area_b_ctrl_ticks: Dict[int, int] = {}   # b_active + b_passive ticks
        area_contested_ticks: Dict[int, int] = {}  # contested ticks

        for tick in range(freeze_end, round_end + 1, SAMPLE_INTERVAL):
            rows = tick_rows.get(tick, [])
            if not rows:
                continue

            _health_map: Dict[str, int] = {
                str(r["_sid"]): int(r.get("health") or 0) for r in rows
            }

            # --- Build alive player snapshots ---
            # Entry layout: (steamid, x, y, z, yaw_deg, pitch_deg, area_id)
            alive_a: List[Tuple[str, float, float, float, float, float, int]] = []
            alive_b: List[Tuple[str, float, float, float, float, float, int]] = []

            for r in rows:
                if (r.get("health") or 0) <= 0:
                    continue
                sid = str(r["_sid"])
                x = float(r["X"] or 0)
                y = float(r["Y"] or 0)
                z = float(r["Z"] or 0)
                yaw = float(r["yaw"] or 0)
                pitch = float(r["pitch"] or 0)
                area_id = self._geometry.get_area_at(x, y, z)
                if area_id is None:
                    continue
                entry = (sid, x, y, z, yaw, pitch, area_id)
                if sid in set_a:
                    alive_a.append(entry)
                elif sid in set_b:
                    alive_b.append(entry)

            # --- Expire last-seen records for players who have died ---
            alive_b_ids: Set[str] = {e[0] for e in alive_b}
            alive_a_ids: Set[str] = {e[0] for e in alive_a}
            last_seen_by_a = {k: v for k, v in last_seen_by_a.items() if k in alive_b_ids}
            last_seen_by_b = {k: v for k, v in last_seen_by_b.items() if k in alive_a_ids}
            b_frontier_dist = {k: v for k, v in b_frontier_dist.items() if k in alive_b_ids}
            a_frontier_dist = {k: v for k, v in a_frontier_dist.items() if k in alive_a_ids}

            # --- Compute vision cones (per-player, then union into team sets) ---
            player_vision_a: Dict[str, Set[int]] = {}
            for sid, x, y, z, yaw, pitch, _ in alive_a:
                player_vision_a[sid] = self._geometry.get_areas_in_vision_cone(
                    (x, y, z + EYE_HEIGHT), yaw, pitch, FOV_HALF_DEG, _MAX_VISION_DISTANCE,
                )

            a_vision: Set[int] = set()
            for _cone in player_vision_a.values():
                a_vision |= _cone

            player_vision_b: Dict[str, Set[int]] = {}
            for sid, x, y, z, yaw, pitch, _ in alive_b:
                player_vision_b[sid] = self._geometry.get_areas_in_vision_cone(
                    (x, y, z + EYE_HEIGHT), yaw, pitch, FOV_HALF_DEG, _MAX_VISION_DISTANCE,
                )

            b_vision: Set[int] = set()
            for _cone in player_vision_b.values():
                b_vision |= _cone

            # Physical-presence injection into vision sets.
            # get_areas_in_vision_cone tests from eye height, but a player's own nav
            # area centroid is at roughly foot level (~64 units below the eye).  The
            # downward vector to that centroid is essentially perpendicular to the
            # forward view direction, so it consistently fails the FOV dot-product
            # gate and is NOT added by the cone test.  Yet a player standing in an
            # area will instantly spot any enemy who enters it, so for mobility
            # purposes the area must be treated as watched.  Adding it directly here
            # closes the gap: enemy Dijkstra will treat occupied areas as barriers,
            # preventing the mobility bubble from leaking through chokepoints or
            # corridors where a player is physically present.
            for sid, _, _, _, _, _, area_id in alive_a:
                a_vision.add(area_id)
                player_vision_a[sid].add(area_id)
            for sid, _, _, _, _, _, area_id in alive_b:
                b_vision.add(area_id)
                player_vision_b[sid].add(area_id)

            # Accumulate clearance history for passive control gating
            cleared_by_a |= a_vision
            cleared_by_b |= b_vision
            # Physical presence: a player's actual nav area is cleared for their team
            # regardless of where the centroid sits relative to their FOV cone.  This
            # fills the gap when a player is at the leading edge of a large area and
            # the centroid falls behind their cone boundary at the sample tick.
            for _, _, _, _, _, _, area_id in alive_a:
                cleared_by_a.add(area_id)
            for _, _, _, _, _, _, area_id in alive_b:
                cleared_by_b.add(area_id)

            # --- Record first-clearer for newly cleared areas (before enemy invalidation) ---
            # Vision cone areas take precedence; physical presence used as a fallback.
            # Once recorded, an entry is never overwritten (first clearer sticks even
            # if the area is later invalidated by enemy presence and re-cleared).
            for sid, _, _, _, _, _, area_id in alive_a:
                if area_id not in first_cleared_by_a:
                    first_cleared_by_a[area_id] = sid
                for _aid in player_vision_a.get(sid, set()):
                    if _aid not in first_cleared_by_a:
                        first_cleared_by_a[_aid] = sid
            for sid, _, _, _, _, _, area_id in alive_b:
                if area_id not in first_cleared_by_b:
                    first_cleared_by_b[area_id] = sid
                for _aid in player_vision_b.get(sid, set()):
                    if _aid not in first_cleared_by_b:
                        first_cleared_by_b[_aid] = sid

            # Enemy presence (physical or via vision cone) invalidates the opposing
            # team's prior clearance.  An area an enemy is watching or standing in
            # cannot be passively held — passive is only restored once the owning team
            # actively re-watches the area at a later tick.
            cleared_by_a -= b_vision
            cleared_by_b -= a_vision
            for _, _, _, _, _, _, area_id in alive_b:
                cleared_by_a.discard(area_id)
            for _, _, _, _, _, _, area_id in alive_a:
                cleared_by_b.discard(area_id)

            # --- Clear watched areas from each player's mobility frontier ---
            # If a CT is watching an area right now and the T is not spotted there, the T
            # is provably NOT in that area.  Remove it from the frontier so it cannot act
            # as a stepping-stone seed in the next expansion step.  Areas past the watched
            # corridor that entered the frontier when it was open are NOT removed — the T
            # may have crossed before guarding began, which is correct behaviour.
            for sid in b_frontier_dist:
                b_frontier_dist[sid] = {
                    k: v for k, v in b_frontier_dist[sid].items() if k not in a_vision
                }
            for sid in a_frontier_dist:
                a_frontier_dist[sid] = {
                    k: v for k, v in a_frontier_dist[sid].items() if k not in b_vision
                }

            # --- Update last-seen records ---
            # A record is created/refreshed when a player's area falls inside an enemy cone.
            # The frontier is also reset to a single seed at the re-spotted area: ground
            # truth that the player IS here now wipes any stale incremental history.
            _sighting_rows: List[dict] = []
            for sid, _, _, _, _, _, area_id in alive_b:
                if area_id in a_vision:
                    last_seen_by_a[sid] = (area_id, tick)
                    b_frontier_dist[sid] = {area_id: 0.0}
                    if writer is not None and demo_id:
                        _spotter = next(
                            (s for s, cone in player_vision_a.items() if area_id in cone), None
                        )
                        _sighting_rows.append({
                            "demo_id": demo_id, "round_num": rnum, "tick": tick,
                            "spotter_team": "A",
                            "spotter_player_id": player_id_map.get(_spotter) if (_spotter and player_id_map) else None,
                            "spotted_player_id": player_id_map[sid] if player_id_map else 0, "area_id": area_id,
                        })
            for sid, _, _, _, _, _, area_id in alive_a:
                if area_id in b_vision:
                    last_seen_by_b[sid] = (area_id, tick)
                    a_frontier_dist[sid] = {area_id: 0.0}
                    if writer is not None and demo_id:
                        _spotter = next(
                            (s for s, cone in player_vision_b.items() if area_id in cone), None
                        )
                        _sighting_rows.append({
                            "demo_id": demo_id, "round_num": rnum, "tick": tick,
                            "spotter_team": "B",
                            "spotter_player_id": player_id_map.get(_spotter) if (_spotter and player_id_map) else None,
                            "spotted_player_id": player_id_map[sid] if player_id_map else 0, "area_id": area_id,
                        })
            if writer is not None and demo_id and _sighting_rows:
                writer.put("mc_sighting_events", _sighting_rows)

            # --- Compute mobility bubbles ---
            # Incremental frontier expansion: each player's dist map (areas reachable from
            # their last-seen origin, with walk distances) is maintained across ticks.
            # This tick it is grown by Dijkstra using the current vision as barriers (cannot
            # cross a watched corridor) and a total-budget cap (cannot exceed total walk
            # distance since last-seen).  Areas that a CT is watching have already been
            # cleared from the frontier above, preventing stale nodes from acting as seeds
            # through newly-guarded corridors.
            #
            # An empty frontier (all reachable positions were watched and cleared) means
            # the T's location is unknown; only their ground-truth position is injected.
            # Per-tick dist maps are stored for debug arrow reconstruction.
            _tick_b_mob_dist: Dict[str, Dict[int, float]] = {}
            b_mobility: Set[int] = set()
            for sid, _, _, _, _, _, cur_area in alive_b:
                b_mobility.add(cur_area)
                if sid not in last_seen_by_a:
                    continue
                frontier = b_frontier_dist.get(sid)
                if not frontier:
                    continue  # frontier fully cleared; T position unknown
                _, seen_tick = last_seen_by_a[sid]
                budget = (tick - seen_tick) * WALK_SPEED_UNITS_PER_TICK
                dist_map = _expand_mobility_frontier(frontier, budget, a_vision, self._geometry)
                b_frontier_dist[sid] = dist_map
                _tick_b_mob_dist[sid] = dist_map
                b_mobility |= dist_map.keys()

            # a_mobility: symmetric — reachable areas for each Team A player known to Team B.
            _tick_a_mob_dist: Dict[str, Dict[int, float]] = {}
            a_mobility: Set[int] = set()
            for sid, _, _, _, _, _, cur_area in alive_a:
                a_mobility.add(cur_area)
                if sid not in last_seen_by_b:
                    continue
                frontier = a_frontier_dist.get(sid)
                if not frontier:
                    continue  # frontier fully cleared; player position unknown
                _, seen_tick = last_seen_by_b[sid]
                budget = (tick - seen_tick) * WALK_SPEED_UNITS_PER_TICK
                dist_map = _expand_mobility_frontier(frontier, budget, b_vision, self._geometry)
                a_frontier_dist[sid] = dist_map
                _tick_a_mob_dist[sid] = dist_map
                a_mobility |= dist_map.keys()

            # --- Classify areas and accumulate ---
            tick_sums = _classify_areas(
                area_sizes, total_map_size, a_vision, b_vision, a_mobility, b_mobility,
                cleared_by_a, cleared_by_b,
            )
            for key in state_sums:
                state_sums[key] += tick_sums[key]
            sample_count += 1

            # --- DB: area tick count accumulation and per-tick aggregates ---
            if writer is not None and demo_id:
                _active_u = a_vision | b_vision
                _pa_set = cleared_by_a - b_mobility - _active_u
                _pb_set = (cleared_by_b - a_mobility - _active_u) - _pa_set
                for _aid in a_vision - b_vision:
                    area_a_ctrl_ticks[_aid] = area_a_ctrl_ticks.get(_aid, 0) + 1
                for _aid in b_vision - a_vision:
                    area_b_ctrl_ticks[_aid] = area_b_ctrl_ticks.get(_aid, 0) + 1
                for _aid in a_vision & b_vision:
                    area_contested_ticks[_aid] = area_contested_ticks.get(_aid, 0) + 1
                for _aid in _pa_set:
                    area_a_ctrl_ticks[_aid] = area_a_ctrl_ticks.get(_aid, 0) + 1
                for _aid in _pb_set:
                    area_b_ctrl_ticks[_aid] = area_b_ctrl_ticks.get(_aid, 0) + 1
                _pct = 100.0 / total_map_size
                writer.put("mc_tick_aggregates", [{
                    "demo_id": demo_id, "round_num": rnum, "tick": tick,
                    "team_a_active_pct": round(tick_sums["a_active"] * _pct, 2),
                    "team_a_passive_pct": round(tick_sums["a_passive"] * _pct, 2),
                    "team_b_active_pct": round(tick_sums["b_active"] * _pct, 2),
                    "team_b_passive_pct": round(tick_sums["b_passive"] * _pct, 2),
                    "contested_pct": round(tick_sums["contested"] * _pct, 2),
                    "neutral_pct": round(tick_sums["neutral"] * _pct, 2),
                }])

            # --- Per-player attribution ---
            # Build per-area coverage count (how many teammates watch each area).
            _area_count_a: Dict[int, int] = {}
            for _cone in player_vision_a.values():
                for _aid in _cone:
                    _area_count_a[_aid] = _area_count_a.get(_aid, 0) + 1
            _area_count_b: Dict[int, int] = {}
            for _cone in player_vision_b.values():
                for _aid in _cone:
                    _area_count_b[_aid] = _area_count_b.get(_aid, 0) + 1

            # Passive areas this tick: set arithmetic mirrors _classify_areas logic.
            _active_union = a_vision | b_vision
            _passive_areas_a: Set[int] = cleared_by_a - b_mobility - _active_union
            _passive_areas_b: Set[int] = cleared_by_b - a_mobility - _active_union

            # Team A: accumulate per-player active / unique / denial sums.
            _tick_player_rows: List[dict] = []
            for sid, _px, _py, _pz, _pyaw, _ppitch, _paid in alive_a:
                _cone = player_vision_a.get(sid, set())
                _active_sz = sum(area_sizes[_aid] for _aid in _cone if _aid in area_sizes)
                _unique_sz = sum(
                    area_sizes[_aid] for _aid in _cone
                    if _aid in area_sizes and _area_count_a.get(_aid, 0) == 1
                )
                _denial_sz = sum(
                    area_sizes[_aid] for _aid in _cone
                    if _aid in area_sizes and _aid in b_mobility
                )
                _pmc_active_a[sid] = _pmc_active_a.get(sid, 0.0) + _active_sz
                _pmc_unique_a[sid] = _pmc_unique_a.get(sid, 0.0) + _unique_sz
                _pmc_denial_a[sid] = _pmc_denial_a.get(sid, 0.0) + _denial_sz
                _pmc_ticks_a[sid] = _pmc_ticks_a.get(sid, 0) + 1
                _pmc_last_unique_a[sid] = _unique_sz
                if writer is not None and demo_id:
                    _tick_player_rows.append({
                        "demo_id": demo_id, "round_num": rnum, "tick": tick,
                        "player_id": player_id_map[sid] if player_id_map else 0, "team": "A",
                        "x": round(_px, 1), "y": round(_py, 1), "z": round(_pz, 1),
                        "yaw": round(_pyaw, 1), "pitch": round(_ppitch, 1),
                        "area_id": _paid, "health": _health_map.get(sid, 0),
                        "active_size": round(_active_sz, 1),
                        "unique_size": round(_unique_sz, 1),
                        "denial_size": round(_denial_sz, 1),
                    })
            # Attribute passive areas to their first clearer (Team A).
            for _aid in _passive_areas_a:
                _clearer = first_cleared_by_a.get(_aid)
                if _clearer is not None:
                    _pmc_passive_a[_clearer] = _pmc_passive_a.get(_clearer, 0.0) + area_sizes.get(_aid, 0.0)

            # Team B: accumulate per-player active / unique / denial sums.
            for sid, _px, _py, _pz, _pyaw, _ppitch, _paid in alive_b:
                _cone = player_vision_b.get(sid, set())
                _active_sz = sum(area_sizes[_aid] for _aid in _cone if _aid in area_sizes)
                _unique_sz = sum(
                    area_sizes[_aid] for _aid in _cone
                    if _aid in area_sizes and _area_count_b.get(_aid, 0) == 1
                )
                _denial_sz = sum(
                    area_sizes[_aid] for _aid in _cone
                    if _aid in area_sizes and _aid in a_mobility
                )
                _pmc_active_b[sid] = _pmc_active_b.get(sid, 0.0) + _active_sz
                _pmc_unique_b[sid] = _pmc_unique_b.get(sid, 0.0) + _unique_sz
                _pmc_denial_b[sid] = _pmc_denial_b.get(sid, 0.0) + _denial_sz
                _pmc_ticks_b[sid] = _pmc_ticks_b.get(sid, 0) + 1
                _pmc_last_unique_b[sid] = _unique_sz
                if writer is not None and demo_id:
                    _tick_player_rows.append({
                        "demo_id": demo_id, "round_num": rnum, "tick": tick,
                        "player_id": player_id_map[sid] if player_id_map else 0, "team": "B",
                        "x": round(_px, 1), "y": round(_py, 1), "z": round(_pz, 1),
                        "yaw": round(_pyaw, 1), "pitch": round(_ppitch, 1),
                        "area_id": _paid, "health": _health_map.get(sid, 0),
                        "active_size": round(_active_sz, 1),
                        "unique_size": round(_unique_sz, 1),
                        "denial_size": round(_denial_sz, 1),
                    })
            # Attribute passive areas to their first clearer (Team B).
            for _aid in _passive_areas_b:
                _clearer = first_cleared_by_b.get(_aid)
                if _clearer is not None:
                    _pmc_passive_b[_clearer] = _pmc_passive_b.get(_clearer, 0.0) + area_sizes.get(_aid, 0.0)

            if writer is not None and demo_id and _tick_player_rows:
                writer.put("mc_tick_players", _tick_player_rows)

            # Death detection: players alive last tick but gone this tick.
            for sid in _prev_alive_a_ids - alive_a_ids:
                _pmc_death_impact_a[sid] = _pmc_last_unique_a.get(sid, 0.0)
            for sid in _prev_alive_b_ids - alive_b_ids:
                _pmc_death_impact_b[sid] = _pmc_last_unique_b.get(sid, 0.0)
            _prev_alive_a_ids = alive_a_ids
            _prev_alive_b_ids = alive_b_ids

            # --- Debug frame rendering ---
            if self._debug_output_dir is not None:
                # Passive areas this tick: cleared by the team, enemy cannot reach them,
                # and neither team is actively watching (active takes priority).
                curr_passive_a: Set[int] = {
                    aid for aid in areas
                    if aid in cleared_by_a
                    and aid not in b_mobility
                    and aid not in b_vision
                    and aid not in a_vision
                }
                curr_passive_b: Set[int] = {
                    aid for aid in areas
                    if aid in cleared_by_b
                    and aid not in a_mobility
                    and aid not in a_vision
                    and aid not in b_vision
                }
                # Staleness decay: was passive last tick but is now neutral because
                # the enemy mobility bubble grew to include it.  Areas that are still
                # in cleared_by_{team} (not directly recaptured by vision/presence)
                # but no longer satisfy the passive condition.
                # Exclude areas now in the owning team's active vision — those are
                # transitioning to active control, not decaying to neutral.
                stale_a: Set[int] = (
                    (_last_tick_passive_a - curr_passive_a) & cleared_by_a - a_vision
                )
                stale_b: Set[int] = (
                    (_last_tick_passive_b - curr_passive_b) & cleared_by_b - b_vision
                )
                _last_tick_passive_a = curr_passive_a
                _last_tick_passive_b = curr_passive_b
                decay_lines_a = _attribute_decay(
                    stale_a, alive_b, last_seen_by_a, _tick_b_mob_dist,
                    self._geometry._centroid_map, self._geometry,
                )
                decay_lines_b = _attribute_decay(
                    stale_b, alive_a, last_seen_by_b, _tick_a_mob_dist,
                    self._geometry._centroid_map, self._geometry,
                )
                out_dir = self._debug_output_dir / f"r{rnum:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                self._render_tick_frame(
                    rnum, tick, areas,
                    a_vision, b_vision, a_mobility, b_mobility,
                    cleared_by_a, cleared_by_b,
                    alive_a, alive_b,
                    decay_lines_a, decay_lines_b,
                    out_dir,
                )

        if sample_count == 0:
            return None

        scale = 100.0 / total_map_size / sample_count
        pct_scale = 100.0 / total_map_size
        round_record = {
            "round_num": rnum,
            "winner_team": winner_team,
            "team_a_active_pct": round(state_sums["a_active"] * scale, 2),
            "team_a_passive_pct": round(state_sums["a_passive"] * scale, 2),
            "team_b_active_pct": round(state_sums["b_active"] * scale, 2),
            "team_b_passive_pct": round(state_sums["b_passive"] * scale, 2),
            "contested_pct": round(state_sums["contested"] * scale, 2),
            "neutral_pct": round(state_sums["neutral"] * scale, 2),
        }
        if writer is not None and demo_id:
            writer.put("mc_rounds", [{**round_record, "demo_id": demo_id}])
            _all_touched = set(area_a_ctrl_ticks) | set(area_b_ctrl_ticks) | set(area_contested_ticks)
            if _all_touched:
                writer.put("mc_area_round_stats", [
                    {
                        "demo_id": demo_id,
                        "round_num": rnum,
                        "area_id": _aid,
                        "ticks_a_ctrl": area_a_ctrl_ticks.get(_aid, 0),
                        "ticks_b_ctrl": area_b_ctrl_ticks.get(_aid, 0),
                        "ticks_contested": area_contested_ticks.get(_aid, 0),
                        "ticks_sampled": sample_count,
                    }
                    for _aid in _all_touched
                ])

        # --- Build PlayerMapControlStats objects ---
        # Total clearance: cumulative size of areas first opened by each player.
        _clearance_size_a: Dict[str, float] = {}
        for _aid, _sid in first_cleared_by_a.items():
            if _aid in area_sizes:
                _clearance_size_a[_sid] = _clearance_size_a.get(_sid, 0.0) + area_sizes[_aid]
        _clearance_size_b: Dict[str, float] = {}
        for _aid, _sid in first_cleared_by_b.items():
            if _aid in area_sizes:
                _clearance_size_b[_sid] = _clearance_size_b.get(_sid, 0.0) + area_sizes[_aid]

        player_stats: Dict[str, PlayerMapControlStats] = {}
        for sid in set(_pmc_ticks_a) | set(_pmc_death_impact_a):
            n = _pmc_ticks_a.get(sid, 1) or 1
            player_stats[sid] = PlayerMapControlStats(
                player_id=player_id_map[sid] if player_id_map else 0,
                round_num=rnum,
                team="A",
                avg_active_pct=round(_pmc_active_a.get(sid, 0.0) / n * pct_scale, 3),
                avg_unique_pct=round(_pmc_unique_a.get(sid, 0.0) / n * pct_scale, 3),
                avg_denial_pct=round(_pmc_denial_a.get(sid, 0.0) / n * pct_scale, 3),
                total_clearance_pct=round(_clearance_size_a.get(sid, 0.0) * pct_scale, 3),
                passive_attributed_pct=round(
                    _pmc_passive_a.get(sid, 0.0) / sample_count * pct_scale, 3
                ),
                death_impact_pct=round(_pmc_death_impact_a.get(sid, 0.0) * pct_scale, 3),
                survived=sid not in _pmc_death_impact_a,
                round_alive_pct=round(_pmc_ticks_a.get(sid, 0) / sample_count * 100, 1),
            )
        for sid in set(_pmc_ticks_b) | set(_pmc_death_impact_b):
            n = _pmc_ticks_b.get(sid, 1) or 1
            player_stats[sid] = PlayerMapControlStats(
                player_id=player_id_map[sid] if player_id_map else 0,
                round_num=rnum,
                team="B",
                avg_active_pct=round(_pmc_active_b.get(sid, 0.0) / n * pct_scale, 3),
                avg_unique_pct=round(_pmc_unique_b.get(sid, 0.0) / n * pct_scale, 3),
                avg_denial_pct=round(_pmc_denial_b.get(sid, 0.0) / n * pct_scale, 3),
                total_clearance_pct=round(_clearance_size_b.get(sid, 0.0) * pct_scale, 3),
                passive_attributed_pct=round(
                    _pmc_passive_b.get(sid, 0.0) / sample_count * pct_scale, 3
                ),
                death_impact_pct=round(_pmc_death_impact_b.get(sid, 0.0) * pct_scale, 3),
                survived=sid not in _pmc_death_impact_b,
                round_alive_pct=round(_pmc_ticks_b.get(sid, 0) / sample_count * 100, 1),
            )

        if writer is not None and demo_id:
            writer.put("mc_player_rounds", [
                {**ps.to_dict(), "demo_id": demo_id}
                for ps in player_stats.values()
            ])

        return round_record, player_stats

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def _render_tick_frame(
        self,
        rnum: int,
        tick: int,
        areas: dict,
        a_vision: Set[int],
        b_vision: Set[int],
        a_mobility: Set[int],
        b_mobility: Set[int],
        cleared_by_a: Set[int],
        cleared_by_b: Set[int],
        alive_a: List[Tuple[str, float, float, float, float, float, int]],
        alive_b: List[Tuple[str, float, float, float, float, float, int]],
        decay_lines_a: List[Tuple[int, List[Tuple[float, float, float]]]],
        decay_lines_b: List[Tuple[int, List[Tuple[float, float, float]]]],
        out_dir: pathlib.Path,
    ) -> None:
        """Render and save one PNG debug frame for a sampled tick.

        Background: AWPY map overview image (if available).
        Nav areas filled by control state; areas whose passive clearance
        decayed this tick are drawn black.  A dashed arrow connects the
        responsible enemy to each decayed area centroid.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from awpy.plot.utils import game_to_pixel
        from awpy.data import MAPS_DIR

        plt.switch_backend("Agg")  # Non-interactive rendering — safe to call repeatedly

        map_name = self._geometry._map_name

        # --- Load / cache map overview image ---
        if self._debug_map_img is None:
            img_path = MAPS_DIR / f"{map_name}.png"
            if img_path.exists():
                self._debug_map_img = plt.imread(str(img_path))
            else:
                logfire.warning(
                    "debug render: map overview image not found",
                    map_name=map_name,
                    expected_path=str(img_path),
                )
                self._debug_map_img = False  # sentinel: unavailable

        fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
        ax.set_aspect("equal")

        if self._debug_map_img is not False:
            img = self._debug_map_img
            h, w = img.shape[:2]
            ax.imshow(img, origin="upper", extent=(0, w, h, 0), zorder=0)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        else:
            ax.set_facecolor("#1a1a1a")
            ax.set_xlim(0, 1024)
            ax.set_ylim(1024, 0)

        # Decayed area IDs (union of both teams) for fast lookup
        decayed_areas: Set[int] = (
            {aid for aid, _ in decay_lines_a} | {aid for aid, _ in decay_lines_b}
        )

        corners_map = self._geometry._corners
        centroid_map = self._geometry._centroid_map

        # --- Draw nav area polygons ---
        for area_id in areas:
            state = _area_debug_state(
                area_id, a_vision, b_vision,
                cleared_by_a, cleared_by_b,
                b_mobility, a_mobility,
                decayed_areas,
            )
            facecolor, alpha = _STATE_COLORS[state]
            area_corners = corners_map.get(area_id)

            if area_corners and len(area_corners) >= 3:
                pixels = []
                for cx, cy, cz in area_corners:
                    px, py, _ = game_to_pixel(map_name, (cx, cy, cz))
                    pixels.append((px, py))
                poly = mpatches.Polygon(
                    pixels, closed=True,
                    facecolor=facecolor, alpha=alpha,
                    edgecolor="#333333", linewidth=0.3, zorder=1,
                )
                ax.add_patch(poly)
            else:
                # Fallback: dot at centroid
                c = centroid_map.get(area_id)
                if c:
                    px, py, _ = game_to_pixel(map_name, c)
                    ax.plot(px, py, ".", color=facecolor, markersize=2,
                            alpha=alpha, zorder=1)

        # --- Decay paths: CT last-seen → T stale area (via nav mesh) ---
        for area_id, path_points in decay_lines_a:
            if len(path_points) < 2:
                continue
            pixel_path = [game_to_pixel(map_name, p) for p in path_points]
            xs = [p[0] for p in pixel_path]
            ys = [p[1] for p in pixel_path]
            ax.plot(xs, ys, "--", color="#003399", lw=1.5, alpha=0.85, zorder=3)
            ax.annotate(
                "",
                xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="-|>", color="#003399", lw=1.5),
                zorder=3,
            )

        # --- Decay paths: T last-seen → CT stale area (via nav mesh) ---
        for area_id, path_points in decay_lines_b:
            if len(path_points) < 2:
                continue
            pixel_path = [game_to_pixel(map_name, p) for p in path_points]
            xs = [p[0] for p in pixel_path]
            ys = [p[1] for p in pixel_path]
            ax.plot(xs, ys, "--", color="#997700", lw=1.5, alpha=0.85, zorder=3)
            ax.annotate(
                "",
                xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="-|>", color="#997700", lw=1.5),
                zorder=3,
            )

        # --- Draw alive players ---
        for _, x, y, z, _, _, _ in alive_a:
            px, py, _ = game_to_pixel(map_name, (x, y, z))
            ax.plot(px, py, "o", color="#FFD700", markersize=8,
                    markeredgecolor="#333300", markeredgewidth=0.8, zorder=4)

        for _, x, y, z, _, _, _ in alive_b:
            px, py, _ = game_to_pixel(map_name, (x, y, z))
            ax.plot(px, py, "o", color="#3399FF", markersize=7,
                    markeredgecolor="#003366", markeredgewidth=0.8, zorder=4)

        # --- Legend and title ---
        legend_handles = [
            mpatches.Patch(facecolor="#FFD700", alpha=0.8, label="T active"),
            mpatches.Patch(facecolor="#3399FF", alpha=0.8, label="CT active"),
            mpatches.Patch(facecolor="#FFF0A0", alpha=0.8, label="T passive"),
            mpatches.Patch(facecolor="#B3D9FF", alpha=0.8, label="CT passive"),
            mpatches.Patch(facecolor="#FF4444", alpha=0.8, label="Contested"),
            mpatches.Patch(facecolor="#888888", alpha=0.5, label="Neutral"),
            mpatches.Patch(facecolor="#111111", alpha=0.9, label="Stale passive (this tick)"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.85)
        ax.set_title(
            f"R{rnum:02d} | Tick {tick:,} | "
            f"Stale T: {len(decay_lines_a)}  Stale CT: {len(decay_lines_b)}",
            fontsize=9,
        )
        ax.axis("off")

        out_path = out_dir / f"t{tick:07d}.png"
        fig.savefig(str(out_path), dpi=80, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions)
# ---------------------------------------------------------------------------

def _expand_mobility_frontier(
    current_dist: Dict[int, float],
    total_budget: float,
    blocked_areas: Set[int],
    geometry: GeometryService,
) -> Dict[int, float]:
    """Single-source Dijkstra returning reachable nav areas and their walk distances.

    Called fresh every sample tick from a single seed ``{from_area: 0.0}``.  The
    budget cap and the barrier set together constrain reachability: edges into
    *blocked_areas* are treated as impassable, and any path whose cumulative distance
    exceeds *total_budget* is not pursued.  The result is a dist map of every area
    reachable from *from_area* without crossing any barrier and within the budget.

    The caller is responsible for assembling *blocked_areas* as the *union* of
    the current tick's opponent vision set and the per-player accumulated barrier
    memory (``b_seen_barriers`` / ``a_seen_barriers`` in ``_compute_round``).  This
    composite barrier set encodes both *currently-watched* areas and *historically-
    watched* corridors that the enemy could not have crossed without being seen,
    preventing the mobility bubble from retroactively routing through a chokepoint
    that was guarded while the enemy was pinned behind it.

    Args:
        current_dist:  Seed mapping — typically ``{from_area_id: 0.0}`` for a fresh
                       per-tick computation.  Must not be empty.
        total_budget:  Maximum cumulative walking distance (CS units) from the seed.
                       Equal to ``(current_tick - seen_tick) * WALK_SPEED_UNITS_PER_TICK``.
        blocked_areas: Nav area IDs that act as impassable barriers for this tick.
                       Edges *into* these areas are skipped; the source area is always
                       included even if it appears in *blocked_areas*.
        geometry:      Shared geometry service (provides nav mesh adjacency).

    Returns:
        New dist mapping ``{area_id → min walk distance from seed}`` for every
        reachable area.  *current_dist* is never mutated.
    """
    adj = geometry.adj_weights
    if not adj or not current_dist:
        return dict(current_dist)

    dist: Dict[int, float] = dict(current_dist)
    heap: List[Tuple[float, int]] = [(d, aid) for aid, d in dist.items()]
    heapq.heapify(heap)

    while heap:
        d, u_id = heapq.heappop(heap)
        if d > dist.get(u_id, math.inf):
            continue
        for v_id, edge_w in adj.get(u_id, ()):
            if v_id in blocked_areas:
                continue  # Currently watched — cannot expand through here this tick
            new_d = d + edge_w
            if new_d <= total_budget and new_d < dist.get(v_id, math.inf):
                dist[v_id] = new_d
                heapq.heappush(heap, (new_d, v_id))

    return dist


def _mobility_dijkstra(
    from_area_id: int,
    budget_units: float,
    blocked_areas: Set[int],
    geometry: GeometryService,
) -> Set[int]:
    """Return all nav area IDs reachable from *from_area_id* within *budget_units*.

    Edges into any area in *blocked_areas* are treated as impassable — the enemy
    cannot cross through areas actively watched by the opposing team's vision cones
    without being spotted.  The source area is always included even if it sits inside
    *blocked_areas* (the player was standing there when last seen).

    Args:
        from_area_id:  Starting nav area (enemy's last known position).
        budget_units:  Maximum cumulative walk distance (CS units) allowed.
        blocked_areas: Nav area IDs that act as vision barriers.
        geometry:      Shared geometry service (provides nav mesh).

    Returns:
        Set of area IDs reachable within the budget without crossing vision barriers.
    """
    adj = geometry.adj_weights
    if not adj:
        return {from_area_id}

    dist: Dict[int, float] = {from_area_id: 0.0}
    heap: List[Tuple[float, int]] = [(0.0, from_area_id)]
    reachable: Set[int] = {from_area_id}

    while heap:
        d, u_id = heapq.heappop(heap)
        if d > dist.get(u_id, math.inf):
            continue
        for v_id, edge_w in adj.get(u_id, ()):
            if v_id in blocked_areas:
                continue  # Vision barrier — enemy cannot cross unseen
            new_d = d + edge_w
            if new_d <= budget_units and new_d < dist.get(v_id, math.inf):
                dist[v_id] = new_d
                reachable.add(v_id)
                heapq.heappush(heap, (new_d, v_id))

    return reachable


def _find_path(
    from_area_id: int,
    to_area_id: int,
    budget_units: float,
    blocked_areas: Set[int],
    geometry: GeometryService,
) -> Optional[List[Tuple[float, float, float]]]:
    """Return the shortest nav-mesh centroid path from *from_area_id* to *to_area_id*.

    Traversal respects the same vision-barrier rule as ``_mobility_dijkstra``:
    edges into *blocked_areas* are treated as impassable.  The path is guaranteed
    to lie within *budget_units* of cumulative walking distance.

    Returns:
        Ordered list of (cx, cy, cz) centroid world-coordinates, starting at
        *from_area_id* and ending at *to_area_id*, or ``None`` when the target
        is unreachable within the budget.
    """
    if from_area_id == to_area_id:
        c = geometry._centroid_map.get(from_area_id)
        return [c] if c else None

    adj = geometry.adj_weights
    centroid_map = geometry._centroid_map
    if not adj:
        return None

    dist: Dict[int, float] = {from_area_id: 0.0}
    prev: Dict[int, Optional[int]] = {from_area_id: None}
    heap: List[Tuple[float, int]] = [(0.0, from_area_id)]

    while heap:
        d, u_id = heapq.heappop(heap)
        if d > dist.get(u_id, math.inf):
            continue
        if u_id == to_area_id:
            # Reconstruct path
            path_ids: List[int] = []
            cur: Optional[int] = to_area_id
            while cur is not None:
                path_ids.append(cur)
                cur = prev.get(cur)
            path_ids.reverse()
            result: List[Tuple[float, float, float]] = []
            for aid in path_ids:
                c = centroid_map.get(aid)
                if c:
                    result.append(c)
            return result if result else None
        for v_id, edge_w in adj.get(u_id, ()):
            if v_id in blocked_areas:
                continue
            new_d = d + edge_w
            if new_d <= budget_units and new_d < dist.get(v_id, math.inf):
                dist[v_id] = new_d
                prev[v_id] = u_id
                heapq.heappush(heap, (new_d, v_id))

    return None  # unreachable within budget


def _classify_areas(
    area_sizes: Dict[int, float],
    total_area_size: float,
    a_vision: Set[int],
    b_vision: Set[int],
    a_mobility: Set[int],
    b_mobility: Set[int],
    cleared_by_a: Optional[Set[int]] = None,
    cleared_by_b: Optional[Set[int]] = None,
) -> Dict[str, float]:
    """Classify every nav area into a control state and return summed sizes per state.

    Classification priority (first match wins):

    1. **contested**  — area inside both ``a_vision`` and ``b_vision``.
    2. **a_active**   — area inside ``a_vision`` only.
    3. **b_active**   — area inside ``b_vision`` only.
    4. **a_passive**  — Team A has looked here this round (``cleared_by_a``) AND no
                        spotted Team B player could have walked back (not in ``b_mobility``).
    5. **b_passive**  — symmetric.
    6. **neutral**    — everything else (not cleared, or enemy could be there).

    Args:
        a_mobility:   Areas spotted Team A players could currently have walked to
                      (Team B’s perspective).
        b_mobility:   Areas spotted Team B players could currently have walked to
                      (Team A’s perspective).
        cleared_by_a: Cumulative union of areas Team A’s vision cones have covered
                      this round. ``None`` treats all areas as cleared (backward-compat).
        cleared_by_b: Symmetric for Team B.
    """
    # Set arithmetic avoids iterating all nav areas on every sample tick.
    # Only the (small) classified sets are walked; neutral is computed by subtraction.
    _active = a_vision | b_vision
    _both = a_vision & b_vision
    _only_a = a_vision - b_vision
    _only_b = b_vision - a_vision

    _all_ids: Set[int] = set(area_sizes.keys())
    _cleared_a: Set[int] = _all_ids if cleared_by_a is None else cleared_by_a
    _cleared_b: Set[int] = _all_ids if cleared_by_b is None else cleared_by_b

    # Priority matches the original elif chain: a_passive is resolved before b_passive.
    _passive_a = _cleared_a - b_mobility - _active
    _passive_b = (_cleared_b - a_mobility - _active) - _passive_a

    def _sz(s: Set[int]) -> float:
        return sum(area_sizes[aid] for aid in s if aid in area_sizes)

    c_sz  = _sz(_both)
    aa_sz = _sz(_only_a)
    ba_sz = _sz(_only_b)
    pa_sz = _sz(_passive_a)
    pb_sz = _sz(_passive_b)
    return {
        "a_active":  aa_sz,
        "a_passive": pa_sz,
        "b_active":  ba_sz,
        "b_passive": pb_sz,
        "contested": c_sz,
        "neutral":   max(0.0, total_area_size - c_sz - aa_sz - ba_sz - pa_sz - pb_sz),
    }


def aggregate_player_stats(player_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-round player map control stats into a single row per player.

    Averages the per-tick metrics (avg_active_pct, avg_unique_pct, avg_denial_pct,
    passive_attributed_pct) across all rounds the player participated in.  Clearance
    is summed then re-scaled to a per-round average.  Death impact takes the maximum
    over all death events (worst single-round loss).  Survival rate and total ticks
    alive are also included.

    Args:
        player_df: DataFrame produced by ``MapControlService.compute()`` containing
                   per-round player rows.

    Returns:
        DataFrame with one row per steamid and columns: steamid, rounds_played,
        avg_active_pct, avg_unique_pct, avg_denial_pct, avg_clearance_pct,
        avg_passive_attributed_pct, max_death_impact_pct, survival_rate,
        total_ticks_alive.
    """
    return (
        player_df
        .group_by("steamid")
        .agg([
            pl.len().alias("rounds_played"),
            pl.col("avg_active_pct").mean().round(3).alias("avg_active_pct"),
            pl.col("avg_unique_pct").mean().round(3).alias("avg_unique_pct"),
            pl.col("avg_denial_pct").mean().round(3).alias("avg_denial_pct"),
            pl.col("total_clearance_pct").mean().round(3).alias("avg_clearance_pct"),
            pl.col("passive_attributed_pct").mean().round(3).alias("avg_passive_attributed_pct"),
            pl.col("death_impact_pct").max().round(3).alias("max_death_impact_pct"),
            pl.col("survived").mean().round(3).alias("survival_rate"),
            pl.col("round_alive_pct").mean().round(1).alias("avg_alive_pct"),
        ])
        .sort("avg_active_pct", descending=True)
    )


# Colour-map for per-area debug rendering: state → (hex_colour, alpha)
# Team A = T side (yellow/gold), Team B = CT side (blue)
_STATE_COLORS: Dict[str, Tuple[str, float]] = {
    "decayed":   ("#111111", 0.85),
    "contested": ("#FF4444", 0.65),
    "a_active":  ("#FFD700", 0.70),  # T active — gold
    "b_active":  ("#3399FF", 0.70),  # CT active — blue
    "a_passive": ("#FFF0A0", 0.50),  # T passive — pale yellow
    "b_passive": ("#B3D9FF", 0.50),  # CT passive — pale blue
    "neutral":   ("#888888", 0.15),
}


def _area_debug_state(
    area_id: int,
    a_vision: Set[int],
    b_vision: Set[int],
    cleared_by_a: Set[int],
    cleared_by_b: Set[int],
    b_mobility: Set[int],
    a_mobility: Set[int],
    decayed_areas: Set[int],
) -> str:
    """Return the display-state key for one nav area in a debug frame.

    Priority order (first match wins): decayed > contested > a_active > b_active
    > a_passive > b_passive > neutral.

    'decayed' overrides all other states so recently-lost clearance is always
    visible regardless of the current vision/mobility situation.
    """
    if area_id in decayed_areas:
        return "decayed"
    if area_id in a_vision and area_id in b_vision:
        return "contested"
    if area_id in a_vision:
        return "a_active"
    if area_id in b_vision:
        return "b_active"
    if area_id in cleared_by_a and area_id not in b_mobility:
        return "a_passive"
    if area_id in cleared_by_b and area_id not in a_mobility:
        return "b_passive"
    return "neutral"


def _trace_frontier_path(
    target_area_id: int,
    mob_dist: Dict[int, float],
    centroid_map: Dict[int, Tuple[float, float, float]],
    geometry: GeometryService,
    avoid_areas: Optional[Set[int]] = None,
) -> Optional[List[Tuple[float, float, float]]]:
    """Reconstruct the path from the last-seen origin to *target_area_id* via the frontier.

    Performs a greedy gradient descent on *mob_dist*: starting at the target area,
    repeatedly move to the adjacent neighbour with the smallest dist value until
    dist == 0.0 (the last-seen origin).  The resulting list is reversed to give an
    origin-to-target sequence of centroid coordinates.

    *avoid_areas* (typically the owning team's current vision set) biases neighbour
    selection: a non-avoided neighbour is always preferred over an avoided one when
    both have a lower dist than the current node.  An avoided neighbour is only chosen
    as a last resort when no non-avoided frontier neighbour exists — the path is still
    historically valid; the arrow just faithfully shows that all routes passed through
    the now-watched corridor.

    Args:
        target_area_id: The stale area to trace back to its origin.
        mob_dist:       Per-player frontier dist map ``{area_id → min walk distance}``.
        centroid_map:   Area centroid world coordinates.
        geometry:       Shared geometry service (provides nav mesh adjacency).
        avoid_areas:    Nav area IDs to route around when a non-avoided alternative
                        exists (typically the owning team's current vision set).

    Returns:
        Ordered list of ``(cx, cy, cz)`` coordinates from origin to target, or
        ``None`` if the target is not in the dist map or centroid data is missing.
    """
    if target_area_id not in mob_dist:
        return None

    areas = geometry.all_areas()
    path_ids: List[int] = [target_area_id]
    cur = target_area_id
    visited: Set[int] = {cur}

    for _ in range(4096):  # guard against degenerate graphs
        cur_d = mob_dist.get(cur, math.inf)
        if cur_d <= 0.0:
            break  # reached the last-seen origin
        u_area = areas.get(cur)
        if u_area is None:
            break

        # Prefer non-avoided neighbours; only accept an avoided one if no non-avoided
        # frontier neighbour with a lower dist exists.
        best_next: Optional[int] = None
        best_next_d: float = cur_d
        best_next_avoided: bool = True  # True = best candidate is in avoid_areas

        for v_id in u_area.connections:
            if v_id in visited:
                continue
            v_d = mob_dist.get(v_id)
            if v_d is None or v_d >= cur_d:
                continue
            v_avoided = avoid_areas is not None and v_id in avoid_areas
            if best_next is None:
                best_next, best_next_d, best_next_avoided = v_id, v_d, v_avoided
            elif not v_avoided and best_next_avoided:
                # Non-avoided always beats avoided regardless of distance.
                best_next, best_next_d, best_next_avoided = v_id, v_d, False
            elif v_avoided == best_next_avoided and v_d < best_next_d:
                best_next, best_next_d = v_id, v_d

        if best_next is None:
            break
        path_ids.append(best_next)
        visited.add(best_next)
        cur = best_next

    path_ids.reverse()
    result: List[Tuple[float, float, float]] = [
        centroid_map[aid] for aid in path_ids if aid in centroid_map
    ]
    return result if len(result) >= 2 else None


def _attribute_decay(
    stale_areas: Set[int],
    enemy_players: List[Tuple[str, float, float, float, float, float, int]],
    last_seen_by_enemy: Dict[str, Tuple[int, int]],
    enemy_mob_dist: Dict[str, Dict[int, float]],
    centroid_map: Dict[int, Tuple[float, float, float]],
    geometry: GeometryService,
) -> List[Tuple[int, List[Tuple[float, float, float]]]]:
    """Pair each stale passive area with the path that caused it to decay.

    For each stale area, finds the enemy player whose per-tick dist map contains
    the area with the smallest distance (the most direct cause), then reconstructs
    the path via ``_trace_frontier_path``.

    Because dist maps are produced by fresh per-tick Dijkstra that blocks entry into
    currently-watched areas, every area in the dist map is non-watched and the
    reconstructed arrow is guaranteed to traverse only non-active polygons.

    Args:
        stale_areas:        Areas that transitioned from passive to neutral this tick.
        enemy_players:      Alive enemy player snapshots ``(sid, x, y, z, yaw, pitch, area_id)``.
        last_seen_by_enemy: Dict mapping enemy steamid to ``(area_id, tick)`` of last sighting.
        enemy_mob_dist:     Dict mapping enemy steamid to their fresh per-tick dist map.
        centroid_map:       Area centroid world coordinates.
        geometry:           Shared geometry service.

    Returns:
        List of ``(area_id, path_points)`` where ``path_points`` traces the route
        from the last-seen origin to the stale area.
    """
    result: List[Tuple[int, List[Tuple[float, float, float]]]] = []

    # Collect dist maps only for players who have a frontier entry.
    spotted_dists: List[Dict[int, float]] = [
        enemy_mob_dist[sid]
        for sid, *_ in enemy_players
        if sid in enemy_mob_dist and enemy_mob_dist[sid]
    ]
    if not spotted_dists:
        return result

    for area_id in stale_areas:
        # Pick the enemy whose frontier reached this area with the smallest distance
        # (most direct cause of the decay).
        best_dist_map: Optional[Dict[int, float]] = None
        best_d = math.inf
        for dm in spotted_dists:
            d = dm.get(area_id)
            if d is not None and d < best_d:
                best_d = d
                best_dist_map = dm

        if best_dist_map is None:
            # Area decayed due to an unspotted player's ground-truth area injection;
            # no frontier entry exists so no path arrow can be drawn.
            continue

        path = _trace_frontier_path(
            area_id, best_dist_map, centroid_map, geometry
        )
        if path is not None:
            result.append((area_id, path))

    return result

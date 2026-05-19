"""Full-game map control analysis using the production MapControlService.

Parses every round of a CS2 demo at 16-tick resolution and prints
per-round averages plus a whole-game summary.  No images are produced.

Usage
-----
    python scripts/map_control_test.py [path/to/demo.dem]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Project root on sys.path so internal imports resolve
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import polars as pl
from awpy import Demo

from internal.statistics.geometry import GeometryService
from internal.statistics.mapcontrol import (
    SAMPLE_INTERVAL,
    MapControlService,
    aggregate_player_stats,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DEMO = Path(r"Z:\CoachDemos\dem2.dem")

# Set to a Path to save per-tick debug PNG frames; set to None to skip.
DEBUG_OUTPUT: Optional[Path] = Path("scripts/debug_output")

_PLAYER_PROPS = ["health", "team_num", "X", "Y", "Z", "yaw", "pitch"]

_RESULT_COLS = [
    "team_a_active_pct",
    "team_a_passive_pct",
    "team_b_active_pct",
    "team_b_passive_pct",
    "contested_pct",
    "neutral_pct",
]

_LABELS = {
    "team_a_active_pct":  "A active",
    "team_a_passive_pct": "A passive",
    "team_b_active_pct":  "B active",
    "team_b_passive_pct": "B passive",
    "contested_pct":      "contested",
    "neutral_pct":        "neutral",
}

_PLAYER_STAT_COLS = [
    "avg_active_pct",
    "avg_unique_pct",
    "avg_denial_pct",
    "total_clearance_pct",
    "passive_attributed_pct",
    "death_impact_pct",
]

_PLAYER_STAT_LABELS = {
    "avg_active_pct":          "active",
    "avg_unique_pct":          "unique",
    "avg_denial_pct":          "denial",
    "total_clearance_pct":     "clearance",
    "passive_attributed_pct": "passive attr",
    "death_impact_pct":        "death impact",
}


def _fmt_player_row(steamid: str, row: Dict[str, float], survived: bool, ticks: int) -> str:
    parts = [f"  {steamid:<20s}"]
    for col in _PLAYER_STAT_COLS:
        val = row.get(col, 0.0)
        parts.append(f"{_PLAYER_STAT_LABELS[col]:>14s}: {val:5.2f}%")
    parts.append(f"  survived={survived}  ticks={ticks}")
    return "  ".join(parts)


def _fmt_player_summary_row(row: dict) -> str:
    sid = row.get("steamid", "?")
    parts = [
        f"  {sid:<20s}",
        f"rounds={row.get('rounds_played', 0):3d}",
        f"active={row.get('avg_active_pct', 0.0):5.2f}%",
        f"unique={row.get('avg_unique_pct', 0.0):5.2f}%",
        f"denial={row.get('avg_denial_pct', 0.0)*100}",
        f"clearance={row.get('avg_clearance_pct', 0.0):5.2f}%",
        f"passive={row.get('avg_passive_attributed_pct', 0.0):5.2f}%",
        f"death_impact={row.get('max_death_impact_pct', 0.0):5.2f}%",
        f"survival={row.get('survival_rate', 0.0):.0%}",
    ]
    return "  ".join(parts)


def _split_teams(ticks_df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Infer team assignments from the modal team_num per steamid."""
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


def _fmt_row(record: Dict[str, float]) -> str:
    lines = []
    for col in _RESULT_COLS:
        val = record.get(col, 0.0)
        bar = "#" * int(val / 2)
        lines.append(f"  {_LABELS[col]:12s} {val:6.2f}%  {bar}")
    return "\n".join(lines)


def run(demo_path: Path) -> Optional[pl.DataFrame]:
    if not demo_path.exists():
        print(f"[ERROR] Demo not found: {demo_path}")
        return None

    # ------------------------------------------------------------------
    # Parse demo
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[PARSE] {demo_path}")
    t0 = time.perf_counter()
    demo = Demo(path=demo_path)
    map_name: str = demo.header.get("map_name", "")
    print(f"[PARSE] Map: {map_name or '(unknown)'}")
    demo.parse(player_props=_PLAYER_PROPS)
    print(f"[PARSE] Done in {time.perf_counter() - t0:.1f}s")

    ticks_df: Optional[pl.DataFrame] = getattr(demo, "ticks", None)
    if ticks_df is None or ticks_df.is_empty():
        print("[ERROR] No tick data — aborting.")
        return None
    print(f"[PARSE] {len(ticks_df):,} tick rows")

    team_a, team_b = _split_teams(ticks_df)
    print(f"[PARSE] Team A ({len(team_a)} players): {team_a}")
    print(f"[PARSE] Team B ({len(team_b)} players): {team_b}")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[GEO] Loading geometry for '{map_name}' ...")
    t0 = time.perf_counter()
    geometry = GeometryService(map_name)
    print(f"[GEO] Nav: {geometry.is_nav_available}  "
          f"Visibility: {geometry.is_visibility_available}  "
          f"({time.perf_counter() - t0:.1f}s)")

    print("[GEO] Precomputing visibility matrix ...")
    t0 = time.perf_counter()
    geometry.precompute_visibility()
    print(f"[GEO] Vis matrix: {geometry.is_vis_matrix_available}  "
          f"({time.perf_counter() - t0:.1f}s)")

    if not geometry.is_nav_available:
        print("[ERROR] Nav unavailable — run 'awpy get navs' and retry.")
        return None

    areas = geometry.all_areas()
    total_map_size = sum(float(a.size) for a in areas.values())
    print(f"[GEO] {len(areas)} nav areas  |  total size: {total_map_size:.0f}")

    # ------------------------------------------------------------------
    # Round loop — call service._compute_round per round for live progress
    # ------------------------------------------------------------------
    service = MapControlService(geometry, debug_output_dir=None)

    if DEBUG_OUTPUT is not None:
        print(f"[DEBUG] Per-tick frames will be saved to: {DEBUG_OUTPUT.resolve()}")

    rounds_df = demo.rounds
    valid_rounds = rounds_df.filter(
        (pl.col("round_num") >= 1) & (pl.col("end") > pl.col("freeze_end"))
    )
    total_rounds = len(valid_rounds)

    ticks_norm = ticks_df.with_columns(pl.col("steamid").cast(pl.Utf8).alias("_sid"))
    tick_rows: Dict[int, List[dict]] = {}
    for _row in ticks_norm.iter_rows(named=True):
        _t = int(_row["tick"])
        if _t not in tick_rows:
            tick_rows[_t] = []
        tick_rows[_t].append(_row)

    area_sizes: Dict[int, float] = {
        aid: float(a.size) for aid, a in areas.items() if float(a.size) > 0
    }

    set_a: Set[str] = set(team_a)
    set_b: Set[str] = set(team_b)

    print(f"\n[INFO] {total_rounds} rounds  |  sample interval: {SAMPLE_INTERVAL} ticks/sample")
    print(f"{'='*60}")

    records: List[Dict] = []
    player_records: List[Dict] = []
    t_total = time.perf_counter()

    for i, round_row in enumerate(valid_rounds.iter_rows(named=True), 1):
        rnum: int = round_row["round_num"]
        freeze_end: int = round_row.get("freeze_end") or 0
        round_end: int = round_row.get("end") or 0
        n_ticks = round_end - freeze_end
        n_samples = max(1, n_ticks // SAMPLE_INTERVAL)

        print(
            f"[R{rnum:02d}] ({i}/{total_rounds})  "
            f"{n_ticks / 128:.1f}s  ~{n_samples} samples ...",
            end="  ",
            flush=True,
        )

        t_round = time.perf_counter()
        result = service._compute_round(
            tick_rows, rnum, freeze_end, round_end,
            set_a, set_b, areas, area_sizes, total_map_size,
        )
        elapsed = time.perf_counter() - t_round

        if result is None:
            print("SKIP (no data)")
            continue

        record, player_stats_dict = result
        records.append(record)
        for ps in player_stats_dict.values():
            player_records.append(ps.to_dict())
        print(f"{elapsed:.1f}s")
        print(_fmt_row(record))
        total_pct = sum(record[c] for c in _RESULT_COLS)
        print(f"  {'sum':12s} {total_pct:6.2f}%")
        if player_stats_dict:
            print(f"  --- per-player (round {rnum}) ---")
            for ps in sorted(player_stats_dict.values(), key=lambda p: p.avg_active_pct, reverse=True):
                print(_fmt_player_row(
                    ps.steamid,
                    ps.to_dict(),
                    ps.survived,
                    ps.round_alive_pct,
                ))

    # ------------------------------------------------------------------
    # Game average
    # ------------------------------------------------------------------
    if not records:
        print("\n[WARN] No rounds produced data.")
        return None

    result_df = pl.DataFrame(records)
    game_elapsed = time.perf_counter() - t_total

    print(f"\n{'='*60}")
    print(f"[AVERAGE] {len(records)} rounds processed in {game_elapsed:.1f}s\n")
    avg = {col: float(result_df[col].mean()) for col in _RESULT_COLS}
    print(_fmt_row(avg))
    print(f"  {'sum':12s} {sum(avg.values()):6.2f}%")

    # Per-player game summary
    if player_records:
        player_df = pl.DataFrame(player_records)
        summary_df = aggregate_player_stats(player_df)
        print(f"\n{'='*60}")
        print(f"[PLAYERS] Game-level map control summary ({len(summary_df)} players)\n")
        for row in summary_df.iter_rows(named=True):
            print(_fmt_player_summary_row(row))

    return result_df


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEMO
    run(path)

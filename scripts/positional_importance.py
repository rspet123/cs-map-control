"""Positional importance heatmap — visualises which nav areas correlate most
strongly with round win rate based on where players *position* themselves,
rather than how much map control they hold.

For each (map, area) the query computes the Pearson r between:
  • the fraction of a round a team spent in that area (presence %)
  • whether that team won the round

Four side-by-side subplots are rendered per map:
  1. t_presence_win_corr   — T-side  presence vs win correlation
  2. ct_presence_win_corr  — CT-side presence vs win correlation
  3. t_avg_pct_of_round    — average % of round T-side spends in area
  4. ct_avg_pct_of_round   — average % of round CT-side spends in area

Areas without enough data (<MIN_ROUNDS rounds present) are drawn as dark gray.
Output: saves ``{map_name}_positional_importance.png`` to ``storage/heatmaps/``
and opens each figure interactively.

Usage
-----
    python scripts/positional_importance.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root on sys.path so internal imports resolve
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.chdir(_ROOT)

import numpy as np
import matplotlib
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import logfire
logfire.configure(send_to_logfire=False)

from awpy.data import MAPS_DIR
from awpy.plot.utils import game_to_pixel

from internal.database.connections import DBConnections, DatabasePool
from internal.statistics.geometry import GeometryService

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path("storage") / "heatmaps"

# Minimum number of distinct rounds a (map, area) must appear in to receive
# a coloured tile.
MIN_ROUNDS = 10

_POSITIONAL_IMPORTANCE_SQL = """
WITH player_area_ticks AS (
    SELECT
        tp.demo_id,
        tp.round_num,
        tp.area_id,
        COUNT(*) AS ticks_in_area,
        COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (
            PARTITION BY tp.demo_id, tp.round_num
        ), 0) AS pct_of_round_in_area
    FROM mc_tick_players tp
    WHERE tp.health > 0
      AND tp.area_id IS NOT NULL
    GROUP BY tp.demo_id, tp.round_num, tp.area_id
),
area_rounds AS (
    SELECT
        pat.demo_id,
        pat.round_num,
        pat.area_id,
        pat.ticks_in_area,
        pat.pct_of_round_in_area,
        dm.map_name,
        rs.team_a_side,
        CASE WHEN r.winner_team = 'A' THEN 1.0 ELSE 0.0 END AS team_a_won
    FROM player_area_ticks pat
    JOIN mc_rounds r
        ON pat.demo_id = r.demo_id
        AND pat.round_num = r.round_num
    JOIN mc_round_sides rs
        ON pat.demo_id = rs.demo_id
        AND pat.round_num = rs.round_num
    JOIN demo_matches dm
        ON pat.demo_id = dm.demo_id
    WHERE r.winner_team IS NOT NULL
),
area_stats AS (
    SELECT
        map_name,
        area_id,
        -- T side stats
        COUNT(DISTINCT CASE WHEN team_a_side = 'T'
            THEN CONCAT(demo_id, '-', round_num) END)              AS t_rounds_present,
        AVG(CASE WHEN team_a_side = 'T'
            THEN pct_of_round_in_area END)                         AS t_avg_pct_of_round,
        AVG(CASE WHEN team_a_side = 'T'
            THEN team_a_won END)                                   AS t_win_rate,
        (
            AVG(CASE WHEN team_a_side = 'T'
                THEN pct_of_round_in_area * team_a_won END) -
            AVG(CASE WHEN team_a_side = 'T'
                THEN pct_of_round_in_area END) *
            AVG(CASE WHEN team_a_side = 'T'
                THEN team_a_won END)
        ) / NULLIF(
            STD(CASE WHEN team_a_side = 'T' THEN pct_of_round_in_area END) *
            STD(CASE WHEN team_a_side = 'T' THEN team_a_won END)
        , 0)                                                        AS t_presence_win_corr,
        -- CT side stats
        COUNT(DISTINCT CASE WHEN team_a_side = 'CT'
            THEN CONCAT(demo_id, '-', round_num) END)              AS ct_rounds_present,
        AVG(CASE WHEN team_a_side = 'CT'
            THEN pct_of_round_in_area END)                         AS ct_avg_pct_of_round,
        AVG(CASE WHEN team_a_side = 'CT'
            THEN 1.0 - team_a_won END)                             AS ct_win_rate,
        (
            AVG(CASE WHEN team_a_side = 'CT'
                THEN pct_of_round_in_area * (1.0 - team_a_won) END) -
            AVG(CASE WHEN team_a_side = 'CT'
                THEN pct_of_round_in_area END) *
            AVG(CASE WHEN team_a_side = 'CT'
                THEN 1.0 - team_a_won END)
        ) / NULLIF(
            STD(CASE WHEN team_a_side = 'CT' THEN pct_of_round_in_area END) *
            STD(CASE WHEN team_a_side = 'CT' THEN 1.0 - team_a_won END)
        , 0)                                                         AS ct_presence_win_corr
    FROM area_rounds
    GROUP BY map_name, area_id
    HAVING COUNT(DISTINCT CONCAT(demo_id, '-', round_num)) >= %s
)
SELECT
    map_name,
    area_id,
    t_rounds_present,
    ROUND(t_avg_pct_of_round * 100, 2)  AS t_avg_pct_of_round,
    ROUND(t_win_rate, 3)                AS t_win_rate,
    ROUND(t_presence_win_corr, 3)       AS t_presence_win_corr,
    ct_rounds_present,
    ROUND(ct_avg_pct_of_round * 100, 2) AS ct_avg_pct_of_round,
    ROUND(ct_win_rate, 3)               AS ct_win_rate,
    ROUND(ct_presence_win_corr, 3)      AS ct_presence_win_corr
FROM area_stats
ORDER BY map_name, t_presence_win_corr DESC
"""

# (column_key, display_title, colormap_name, is_diverging)
_METRICS: List[Tuple[str, str, str, bool]] = [
    ("t_presence_win_corr",  "T Presence–Win Corr",    "RdYlGn", True),
    ("ct_presence_win_corr", "CT Presence–Win Corr",   "RdYlGn", True),
    ("t_avg_pct_of_round",   "T Avg % of Round Here",  "plasma",  False),
    ("ct_avg_pct_of_round",  "CT Avg % of Round Here", "plasma",  False),
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_scores(pool: DatabasePool) -> Dict[str, List[dict]]:
    """Run the positional importance CTE and return rows grouped by map_name."""
    cur = pool.cursor(dictionary=True)
    try:
        cur.execute(_POSITIONAL_IMPORTANCE_SQL, (MIN_ROUNDS,))
        rows = cur.fetchall()
    finally:
        cur.close()

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["map_name"]].append(row)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Colormap / normalisation helpers
# ---------------------------------------------------------------------------

_GAMMA: float = 2.0


class _SignedPowerNorm(mcolors.TwoSlopeNorm):
    """TwoSlopeNorm with a symmetric power curve applied after mapping.

    Values near the centre (0.0) are compressed into a near-neutral band;
    strong positive/negative correlations are pulled toward the colormap ends.
    """

    def __init__(self, vmin: float, vmax: float, gamma: float = _GAMMA) -> None:
        super().__init__(vmin=vmin, vcenter=0.0, vmax=vmax)
        self._gamma = gamma

    def __call__(self, value, clip=None):  # type: ignore[override]
        result = super().__call__(value, clip=clip)
        data = np.ma.getdata(result).astype(float)
        mask = np.ma.getmaskarray(result)
        centred = data * 2.0 - 1.0
        sharpened = np.sign(centred) * np.abs(centred) ** (1.0 / self._gamma)
        out = (sharpened + 1.0) / 2.0
        return np.ma.array(out, mask=mask)


def _make_norm(values: List[float], is_diverging: bool) -> mcolors.Normalize:
    if not values:
        return mcolors.Normalize(vmin=0, vmax=1)

    vmin = min(values)
    vmax = max(values)

    if is_diverging:
        if vmin < 0 < vmax:
            return _SignedPowerNorm(vmin=vmin, vmax=vmax)
        bound = max(abs(vmin), abs(vmax))
        return _SignedPowerNorm(vmin=-bound, vmax=bound)

    return mcolors.PowerNorm(gamma=_GAMMA, vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Per-map rendering
# ---------------------------------------------------------------------------

def render_map(
    map_name: str,
    rows: List[dict],
    geo: GeometryService,
) -> None:
    """Render and save the 4-subplot positional importance heatmap for one map."""

    scores: Dict[int, dict] = {int(r["area_id"]): r for r in rows}

    # Collect valid (non-None) values per metric for normalisation
    metric_values: Dict[str, List[float]] = {m[0]: [] for m in _METRICS}
    for row in rows:
        for col, _, _, _ in _METRICS:
            v = row.get(col)
            if v is not None:
                metric_values[col].append(float(v))

    # --- Load map overview image ---
    map_img = None
    img_path = MAPS_DIR / f"{map_name}.png"
    if img_path.exists():
        map_img = plt.imread(str(img_path))
    else:
        print(f"  [warn] map image not found: {img_path}")

    # --- Figure setup (2×2 grid) ---
    fig, axes_grid = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes_grid.flat  # type: ignore[union-attr]
    fig.suptitle(f"{map_name} — Positional Importance", fontsize=20, fontweight="bold")

    corners_map = geo._corners
    all_area_ids = sorted(corners_map.keys())

    for ax, (col, title, cmap_name, is_diverging) in zip(axes, _METRICS):
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.axis("off")

        if map_img is not None:
            h, w = map_img.shape[:2]
            ax.imshow(map_img, origin="upper", extent=(0, w, h, 0), zorder=0)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        else:
            ax.set_facecolor("#1a1a1a")
            ax.set_xlim(0, 1024)
            ax.set_ylim(1024, 0)

        cmap = matplotlib.colormaps[cmap_name]  # type: ignore[attr-defined]
        norm = _make_norm(metric_values[col], is_diverging)

        for area_id in all_area_ids:
            area_corners = corners_map.get(area_id)
            if not area_corners or len(area_corners) < 3:
                centroid = geo._centroid_map.get(area_id)
                if centroid is None:
                    continue
                px, py, _ = game_to_pixel(map_name, centroid)
                row = scores.get(area_id)
                if row is not None and row.get(col) is not None:
                    color = cmap(norm(float(row[col])))
                    ax.plot(px, py, "o", color=color, markersize=3, alpha=0.75, zorder=2)
                else:
                    ax.plot(px, py, "o", color="#333333", markersize=2, alpha=0.35, zorder=2)
                continue

            pixels: List[Tuple[float, float]] = []
            for cx, cy, cz in area_corners:
                px, py, _ = game_to_pixel(map_name, (cx, cy, cz))
                pixels.append((px, py))

            row = scores.get(area_id)
            if row is not None and row.get(col) is not None:
                facecolor = cmap(norm(float(row[col])))
                alpha = 0.72
            else:
                facecolor = "#333333"
                alpha = 0.35

            poly = mpatches.Polygon(
                np.array(pixels),
                closed=True,
                facecolor=facecolor,
                alpha=alpha,
                edgecolor=None,
                linewidth=0,
                zorder=1,
            )
            ax.add_patch(poly)

        sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{map_name}_positional_importance.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Connecting to database…")
    db = DBConnections()
    if not db.connect_prefire_db():
        print("ERROR: could not connect to database.")
        sys.exit(1)

    print("Running positional importance query…")
    assert db.prefire_db is not None
    scores_by_map = fetch_scores(db.prefire_db)
    if not scores_by_map:
        print("No data returned — make sure mc_tick_players is populated.")
        sys.exit(0)

    print(f"Maps with data: {sorted(scores_by_map)}")

    for map_name, rows in sorted(scores_by_map.items()):
        print(f"\nRendering {map_name}  ({len(rows)} scored areas)…")
        try:
            geo = GeometryService(map_name)
        except Exception as exc:
            print(f"  [warn] Could not load geometry for {map_name}: {exc}")
            continue

        if not geo.is_nav_available:
            print(f"  [warn] Nav mesh unavailable for {map_name} — skipping.")
            continue

        render_map(map_name, rows, geo)

    print("\nDone.")


if __name__ == "__main__":
    main()

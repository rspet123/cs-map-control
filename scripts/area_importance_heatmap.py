"""Area importance heatmap — visualises which nav areas correlate most strongly
with round win rate for T-side, CT-side, and an aggregate importance score.

Three side-by-side subplots are rendered per map:
  1. importance_score  (avg of t_win_corr + ct_win_corr)
  2. t_win_corr        (T-side Pearson r vs win)
  3. ct_win_corr       (CT-side Pearson r vs win)

Nav areas without enough data (<20 rounds present) are drawn as dark gray.
Output: saves ``{map_name}_area_importance.png`` to ``storage/heatmaps/`` and
opens each figure interactively.

Usage
-----
    python scripts/area_importance_heatmap.py
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

# Ensure the working directory is the project root so GeometryService can find
# the vis_cache and other relative-path resources.
os.chdir(_ROOT)

import numpy as np
import matplotlib
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import logfire
# Suppress logfire telemetry — this script has no logfire token configured.
logfire.configure(send_to_logfire=False)

from awpy.data import MAPS_DIR
from awpy.plot.utils import game_to_pixel

from internal.database.connections import DBConnections, DatabasePool
from internal.statistics.geometry import GeometryService

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path("storage") / "heatmaps"

# Minimum rounds a (map, area) pair must appear in to receive a colored tile.
MIN_ROUNDS = 20

_AREA_IMPORTANCE_SQL = """
WITH area_control AS (
    SELECT
        ars.demo_id,
        dm.map_name,
        ars.round_num,
        ars.area_id,
        rs.team_a_side,
        -- Remap A/B to T/CT based on which side team A was on
        CASE WHEN rs.team_a_side = 'T'
            THEN ars.ticks_a_ctrl / NULLIF(ars.ticks_sampled, 0)
            ELSE ars.ticks_b_ctrl / NULLIF(ars.ticks_sampled, 0)
        END AS t_ctrl_ratio,
        CASE WHEN rs.team_a_side = 'CT'
            THEN ars.ticks_a_ctrl / NULLIF(ars.ticks_sampled, 0)
            ELSE ars.ticks_b_ctrl / NULLIF(ars.ticks_sampled, 0)
        END AS ct_ctrl_ratio,
        -- Remap winner to T/CT as well
        CASE WHEN r.winner_team = 'A' AND rs.team_a_side = 'T' THEN 1.0
             WHEN r.winner_team = 'B' AND rs.team_a_side = 'CT' THEN 1.0
             ELSE 0.0
        END AS t_won,
        CASE WHEN r.winner_team = 'A' AND rs.team_a_side = 'CT' THEN 1.0
             WHEN r.winner_team = 'B' AND rs.team_a_side = 'T' THEN 1.0
             ELSE 0.0
        END AS ct_won
    FROM mc_area_round_stats ars
    JOIN mc_rounds r
        ON ars.demo_id = r.demo_id
        AND ars.round_num = r.round_num
    JOIN demo_matches dm
        ON ars.demo_id = dm.demo_id
    JOIN mc_round_sides rs
        ON ars.demo_id = rs.demo_id
        AND ars.round_num = rs.round_num
    WHERE r.winner_team IS NOT NULL
),
area_win_corr AS (
    SELECT
        map_name,
        area_id,
        COUNT(*) AS rounds_present,
        (AVG(t_ctrl_ratio * t_won) - AVG(t_ctrl_ratio) * AVG(t_won))
            / NULLIF(STD(t_ctrl_ratio) * STD(t_won), 0) AS t_win_corr,
        (AVG(ct_ctrl_ratio * ct_won) - AVG(ct_ctrl_ratio) * AVG(ct_won))
            / NULLIF(STD(ct_ctrl_ratio) * STD(ct_won), 0) AS ct_win_corr,
        AVG(t_ctrl_ratio)  AS avg_t_ctrl,
        AVG(ct_ctrl_ratio) AS avg_ct_ctrl
    FROM area_control
    GROUP BY map_name, area_id
)
SELECT
    map_name,
    area_id,
    rounds_present,
    ROUND(avg_t_ctrl, 3)                           AS avg_t_ctrl,
    ROUND(avg_ct_ctrl, 3)                          AS avg_ct_ctrl,
    ROUND(t_win_corr, 3)                           AS t_win_corr,
    ROUND(ct_win_corr, 3)                          AS ct_win_corr,
    ROUND((t_win_corr + ct_win_corr) / 2, 3)       AS importance_score
FROM area_win_corr
WHERE rounds_present >= %s
ORDER BY map_name, importance_score DESC
"""

# (column_key, display_title, colormap_name, is_diverging)
_METRICS: List[Tuple[str, str, str, bool]] = [
    ("importance_score", "Importance Score",  "plasma",   False),  # top-left
    ("ct_minus_t_corr",  "CT − T Corr Diff",  "coolwarm", True),   # top-right
    ("ct_win_corr",      "CT Win Correlation","RdYlGn",   True),   # bottom-left
    ("t_win_corr",       "T Win Correlation", "RdYlGn",   True),   # bottom-right
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def fetch_scores(pool: DatabasePool) -> Dict[str, List[dict]]:
    """Run the CTE query and return rows grouped by map_name."""
    cur = pool.cursor(dictionary=True)
    try:
        cur.execute(_AREA_IMPORTANCE_SQL, (MIN_ROUNDS,))
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

# Exponent applied to the normalised position before colour lookup.
# gamma > 1  →  compresses low/neutral values toward the same dull colour and
#               stretches high values apart, so important areas "pop".
_GAMMA: float = 2.0


class _SignedPowerNorm(mcolors.TwoSlopeNorm):
    """TwoSlopeNorm with a symmetric power curve applied after mapping.

    Values close to *vcenter* (0.0) are compressed into a narrow band of
    near-neutral colour; strong positive/negative correlations are pulled
    visually toward the ends of the colormap.

    Concretely, after the linear TwoSlopeNorm maps each value to [0, 1] with
    0.5 at the centre, this class applies::

        centred  = mapped * 2 - 1          # → [-1, 1]
        sharpened = sign(centred) * |centred| ** (1 / gamma)   # γ > 1 compresses towards 0
        result   = (sharpened + 1) / 2    # → [0, 1]
    """

    def __init__(self, vmin: float, vmax: float, gamma: float = _GAMMA) -> None:
        super().__init__(vmin=vmin, vcenter=0.0, vmax=vmax)
        self._gamma = gamma

    def __call__(self, value, clip=None):  # type: ignore[override]
        result = super().__call__(value, clip=clip)
        # result may be a masked array; work on the underlying data
        data = np.ma.getdata(result).astype(float)
        mask = np.ma.getmaskarray(result)
        centred = data * 2.0 - 1.0
        sharpened = np.sign(centred) * np.abs(centred) ** (1.0 / self._gamma)
        out = (sharpened + 1.0) / 2.0
        return np.ma.array(out, mask=mask)


def _make_norm(
    values: List[float],
    is_diverging: bool,
) -> mcolors.Normalize:
    """Return a power-curve Normalize for *values*.

    Sequential metrics use PowerNorm(gamma=_GAMMA): low values are compressed
    together (all look dark) while high-importance areas spread apart and
    appear vivid.

    Diverging metrics use _SignedPowerNorm: values near 0 collapse to neutral
    grey/yellow while strong positive/negative correlations are pulled toward
    the colormap extremes.
    """
    if not values:
        return mcolors.Normalize(vmin=0, vmax=1)

    vmin = min(values)
    vmax = max(values)

    if is_diverging:
        if vmin < 0 < vmax:
            return _SignedPowerNorm(vmin=vmin, vmax=vmax)
        # All same sign — use a symmetric range so neutral (0) is always the
        # midpoint of the diverging colormap.
        bound = max(abs(vmin), abs(vmax))
        return _SignedPowerNorm(vmin=-bound, vmax=bound)

    # Sequential: PowerNorm on the [vmin, vmax] range.
    # gamma > 1 means the colour ramp is pulled toward the high end.
    return mcolors.PowerNorm(gamma=_GAMMA, vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Per-map rendering
# ---------------------------------------------------------------------------

def render_map(
    map_name: str,
    rows: List[dict],
    geo: GeometryService,
) -> None:
    """Render and save the 3-subplot heatmap for one map."""

    # --- Score lookup -------------------------------------------------------
    scores: Dict[int, dict] = {int(r["area_id"]): r for r in rows}

    # Compute derived metric: CT−T win correlation difference
    for row in scores.values():
        t = row.get("t_win_corr")
        ct = row.get("ct_win_corr")
        row["ct_minus_t_corr"] = (float(ct) - float(t)) if (t is not None and ct is not None) else None

    # Collect valid (non-None) values per metric for normalisation
    metric_values: Dict[str, List[float]] = {m[0]: [] for m in _METRICS}
    for row in rows:
        for col, _, _, _ in _METRICS:
            v = row.get(col)
            if v is not None:
                metric_values[col].append(float(v))

    # --- Load map overview image -------------------------------------------
    map_img = None
    img_path = MAPS_DIR / f"{map_name}.png"
    if img_path.exists():
        map_img = plt.imread(str(img_path))
    else:
        print(f"  [warn] map image not found: {img_path}")

    # --- Figure setup -------------------------------------------------------
    fig, axes_grid = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes_grid.flat  # type: ignore[union-attr]
    fig.suptitle(map_name, fontsize=20, fontweight="bold")

    corners_map = geo._corners
    all_area_ids = sorted(corners_map.keys())

    for ax, (col, title, cmap_name, is_diverging) in zip(axes, _METRICS):
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.axis("off")

        # --- Background ---
        if map_img is not None:
            h, w = map_img.shape[:2]
            ax.imshow(map_img, origin="upper", extent=(0, w, h, 0), zorder=0)
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
        else:
            ax.set_facecolor("#1a1a1a")
            ax.set_xlim(0, 1024)
            ax.set_ylim(1024, 0)

        # --- Colormap / norm ---
        cmap = matplotlib.colormaps[cmap_name]  # type: ignore[attr-defined]
        norm = _make_norm(metric_values[col], is_diverging)

        # --- Draw nav area polygons ---
        for area_id in all_area_ids:
            area_corners = corners_map.get(area_id)
            if not area_corners or len(area_corners) < 3:
                # Fall back to a centroid dot when no polygon is available
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

            # Convert game-world corners to pixel coords
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

        # --- Colorbar ---
        sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # --- Save ---------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{map_name}_area_importance.png"
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

    print("Running area win-correlation query…")
    assert db.prefire_db is not None  # guaranteed by connect_prefire_db() returning True
    scores_by_map = fetch_scores(db.prefire_db)
    if not scores_by_map:
        print("No data returned — make sure mc_area_round_stats is populated.")
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

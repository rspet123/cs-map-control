"""Plant-location win-rate heatmap.

For each map with bomb-plant data, renders a figure with one **row per site**
(A / B) and two **columns**:

  Col 0 : T-side win rate when planted at that cell
  Col 1 : Bomb-detonation rate when planted at that cell

Plant (x, y) positions are bucketed into fixed CELL_SIZE × CELL_SIZE game-unit squares.
The number of cells is determined by the spatial extent of the plant locations at each
site.  Each subplot is **zoomed** to that bounding box plus a pixel buffer so you see
a tight view of the site rather than the full map overview.

Usage
-----
    python scripts/plant_location_importance.py

Output → ``storage/heatmaps/{map_name}_plant_importance.png``
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root on sys.path
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
from awpy.plot.utils import game_to_pixel, is_position_on_lower_level

from internal.database.connections import DBConnections, DatabasePool

# ---------------------------------------------------------------------------
# Constants : tune freely
# ---------------------------------------------------------------------------

CELL_SIZE: float = 20.0    # side length of each bucket cell in game units
ZOOM_BUFFER: int = 80       # extra pixel padding around plant bounding box
MIN_SAMPLES: int = 3        # cells with fewer plants are drawn dark-gray
OUT_DIR = Path("storage") / "heatmaps"

# Colormap for each metric (sequential)
_CMAP_WIN = "YlOrRd"
_CMAP_DET = "Blues"

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_SQL = """
SELECT
    dm.map_name,
    bp.site,
    bp.x,
    bp.y,
    bp.z,
    bp.detonated,
    CASE
        WHEN r.winner_team = 'A' AND rs.team_a_side = 'T' THEN 1
        WHEN r.winner_team = 'B' AND rs.team_a_side = 'CT' THEN 1
        ELSE 0
    END AS t_won
FROM bomb_plants bp
JOIN demo_matches dm
    ON bp.demo_id = dm.demo_id
JOIN mc_rounds r
    ON bp.demo_id = r.demo_id
    AND bp.round_num = r.round_num
JOIN mc_round_sides rs
    ON bp.demo_id = rs.demo_id
    AND bp.round_num = rs.round_num
WHERE bp.x IS NOT NULL
  AND bp.y IS NOT NULL
  AND bp.site IS NOT NULL
  AND r.winner_team IS NOT NULL
ORDER BY dm.map_name, bp.site
"""


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

PlantRecord = Dict  # keys: x, y, z, t_won, detonated


def fetch_plants(pool: DatabasePool) -> Dict[str, Dict[str, List[PlantRecord]]]:
    """Return ``{map_name: {site: [record, ...]}}``."""
    cur = pool.cursor(dictionary=True)
    try:
        cur.execute(_SQL)
        rows = cur.fetchall()
    finally:
        cur.close()

    result: Dict[str, Dict[str, List[PlantRecord]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        result[row["map_name"]][row["site"]].append({
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]) if row["z"] is not None else 0.0,
            "t_won": int(row["t_won"]),
            "detonated": int(row["detonated"]) if row["detonated"] is not None else 0,
        })
    return {m: dict(s) for m, s in result.items()}


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

CellKey = Tuple[int, int]  # (col, row) indices, zero-based


class GridBuckets:
    """Buckets plants into fixed CELL_SIZE×CELL_SIZE game-unit squares.

    The number of columns and rows is determined by the spatial extent of the
    plant locations, not fixed in advance.
    """

    def __init__(self, plants: List[PlantRecord]) -> None:
        xs = [p["x"] for p in plants]
        ys = [p["y"] for p in plants]

        self.x_min = min(xs)
        self.x_max = max(xs)
        self.y_min = min(ys)
        self.y_max = max(ys)

        # Guard against degenerate single-point bounding boxes
        if self.x_max == self.x_min:
            self.x_min -= CELL_SIZE / 2
            self.x_max += CELL_SIZE / 2
        if self.y_max == self.y_min:
            self.y_min -= CELL_SIZE / 2
            self.y_max += CELL_SIZE / 2

        self.dx = CELL_SIZE
        self.dy = CELL_SIZE
        self.n_cols = math.ceil((self.x_max - self.x_min) / CELL_SIZE)
        self.n_rows = math.ceil((self.y_max - self.y_min) / CELL_SIZE)

        # {(col, row): {"n": int, "t_wins": int, "det": int}}
        self.cells: Dict[CellKey, dict] = {}

        for p in plants:
            col = min(int(math.floor((p["x"] - self.x_min) / self.dx)), self.n_cols - 1)
            row = min(int(math.floor((p["y"] - self.y_min) / self.dy)), self.n_rows - 1)
            key = (col, row)
            if key not in self.cells:
                self.cells[key] = {"n": 0, "t_wins": 0, "det": 0}
            self.cells[key]["n"] += 1
            self.cells[key]["t_wins"] += p["t_won"]
            self.cells[key]["det"] += p["detonated"]

    def game_corners(self, col: int, row: int) -> List[Tuple[float, float]]:
        """Return the 4 game-world (x, y) corners of a grid cell (CCW)."""
        x0 = self.x_min + col * self.dx
        x1 = x0 + self.dx
        y0 = self.y_min + row * self.dy
        y1 = y0 + self.dy
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def game_centroid(self, col: int, row: int) -> Tuple[float, float]:
        x0 = self.x_min + col * self.dx
        y0 = self.y_min + row * self.dy
        return (x0 + self.dx / 2.0, y0 + self.dy / 2.0)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _cell_metric(cell: dict, metric: str) -> float:
    n = cell["n"]
    if metric == "t_win":
        return cell["t_wins"] / n
    if metric == "det":
        return cell["det"] / n
    raise ValueError(metric)


def _draw_site_subplot(
    ax: matplotlib.axes.Axes,
    map_name: str,
    site: str,
    metric: str,
    metric_title: str,
    cmap_name: str,
    buckets: GridBuckets,
    map_img: Optional[np.ndarray],
    mean_z: float,
) -> None:
    """Draw one subplot: zoomed map + colored grid cells."""
    ax.set_aspect("equal")
    ax.set_title(f"Site {site} : {metric_title}", fontsize=12)
    ax.axis("off")

    # ---- Background map image ----
    if map_img is not None:
        h, w = map_img.shape[:2]
        ax.imshow(map_img, origin="upper", extent=(0, w, h, 0), zorder=0)
        img_w, img_h = w, h
    else:
        ax.set_facecolor("#1a1a1a")
        img_w, img_h = 1024, 1024

    # ---- Colormap / norm ----
    valid_vals = [
        _cell_metric(c, metric)
        for c in buckets.cells.values()
        if c["n"] >= MIN_SAMPLES
    ]
    if valid_vals:
        norm = mcolors.Normalize(vmin=0.0, vmax=max(valid_vals) if max(valid_vals) > 0 else 1.0)
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = matplotlib.colormaps[cmap_name]

    # ---- Collect pixel bounding box of all grid corners for zoom ----
    px_all: List[float] = []
    py_all: List[float] = []

    for col in range(buckets.n_cols):
        for row in range(buckets.n_rows):
            for gx, gy in buckets.game_corners(col, row):
                px, py, _ = game_to_pixel(map_name, (gx, gy, mean_z))
                px_all.append(px)
                py_all.append(py)

    px_min = min(px_all) - ZOOM_BUFFER
    px_max = max(px_all) + ZOOM_BUFFER
    py_min = min(py_all) - ZOOM_BUFFER
    py_max = max(py_all) + ZOOM_BUFFER

    # Clamp to image bounds
    px_min = max(px_min, 0)
    px_max = min(px_max, img_w)
    py_min = max(py_min, 0)
    py_max = min(py_max, img_h)

    # ---- Draw grid cells ----
    annotation_items: List[Tuple[float, float, str, bool]] = []  # (px, py, text, has_data)

    for col in range(buckets.n_cols):
        for row in range(buckets.n_rows):
            corners_game = buckets.game_corners(col, row)
            pixels = [
                game_to_pixel(map_name, (gx, gy, mean_z))[:2]
                for gx, gy in corners_game
            ]

            cell = buckets.cells.get((col, row))
            if cell is not None and cell["n"] >= MIN_SAMPLES:
                val = _cell_metric(cell, metric)
                facecolor = cmap(norm(val))
                alpha = 0.78
                has_data = True
            else:
                facecolor = "#333333"
                alpha = 0.35
                has_data = False

            poly = mpatches.Polygon(
                np.array(pixels),
                closed=True,
                facecolor=facecolor,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.4,
                zorder=1,
            )
            ax.add_patch(poly)

            # Centroid for annotation
            cpx = sum(p[0] for p in pixels) / 4
            cpy = sum(p[1] for p in pixels) / 4
            n = cell["n"] if cell is not None else 0
            annotation_items.append((cpx, cpy, str(n) if n > 0 else "", has_data))

    # ---- Annotations ----
    for cpx, cpy, label, has_data in annotation_items:
        if not label:
            continue
        ax.text(
            cpx, cpy, label,
            ha="center", va="center",
            fontsize=7,
            color="white" if has_data else "#777777",
            fontweight="bold" if has_data else "normal",
            zorder=2,
        )

    # ---- Zoom ----
    ax.set_xlim(px_min, px_max)
    # image is origin="upper" → y increases downward
    ax.set_ylim(py_max, py_min)

    # ---- Colorbar ----
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(metric_title, fontsize=8)


# ---------------------------------------------------------------------------
# Per-map render
# ---------------------------------------------------------------------------

_METRICS: List[Tuple[str, str, str]] = [
    ("t_win", "T Win Rate",   _CMAP_WIN),
    ("det",   "Detonation Rate", _CMAP_DET),
]


def render_map(
    map_name: str,
    site_data: Dict[str, List[PlantRecord]],
) -> None:
    sites = sorted(site_data.keys())  # alphabetical: A, B
    n_sites = len(sites)
    n_metrics = len(_METRICS)

    fig, axes = plt.subplots(
        n_sites, n_metrics,
        figsize=(10 * n_metrics, 9 * n_sites),
        squeeze=False,
    )
    fig.suptitle(f"{map_name} : Bomb Plant Location Analysis", fontsize=16, fontweight="bold")

    for r_idx, site in enumerate(sites):
        plants = site_data[site]
        buckets = GridBuckets(plants)
        mean_z = sum(p["z"] for p in plants) / len(plants)

        # Pick upper or lower map image based on the site's mean Z level
        lower = is_position_on_lower_level(map_name, (0.0, 0.0, mean_z))
        img_suffix = "_lower" if lower else ""
        img_path = MAPS_DIR / f"{map_name}{img_suffix}.png"
        if not img_path.exists() and lower:
            # Fall back to the upper image if a _lower variant doesn't exist
            img_path = MAPS_DIR / f"{map_name}.png"
        map_img = plt.imread(str(img_path)) if img_path.exists() else None
        if map_img is None:
            print(f"  [warn] map image not found: {img_path}")

        total = len(plants)
        print(f"  Site {site}: {total} plants, {buckets.n_cols}×{buckets.n_rows} grid ({len(buckets.cells)} non-empty cells)")

        for c_idx, (metric, metric_title, cmap_name) in enumerate(_METRICS):
            ax = axes[r_idx][c_idx]
            _draw_site_subplot(
                ax=ax,
                map_name=map_name,
                site=site,
                metric=metric,
                metric_title=metric_title,
                cmap_name=cmap_name,
                buckets=buckets,
                map_img=map_img,
                mean_z=mean_z,
            )
            ax.set_xlabel(f"n={total} plants", fontsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{map_name}_plant_importance.png"
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
    assert db.prefire_db is not None

    print("Fetching plant data…")
    plants_by_map = fetch_plants(db.prefire_db)

    if not plants_by_map:
        print("No plant data found : run extract_bomb_plants.py first.")
        sys.exit(0)

    print(f"Maps with data: {sorted(plants_by_map)}")

    for map_name, site_data in sorted(plants_by_map.items()):
        total = sum(len(v) for v in site_data.values())
        print(f"\nRendering {map_name}  ({total} plants across {sorted(site_data)} sites)…")
        try:
            render_map(map_name, site_data)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"  [FAILED] {map_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()

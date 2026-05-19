"""Shared geometry service providing nav-mesh and visibility operations for a single CS2 map."""

from __future__ import annotations

import heapq
import json
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import logfire
from awpy.data import NAVS_DIR, TRIS_DIR
from awpy.nav import Nav
from awpy.visibility import VisibilityChecker


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Eye height above the player's foot origin (Z), in CS units.
EYE_HEIGHT: float = 64.0

# Directory for pre-baked per-map area visibility caches (project-local).
VIS_CACHE_DIR: pathlib.Path = pathlib.Path("storage") / "vis_cache"

# Number of worker processes used when building the visibility matrix.
_VIS_BUILD_WORKERS: int = 10

# Pre-cull radius for the matrix build: matches the runtime max_distance default
# so areas beyond this can never appear in a vision cone result anyway.
_VIS_MAX_AREA_DIST_SQ: float = 3_000.0 ** 2


# ---------------------------------------------------------------------------
# Module-level worker helpers for ProcessPoolExecutor.
# Must live at module level so worker processes can import them on all platforms.
# ---------------------------------------------------------------------------

# Per-process state populated once by _vis_initializer before tasks run.
_worker_checker_inst: Optional[VisibilityChecker] = None
_worker_corners_map: Dict[int, List[Tuple[float, float, float]]] = {}


def _vis_initializer(
    tri_path: str,
    corners: Dict[int, List[Tuple[float, float, float]]],
) -> None:
    """One-time setup run in each worker process before any tasks are dispatched."""
    global _worker_checker_inst, _worker_corners_map
    _worker_checker_inst = VisibilityChecker(path=pathlib.Path(tri_path))
    _worker_corners_map = corners


def _vis_compute_area(
    src_id: int,
    src_eye: Tuple[float, float, float],
    nearby_targets: List[Tuple[int, float, float, float]],
) -> Tuple[int, List[int]]:
    """Return *(src_id, visible_area_ids)* for one source area."""
    visible: List[int] = [src_id]
    for tgt_id, tcx, tcy, tcz in nearby_targets:
        if _worker_checker_inst.is_visible(src_eye, (tcx, tcy, tcz)):  # type: ignore[union-attr]
            visible.append(tgt_id)
            continue
        for cx, cy, cz in _worker_corners_map.get(tgt_id, ()):
            if _worker_checker_inst.is_visible(src_eye, (cx, cy, cz)):  # type: ignore[union-attr]
                visible.append(tgt_id)
                break
    return src_id, visible


class GeometryService:
    """Loads and caches AWPY nav mesh and VisibilityChecker for one map.

    Construct once per map name and inject into any service that needs
    geometric operations (ray-casting, walk-distance, area lookup).  Both
    nav and visibility loading are optional — the service degrades gracefully
    when the required data files are absent, and callers should guard with
    ``is_nav_available`` / ``is_visibility_available`` before use.

    Thread safety: the object is safe to read from multiple threads after
    construction; writes only happen during lazy Dijkstra caching inside
    ``walk_distance`` / ``get_distances_from``, which should be called from
    a single thread (e.g. the demo-parser worker).
    """

    def __init__(self, map_name: str) -> None:
        self._map_name = map_name
        self._checker: Optional[VisibilityChecker] = self._load_checker(map_name)
        self._nav: Optional[Nav] = self._load_nav(map_name)
        # Pre-built list of (area_id, cx, cy, cz) for nearest-area lookup
        self._centroids: List[Tuple[int, float, float, float]] = self._build_centroids()
        # Per-area corner lists for polygon-level vision cone testing
        self._corners: Dict[int, List[Tuple[float, float, float]]] = self._build_corners()
        # Lazy single-source Dijkstra cache: {from_area_id: {to_area_id: distance}}
        self._dist_cache: Dict[int, Dict[int, float]] = {}
        # O(1) centroid lookup by area_id (derived from _centroids list)
        self._centroid_map: Dict[int, Tuple[float, float, float]] = {
            aid: (cx, cy, cz) for aid, cx, cy, cz in self._centroids
        }
        # Pre-baked inter-area visibility matrix; None until precompute_visibility() is called
        self._vis_matrix: Optional[Dict[int, frozenset]] = None
        # Lazy-built adjacency list with precomputed centroid-to-centroid edge weights.
        # {area_id: [(neighbor_id, edge_distance), ...]}; populated on first access.
        self._adj_weights_cache: Optional[Dict[int, List[Tuple[int, float]]]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_visibility_available(self) -> bool:
        """True when the .tri geometry file was loaded successfully."""
        return self._checker is not None

    @property
    def is_nav_available(self) -> bool:
        """True when the .json nav mesh file was loaded successfully."""
        return self._nav is not None

    @property
    def is_vis_matrix_available(self) -> bool:
        """True when the inter-area visibility matrix has been loaded or built."""
        return self._vis_matrix is not None

    @property
    def adj_weights(self) -> Dict[int, List[Tuple[int, float]]]:
        """Adjacency list with precomputed centroid-to-centroid edge distances.

        Built lazily on first access and cached for all subsequent Dijkstra calls.
        Returns ``{}`` when nav is unavailable.
        """
        if self._adj_weights_cache is not None:
            return self._adj_weights_cache
        if self._nav is None:
            self._adj_weights_cache = {}
            return {}
        areas = self._nav.areas
        result: Dict[int, List[Tuple[int, float]]] = {}
        for u_id, u_area in areas.items():
            u_c = u_area.centroid
            neighbours: List[Tuple[int, float]] = []
            for v_id in u_area.connections:
                v_area = areas.get(v_id)
                if v_area is None:
                    continue
                v_c = v_area.centroid
                w = math.sqrt(
                    (u_c.x - v_c.x) ** 2
                    + (u_c.y - v_c.y) ** 2
                    + (u_c.z - v_c.z) ** 2
                )
                neighbours.append((v_id, w))
            result[u_id] = neighbours
        self._adj_weights_cache = result
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_visible(
        self,
        pos_a: Tuple[float, float, float],
        pos_b: Tuple[float, float, float],
    ) -> bool:
        """Return True if the ray from *pos_a* to *pos_b* is unobstructed.

        Returns False (not raising) when the visibility checker is unavailable.
        """
        if self._checker is None:
            return False
        return self._checker.is_visible(pos_a, pos_b)

    def all_areas(self) -> Dict[int, Any]:
        """Return the full ``{area_id: NavArea}`` mapping, or ``{}`` if unavailable."""
        if self._nav is None:
            return {}
        return self._nav.areas

    def get_area_at(self, x: float, y: float, z: float) -> Optional[int]:
        """Return the area_id of the nearest nav area to *(x, y, z)*.

        Strategy: nearest XY-plane centroid whose Z value is within 200 units.
        Falls back to absolute nearest centroid (3D Euclidean) when no area
        passes the vertical proximity filter, which handles maps with multiple
        overlapping floors.

        Returns None when nav is unavailable or no areas exist.
        """
        if not self._centroids:
            return None

        best_id: Optional[int] = None
        best_dxy_sq: float = math.inf
        best_fallback_id: Optional[int] = None
        best_fallback_d3_sq: float = math.inf

        for area_id, cx, cy, cz in self._centroids:
            dxy_sq = (x - cx) ** 2 + (y - cy) ** 2
            dz = abs(z - cz)

            if dz < 200.0 and dxy_sq < best_dxy_sq:
                best_dxy_sq = dxy_sq
                best_id = area_id

            d3_sq = dxy_sq + (z - cz) ** 2
            if d3_sq < best_fallback_d3_sq:
                best_fallback_d3_sq = d3_sq
                best_fallback_id = area_id

        return best_id if best_id is not None else best_fallback_id

    def get_distances_from(self, from_area_id: int) -> Dict[int, float]:
        """Return ``{area_id: walk_distance}`` for all areas reachable from *from_area_id*.

        Runs Dijkstra on first call for a given source; subsequent calls return
        the cached result.  Returns ``{}`` when nav is unavailable.
        """
        if self._nav is None:
            return {}
        if from_area_id not in self._dist_cache:
            self._dist_cache[from_area_id] = self._dijkstra(from_area_id)
        return self._dist_cache[from_area_id]

    def walk_distance(self, from_area_id: int, to_area_id: int) -> float:
        """Return the walk distance (sum of centroid-to-centroid edge lengths) between two areas.

        Uses ``get_distances_from`` with lazy Dijkstra caching.
        Returns ``math.inf`` when no path exists or nav is unavailable.
        """
        if self._nav is None:
            return math.inf
        if from_area_id == to_area_id:
            return 0.0
        return self.get_distances_from(from_area_id).get(to_area_id, math.inf)

    def get_areas_in_vision_cone(
        self,
        eye_pos: Tuple[float, float, float],
        yaw_deg: float,
        pitch_deg: float,
        fov_half_deg: float = 53.0,
        max_distance: float = 3000.0,
    ) -> Set[int]:
        """Return the set of nav area IDs visible within the player's vision cone.

        Uses three-gate filtering for efficiency:

        1. **Distance cull** — skip centroids beyond *max_distance*.
        2. **FOV angle** — dot-product filter against the view-direction vector.
        3. **LOS ray-cast** — confirm geometric line-of-sight to the centroid.
           Gate 3 is omitted when the visibility checker is unavailable, in which
           case angle-filtered results are returned directly.

        Coordinate conventions (CS2 / AWPY / demoparser2):

        * yaw 0° = +X axis; increases counter-clockwise viewed from above.
        * pitch 0° = horizontal; positive pitch = looking downward.

        Args:
            eye_pos:      Player eye position ``(x, y, z)`` in world space.
            yaw_deg:      Horizontal view angle from the ticks DataFrame.
            pitch_deg:    Vertical view angle from the ticks DataFrame.
            fov_half_deg: Half the horizontal FOV cone (default 53° = 106° full).
            max_distance: Distance beyond which areas are ignored (CS units).

        Returns:
            Set of area IDs whose centroids are inside the cone and pass LOS.
            Returns an empty set when nav is unavailable.
        """
        if not self._centroids:
            return set()

        ex, ey, ez = eye_pos
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        # View-direction unit vector (yaw CCW from +X; pitch positive = down)
        vx = math.cos(yaw_rad) * math.cos(pitch_rad)
        vy = math.sin(yaw_rad) * math.cos(pitch_rad)
        vz = -math.sin(pitch_rad)

        cos_fov = math.cos(math.radians(fov_half_deg))
        max_dist_sq = max_distance * max_distance

        result: Set[int] = set()

        if self._vis_matrix is not None:
            # Fast path: structural LOS is pre-baked — only gates 1 and 2 needed.
            # Resolve the player's nav area from foot position (eye_pos minus eye height).
            player_area_id = self.get_area_at(ex, ey, ez - EYE_HEIGHT)
            candidates = (
                self._vis_matrix.get(player_area_id) if player_area_id is not None else None
            )
            if candidates is not None:
                for area_id in candidates:
                    pos = self._centroid_map.get(area_id)
                    if pos is None:
                        continue
                    cx, cy, cz = pos
                    dx = cx - ex
                    dy = cy - ey
                    dz = cz - ez
                    dist_sq = dx * dx + dy * dy + dz * dz
                    # Gate 1: distance
                    if dist_sq > max_dist_sq:
                        continue
                    dist = math.sqrt(dist_sq)
                    if dist < 1e-6:
                        result.add(area_id)
                        continue
                    # Gate 2: FOV angle (structural LOS confirmed by matrix)
                    dot = (dx * vx + dy * vy + dz * vz) / dist
                    if dot >= cos_fov:
                        result.add(area_id)
                        continue
                    # Centroid behind player — test corners for FOV (no LOS re-cast needed)
                    for corner_x, corner_y, corner_z in self._corners.get(area_id, ()):
                        cdx = corner_x - ex
                        cdy = corner_y - ey
                        cdz = corner_z - ez
                        cdist_sq = cdx * cdx + cdy * cdy + cdz * cdz
                        if cdist_sq > max_dist_sq:
                            continue
                        cdist = math.sqrt(cdist_sq)
                        if cdist < 1e-6:
                            result.add(area_id)
                            break
                        cdot = (cdx * vx + cdy * vy + cdz * vz) / cdist
                        if cdot >= cos_fov:
                            result.add(area_id)
                            break
                return result
            # Player area unknown — fall through to full scan path

        # Full scan path: per-frame ray-casts (used when vis matrix is absent or
        # when the player's current nav area could not be resolved).
        for area_id, cx, cy, cz in self._centroids:
            dx = cx - ex
            dy = cy - ey
            dz = cz - ez
            dist_sq = dx * dx + dy * dy + dz * dz

            # Gate 1: distance pre-filter
            if dist_sq > max_dist_sq:
                continue

            dist = math.sqrt(dist_sq)
            if dist < 1e-6:
                # Player is standing in this area — always in cone
                result.add(area_id)
                continue

            # Gate 2: FOV angle (dot product with unit view vector)
            dot = (dx * vx + dy * vy + dz * vz) / dist
            centroid_in_fov = dot >= cos_fov

            # Gate 3: LOS ray-cast on centroid (skipped when checker unavailable)
            centroid_visible = centroid_in_fov and (
                self._checker is None
                or self._checker.is_visible(eye_pos, (cx, cy, cz))
            )

            if centroid_visible:
                result.add(area_id)
                continue

            # Centroid failed Gate 2 or Gate 3 — try individual polygon corners.
            # A large nav area may still be partly visible even when its centroid
            # is occluded or sits just outside the FOV cone boundary.
            for corner_x, corner_y, corner_z in self._corners.get(area_id, ()):
                cdx = corner_x - ex
                cdy = corner_y - ey
                cdz = corner_z - ez
                cdist_sq = cdx * cdx + cdy * cdy + cdz * cdz
                if cdist_sq > max_dist_sq:
                    continue
                cdist = math.sqrt(cdist_sq)
                if cdist < 1e-6:
                    result.add(area_id)
                    break
                cdot = (cdx * vx + cdy * vy + cdz * vz) / cdist
                if cdot < cos_fov:
                    continue
                if self._checker is not None and not self._checker.is_visible(
                    eye_pos, (corner_x, corner_y, corner_z)
                ):
                    continue
                result.add(area_id)
                break

        return result

    def precompute_distances(self) -> None:
        """Eagerly run Dijkstra from every area.

        Call before a hot loop that will query many source/destination pairs.
        For a typical CS2 map (~1 500 areas) this takes a few seconds in Python.
        """
        if self._nav is None:
            return
        area_ids = list(self._nav.areas.keys())
        for area_id in area_ids:
            if area_id not in self._dist_cache:
                self._dist_cache[area_id] = self._dijkstra(area_id)
        logfire.info(
            "geometry: precompute complete",
            map_name=self._map_name,
            area_count=len(area_ids),
        )

    def precompute_visibility(self) -> None:
        """Load or build the per-map inter-area structural visibility matrix.

        If a valid cache exists at ``VIS_CACHE_DIR/{map_name}.vis.json`` it is
        loaded immediately (no VisibilityChecker required).  Otherwise the matrix
        is built from scratch via ray-casts and written out for future use.

        Once this method returns, :meth:`get_areas_in_vision_cone` switches to a
        fast path that omits per-frame ray-casts (gate 3) and applies only the
        distance and FOV-angle filters.  The structural LOS result is read from
        the pre-baked matrix keyed on the player's current nav area.

        Does nothing when:

        * Nav mesh is unavailable (no areas to process).
        * No cache exists **and** the .tri file is absent (cannot build matrix).
        """
        if not self.is_nav_available:
            return

        loaded = self._load_vis_cache()
        if loaded is not None:
            self._vis_matrix = loaded
            logfire.info(
                "geometry: vis matrix loaded from cache",
                map_name=self._map_name,
                area_count=len(loaded),
            )
            return

        if not self.is_visibility_available:
            logfire.warning(
                "geometry: cannot build vis matrix — .tri file missing",
                map_name=self._map_name,
                hint="run 'awpy get tris' then parse a demo to build the cache",
            )
            return

        matrix = self._build_vis_matrix()
        self._save_vis_cache(matrix)
        self._vis_matrix = matrix

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_checker(self, map_name: str) -> Optional[VisibilityChecker]:
        tri_path: pathlib.Path = TRIS_DIR / f"{map_name}.tri"
        if not tri_path.exists():
            logfire.warning(
                "geometry: .tri file not found — visibility checks unavailable",
                map_name=map_name,
                expected_path=str(tri_path),
                hint="run 'awpy get tris' to download geometry files",
            )
            return None
        logfire.info("geometry: loading visibility checker", map_name=map_name, tri_path=str(tri_path))
        return VisibilityChecker(path=tri_path)

    def _load_nav(self, map_name: str) -> Optional[Nav]:
        nav_path: pathlib.Path = NAVS_DIR / f"{map_name}.json"
        if not nav_path.exists():
            logfire.warning(
                "geometry: .json nav file not found — nav operations unavailable",
                map_name=map_name,
                expected_path=str(nav_path),
                hint="run 'awpy get navs' to download navigation meshes",
            )
            return None
        try:
            logfire.info("geometry: loading nav mesh", map_name=map_name, nav_path=str(nav_path))
            return Nav.from_json(nav_path)
        except Exception:
            logfire.exception("geometry: failed to load nav mesh", map_name=map_name)
            return None

    def _build_centroids(self) -> List[Tuple[int, float, float, float]]:
        """Pre-build the centroid list once so ``get_area_at`` is O(N) not O(N log N)."""
        if self._nav is None:
            return []
        result: List[Tuple[int, float, float, float]] = []
        for area_id, area in self._nav.areas.items():
            c = area.centroid
            result.append((area_id, float(c.x), float(c.y), float(c.z)))
        return result

    def _build_corners(self) -> Dict[int, List[Tuple[float, float, float]]]:
        """Pre-build per-area corner lists for polygon-level vision cone testing."""
        if self._nav is None:
            return {}
        result: Dict[int, List[Tuple[float, float, float]]] = {}
        for area_id, area in self._nav.areas.items():
            corners = getattr(area, "corners", None)
            if corners:
                fallback_z = float(area.centroid.z)
                result[area_id] = [
                    (float(c.x), float(c.y), float(getattr(c, "z", fallback_z)))
                    for c in corners
                ]
        return result

    def _dijkstra(self, source_id: int) -> Dict[int, float]:
        """Single-source Dijkstra over the nav mesh from *source_id*.

        Edge weight is the Euclidean distance between area centroids.
        Returns ``{area_id: distance}`` for all reachable areas.
        """
        areas = self._nav.areas  # type: ignore[union-attr]
        dist: Dict[int, float] = {source_id: 0.0}
        heap: List[Tuple[float, int]] = [(0.0, source_id)]

        while heap:
            d, u_id = heapq.heappop(heap)
            if d > dist.get(u_id, math.inf):
                continue
            u_area = areas.get(u_id)
            if u_area is None:
                continue
            u_c = u_area.centroid
            for v_id in u_area.connections:
                v_area = areas.get(v_id)
                if v_area is None:
                    continue
                v_c = v_area.centroid
                edge_w = math.sqrt(
                    (u_c.x - v_c.x) ** 2
                    + (u_c.y - v_c.y) ** 2
                    + (u_c.z - v_c.z) ** 2
                )
                new_d = d + edge_w
                if new_d < dist.get(v_id, math.inf):
                    dist[v_id] = new_d
                    heapq.heappush(heap, (new_d, v_id))

        return dist

    def _build_vis_matrix(self) -> Dict[int, frozenset]:
        """Build the inter-area structural visibility matrix using ray-casts.

        For each source area the eye position is the centroid XY with
        Z = centroid.z + ``EYE_HEIGHT``.  Each target area is tested centroid-first;
        if the centroid cast fails, individual corners are tried.  This mirrors the
        full gate-3 logic used in :meth:`get_areas_in_vision_cone`.

        Work is parallelised across ``_VIS_BUILD_WORKERS`` *processes* so ray-casts
        run with true CPU parallelism regardless of GIL behaviour.  Target areas
        beyond ``_VIS_MAX_AREA_DIST_SQ`` are pre-culled before dispatch so workers
        never fire casts that gate 1 would discard at runtime.

        Requires ``_checker`` (VisibilityChecker) to be non-None.

        Returns:
            ``{source_area_id: frozenset(visible_area_ids)}`` for every area.
        """
        areas = self.all_areas()
        target_centroids: List[Tuple[int, float, float, float]] = list(self._centroids)

        # Pre-build per-source task tuples with distance-culled target lists.
        source_tasks: List[Tuple[int, Tuple[float, float, float], List[Tuple[int, float, float, float]]]] = []
        for src_id, src_area in areas.items():
            src_c = src_area.centroid
            scx, scy, scz = float(src_c.x), float(src_c.y), float(src_c.z)
            src_eye: Tuple[float, float, float] = (scx, scy, scz + EYE_HEIGHT)
            nearby = [
                (tid, tcx, tcy, tcz)
                for tid, tcx, tcy, tcz in target_centroids
                if tid != src_id
                and (tcx - scx) ** 2 + (tcy - scy) ** 2 + (tcz - scz) ** 2 <= _VIS_MAX_AREA_DIST_SQ
            ]
            source_tasks.append((src_id, src_eye, nearby))

        tri_path = str(TRIS_DIR / f"{self._map_name}.tri")
        total = len(source_tasks)
        matrix: Dict[int, frozenset] = {}
        done_count = 0
        _bar_width = 40
        t0 = time.monotonic()

        avg_nearby = sum(len(t[2]) for t in source_tasks) / max(total, 1)
        logfire.info(
            "geometry: building visibility matrix",
            map_name=self._map_name,
            area_count=total,
            workers=_VIS_BUILD_WORKERS,
            avg_nearby_targets=round(avg_nearby, 1),
        )

        with ProcessPoolExecutor(
            max_workers=_VIS_BUILD_WORKERS,
            initializer=_vis_initializer,
            initargs=(tri_path, self._corners),
        ) as executor:
            futures = {
                executor.submit(_vis_compute_area, sid, seye, ntgts): sid
                for sid, seye, ntgts in source_tasks
            }
            for future in as_completed(futures):
                src_id, vis_list = future.result()
                matrix[src_id] = frozenset(vis_list)
                done_count += 1
                if done_count % 10 == 0 or done_count == total:
                    filled = int(_bar_width * done_count / total)
                    elapsed = time.monotonic() - t0
                    eta = (elapsed / done_count * (total - done_count)) if done_count else 0.0
                    bar = "|" * filled + " " * (_bar_width - filled)
                    sys.stdout.write(
                        f"\r  [{bar}] {done_count}/{total}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s  "
                    )
                    sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()
        elapsed = time.monotonic() - t0
        logfire.info(
            "geometry: visibility matrix built",
            map_name=self._map_name,
            area_count=total,
            elapsed_s=round(elapsed, 1),
        )
        return matrix

    def _load_vis_cache(self) -> Optional[Dict[int, frozenset]]:
        """Load the visibility matrix from the JSON cache file.

        Returns:
            Parsed matrix dict or ``None`` if the file is absent, malformed,
            or has a map-name / version mismatch.
        """
        cache_path = VIS_CACHE_DIR / f"{self._map_name}.vis.json"
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("version") != 1 or data.get("map_name") != self._map_name:
                logfire.warning(
                    "geometry: vis cache version/map mismatch — will rebuild",
                    map_name=self._map_name,
                    cache_path=str(cache_path),
                )
                return None
            return {int(k): frozenset(v) for k, v in data["areas"].items()}
        except Exception:
            logfire.exception(
                "geometry: failed to load vis cache — will rebuild if possible",
                map_name=self._map_name,
                cache_path=str(cache_path),
            )
            return None

    def _save_vis_cache(self, matrix: Dict[int, frozenset]) -> None:
        """Serialise the visibility matrix to a JSON cache file atomically."""
        VIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = VIS_CACHE_DIR / f"{self._map_name}.vis.json"
        tmp_path = cache_path.with_suffix(".tmp")
        data = {
            "version": 1,
            "map_name": self._map_name,
            "areas": {str(k): sorted(v) for k, v in matrix.items()},
        }
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, separators=(",", ":"))
            os.replace(tmp_path, cache_path)
            logfire.info(
                "geometry: vis cache saved",
                map_name=self._map_name,
                cache_path=str(cache_path),
            )
        except Exception:
            logfire.exception(
                "geometry: failed to save vis cache",
                map_name=self._map_name,
                cache_path=str(cache_path),
            )
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

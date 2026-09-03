"""
The discretised field: grid construction, navigation, and travel-time model.

:class:`Field` is a pure, stateless-after-construction geometry service. It
knows nothing about scores, inventory, or the match clock -- it answers only
"what zone is this cell?", "how far is the nearest pantry cell?", and "how long
does driving that far take?".

Performance note
----------------
A multi-source Dijkstra distance field is precomputed **once per zone** at
construction. Macro-action travel distance is then an O(1) array lookup from
any cell, and path reconstruction is a greedy descent. This keeps `step()`
cheap enough for millions of RL timesteps.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import FieldConfig
from .timing import cruise_time, traverse_time  # re-exported for convenience
from .zones import DEFAULT_LAYOUT, Zone, ZoneRect

#: A grid coordinate, ``(col, row)`` == ``(x_index, y_index)``.
Cell = Tuple[int, int]

_CARDINAL: Tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL: Tuple[Cell, ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class Field:
    """Discrete grid representation of the Harvest Havoc field.

    Parameters
    ----------
    config:
        Geometry and grid resolution.
    layout:
        Ordered zone rectangles; later entries override earlier ones. Defaults
        to :data:`harvest_havoc.zones.DEFAULT_LAYOUT`.
    blocked:
        Optional boolean array of shape ``(n_cols, n_rows)`` marking
        impassable cells (field obstacles, the pantry structure itself, ...).
        Defaults to fully open. Provided as an extension point -- navigation
        already routes around blocked cells correctly.

    Attributes
    ----------
    zone_grid: np.ndarray
        ``(n_cols, n_rows)`` int array of :class:`~harvest_havoc.zones.Zone`
        values. Exactly one label per cell.
    """

    def __init__(
        self,
        config: Optional[FieldConfig] = None,
        layout: Optional[Sequence[ZoneRect]] = None,
        blocked: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config or FieldConfig()
        self.layout: List[ZoneRect] = list(
            layout if layout is not None else DEFAULT_LAYOUT
        )

        self.n_cols, self.n_rows = self.config.shape
        self.cell_size = self.config.cell_size_ft

        if blocked is None:
            self.blocked = np.zeros((self.n_cols, self.n_rows), dtype=bool)
        else:
            if blocked.shape != (self.n_cols, self.n_rows):
                raise ValueError(
                    f"blocked has shape {blocked.shape}, expected "
                    f"{(self.n_cols, self.n_rows)}"
                )
            self.blocked = blocked.astype(bool)

        self.zone_grid = self._build_zone_grid()
        # Keyed by either a Zone member or an arbitrary string label; see
        # `register_target`. One multi-source Dijkstra field per key.
        self._distance_fields: Dict[Hashable, np.ndarray] = {}
        self._precompute_distance_fields()

    # ------------------------------------------------------------------ build

    def _build_zone_grid(self) -> np.ndarray:
        """Rasterise the layout rectangles into a per-cell zone label array."""
        grid = np.full((self.n_cols, self.n_rows), int(Zone.NEUTRAL), dtype=np.int16)
        for rect in self.layout:                      # later rects override
            for cx in range(self.n_cols):
                for cy in range(self.n_rows):
                    x_ft, y_ft = self.cell_to_feet((cx, cy))
                    if rect.contains(x_ft, y_ft):
                        grid[cx, cy] = int(rect.zone)
        return grid

    def _precompute_distance_fields(self) -> None:
        """Run one multi-source Dijkstra per zone that actually has cells."""
        for zone in Zone:
            cells = self.cells_of_zone(zone)
            if cells:
                self._distance_fields[zone] = self._dijkstra(cells)

    def _dijkstra(self, sources: Sequence[Cell]) -> np.ndarray:
        """Shortest driving distance (feet) from every cell to any source.

        8-connected if ``allow_diagonal_movement``; diagonal steps cost
        ``diagonal_cost_multiplier * cell_size``. Blocked cells are
        unreachable and receive ``inf``.
        """
        dist = np.full((self.n_cols, self.n_rows), np.inf, dtype=np.float64)
        heap: List[Tuple[float, int, int]] = []

        for (cx, cy) in sources:
            if self.blocked[cx, cy]:
                continue
            dist[cx, cy] = 0.0
            heapq.heappush(heap, (0.0, cx, cy))

        steps = self._neighbour_offsets()
        while heap:
            d, cx, cy = heapq.heappop(heap)
            if d > dist[cx, cy]:
                continue
            for dx, dy, step_cost in steps:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.n_cols and 0 <= ny < self.n_rows):
                    continue
                if self.blocked[nx, ny]:
                    continue
                nd = d + step_cost
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    heapq.heappush(heap, (nd, nx, ny))
        return dist

    def _neighbour_offsets(self) -> Tuple[Tuple[int, int, float], ...]:
        """``(dx, dy, cost_ft)`` triples for every legal single-cell move."""
        out: List[Tuple[int, int, float]] = [
            (dx, dy, self.cell_size) for dx, dy in _CARDINAL
        ]
        if self.config.allow_diagonal_movement:
            diag_cost = self.cell_size * self.config.diagonal_cost_multiplier
            out.extend((dx, dy, diag_cost) for dx, dy in _DIAGONAL)
        return tuple(out)

    # ------------------------------------------------------- coordinate utils

    def cell_to_feet(self, cell: Cell) -> Tuple[float, float]:
        """Centre of `cell` in field coordinates (feet)."""
        cx, cy = cell
        return ((cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size)

    def feet_to_cell(self, x_ft: float, y_ft: float) -> Cell:
        """Grid cell containing the point ``(x_ft, y_ft)``, clamped in-bounds."""
        cx = int(x_ft / self.cell_size)
        cy = int(y_ft / self.cell_size)
        cx = max(0, min(self.n_cols - 1, cx))
        cy = max(0, min(self.n_rows - 1, cy))
        return (cx, cy)

    def in_bounds(self, cell: Cell) -> bool:
        """True if `cell` lies on the grid."""
        cx, cy = cell
        return 0 <= cx < self.n_cols and 0 <= cy < self.n_rows

    def is_passable(self, cell: Cell) -> bool:
        """True if `cell` is on the grid and not blocked."""
        return self.in_bounds(cell) and not bool(self.blocked[cell[0], cell[1]])

    # --------------------------------------------------------------- queries

    def zone_at(self, cell: Cell) -> Zone:
        """Zone label of `cell`."""
        return Zone(int(self.zone_grid[cell[0], cell[1]]))

    def cells_of_zone(self, zone: Zone) -> List[Cell]:
        """Every cell carrying the given zone label."""
        xs, ys = np.nonzero(self.zone_grid == int(zone))
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def has_zone(self, zone: Zone) -> bool:
        """True if any cell carries this label in the current layout."""
        return zone in self._distance_fields

    # ---------------------------------------------------- navigation targets
    #
    # A "target" is any set of goal cells with a precomputed distance field.
    # Zones are registered automatically; `register_target` adds named ones,
    # which is how "the nearest cell outside the kitchen complex" is modelled
    # without inventing a pseudo-zone label for it.

    def register_target(self, key: str, cells: Iterable[Cell]) -> bool:
        """Precompute a distance field to an arbitrary set of goal `cells`.

        Parameters
        ----------
        key:
            Label to retrieve this target by. Re-registering replaces it.
        cells:
            Goal cells. Blocked and out-of-bounds cells are ignored.

        Returns
        -------
        bool
            False if no usable goal cell was supplied (the target is not
            registered in that case).
        """
        usable = [c for c in cells if self.is_passable(c)]
        if not usable:
            self._distance_fields.pop(key, None)
            return False
        self._distance_fields[key] = self._dijkstra(usable)
        return True

    def has_target(self, key: Hashable) -> bool:
        """True if a distance field is registered under `key`."""
        return key in self._distance_fields

    def distance_to_target(self, cell: Cell, key: Hashable) -> float:
        """Driving distance in feet from `cell` to the nearest goal of `key`.

        Returns ``inf`` if the target is unregistered or unreachable.
        """
        dfield = self._distance_fields.get(key)
        if dfield is None:
            return math.inf
        return float(dfield[cell[0], cell[1]])

    def distance_to_zone(self, cell: Cell, zone: Zone) -> float:
        """Driving distance in feet from `cell` to the nearest cell of `zone`.

        Returns ``inf`` if the zone is absent from the layout or unreachable.
        """
        return self.distance_to_target(cell, zone)

    def distance_field(self, key: Hashable) -> Optional[np.ndarray]:
        """Read-only view of the precomputed distance field for `key`.

        `key` may be a :class:`~harvest_havoc.zones.Zone` or a string
        registered via :meth:`register_target`.
        """
        dfield = self._distance_fields.get(key)
        if dfield is None:
            return None
        view = dfield.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------ navigation

    def path_to_zone(self, start: Cell, zone: Zone) -> Optional[List[Cell]]:
        """Cell-by-cell shortest path from `start` into `zone`.

        See :meth:`path_to_target`.
        """
        return self.path_to_target(start, zone)

    def path_to_target(self, start: Cell, key: Hashable) -> Optional[List[Cell]]:
        """Cell-by-cell shortest path from `start` to the nearest goal of `key`.

        The returned list begins with `start` and ends on the first goal cell
        reached. A robot already on a goal cell gets ``[start]``.

        Returns ``None`` if the target is unregistered or unreachable.
        """
        dfield = self._distance_fields.get(key)
        if dfield is None or not math.isfinite(dfield[start[0], start[1]]):
            return None

        steps = self._neighbour_offsets()
        path = [start]
        current = start
        # Bound the walk; a correct descent can never exceed the cell count.
        for _ in range(self.n_cols * self.n_rows):
            if dfield[current[0], current[1]] <= 0.0:
                return path
            best: Optional[Cell] = None
            best_d = dfield[current[0], current[1]]
            for dx, dy, _cost in steps:
                nx, ny = current[0] + dx, current[1] + dy
                if not self.is_passable((nx, ny)):
                    continue
                nd = dfield[nx, ny]
                if nd < best_d - 1e-12:
                    best_d = nd
                    best = (nx, ny)
            if best is None:
                return path  # local minimum: already as close as possible
            path.append(best)
            current = best
        return path

    def path_length_ft(self, path: Sequence[Cell]) -> float:
        """Total driving distance along `path`, in feet."""
        total = 0.0
        for (ax, ay), (bx, by) in zip(path, path[1:]):
            dx, dy = abs(bx - ax), abs(by - ay)
            if dx and dy:
                total += self.cell_size * self.config.diagonal_cost_multiplier
            else:
                total += self.cell_size * max(dx, dy)
        return total

    def step_cost_ft(self, a: Cell, b: Cell) -> float:
        """Driving distance of a single grid step from `a` to `b`, in feet."""
        dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
        if dx > 1 or dy > 1:
            raise ValueError(f"{a} -> {b} is not a single-cell step")
        if dx and dy:
            return self.cell_size * self.config.diagonal_cost_multiplier
        return self.cell_size * max(dx, dy)

    # ------------------------------------------------------------------ misc

    def zone_centroid_cell(self, zone: Zone) -> Optional[Cell]:
        """Cell of `zone` closest to that zone's centroid.

        Handy for rendering and for reporting a canonical "site" position.
        """
        cells = self.cells_of_zone(zone)
        if not cells:
            return None
        arr = np.asarray(cells, dtype=np.float64)
        centre = arr.mean(axis=0)
        idx = int(np.argmin(((arr - centre) ** 2).sum(axis=1)))
        return cells[idx]

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"Field({self.n_cols}x{self.n_rows} cells @ {self.cell_size} ft, "
            f"{self.config.field_length_ft}x{self.config.field_width_ft} ft)"
        )


__all__ = ["Field", "Cell", "traverse_time", "cruise_time"]

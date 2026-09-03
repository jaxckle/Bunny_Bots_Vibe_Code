"""
Mutable match state: the robot, the pantry, the oven, and the clock.

This module defines *what can be true* about a match at an instant. It holds no
rules -- point values and ranking-point logic live in
:mod:`harvest_havoc.scoring`, and time costs live in
:mod:`harvest_havoc.config`. Keeping state dumb makes it trivial to snapshot,
diff, log, and unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple

from .config import TimeConfig
from .field import Cell


class Piece(IntEnum):
    """The two game pieces a robot can hold."""

    CARROT = 0
    CARROT_CAKE = 1

    @property
    def label(self) -> str:
        """Short human-readable name."""
        return "carrot" if self is Piece.CARROT else "cake"


class Phase(IntEnum):
    """Match period. Ordered, so ``phase >= Phase.ENDGAME`` is meaningful."""

    AUTO = 0
    TELEOP = 1
    ENDGAME = 2

    @staticmethod
    def at(t: float, cfg: TimeConfig) -> "Phase":
        """Which phase wall-clock time `t` falls in.

        The boundaries are half-open: ``[0, auto_end)`` is auto,
        ``[auto_end, endgame_start)`` is teleop, and ``[endgame_start, end]``
        is endgame.
        """
        if t < cfg.auto_end_s:
            return Phase.AUTO
        if t < cfg.endgame_start_s:
            return Phase.TELEOP
        return Phase.ENDGAME


class ScoreCategory(str, Enum):
    """Where a point came from. Used for time-allocation attribution."""

    PANTRY_CARROT = "pantry_carrot"
    PANTRY_CAKE = "pantry_cake"
    OVEN = "oven"
    AUTO_LEAVE = "auto_leave"
    PARK = "park"
    HARVEST_HAUL = "harvest_haul"


@dataclass
class ScoreEvent:
    """One scoring occurrence, timestamped.

    The timestamp is what makes the "all points during the endgame period"
    interpretation of the Dinner RP computable, and it is also what lets you
    plot points-per-second over the match after a rollout.
    """

    t: float
    category: ScoreCategory
    points: int
    #: Pantry level 1/2/3 where applicable, else ``None``.
    level: Optional[int] = None
    #: Piece involved where applicable.
    piece: Optional[Piece] = None


@dataclass
class RobotState:
    """Pose, inventory, and motion status of the single robot."""

    #: Current grid cell.
    cell: Cell = (0, 0)

    #: Carrots currently held.
    carrots: int = 0
    #: Carrot cakes currently held.
    cakes: int = 0

    #: True if the robot ended its last action still rolling. Used by the
    #: micro-movement time model so that a straight run of MOVE actions is not
    #: charged a full accel/decel cycle per cell.
    in_motion: bool = False

    #: True once a PARK action has completed successfully. Cleared by any
    #: subsequent action other than WAIT, since driving away or reaching out
    #: to score breaks the parked configuration. The park bonus at the buzzer
    #: requires BOTH this flag and actual occupancy of the own table zone.
    parked: bool = False

    @property
    def held(self) -> int:
        """Total pieces held."""
        return self.carrots + self.cakes

    def free_slots(self, max_inventory: int) -> int:
        """Remaining inventory capacity."""
        return max(0, max_inventory - self.held)

    def add(self, piece: Piece, max_inventory: int) -> bool:
        """Try to pick up `piece`. Returns False if the inventory cap is hit."""
        if self.held >= max_inventory:
            return False
        if piece is Piece.CARROT:
            self.carrots += 1
        else:
            self.cakes += 1
        return True

    def remove(self, piece: Piece) -> bool:
        """Try to release `piece`. Returns False if none is held."""
        if piece is Piece.CARROT and self.carrots > 0:
            self.carrots -= 1
            return True
        if piece is Piece.CARROT_CAKE and self.cakes > 0:
            self.cakes -= 1
            return True
        return False

    def has(self, piece: Piece) -> bool:
        """True if at least one `piece` is held."""
        return (self.carrots if piece is Piece.CARROT else self.cakes) > 0


@dataclass
class Shelf:
    """One pantry shelf (level 1, 2, or 3), with a hard piece capacity.

    A shelf holds at most :attr:`capacity` pieces, carrots and cakes
    **combined**. Pieces can never be removed, so the composition of a shelf
    is a one-way decision: filling it with carrots permanently forfeits Baked
    Up on that shelf.
    """

    level: int
    carrots: int = 0
    cakes: int = 0
    #: Maximum pieces of any type this shelf can hold.
    capacity: int = 3

    @property
    def total(self) -> int:
        """Pieces of any type on this shelf."""
        return self.carrots + self.cakes

    @property
    def free_slots(self) -> int:
        """Remaining space on this shelf."""
        return max(0, self.capacity - self.total)

    @property
    def is_full(self) -> bool:
        """True if no more pieces will fit."""
        return self.total >= self.capacity


@dataclass
class Pantry:
    """The three-shelf pantry."""

    shelves: Dict[int, Shelf] = field(default_factory=dict)

    @classmethod
    def create(cls, levels: Tuple[int, ...], capacity: int = 3) -> "Pantry":
        """Build an empty pantry with the given shelf levels and capacity."""
        return cls(
            shelves={lvl: Shelf(level=lvl, capacity=capacity) for lvl in levels}
        )

    def can_add(self, level: int) -> bool:
        """True if shelf `level` has room for another piece."""
        return not self.shelves[level].is_full

    def add(self, level: int, piece: Piece) -> bool:
        """Record a scored piece on `level`.

        Returns
        -------
        bool
            False if the shelf was already full, in which case nothing is
            recorded. Callers should have checked :meth:`can_add` first; this
            is a safety net so a capacity violation can never be silent.
        """
        shelf = self.shelves[level]
        if shelf.is_full:
            return False
        if piece is Piece.CARROT:
            shelf.carrots += 1
        else:
            shelf.cakes += 1
        return True

    @property
    def levels(self) -> Tuple[int, ...]:
        """Shelf levels, ascending."""
        return tuple(sorted(self.shelves))

    def total_pieces(self) -> int:
        """Every piece on every shelf."""
        return sum(s.total for s in self.shelves.values())

    def total_capacity(self) -> int:
        """Total slots in the pantry across all shelves."""
        return sum(s.capacity for s in self.shelves.values())

    def is_full(self) -> bool:
        """True if every shelf is at capacity."""
        return all(s.is_full for s in self.shelves.values())


@dataclass
class OvenState:
    """The oven and the human-player carrot-to-cake exchange pipeline.

    Only *carrots* deposited into the oven advance the exchange counter; a cake
    put into the oven scores its 2 points and is simply consumed. That is a
    deliberately bad trade, and the agent should learn to avoid it.
    """

    #: Carrots deposited into the oven over the whole match.
    carrots_deposited: int = 0
    #: Cakes deposited into the oven over the whole match.
    cakes_deposited: int = 0
    #: Carrots counted toward the *next* exchange (resets on each conversion).
    carrots_toward_next: int = 0
    #: Wall-clock completion times of exchanges currently in flight.
    pending_exchange_times: List[float] = field(default_factory=list)
    #: Exchanges that were rolled and failed (diagnostics only).
    failed_exchanges: int = 0

    @property
    def pending_count(self) -> int:
        """Number of cakes currently being prepared by the human player."""
        return len(self.pending_exchange_times)


@dataclass
class MatchState:
    """Complete state of a Harvest Havoc match at wall-clock time `t`.

    This object is the single source of truth for the environment. Everything
    the agent observes is a pure function of it.
    """

    #: Wall-clock seconds since the match started. Always in ``[0, 150]``.
    t: float = 0.0

    robot: RobotState = field(default_factory=RobotState)
    pantry: Pantry = field(default_factory=Pantry)
    oven: OvenState = field(default_factory=OvenState)

    #: Carrots the depot can still supply.
    depot_carrots: int = 0
    #: Finished cakes waiting at the farm depot for reintroduction.
    cakes_available: int = 0

    #: One-time autonomous leave-the-kitchen bonus already earned?
    auto_leave_earned: bool = False

    #: Running raw score, excluding endgame park/haul (which are only
    #: evaluated at the buzzer and are added by
    #: :func:`harvest_havoc.scoring.finalize_endgame`).
    raw_score: int = 0

    #: Every scoring occurrence, in time order.
    score_events: List[ScoreEvent] = field(default_factory=list)

    #: Set by the finaliser once the buzzer has sounded.
    finalized: bool = False

    # ------------------------------------------------------------------ clock

    def phase(self, cfg: TimeConfig) -> Phase:
        """Current match phase."""
        return Phase.at(self.t, cfg)

    def time_remaining(self, cfg: TimeConfig) -> float:
        """Seconds left in the match (never negative)."""
        return max(0.0, cfg.match_duration_s - self.t)

    def phase_time_remaining(self, cfg: TimeConfig) -> float:
        """Seconds left in the *current* phase."""
        phase = self.phase(cfg)
        if phase is Phase.AUTO:
            return max(0.0, cfg.auto_end_s - self.t)
        if phase is Phase.TELEOP:
            return max(0.0, cfg.endgame_start_s - self.t)
        return max(0.0, cfg.match_duration_s - self.t)

    def is_over(self, cfg: TimeConfig) -> bool:
        """True once the match clock has expired."""
        return self.t >= cfg.match_duration_s - 1e-9

    # ----------------------------------------------------------------- points

    def award(
        self,
        points: int,
        category: ScoreCategory,
        *,
        level: Optional[int] = None,
        piece: Optional[Piece] = None,
    ) -> int:
        """Add `points` to the raw score and log a :class:`ScoreEvent`.

        Returns the points awarded, so callers can accumulate a step delta.
        """
        self.raw_score += points
        self.score_events.append(
            ScoreEvent(t=self.t, category=category, points=points,
                       level=level, piece=piece)
        )
        return points

    def points_scored_since(self, t0: float) -> int:
        """Total points from events at or after wall-clock time `t0`."""
        return sum(e.points for e in self.score_events if e.t >= t0 - 1e-9)

    def points_by_category(self) -> Dict[ScoreCategory, int]:
        """Raw score broken down by where it came from."""
        out: Dict[ScoreCategory, int] = {c: 0 for c in ScoreCategory}
        for e in self.score_events:
            out[e.category] += e.points
        return out


@dataclass
class ActionRecord:
    """One entry in the environment's time ledger.

    The ledger is the primary artifact for answering the project's actual
    question. Every second of the match belongs to exactly one record, split
    into travel / align / manipulate, and tagged with the points it produced.
    """

    #: Index of the step within the episode.
    step: int
    #: Action id that was requested.
    action: int
    #: Human-readable action name.
    action_name: str
    #: Match phase at the moment the action started.
    phase: Phase
    #: Wall-clock time the action started.
    t_start: float
    #: Wall-clock time the action finished (or was cut off by the buzzer).
    t_end: float
    #: Seconds spent driving.
    travel_s: float = 0.0
    #: Seconds spent aligning to a target.
    align_s: float = 0.0
    #: Seconds spent manipulating a game piece.
    manipulate_s: float = 0.0
    #: Seconds spent idle (WAIT actions).
    idle_s: float = 0.0
    #: Raw points produced by this action.
    points: int = 0
    #: Did the intended interaction complete successfully?
    success: bool = True
    #: Was the action rejected as illegal (no time consumed)?
    illegal: bool = False
    #: Was the action cut short by the end of the match?
    truncated_by_buzzer: bool = False
    #: Free-form detail for debugging.
    note: str = ""

    @property
    def duration_s(self) -> float:
        """Total wall-clock seconds consumed."""
        return self.t_end - self.t_start

    @property
    def points_per_second(self) -> float:
        """Raw scoring rate of this action; 0.0 for zero-duration actions."""
        d = self.duration_s
        return self.points / d if d > 1e-9 else 0.0


__all__ = [
    "Piece",
    "Phase",
    "ScoreCategory",
    "ScoreEvent",
    "RobotState",
    "Shelf",
    "Pantry",
    "OvenState",
    "MatchState",
    "ActionRecord",
]

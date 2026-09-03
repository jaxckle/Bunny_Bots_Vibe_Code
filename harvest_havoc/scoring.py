"""
Scoring rules: point values, endgame finalisation, and ranking points.

Split out from :mod:`harvest_havoc.state` so that "what the rules say" is
readable in one place and testable without constructing an environment.

Design notes
------------
* **Endgame park and harvest haul are evaluated exactly once, at the buzzer.**
  Nothing awards them mid-match. :func:`finalize_endgame` is the only writer.
* **Park is positional.** The rules say the bonus applies only if the robot is
  actually in its table zone at the end of the match, so the environment
  passes the robot's final occupancy in rather than trusting an earlier
  "I parked" action.
* **Only oven-scored carrots feed the human-player exchange.** A cake dropped
  in the oven scores 2 and is gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

from .config import ScoringConfig, TimeConfig
from .state import MatchState, Piece, ScoreCategory


class DinnerRPMode(str, Enum):
    """Two defensible readings of "at least ten endgame points".

    PARK_AND_HAUL
        Endgame points are the park bonus plus the harvest haul only. Maximum
        achievable is ``2 + 3*3 = 11``, so the RP effectively demands parking
        while holding three carrot cakes. This matches how FRC manuals
        normally scope an "endgame points" phrase, and is the default.

    ALL_IN_ENDGAME
        Endgame points are every point earned at or after the endgame start
        whistle, including pantry and oven scoring done in the last 20
        seconds. Much easier to satisfy, and it creates a very different
        optimal time allocation.

    Selected via ``ScoringConfig.dinner_rp_mode``.
    """

    PARK_AND_HAUL = "park_and_haul"
    ALL_IN_ENDGAME = "all_in_endgame"


# =============================================================================
# Point values
# =============================================================================

def pantry_points(piece: Piece, level: int, cfg: ScoringConfig) -> int:
    """Points for placing `piece` on pantry `level`."""
    table = cfg.carrot_points if piece is Piece.CARROT else cfg.cake_points
    return table[level]


def haul_value(carrots: int, cakes: int, cfg: ScoringConfig) -> int:
    """Harvest haul points for an inventory of `carrots` and `cakes`.

    Only ``cfg.haul_max_pieces`` pieces count. Cakes are worth strictly more
    than carrots, so the greedy choice (count cakes first) is optimal.
    """
    slots = cfg.haul_max_pieces
    counted_cakes = min(cakes, slots)
    slots -= counted_cakes
    counted_carrots = min(carrots, slots)
    return (counted_cakes * cfg.haul_cake_points
            + counted_carrots * cfg.haul_carrot_points)


# =============================================================================
# Endgame
# =============================================================================

def finalize_endgame(
    state: MatchState,
    cfg: ScoringConfig,
    *,
    parked: bool,
) -> int:
    """Award park and harvest-haul points. Call exactly once, at the buzzer.

    Parameters
    ----------
    state:
        The match state, mutated in place. ``state.finalized`` is set.
    cfg:
        Scoring configuration.
    parked:
        True if the robot is legally parked in its own table zone at the end
        of the match. The caller (the environment) is responsible for this
        determination because it owns the field geometry and the park
        success roll.

    Returns
    -------
    int
        Points added. Zero if not parked, or if already finalised.
    """
    if state.finalized:
        return 0
    state.finalized = True
    if not parked:
        return 0

    gained = state.award(cfg.park_points, ScoreCategory.PARK)
    haul = haul_value(state.robot.carrots, state.robot.cakes, cfg)
    if haul:
        gained += state.award(haul, ScoreCategory.HARVEST_HAUL)
    return gained


def endgame_points(
    state: MatchState,
    cfg: ScoringConfig,
    time_cfg: TimeConfig,
) -> int:
    """Points that count toward the Dinner RP, per the configured mode.

    Only meaningful after :func:`finalize_endgame` has run; before that it
    reports what has been banked so far (see
    :func:`projected_endgame_points` for a forward-looking estimate).
    """
    mode = DinnerRPMode(cfg.dinner_rp_mode)
    if mode is DinnerRPMode.ALL_IN_ENDGAME:
        return state.points_scored_since(time_cfg.endgame_start_s)
    return sum(
        e.points for e in state.score_events
        if e.category in (ScoreCategory.PARK, ScoreCategory.HARVEST_HAUL)
    )


def projected_endgame_points(
    state: MatchState,
    cfg: ScoringConfig,
    time_cfg: TimeConfig,
    *,
    in_table_zone: bool,
) -> int:
    """Endgame points the robot would bank if the buzzer sounded right now.

    Used for reward shaping, and for the human-readable status line. Assumes
    the current position and inventory are the final ones.
    """
    projected = 0
    if in_table_zone:
        projected += cfg.park_points
        projected += haul_value(state.robot.carrots, state.robot.cakes, cfg)
    if DinnerRPMode(cfg.dinner_rp_mode) is DinnerRPMode.ALL_IN_ENDGAME:
        already = state.points_scored_since(time_cfg.endgame_start_s)
        # Exclude park/haul already banked so we do not double count.
        already -= sum(
            e.points for e in state.score_events
            if e.category in (ScoreCategory.PARK, ScoreCategory.HARVEST_HAUL)
        )
        projected += already
    return projected


# =============================================================================
# Ranking points
# =============================================================================

@dataclass
class RankingPoints:
    """Which ranking points have been earned."""

    stocked_up: bool = False
    baked_up: bool = False
    dinner_rp: bool = False

    @property
    def total(self) -> int:
        """Number of RPs earned (0-3)."""
        return int(self.stocked_up) + int(self.baked_up) + int(self.dinner_rp)

    def as_dict(self) -> Dict[str, bool]:
        """Dict view, for logging into ``info``."""
        return {
            "stocked_up": self.stocked_up,
            "baked_up": self.baked_up,
            "dinner_rp": self.dinner_rp,
        }


@dataclass
class RPProgress:
    """Continuous 0-1 progress toward each ranking point.

    These drive the dense shaping term. Each is a *monotone* function of
    scoring actions, so shaping never rewards undoing progress.

    ``stocked_up`` and ``baked_up`` average the per-shelf progress rather than
    taking the minimum. Averaging gives a gradient for filling the second and
    third shelves even before the first is complete, which is what makes the
    RP learnable; the minimum would be flat over most of the state space.
    """

    stocked_up: float = 0.0
    baked_up: float = 0.0
    dinner_rp: float = 0.0

    #: Per-shelf detail, useful for diagnostics and rendering.
    shelf_stock: Tuple[float, ...] = ()
    shelf_bake: Tuple[float, ...] = ()

    def as_dict(self) -> Dict[str, float]:
        """Dict view, for logging into ``info``."""
        return {
            "stocked_up": self.stocked_up,
            "baked_up": self.baked_up,
            "dinner_rp": self.dinner_rp,
        }


def evaluate_ranking_points(
    state: MatchState,
    cfg: ScoringConfig,
    time_cfg: TimeConfig,
) -> RankingPoints:
    """Evaluate all three ranking points against the current state."""
    shelves = list(state.pantry.shelves.values())
    stocked = bool(shelves) and all(
        s.total >= cfg.stocked_up_threshold for s in shelves
    )
    baked = bool(shelves) and all(
        s.cakes >= cfg.baked_up_threshold for s in shelves
    )
    dinner = endgame_points(state, cfg, time_cfg) >= cfg.dinner_rp_threshold
    return RankingPoints(stocked_up=stocked, baked_up=baked, dinner_rp=dinner)


def rp_progress(
    state: MatchState,
    cfg: ScoringConfig,
    time_cfg: TimeConfig,
    *,
    in_table_zone: bool,
) -> RPProgress:
    """Continuous progress toward each ranking point, each in ``[0, 1]``."""
    shelves = [state.pantry.shelves[lvl] for lvl in state.pantry.levels]
    if not shelves:
        return RPProgress()

    stock_terms = tuple(
        min(s.total, cfg.stocked_up_threshold) / cfg.stocked_up_threshold
        for s in shelves
    )
    bake_terms = tuple(
        min(s.cakes, cfg.baked_up_threshold) / cfg.baked_up_threshold
        for s in shelves
    )

    projected = projected_endgame_points(
        state, cfg, time_cfg, in_table_zone=in_table_zone
    )
    dinner = min(1.0, projected / max(1, cfg.dinner_rp_threshold))

    return RPProgress(
        stocked_up=sum(stock_terms) / len(stock_terms),
        baked_up=sum(bake_terms) / len(bake_terms),
        dinner_rp=dinner,
        shelf_stock=stock_terms,
        shelf_bake=bake_terms,
    )


# =============================================================================
# Reference maxima, for normalisation and sanity checks
# =============================================================================

def max_single_piece_value(cfg: ScoringConfig) -> int:
    """Highest points a single game piece can produce in the pantry."""
    return max(max(cfg.carrot_points.values()), max(cfg.cake_points.values()))


def max_pantry_points(cfg: ScoringConfig) -> int:
    """Highest total the pantry can hold, given the per-shelf capacity.

    With the default rules this is ``3 shelves x 3 slots`` filled with cakes,
    i.e. ``3*(8 + 10 + 12) = 90``. It is a small number relative to a 150
    second match, which is precisely what makes the shelf cap interesting: the
    pantry saturates, and every slot spent on a carrot is a slot that can
    never hold a cake.
    """
    return cfg.shelf_capacity * sum(cfg.cake_points[lvl]
                                    for lvl in cfg.shelf_levels)


def stocked_up_is_reachable(cfg: ScoringConfig) -> bool:
    """Can Stocked Up be earned at all? False if the shelf cap forbids it."""
    return cfg.shelf_capacity >= cfg.stocked_up_threshold


def baked_up_is_reachable(cfg: ScoringConfig) -> bool:
    """Can Baked Up be earned at all? False if the shelf cap forbids it."""
    return cfg.shelf_capacity >= cfg.baked_up_threshold


def both_pantry_rps_reachable(cfg: ScoringConfig) -> bool:
    """Can Stocked Up and Baked Up be held simultaneously?

    They compete for the same slots: Stocked Up needs
    ``stocked_up_threshold`` pieces per shelf and Baked Up needs
    ``baked_up_threshold`` of those to be cakes. Both fit as long as the
    capacity covers the stocking requirement -- with the defaults, exactly
    (3 slots, of which at least 1 must be a cake). The margin being zero is
    why "fill the shelves with carrots first" is a losing opening.
    """
    return (stocked_up_is_reachable(cfg)
            and cfg.baked_up_threshold <= cfg.stocked_up_threshold
            and baked_up_is_reachable(cfg))


def max_endgame_points(cfg: ScoringConfig) -> int:
    """Ceiling on park + harvest haul (11 with default values)."""
    return cfg.park_points + cfg.haul_max_pieces * cfg.haul_cake_points


def dinner_rp_is_reachable(cfg: ScoringConfig) -> bool:
    """Sanity check: can the Dinner RP be earned at all under this config?

    With the default PARK_AND_HAUL reading, the maximum is 11 against a
    threshold of 10 -- achievable, but only by parking with three cakes. This
    helper exists so a misconfiguration surfaces loudly rather than as a
    ranking point the agent can never earn.
    """
    if DinnerRPMode(cfg.dinner_rp_mode) is DinnerRPMode.ALL_IN_ENDGAME:
        return True
    return max_endgame_points(cfg) >= cfg.dinner_rp_threshold


__all__ = [
    "DinnerRPMode",
    "RankingPoints",
    "RPProgress",
    "pantry_points",
    "haul_value",
    "finalize_endgame",
    "endgame_points",
    "projected_endgame_points",
    "evaluate_ranking_points",
    "rp_progress",
    "max_single_piece_value",
    "max_pantry_points",
    "max_endgame_points",
    "dinner_rp_is_reachable",
    "stocked_up_is_reachable",
    "baked_up_is_reachable",
    "both_pantry_rps_reachable",
]

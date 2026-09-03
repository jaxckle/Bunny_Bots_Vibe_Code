"""
The hybrid action space: micro moves plus timed macro actions.

Why hybrid
----------
The research question is time allocation, so the agent needs to reason in units
of *"go do a thing"*, not *"nudge one cell north"*. But throwing away the grid
would make travel time an unexaminable constant. So:

* **Micro actions** (``MOVE_*``) step one grid cell and cost the corresponding
  driving time. They exist so positioning is expressible and so travel cost is
  auditable.
* **Macro actions** (``INTAKE_*``, ``SCORE_*``, ``OVEN_*``, ``PARK``,
  ``EXIT_KITCHEN``) are *options*: they path to the required zone over the same
  grid, then perform one manipulation. Their duration is
  ``travel + align + manipulate``, each component computed from the same
  physical model the micro actions use -- never a hand-tuned constant.

Because both levels share one time model, a macro action's cost is exactly the
cost of the micro actions it replaces plus the manipulation overhead. Nothing
is double counted, and a learned macro-only policy can be replayed as a
cell-by-cell trajectory.

One manipulation per action
---------------------------
Each macro moves exactly one game piece. Emptying a full robot onto level 2 is
three ``SCORE_CARROT_L2`` actions; the second and third are cheap because the
travel leg is already zero. This keeps the action space small while letting the
agent express partial unloads -- which matter, because holding cakes into the
endgame is worth 3 points each.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, Optional, Tuple

from .field import Cell
from .state import Piece
from .zones import Zone


class ActionKind(Enum):
    """Coarse category of an action, used for time-ledger attribution."""

    MOVE = "move"
    INTAKE = "intake"
    SCORE_PANTRY = "score_pantry"
    OVEN_DEPOSIT = "oven_deposit"
    PARK = "park"
    EXIT_KITCHEN = "exit_kitchen"
    WAIT = "wait"


class Action(IntEnum):
    """The discrete action space.

    Values are frozen: appending new actions at the end will not invalidate a
    trained policy's head, and the ``MOVE_*`` block is contiguous so it can be
    masked off wholesale for a macro-only ablation.
    """

    # --- Micro: single-cell movement (indices 0-7) ---------------------------
    MOVE_N = 0
    MOVE_S = 1
    MOVE_E = 2
    MOVE_W = 3
    MOVE_NE = 4
    MOVE_NW = 5
    MOVE_SE = 6
    MOVE_SW = 7

    # --- Macro: acquire ------------------------------------------------------
    INTAKE_CARROT = 8
    INTAKE_CAKE = 9

    # --- Macro: pantry -------------------------------------------------------
    SCORE_CARROT_L1 = 10
    SCORE_CARROT_L2 = 11
    SCORE_CARROT_L3 = 12
    SCORE_CAKE_L1 = 13
    SCORE_CAKE_L2 = 14
    SCORE_CAKE_L3 = 15

    # --- Macro: oven ---------------------------------------------------------
    OVEN_DEPOSIT_CARROT = 16
    OVEN_DEPOSIT_CAKE = 17

    # --- Macro: positioning --------------------------------------------------
    PARK = 18
    EXIT_KITCHEN = 19

    # --- No-op ---------------------------------------------------------------
    WAIT = 20


#: First and last index of the contiguous movement block.
MOVE_ACTION_RANGE: Tuple[int, int] = (Action.MOVE_N, Action.MOVE_SW)

#: Grid deltas for the movement actions. +y is "north" (across the field),
#: +x is "east" (away from the own alliance wall).
MOVE_DELTAS: Dict[Action, Cell] = {
    Action.MOVE_N: (0, 1),
    Action.MOVE_S: (0, -1),
    Action.MOVE_E: (1, 0),
    Action.MOVE_W: (-1, 0),
    Action.MOVE_NE: (1, 1),
    Action.MOVE_NW: (-1, 1),
    Action.MOVE_SE: (1, -1),
    Action.MOVE_SW: (-1, -1),
}


@dataclass(frozen=True)
class ActionSpec:
    """Static description of one action.

    The environment reads these fields rather than branching on the action id,
    so adding an action means adding a row to :data:`ACTION_SPECS` and nothing
    else.

    Attributes
    ----------
    action, kind, name:
        Identity.
    target_zone:
        Zone the macro must reach before manipulating. ``None`` for micro
        moves and WAIT.
    piece:
        Game piece consumed or acquired. ``None`` if not applicable.
    level:
        Pantry shelf level. ``None`` outside pantry scoring.
    delta:
        Single-cell grid offset. Only set for ``MOVE_*``.
    """

    action: Action
    kind: ActionKind
    name: str
    target_zone: Optional[Zone] = None
    piece: Optional[Piece] = None
    level: Optional[int] = None
    delta: Optional[Cell] = None

    @property
    def is_macro(self) -> bool:
        """True for option-style actions that navigate before acting."""
        return self.kind not in (ActionKind.MOVE, ActionKind.WAIT)

    @property
    def is_move(self) -> bool:
        """True for single-cell movement actions."""
        return self.kind is ActionKind.MOVE


def _build_specs() -> Dict[Action, ActionSpec]:
    """Construct the static action table."""
    specs: Dict[Action, ActionSpec] = {}

    for act, delta in MOVE_DELTAS.items():
        specs[act] = ActionSpec(
            action=act, kind=ActionKind.MOVE, name=act.name, delta=delta
        )

    specs[Action.INTAKE_CARROT] = ActionSpec(
        Action.INTAKE_CARROT, ActionKind.INTAKE, "INTAKE_CARROT",
        target_zone=Zone.OWN_FARM, piece=Piece.CARROT,
    )
    specs[Action.INTAKE_CAKE] = ActionSpec(
        Action.INTAKE_CAKE, ActionKind.INTAKE, "INTAKE_CAKE",
        target_zone=Zone.OWN_FARM, piece=Piece.CARROT_CAKE,
    )

    for lvl, act in ((1, Action.SCORE_CARROT_L1),
                     (2, Action.SCORE_CARROT_L2),
                     (3, Action.SCORE_CARROT_L3)):
        specs[act] = ActionSpec(
            act, ActionKind.SCORE_PANTRY, act.name,
            target_zone=Zone.OWN_PANTRY, piece=Piece.CARROT, level=lvl,
        )
    for lvl, act in ((1, Action.SCORE_CAKE_L1),
                     (2, Action.SCORE_CAKE_L2),
                     (3, Action.SCORE_CAKE_L3)):
        specs[act] = ActionSpec(
            act, ActionKind.SCORE_PANTRY, act.name,
            target_zone=Zone.OWN_PANTRY, piece=Piece.CARROT_CAKE, level=lvl,
        )

    specs[Action.OVEN_DEPOSIT_CARROT] = ActionSpec(
        Action.OVEN_DEPOSIT_CARROT, ActionKind.OVEN_DEPOSIT,
        "OVEN_DEPOSIT_CARROT", target_zone=Zone.OWN_OVEN, piece=Piece.CARROT,
    )
    specs[Action.OVEN_DEPOSIT_CAKE] = ActionSpec(
        Action.OVEN_DEPOSIT_CAKE, ActionKind.OVEN_DEPOSIT,
        "OVEN_DEPOSIT_CAKE", target_zone=Zone.OWN_OVEN, piece=Piece.CARROT_CAKE,
    )

    specs[Action.PARK] = ActionSpec(
        Action.PARK, ActionKind.PARK, "PARK", target_zone=Zone.OWN_TABLE,
    )
    specs[Action.EXIT_KITCHEN] = ActionSpec(
        Action.EXIT_KITCHEN, ActionKind.EXIT_KITCHEN, "EXIT_KITCHEN",
        target_zone=None,  # target is "any cell outside the kitchen complex"
    )
    specs[Action.WAIT] = ActionSpec(
        Action.WAIT, ActionKind.WAIT, "WAIT",
    )
    return specs


#: Static action table, keyed by :class:`Action`.
ACTION_SPECS: Dict[Action, ActionSpec] = _build_specs()

#: Size of the discrete action space.
N_ACTIONS: int = len(Action)

#: Action ids that are macro options (everything except moves and WAIT).
MACRO_ACTIONS: Tuple[Action, ...] = tuple(
    a for a in Action if ACTION_SPECS[a].is_macro
)

#: Action ids that are single-cell moves.
MICRO_ACTIONS: Tuple[Action, ...] = tuple(
    a for a in Action if ACTION_SPECS[a].is_move
)


def spec(action) -> ActionSpec:
    """Look up the :class:`ActionSpec` for an action id or enum member."""
    return ACTION_SPECS[Action(action)]


def action_name(action) -> str:
    """Human-readable name for an action id."""
    return Action(action).name


__all__ = [
    "Action",
    "ActionKind",
    "ActionSpec",
    "ACTION_SPECS",
    "N_ACTIONS",
    "MOVE_DELTAS",
    "MOVE_ACTION_RANGE",
    "MACRO_ACTIONS",
    "MICRO_ACTIONS",
    "spec",
    "action_name",
]

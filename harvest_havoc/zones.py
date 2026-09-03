"""
Zone taxonomy and the declarative field layout.

Every grid cell carries **exactly one** :class:`Zone` label. Where regions
physically nest -- the pantry and the oven both sit inside the kitchen -- the
more specific label wins, and the coarser "is this inside the kitchen complex"
question is answered by :data:`KITCHEN_COMPLEX` rather than by the cell label.

Extending to a full 2-alliance / 3-robot field
----------------------------------------------
1. :class:`Zone` already reserves values 11-19 for opponent zones, so adding
   them will not renumber existing labels or invalidate a trained policy's
   one-hot encoding.
2. :data:`DEFAULT_LAYOUT` is plain data. Add mirrored rectangles (see
   :func:`mirror_layout`) and nothing else in the package needs to change.
3. Predicates :func:`is_own` / :func:`is_opponent` / :func:`counterpart` let
   game logic be written once and applied per-alliance.

Layout coordinates
------------------
Feet, origin at the own-alliance back-left corner, +x toward the opponent
wall, +y across the field. Rectangles are half-open ``[x0, x1) x [y0, y1)``.
Later rectangles in the list **override** earlier ones, which is how the
pantry and oven are carved out of the kitchen.

.. warning::
   The real Harvest Havoc field drawings were not available when this was
   written. The geometry below is a plausible, internally consistent 54 x 27 ft
   layout chosen so that the travel-time tradeoffs are interesting. Replace
   :data:`DEFAULT_LAYOUT` with surveyed coordinates when you have them --
   that is the only edit required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, FrozenSet, List, Tuple


class Zone(IntEnum):
    """Field zone labels.

    Value ranges are reserved deliberately:

    ==========  ==========================================================
    0           neutral / unclaimed carpet
    1-9         own alliance zones
    11-19       opponent alliance zones (mirror of 1-9, offset +10)
    21-29       reserved for shared / contested zones added later
    ==========  ==========================================================
    """

    NEUTRAL = 0

    # --- Own alliance --------------------------------------------------------
    OWN_KITCHEN = 1
    OWN_FARM = 2
    OWN_PANTRY = 3
    OWN_OVEN = 4
    OWN_TABLE = 5

    # --- Opponent alliance (declared now so the one-hot width is stable; no
    #     cells carry these labels in the single-robot v1 layout) -------------
    OPP_KITCHEN = 11
    OPP_FARM = 12
    OPP_PANTRY = 13
    OPP_OVEN = 14
    OPP_TABLE = 15


#: Offset between an own-alliance zone value and its opponent counterpart.
ALLIANCE_OFFSET = 10

#: Stable ordering used for one-hot encoding a zone in the observation vector.
#: Appending new members to :class:`Zone` extends this list at the end, which
#: keeps existing observation indices valid.
ZONE_ORDER: Tuple[Zone, ...] = tuple(Zone)

#: Index of each zone within :data:`ZONE_ORDER`.
ZONE_INDEX: Dict[Zone, int] = {z: i for i, z in enumerate(ZONE_ORDER)}

#: Number of zone one-hot slots in the observation.
N_ZONES = len(ZONE_ORDER)

#: The pantry and oven are physically *inside* the kitchen. A cell labelled
#: OWN_PANTRY is still "in the kitchen" for the purposes of the autonomous
#: leave-the-kitchen bonus. This set is the authoritative answer to that.
KITCHEN_COMPLEX: FrozenSet[Zone] = frozenset(
    {Zone.OWN_KITCHEN, Zone.OWN_PANTRY, Zone.OWN_OVEN}
)

#: Single-character glyphs used by the ASCII renderer.
ZONE_GLYPH: Dict[Zone, str] = {
    Zone.NEUTRAL: ".",
    Zone.OWN_KITCHEN: "k",
    Zone.OWN_FARM: "F",
    Zone.OWN_PANTRY: "P",
    Zone.OWN_OVEN: "O",
    Zone.OWN_TABLE: "T",
    Zone.OPP_KITCHEN: "K",
    Zone.OPP_FARM: "f",
    Zone.OPP_PANTRY: "p",
    Zone.OPP_OVEN: "o",
    Zone.OPP_TABLE: "t",
}


def is_own(zone: Zone) -> bool:
    """True if `zone` belongs to the agent's own alliance."""
    return 1 <= int(zone) <= 9


def is_opponent(zone: Zone) -> bool:
    """True if `zone` belongs to the opposing alliance."""
    return 11 <= int(zone) <= 19


def counterpart(zone: Zone) -> Zone:
    """Map an own-alliance zone to its opponent twin, or vice versa.

    Returns NEUTRAL unchanged. Raises :class:`ValueError` for zones with no
    defined counterpart.
    """
    if zone is Zone.NEUTRAL:
        return zone
    if is_own(zone):
        return Zone(int(zone) + ALLIANCE_OFFSET)
    if is_opponent(zone):
        return Zone(int(zone) - ALLIANCE_OFFSET)
    raise ValueError(f"{zone!r} has no alliance counterpart")


@dataclass(frozen=True)
class ZoneRect:
    """A half-open axis-aligned rectangle of field, in feet, tagged with a zone.

    Attributes
    ----------
    zone:
        The label applied to every cell whose centre falls inside the rect.
    x0, y0, x1, y1:
        Bounds in feet. ``x0 <= x < x1`` and ``y0 <= y < y1``.
    name:
        Human-readable description, used only for documentation and rendering.
    """

    zone: Zone
    x0: float
    y0: float
    x1: float
    y1: float
    name: str = ""

    def contains(self, x_ft: float, y_ft: float) -> bool:
        """True if the point ``(x_ft, y_ft)`` lies inside this rectangle."""
        return self.x0 <= x_ft < self.x1 and self.y0 <= y_ft < self.y1

    def center(self) -> Tuple[float, float]:
        """Centroid of the rectangle, in feet."""
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


# =============================================================================
# DEFAULT LAYOUT  --  edit these rectangles to match the real field
# =============================================================================
#
#   y=27 +----------------------------------------------------------+
#        |            kitchen            |        |                 |
#        |                               | TABLE  |                 |
#        |  +---+                        |        |                 |
#        |  | P |  (pantry, back wall)   +--------+     neutral     |
#        |  | P |                        |                          |
#        |  +---+                   +----+----+                     |
#        |                          |  FARM   |                     |
#        |  +---+                   |         |                     |
#        |  | O |  (oven)           |         |                     |
#   y=0  +--+---+-------------------+---------+---------------------+
#        x=0                                                     x=54
#
DEFAULT_LAYOUT: List[ZoneRect] = [
    # -- Coarse regions first; later entries override. -----------------------
    ZoneRect(Zone.OWN_KITCHEN, 0.0, 0.0, 15.0, 27.0, "own kitchen"),
    ZoneRect(Zone.OWN_FARM, 15.0, 0.0, 27.0, 10.5, "own farm / depot"),
    ZoneRect(Zone.OWN_TABLE, 16.5, 18.0, 24.0, 27.0, "own table (endgame park)"),

    # -- Fine regions carved out of the kitchen. -----------------------------
    ZoneRect(Zone.OWN_PANTRY, 0.0, 9.0, 3.0, 21.0, "own pantry (3 shelves)"),
    ZoneRect(Zone.OWN_OVEN, 0.0, 1.5, 3.0, 6.0, "own oven"),
]

#: Robot starting pose, in feet. Must lie inside the kitchen complex.
DEFAULT_START_FT: Tuple[float, float] = (10.5, 13.5)


def mirror_layout(
    layout: List[ZoneRect], field_length_ft: float = 54.0
) -> List[ZoneRect]:
    """Return opponent-side rectangles by reflecting `layout` across midfield.

    Not used by the v1 single-robot environment, but provided so that adding
    an opponent later is a one-line change rather than a restructuring::

        layout = DEFAULT_LAYOUT + mirror_layout(DEFAULT_LAYOUT)

    Parameters
    ----------
    layout:
        Own-alliance rectangles. Entries already tagged with an opponent or
        neutral zone are skipped.
    field_length_ft:
        Length of the field along x, used as the reflection axis.
    """
    mirrored: List[ZoneRect] = []
    for rect in layout:
        if not is_own(rect.zone):
            continue
        mirrored.append(
            ZoneRect(
                zone=counterpart(rect.zone),
                x0=field_length_ft - rect.x1,
                y0=rect.y0,
                x1=field_length_ft - rect.x0,
                y1=rect.y1,
                name=f"opponent {rect.name}",
            )
        )
    return mirrored


__all__ = [
    "Zone",
    "ZoneRect",
    "ZONE_ORDER",
    "ZONE_INDEX",
    "N_ZONES",
    "ZONE_GLYPH",
    "KITCHEN_COMPLEX",
    "ALLIANCE_OFFSET",
    "DEFAULT_LAYOUT",
    "DEFAULT_START_FT",
    "is_own",
    "is_opponent",
    "counterpart",
    "mirror_layout",
]

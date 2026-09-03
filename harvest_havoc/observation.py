"""
Observation encoding.

The observation is a flat, fully named ``float32`` vector. Every slot has a
label in :attr:`ObservationEncoder.names`, so a policy's saliency or a debug
dump can be read without counting indices by hand.

What the agent is given, and why
--------------------------------
Because the goal is *time allocation*, the clock is represented three ways --
fraction of match remaining, a phase one-hot, and fraction of the current phase
remaining. An agent that must decide "one more cycle, or leave for the table
zone now?" needs all three: absolute urgency, which rules are in force, and
distance to the next rule change.

Ranking-point progress is included explicitly rather than left to be inferred
from shelf counts. It is a deterministic function of the shelf counts, so this
is redundant information -- but it is the *shape* of the redundancy that
matters: it hands the network the threshold structure for free instead of
making it discover three step functions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import EnvConfig
from .field import Field
from .scoring import RPProgress, rp_progress
from .state import MatchState, Phase
from .zones import N_ZONES, ZONE_INDEX, ZONE_ORDER, Zone

#: Declared upper bound for count-like slots whose normalising divisor is a
#: *typical* value rather than a hard maximum. A 150-second match can put ~30
#: carrots on one shelf, which is 5.0 at the default ``obs_shelf_cap`` of 6, so
#: the space must admit values above 1.0. Clipping instead would silently erase
#: the difference between a well-stocked shelf and an overflowing one.
_COUNT_HEADROOM = 10.0


class ObservationEncoder:
    """Turns a :class:`~harvest_havoc.state.MatchState` into a flat vector.

    Parameters
    ----------
    config:
        Supplies normalisation constants and shelf levels.
    field:
        Supplies grid dimensions and zone lookup.

    Attributes
    ----------
    names: list[str]
        Label for each slot, in order.
    low, high: np.ndarray
        Element-wise bounds, suitable for a ``Box`` space. Every slot is
        normalised into ``[0, 1]``, but counts that can exceed their
        normalising cap (shelf contents, score) are given headroom rather than
        being clipped, so information is never silently destroyed.
    """

    def __init__(self, config: EnvConfig, field: Field) -> None:
        self.config = config
        self.field = field
        self.levels: Tuple[int, ...] = config.scoring.shelf_levels

        self.names: List[str] = []
        lows: List[float] = []
        highs: List[float] = []

        def add(name: str, lo: float = 0.0, hi: float = 1.0) -> None:
            self.names.append(name)
            lows.append(lo)
            highs.append(hi)

        # --- clock -----------------------------------------------------------
        add("time_remaining_frac")
        for phase in Phase:
            add(f"phase_{phase.name.lower()}")
        add("phase_time_remaining_frac")

        # --- robot -----------------------------------------------------------
        add("robot_x_norm")
        add("robot_y_norm")
        for zone in ZONE_ORDER:
            add(f"zone_{zone.name.lower()}")
        add("inv_carrots_norm")
        add("inv_cakes_norm")
        add("inv_free_norm")
        add("in_motion")

        # --- pantry ----------------------------------------------------------
        # Shelf counts are hard-capped by ScoringConfig.shelf_capacity, so
        # these normalise cleanly into [0, 1] with no headroom needed. The
        # free-slot signal is given explicitly: with only nine slots in the
        # whole pantry, "can I still put a cake here?" is a first-class
        # question, not something to infer from two counts.
        for lvl in self.levels:
            add(f"shelf{lvl}_carrots_norm")
        for lvl in self.levels:
            add(f"shelf{lvl}_cakes_norm")
        for lvl in self.levels:
            add(f"shelf{lvl}_free_norm")

        # --- ranking-point progress -----------------------------------------
        add("rp_stocked_up_progress")
        add("rp_baked_up_progress")
        add("rp_dinner_progress")

        # --- resources -------------------------------------------------------
        add("oven_carrots_toward_next_norm")
        add("oven_pending_exchanges_norm", 0.0, _COUNT_HEADROOM)
        add("cakes_available_norm", 0.0, _COUNT_HEADROOM)
        add("depot_carrots_norm")

        # --- flags and score -------------------------------------------------
        add("auto_leave_earned")
        add("in_table_zone")
        add("parked")
        add("raw_score_norm", 0.0, _COUNT_HEADROOM)

        self.low = np.asarray(lows, dtype=np.float32)
        self.high = np.asarray(highs, dtype=np.float32)
        self.size = len(self.names)
        self._index: Dict[str, int] = {n: i for i, n in enumerate(self.names)}

    # ------------------------------------------------------------------ encode

    def encode(self, state: MatchState) -> np.ndarray:
        """Encode `state` as a ``float32`` vector of length :attr:`size`."""
        cfg = self.config
        tcfg = cfg.time
        scfg = cfg.scoring

        zone = self.field.zone_at(state.robot.cell)
        in_table = zone is Zone.OWN_TABLE
        progress: RPProgress = rp_progress(
            state, scfg, tcfg, in_table_zone=in_table
        )
        phase = state.phase(tcfg)

        out = np.zeros(self.size, dtype=np.float32)
        i = 0

        # --- clock -----------------------------------------------------------
        out[i] = state.time_remaining(tcfg) / tcfg.match_duration_s
        i += 1
        for p in Phase:
            out[i] = 1.0 if p is phase else 0.0
            i += 1
        phase_span = {
            Phase.AUTO: tcfg.auto_duration_s,
            Phase.TELEOP: max(1e-9, tcfg.teleop_duration_s),
            Phase.ENDGAME: tcfg.endgame_duration_s,
        }[phase]
        out[i] = state.phase_time_remaining(tcfg) / phase_span
        i += 1

        # --- robot -----------------------------------------------------------
        cx, cy = state.robot.cell
        out[i] = cx / max(1, self.field.n_cols - 1)
        i += 1
        out[i] = cy / max(1, self.field.n_rows - 1)
        i += 1
        out[i + ZONE_INDEX[zone]] = 1.0
        i += N_ZONES
        out[i] = state.robot.carrots / cfg.max_inventory
        i += 1
        out[i] = state.robot.cakes / cfg.max_inventory
        i += 1
        out[i] = state.robot.free_slots(cfg.max_inventory) / cfg.max_inventory
        i += 1
        out[i] = 1.0 if state.robot.in_motion else 0.0
        i += 1

        # --- pantry ----------------------------------------------------------
        cap = max(1, scfg.shelf_capacity)
        for lvl in self.levels:
            out[i] = state.pantry.shelves[lvl].carrots / cap
            i += 1
        for lvl in self.levels:
            out[i] = state.pantry.shelves[lvl].cakes / cap
            i += 1
        for lvl in self.levels:
            out[i] = state.pantry.shelves[lvl].free_slots / cap
            i += 1

        # --- ranking-point progress -----------------------------------------
        out[i] = progress.stocked_up
        i += 1
        out[i] = progress.baked_up
        i += 1
        out[i] = progress.dinner_rp
        i += 1

        # --- resources -------------------------------------------------------
        out[i] = (state.oven.carrots_toward_next
                  / max(1, scfg.oven_carrots_per_cake))
        i += 1
        out[i] = state.oven.pending_count / cfg.max_inventory
        i += 1
        out[i] = state.cakes_available / cfg.max_inventory
        i += 1
        out[i] = state.depot_carrots / max(1, cfg.depot_carrot_supply)
        i += 1

        # --- flags and score -------------------------------------------------
        out[i] = 1.0 if state.auto_leave_earned else 0.0
        i += 1
        out[i] = 1.0 if in_table else 0.0
        i += 1
        out[i] = 1.0 if state.robot.parked else 0.0
        i += 1
        out[i] = state.raw_score / cfg.obs_score_cap
        i += 1

        assert i == self.size, f"encoder wrote {i} slots, expected {self.size}"
        return out

    # ------------------------------------------------------------------ lookup

    def index(self, name: str) -> int:
        """Index of the named observation slot."""
        return self._index[name]

    def as_dict(self, obs: np.ndarray) -> Dict[str, float]:
        """Zip an observation vector back into a labelled dict."""
        return {n: float(v) for n, v in zip(self.names, obs)}


__all__ = ["ObservationEncoder"]

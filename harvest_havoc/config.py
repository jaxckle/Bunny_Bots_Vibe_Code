"""
Harvest Havoc (2026 Bunnybots offseason) -- central configuration.

EVERY tunable quantity in the simulation lives in this module. Nothing else in
the package hard-codes a number that a user might reasonably want to change.

The organising principle of this project is that **time is the scarce
resource**. A 150 second match is a fixed budget, and the research question is
how to spend it. Consequently time costs are not buried inside the step
function -- they are declared here, as data, in `TimeConfig`, so that you can
sweep them, fit them to real robot logs, or ask "how much faster does our
level-3 scorer need to be before L3 beats L1?" without touching logic code.

Configuration layout
--------------------
    FieldConfig       geometry + grid resolution
    TimeConfig        match phase boundaries, robot kinematics, action durations
    ScoringConfig     point values, ranking-point thresholds
    StochasticConfig  success rates, timing noise, human-player exchange delay
    RewardConfig      multi-objective reward weights and shaping scales
    EnvConfig         bundles all of the above

All are frozen-by-convention dataclasses (mutable, but treat instances as
immutable once an env is constructed; call `dataclasses.replace` to vary one).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field as _field
from typing import Dict, Tuple

# =============================================================================
# TOP-LEVEL TUNABLES
# The two constants most likely to be swept are hoisted to the very top.
# =============================================================================

#: Side length of one square grid cell, in feet. The field is discretised into
#: cells of this size. 1.0-2.0 ft is the intended usable range: 1.5 ft gives a
#: 36 x 18 grid, which is coarse enough to be fast and fine enough that the
#: pantry, oven, farm and table zones are each several cells across.
GRID_CELL_SIZE_FT: float = 1.5

#: Total match length in seconds (2:30). The single most important number here.
MATCH_DURATION_S: float = 150.0


# =============================================================================
# FIELD GEOMETRY
# =============================================================================

@dataclass
class FieldConfig:
    """Physical field dimensions and grid discretisation.

    The field is modelled as an axis-aligned rectangle `field_length_ft` (x)
    by `field_width_ft` (y), with the origin at the own-alliance back-left
    corner. +x points away from the own alliance wall (toward the opponent),
    +y points across the field.

    Zone rectangles themselves live in :mod:`harvest_havoc.zones` because they
    are structural (they define what a zone *is*), not merely numeric.
    """

    field_length_ft: float = 54.0
    field_width_ft: float = 27.0
    cell_size_ft: float = GRID_CELL_SIZE_FT

    #: Cost multiplier for a diagonal grid move (sqrt(2) for true Euclidean).
    #: Set to a large number to forbid diagonals entirely.
    diagonal_cost_multiplier: float = 2.0 ** 0.5

    #: Whether the robot may traverse diagonally between cells.
    allow_diagonal_movement: bool = True

    @property
    def n_cols(self) -> int:
        """Number of grid cells along x."""
        return int(round(self.field_length_ft / self.cell_size_ft))

    @property
    def n_rows(self) -> int:
        """Number of grid cells along y."""
        return int(round(self.field_width_ft / self.cell_size_ft))

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as ``(n_cols, n_rows)`` i.e. ``(x, y)``."""
        return (self.n_cols, self.n_rows)


# =============================================================================
# TIME MODEL  --  the heart of this simulation
# =============================================================================

@dataclass
class TimeConfig:
    """Match schedule, robot kinematics, and per-action time costs.

    Total action duration is always decomposed as::

        duration = travel_time + align_time + manipulation_time

    and each component is reported separately in the env's time ledger, so a
    post-hoc analysis can attribute every one of the 150 seconds to either
    driving, aligning, or manipulating.

    Travel time uses a **trapezoidal velocity profile**: the robot accelerates
    at `max_accel_ft_s2` up to `max_velocity_ft_s`, cruises, then decelerates
    symmetrically to rest. For a path of length ``d``::

        d_ramp = v^2 / a                       # distance used by accel + decel
        t = 2 * sqrt(d / a)          if d < d_ramp    (triangular, never cruises)
        t = d / v + v / a            if d >= d_ramp   (trapezoidal)

    This is a genuine two-parameter model, so raising `max_velocity_ft_s`
    alone has diminishing returns on short hops -- exactly the behaviour real
    cycle-time analysis shows.
    """

    # ---- Match schedule -----------------------------------------------------
    match_duration_s: float = MATCH_DURATION_S
    #: Autonomous runs from t=0 to this time.
    auto_duration_s: float = 15.0
    #: Endgame is the final `endgame_duration_s` seconds of the match.
    endgame_duration_s: float = 20.0

    # ---- Robot kinematics ---------------------------------------------------
    #: Free-speed translation, ft/s. ~14 ft/s is a well-geared FRC swerve.
    max_velocity_ft_s: float = 14.0
    #: Translational acceleration, ft/s^2.
    max_accel_ft_s2: float = 12.0

    # ---- Fixed overheads ----------------------------------------------------
    #: Time to square up / align on a scoring target after arriving. Charged
    #: once per macro action that targets a fixed field element.
    align_time_s: float = 0.50

    # ---- Manipulation times -------------------------------------------------
    #: Seconds to intake one carrot from the farm depot.
    intake_carrot_s: float = 1.00
    #: Seconds to intake one carrot cake (bulkier, slower).
    intake_cake_s: float = 1.30

    #: Seconds to place one carrot on pantry level 1 / 2 / 3. Higher shelves
    #: cost more because the elevator must travel further -- this is the
    #: central risk/reward tradeoff of the pantry.
    score_carrot_s: Dict[int, float] = _field(
        default_factory=lambda: {1: 1.40, 2: 1.90, 3: 2.60}
    )
    #: Seconds to place one carrot cake on pantry level 1 / 2 / 3.
    score_cake_s: Dict[int, float] = _field(
        default_factory=lambda: {1: 1.70, 2: 2.20, 3: 3.00}
    )

    #: Seconds to deposit one game piece into the oven.
    oven_deposit_s: float = 1.20

    #: Seconds to settle into a legal parked configuration in the table zone.
    park_settle_s: float = 0.60

    #: Duration consumed by an explicit WAIT / no-op action.
    wait_duration_s: float = 0.50

    #: Floor on how long any single action may take. Guarantees that **every**
    #: action costs time, so no policy can find a zero-cost action to spam and
    #: the step count is always a meaningful budget. Only ever binds on
    #: degenerate cases, e.g. EXIT_KITCHEN when the robot is already outside.
    min_action_duration_s: float = 0.10

    # ---- Micro-movement model ----------------------------------------------
    #: When True, a MOVE action taken while the robot is already in motion is
    #: charged at cruise speed (d / v_max) instead of a start-stop trapezoid.
    #: Without this, single-cell moves are absurdly expensive and the discrete
    #: grid becomes an artifact rather than a model.
    momentum_carries_between_moves: bool = True

    # ---- Derived schedule boundaries ---------------------------------------
    @property
    def auto_end_s(self) -> float:
        """Wall-clock time at which autonomous ends."""
        return self.auto_duration_s

    @property
    def endgame_start_s(self) -> float:
        """Wall-clock time at which endgame begins."""
        return self.match_duration_s - self.endgame_duration_s

    @property
    def teleop_duration_s(self) -> float:
        """Length of the non-auto, non-endgame portion of teleop."""
        return self.endgame_start_s - self.auto_end_s


# =============================================================================
# SCORING
# =============================================================================

@dataclass
class ScoringConfig:
    """Point values and ranking-point thresholds."""

    # ---- Pantry -------------------------------------------------------------
    #: Points for a carrot on pantry level 1 / 2 / 3.
    carrot_points: Dict[int, int] = _field(
        default_factory=lambda: {1: 3, 2: 4, 3: 5}
    )
    #: Points for a carrot cake on pantry level 1 / 2 / 3.
    cake_points: Dict[int, int] = _field(
        default_factory=lambda: {1: 8, 2: 10, 3: 12}
    )
    #: The pantry's shelf identifiers, low to high.
    shelf_levels: Tuple[int, ...] = (1, 2, 3)

    #: Maximum game pieces per shelf -- carrots and carrot cakes **combined**.
    #: This is a hard cap: once a shelf holds this many pieces, scoring on it
    #: becomes an illegal action.
    #:
    #: This single number dominates the strategy. With three shelves at
    #: capacity 3 the whole pantry holds only nine pieces, so the question
    #: stops being "how many pieces can we score?" and becomes "which nine
    #: pieces, and in what order?". Note the trap it creates: filling a shelf
    #: with carrots permanently forfeits Baked Up on that shelf, because
    #: nothing can be removed to make room for a cake.
    shelf_capacity: int = 3

    # ---- Oven ---------------------------------------------------------------
    #: Points for depositing any game piece into the oven.
    oven_points: int = 2
    #: Number of oven-scored CARROTS that the human player converts into one
    #: new carrot cake. Cakes deposited into the oven do NOT count toward this.
    oven_carrots_per_cake: int = 3

    # ---- Autonomous ---------------------------------------------------------
    #: One-time bonus for fully leaving the kitchen zone during autonomous.
    auto_leave_kitchen_points: int = 2

    # ---- Endgame ------------------------------------------------------------
    #: Points for being parked in the own table zone at the buzzer.
    park_points: int = 2
    #: Harvest haul: points per carrot held while parked.
    haul_carrot_points: int = 1
    #: Harvest haul: points per carrot cake held while parked.
    haul_cake_points: int = 3
    #: Maximum number of held pieces that count toward the harvest haul.
    haul_max_pieces: int = 3

    # ---- Ranking points -----------------------------------------------------
    #: Stocked Up: at least this many pieces (of any type) on EVERY shelf.
    stocked_up_threshold: int = 3
    #: Baked Up: at least this many CAKES on EVERY shelf.
    baked_up_threshold: int = 1
    #: Dinner RP: at least this many endgame points.
    dinner_rp_threshold: int = 10

    #: How "endgame points" is defined for the Dinner RP. See
    #: :class:`harvest_havoc.scoring.DinnerRPMode`. The real game manual is
    #: ambiguous to us, so this is an explicit, documented switch.
    #:   "park_and_haul"  -> park + harvest haul only (max 11). Default.
    #:   "all_in_endgame" -> every point earned at t >= endgame_start.
    dinner_rp_mode: str = "park_and_haul"


# =============================================================================
# STOCHASTICITY
# =============================================================================

@dataclass
class StochasticConfig:
    """Failure rates and the probabilistic duration model.

    Action durations are **sampled, not fixed**. Every action's nominal time
    (from :class:`TimeConfig`) is multiplied by a random factor drawn from a
    right-skewed, mean-one distribution -- see :mod:`harvest_havoc.timing`.
    Real robot actions behave this way: there is a floor set by physics, most
    attempts land near it, and the tail is long because bobbles, re-grips and
    missed alignments only ever cost *more* time, never less. A symmetric
    Gaussian would wrongly imply that being lucky saves as much as being
    unlucky costs.

    Because the mean factor is exactly 1.0, turning noise on does not
    secretly make the robot slower on average -- it only adds variance. That
    keeps a noisy run comparable to a deterministic one.

    Set ``duration_model="deterministic"`` (or ``time_noise_scale=0.0``) to
    recover fixed per-action times for debugging.
    """

    # ---- Success rates ------------------------------------------------------
    #: Probability an intake attempt succeeds. On failure the time is still
    #: spent -- that is the whole point of modelling it.
    intake_success_prob: float = 1.0

    #: Probability a pantry placement succeeds, per shelf level. Higher shelves
    #: are the natural place to encode "harder shot".
    score_success_prob: Dict[int, float] = _field(
        default_factory=lambda: {1: 1.0, 2: 1.0, 3: 1.0}
    )

    #: Probability an oven deposit succeeds.
    oven_success_prob: float = 1.0

    #: Probability parking succeeds (fails => robot is in the zone but not
    #: legally parked; note park is scored positionally, so this models a
    #: bumper-out-of-zone or tipped-over outcome).
    park_success_prob: float = 1.0

    # ---- Duration distribution ---------------------------------------------
    #: Which distribution generates the duration multiplier. One of
    #: ``"lognormal"`` (default, heaviest tail), ``"gamma"`` (lighter tail),
    #: or ``"deterministic"`` (no noise at all).
    duration_model: str = "lognormal"

    #: Global multiplier applied to every coefficient of variation below.
    #: The single knob to dial all timing randomness up or down: 0.0 is fully
    #: deterministic, 1.0 is the calibrated default, 2.0 is a scrappy match.
    time_noise_scale: float = 1.0

    #: Coefficient of variation (std / mean) per duration component. Driving
    #: is the most repeatable part of a cycle; alignment is the least, because
    #: it is where a bad approach angle turns into a second attempt.
    travel_noise_cv: float = 0.08
    align_noise_cv: float = 0.25
    manipulate_noise_cv: float = 0.15
    #: Waiting is a decision, not a physical action, so it has no variance.
    wait_noise_cv: float = 0.0

    #: Duration multipliers are clamped to this range so a draw from the tail
    #: can never make an action free or consume the whole match.
    time_noise_clip: Tuple[float, float] = (0.45, 3.0)

    # ---- Human player -------------------------------------------------------
    #: Mean seconds between the third qualifying carrot entering the oven and
    #: a new cake becoming available at the farm depot.
    hp_exchange_delay_s: float = 7.5

    #: Coefficient of variation of that delay. Drawn from the same right-skewed
    #: family as action durations, and for the same reason -- a human player
    #: has a best case they cannot beat and a worst case with no ceiling.
    hp_exchange_delay_cv: float = 0.35

    #: Probability the human player successfully completes an exchange at all.
    hp_exchange_success_prob: float = 1.0

    def cv_for(self, component: str) -> float:
        """Effective coefficient of variation for a duration component.

        Parameters
        ----------
        component:
            One of ``"travel"``, ``"align"``, ``"manipulate"``, ``"wait"``.
        """
        base = {
            "travel": self.travel_noise_cv,
            "align": self.align_noise_cv,
            "manipulate": self.manipulate_noise_cv,
            "wait": self.wait_noise_cv,
        }.get(component, 0.0)
        return max(0.0, base * self.time_noise_scale)

    @classmethod
    def deterministic(cls) -> "StochasticConfig":
        """Every action takes exactly its nominal time. Useful for debugging."""
        return cls(duration_model="deterministic", time_noise_scale=0.0,
                   hp_exchange_delay_cv=0.0)

    @classmethod
    def realistic(cls) -> "StochasticConfig":
        """Timing noise plus the failure rates of a decent competition robot."""
        return cls(
            intake_success_prob=0.93,
            score_success_prob={1: 0.98, 2: 0.95, 3: 0.90},
            oven_success_prob=0.97,
            park_success_prob=0.97,
            time_noise_scale=1.0,
        )


# =============================================================================
# REWARD
# =============================================================================

@dataclass
class RewardConfig:
    """Multi-objective reward weights.

    The reward is deliberately decomposed. `HarvestHavocEnv.step` returns a
    scalar (the weighted sum, for off-the-shelf RL) but *also* puts the full
    component vector in ``info["reward_vector"]`` and a labelled breakdown in
    ``info["reward_breakdown"]``, so you can re-weight objectives offline or
    plug in a genuine multi-objective learner without re-simulating.

    Objectives
    ----------
    0. raw score        -- immediate points delta
    1. Stocked Up       -- progress toward >=3 pieces on every shelf
    2. Baked Up         -- progress toward >=1 cake on every shelf
    3. Dinner RP        -- progress toward >=10 endgame points
    4. RP achievement   -- terminal, one lump per RP actually earned
    """

    #: Weight on raw score delta. 1.0 means "one reward per point".
    w_raw_score: float = 1.0

    #: Global multiplier on all ranking-point *shaping* (dense progress).
    w_rp_shaping: float = 1.0

    #: Per-RP shaping potential scale, in reward units at 100% progress.
    #: These are the "how much is being on track worth" knobs.
    shaping_scale_stocked_up: float = 12.0
    shaping_scale_baked_up: float = 12.0
    shaping_scale_dinner_rp: float = 8.0

    #: Terminal lump-sum bonus for each ranking point actually earned. Set
    #: these high relative to raw score to reflect that RPs drive seeding.
    bonus_stocked_up: float = 30.0
    bonus_baked_up: float = 30.0
    bonus_dinner_rp: float = 20.0

    #: The Dinner RP potential is ramped in over the last
    #: ``dinner_shaping_ramp_s`` seconds before endgame starts, reaching full
    #: weight at the endgame whistle. Without this ramp the shaping would pay
    #: the agent to hoard cakes in the table zone from t=0, which is a
    #: terrible strategy that dense reward would otherwise encourage.
    dinner_shaping_ramp_s: float = 30.0

    #: Optional per-second penalty. Usually 0.0 -- the 150 s budget is already
    #: a hard constraint, so an explicit time penalty double-counts. Useful
    #: when you want to discourage stalling in a shortened-match ablation.
    w_time_penalty: float = 0.0

    #: Penalty applied when an illegal action is selected (no time passes).
    illegal_action_penalty: float = 0.0

    #: If True, force the shaping potential to zero at the terminal state,
    #: making the shaping strictly potential-based (Ng et al. 1999) and hence
    #: policy-invariant -- at the cost of a large negative spike at the buzzer.
    #: If False (default), shaping telescopes to Phi(s_T) - Phi(s_0), which is
    #: a mild, well-behaved bias toward finishing in a high-potential state.
    strict_potential_based_shaping: bool = False

    # =========================================================================
    # SIMPLE PRIORITY INTERFACE
    # =========================================================================
    #
    # The fields above are the full control surface, but tuning eight numbers
    # by hand is a poor way to express "care more about ranking points". These
    # helpers collapse the whole config down to four relative priorities.

    #: Reward units a fully-earned ranking point is worth at priority 1.0.
    #: `from_priorities` multiplies this by each priority to get the bonus,
    #: and uses 40% of it as the dense shaping scale.
    RP_REFERENCE_VALUE: float = 30.0

    @classmethod
    def from_priorities(
        cls,
        score: float = 1.0,
        stocked_up: float = 1.0,
        baked_up: float = 1.0,
        dinner_rp: float = 1.0,
        **overrides,
    ) -> "RewardConfig":
        """Build a reward config from four relative priorities.

        Each priority is a plain multiplier; only their *ratios* matter. All
        four at 1.0 reproduces the balanced default.

        Parameters
        ----------
        score:
            How much one raw point is worth.
        stocked_up, baked_up, dinner_rp:
            How much each ranking point is worth, relative to
            :data:`RP_REFERENCE_VALUE` reward units. Set one to 0.0 to ignore
            that objective entirely.
        **overrides:
            Any other :class:`RewardConfig` field, applied afterwards.

        Examples
        --------
        Chase ranking points, and do not care which::

            RewardConfig.from_priorities(score=0.2, stocked_up=3, baked_up=3)

        Pure raw score, ignoring every RP::

            RewardConfig.from_priorities(score=1, stocked_up=0,
                                         baked_up=0, dinner_rp=0)

        Notes
        -----
        The dense shaping scale is set to 40% of each bonus. Shaping only
        needs to be large enough to point the way toward a threshold; making
        it comparable to the bonus itself lets a policy farm partial progress
        and never actually close out the ranking point.
        """
        ref = cls.RP_REFERENCE_VALUE
        cfg = cls(
            w_raw_score=score,
            bonus_stocked_up=ref * stocked_up,
            bonus_baked_up=ref * baked_up,
            bonus_dinner_rp=ref * dinner_rp,
            shaping_scale_stocked_up=0.4 * ref * stocked_up,
            shaping_scale_baked_up=0.4 * ref * baked_up,
            shaping_scale_dinner_rp=0.4 * ref * dinner_rp,
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"RewardConfig has no field {key!r}")
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def preset(cls, name: str) -> "RewardConfig":
        """Look up a named priority preset.

        ============  =========================================================
        ``balanced``  the default: points matter, RPs matter more
        ``score``     raw score only; every ranking point ignored
        ``rp``        ranking points only; raw score nearly ignored
        ``seeding``   RP-weighted the way tournament seeding actually works
        ``stocked``   single-objective: chase Stocked Up
        ``baked``     single-objective: chase Baked Up
        ``dinner``    single-objective: chase the Dinner RP
        ============  =========================================================
        """
        presets = {
            "balanced": dict(score=1.0, stocked_up=1.0, baked_up=1.0,
                             dinner_rp=0.7),
            "score": dict(score=1.0, stocked_up=0.0, baked_up=0.0,
                          dinner_rp=0.0),
            "rp": dict(score=0.05, stocked_up=1.0, baked_up=1.0,
                       dinner_rp=1.0),
            "seeding": dict(score=0.1, stocked_up=1.0, baked_up=1.0,
                            dinner_rp=1.0),
            "stocked": dict(score=0.1, stocked_up=1.0, baked_up=0.0,
                            dinner_rp=0.0),
            "baked": dict(score=0.1, stocked_up=0.0, baked_up=1.0,
                          dinner_rp=0.0),
            "dinner": dict(score=0.1, stocked_up=0.0, baked_up=0.0,
                           dinner_rp=1.0),
        }
        if name not in presets:
            raise ValueError(
                f"unknown reward preset {name!r}; "
                f"expected one of {sorted(presets)}"
            )
        return cls.from_priorities(**presets[name])


# =============================================================================
# BUNDLE
# =============================================================================

@dataclass
class EnvConfig:
    """Everything the environment needs, in one object."""

    field: FieldConfig = _field(default_factory=FieldConfig)
    time: TimeConfig = _field(default_factory=TimeConfig)
    scoring: ScoringConfig = _field(default_factory=ScoringConfig)
    stochastic: StochasticConfig = _field(default_factory=StochasticConfig)
    reward: RewardConfig = _field(default_factory=RewardConfig)

    # ---- Match setup --------------------------------------------------------
    #: Game pieces preloaded in the robot at t=0, as piece names. The robot
    #: cannot start with more than `max_inventory` pieces.
    preload: Tuple[str, ...] = ("CARROT",)

    #: Hard inventory cap. The rules say three.
    max_inventory: int = 3

    #: Carrots available from the farm depot over the whole match. Set large to
    #: model an effectively unlimited supply; lower it to study scarcity.
    depot_carrot_supply: int = 40

    #: Hard cap on steps per episode. Purely a safety net against a policy
    #: that stalls forever; the match clock is the real terminator. A 150 s
    #: match cannot exceed ~430 steps even with only single-cell moves.
    max_episode_steps: int = 2000

    #: When True an illegal action still burns ``TimeConfig.wait_duration_s``
    #: of match clock, so a policy cannot spin on illegal actions for free and
    #: episodes are guaranteed to terminate. Set False only if you always mask
    #: actions with :meth:`HarvestHavocEnv.action_mask`.
    charge_time_for_illegal_actions: bool = True

    #: When True, a macro action whose travel leg fits in the remaining time
    #: but whose manipulation leg does not will still MOVE the robot (it just
    #: does not score). This matters enormously at the buzzer, because parking
    #: is evaluated positionally.
    allow_partial_actions: bool = True

    # ---- Observation normalisation constants -------------------------------
    #: Divisor used to normalise raw score into the observation vector.
    obs_score_cap: float = 150.0
    #: Divisor used to normalise per-shelf piece counts.
    obs_shelf_cap: float = 6.0

    def replace(self, **kwargs) -> "EnvConfig":
        """Return a copy with top-level fields replaced (see dataclasses)."""
        return dataclasses.replace(self, **kwargs)


__all__ = [
    "GRID_CELL_SIZE_FT",
    "MATCH_DURATION_S",
    "FieldConfig",
    "TimeConfig",
    "ScoringConfig",
    "StochasticConfig",
    "RewardConfig",
    "EnvConfig",
]

"""
:class:`HarvestHavocEnv` -- the single-robot Harvest Havoc simulation.

Gymnasium-compatible (``reset -> (obs, info)``,
``step -> (obs, reward, terminated, truncated, info)``) with no hard dependency
on Gymnasium; see :mod:`harvest_havoc.spaces`.

The time contract
-----------------
The environment is **event-driven, not fixed-tick**. One ``step`` advances the
clock by exactly as long as the chosen action takes -- 0.5 s for a WAIT,
maybe 4.8 s for "drive to the pantry and place a cake on level 3". There is no
notion of a control period. This is the single most important design decision
in the project: it makes the agent's decision variable *how to spend time*
rather than *what to do this 20 ms*, and it makes an episode's length a direct
readout of how many actions the 150-second budget bought.

Every step's duration is decomposed into travel / align / manipulate / idle and
appended to :attr:`HarvestHavocEnv.time_ledger`. After a rollout,
:meth:`HarvestHavocEnv.time_allocation_summary` attributes all 150 seconds and
reports points-per-second by action category -- which is the raw material for
the optimal-allocation question.

Action semantics
----------------
* ``MOVE_*`` -- one grid cell. Charged at cruise speed if the robot was already
  rolling, otherwise a full accel/decel trapezoid.
* Macro actions -- shortest path into the required zone, then align, then one
  manipulation. Zero travel time if already in the zone, which is what makes
  repeated scoring at one location cheap.
* ``EXIT_KITCHEN`` -- drive to the nearest cell outside the kitchen complex.
  Legal only during autonomous while the bonus is unclaimed.
* ``WAIT`` -- burn ``wait_duration_s``. Genuinely useful: it is how the robot
  waits at the depot for the human player to finish an exchange.

Buzzer handling
---------------
An action that cannot finish is cut off. If ``allow_partial_actions`` is set
(default), the travel leg still moves the robot as far as the remaining time
allows even though the manipulation does not happen. This matters because the
endgame park is checked positionally: creeping into the table zone as the
buzzer sounds is a legal, and sometimes correct, play.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .actions import (
    ACTION_SPECS,
    Action,
    ActionKind,
    ActionSpec,
    N_ACTIONS,
    spec as action_spec,
)
from .config import EnvConfig
from .field import Cell, Field
from .observation import ObservationEncoder
from .reward import REWARD_OBJECTIVES, RewardBreakdown, RewardCalculator
from .scoring import (
    both_pantry_rps_reachable,
    evaluate_ranking_points,
    dinner_rp_is_reachable,
    endgame_points,
    finalize_endgame,
    pantry_points,
    projected_endgame_points,
    rp_progress,
    stocked_up_is_reachable,
)
from .timing import DurationBreakdown, TimingModel
from .spaces import Box, Discrete, GymEnv
from .state import (
    ActionRecord,
    MatchState,
    OvenState,
    Pantry,
    Phase,
    Piece,
    RobotState,
    ScoreCategory,
)
from .zones import DEFAULT_START_FT, KITCHEN_COMPLEX, Zone, ZoneRect

#: Distance-field key for "the nearest cell outside the kitchen complex".
_OUTSIDE_KITCHEN = "outside_kitchen_complex"

#: Piece-name strings accepted in ``EnvConfig.preload``.
_PIECE_BY_NAME = {"CARROT": Piece.CARROT, "CARROT_CAKE": Piece.CARROT_CAKE,
                  "CAKE": Piece.CARROT_CAKE}


class HarvestHavocEnv(GymEnv):
    """A 150-second, single-robot Harvest Havoc match.

    Parameters
    ----------
    config:
        Full configuration bundle. Defaults to :class:`EnvConfig`.
    layout:
        Zone rectangles overriding :data:`harvest_havoc.zones.DEFAULT_LAYOUT`.
    blocked:
        Optional impassable-cell mask, shape ``(n_cols, n_rows)``.
    start_ft:
        Robot starting position in feet. Must lie in the kitchen complex.
    render_mode:
        ``"ansi"`` returns a string from :meth:`render`; ``"human"`` prints it.

    Attributes
    ----------
    state: MatchState
        Live match state. Read freely; mutate at your own risk.
    field: Field
        The discretised field and navigation service.
    time_ledger: list[ActionRecord]
        One record per step of the current episode.
    """

    metadata = {"render_modes": ["ansi", "human"], "name": "HarvestHavoc-v0"}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        layout: Optional[Sequence[ZoneRect]] = None,
        blocked: Optional[np.ndarray] = None,
        start_ft: Optional[Tuple[float, float]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.field = Field(self.config.field, layout=layout, blocked=blocked)
        self._register_navigation_targets()
        self._validate_config()

        self.start_ft = start_ft or DEFAULT_START_FT
        self.start_cell: Cell = self.field.feet_to_cell(*self.start_ft)

        self.encoder = ObservationEncoder(self.config, self.field)
        self.rewarder = RewardCalculator(self.config)

        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = Box(
            low=self.encoder.low, high=self.encoder.high, dtype=np.float32
        )

        self._rng = np.random.default_rng()
        #: Owns every duration in the simulation. See :mod:`harvest_havoc.timing`.
        self.timing = TimingModel(
            self.config.time, self.config.stochastic, self._rng
        )

        self.state: MatchState = self._fresh_state()
        self.time_ledger: List[ActionRecord] = []
        self._step_count = 0

    # ================================================================== setup

    def _register_navigation_targets(self) -> None:
        """Precompute the non-zone distance fields the macros need."""
        outside = [
            (cx, cy)
            for cx in range(self.field.n_cols)
            for cy in range(self.field.n_rows)
            if self.field.zone_at((cx, cy)) not in KITCHEN_COMPLEX
        ]
        self._has_outside_target = self.field.register_target(
            _OUTSIDE_KITCHEN, outside
        )

    def _validate_config(self) -> None:
        """Fail loudly on configurations that make the game unplayable.

        A silently unreachable ranking point or missing zone would show up
        much later as a mysteriously flat learning curve.
        """
        cfg = self.config
        for zone in (Zone.OWN_KITCHEN, Zone.OWN_FARM, Zone.OWN_PANTRY,
                     Zone.OWN_OVEN, Zone.OWN_TABLE):
            if not self.field.has_zone(zone):
                raise ValueError(
                    f"layout has no cells labelled {zone.name}; at the current "
                    f"cell size ({self.field.cell_size} ft) some zone "
                    f"rectangle is too small to contain a cell centre"
                )
        if len(cfg.preload) > cfg.max_inventory:
            raise ValueError(
                f"preload of {len(cfg.preload)} pieces exceeds the inventory "
                f"cap of {cfg.max_inventory}"
            )
        for name in cfg.preload:
            if name not in _PIECE_BY_NAME:
                raise ValueError(
                    f"unknown preload piece {name!r}; expected one of "
                    f"{sorted(_PIECE_BY_NAME)}"
                )
        if not dinner_rp_is_reachable(cfg.scoring):
            raise ValueError(
                "Dinner RP is unreachable under this ScoringConfig: the "
                "threshold exceeds the maximum park + harvest haul total. "
                "Lower dinner_rp_threshold or use "
                "dinner_rp_mode='all_in_endgame'."
            )
        if not stocked_up_is_reachable(cfg.scoring):
            raise ValueError(
                f"Stocked Up is unreachable: it needs "
                f"{cfg.scoring.stocked_up_threshold} pieces per shelf but "
                f"shelf_capacity is only {cfg.scoring.shelf_capacity}."
            )
        if not both_pantry_rps_reachable(cfg.scoring):
            raise ValueError(
                f"Stocked Up and Baked Up cannot both be earned: "
                f"shelf_capacity={cfg.scoring.shelf_capacity}, "
                f"stocked_up_threshold={cfg.scoring.stocked_up_threshold}, "
                f"baked_up_threshold={cfg.scoring.baked_up_threshold}."
            )
        if cfg.time.endgame_start_s <= cfg.time.auto_end_s:
            raise ValueError("endgame would begin before autonomous ends")

    def _fresh_state(self) -> MatchState:
        """Build the t=0 match state."""
        cfg = self.config
        robot = RobotState(cell=self.start_cell)
        for name in cfg.preload:
            robot.add(_PIECE_BY_NAME[name], cfg.max_inventory)
        return MatchState(
            t=0.0,
            robot=robot,
            pantry=Pantry.create(cfg.scoring.shelf_levels,
                                 capacity=cfg.scoring.shelf_capacity),
            oven=OvenState(),
            depot_carrots=cfg.depot_carrot_supply,
            cakes_available=0,
        )

    # ================================================================ gym API

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new match.

        Parameters
        ----------
        seed:
            Seeds the stochastic model (intake/scoring failures, timing noise,
            human-player delays). With the default deterministic
            :class:`~harvest_havoc.config.StochasticConfig` the seed has no
            effect on dynamics.
        options:
            ``{"start_cell": (cx, cy)}`` overrides the starting cell for this
            episode only -- handy for evaluating a policy from several
            starting positions.

        Returns
        -------
        (obs, info)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.timing.set_rng(self._rng)
            if hasattr(self.action_space, "seed"):
                self.action_space.seed(seed)

        self.state = self._fresh_state()
        if options and "start_cell" in options:
            cell = tuple(options["start_cell"])  # type: ignore[arg-type]
            if not self.field.is_passable(cell):  # type: ignore[arg-type]
                raise ValueError(f"start_cell {cell} is not passable")
            self.state.robot.cell = cell  # type: ignore[assignment]

        self.time_ledger = []
        self._step_count = 0
        self.rewarder.reset(self.state, in_table_zone=self._in_table_zone())
        return self.encoder.encode(self.state), self._build_info(
            RewardBreakdown(), np.zeros(len(REWARD_OBJECTIVES)), None
        )

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one action, advancing the match clock by its duration.

        Parameters
        ----------
        action:
            An :class:`~harvest_havoc.actions.Action` value.

        Returns
        -------
        (obs, reward, terminated, truncated, info)
            ``terminated`` is True when the match clock expires (and the
            endgame has been finalised). ``truncated`` is True only if the
            ``max_episode_steps`` safety net trips.
        """
        if self.state.finalized:
            raise RuntimeError("step() called after the match ended; reset()")

        act = Action(int(action))
        sp = action_spec(act)
        rec = ActionRecord(
            step=self._step_count,
            action=int(act),
            action_name=sp.name,
            phase=self.state.phase(self.config.time),
            t_start=self.state.t,
            t_end=self.state.t,
        )

        if not self._is_legal(act):
            self._execute_illegal(rec)
        elif sp.is_move:
            self._execute_move(sp, rec)
        elif sp.kind is ActionKind.WAIT:
            self._execute_wait(rec)
        else:
            self._execute_macro(sp, rec)

        # Enforce the minimum action duration so no action is ever free. This
        # only binds on degenerate cases (EXIT_KITCHEN while already outside
        # the kitchen), but it makes "every action costs time" a guarantee
        # rather than a near-certainty.
        floor = self.config.time.min_action_duration_s
        elapsed = self.state.t - rec.t_start
        if elapsed < floor - 1e-12:
            extra = min(floor - elapsed, self._time_left())
            if extra > 0.0:
                rec.idle_s += extra
                self._advance_time(extra)

        rec.t_end = self.state.t
        self.time_ledger.append(rec)
        self._step_count += 1

        terminated = self.state.is_over(self.config.time)
        truncated = (not terminated
                     and self._step_count >= self.config.max_episode_steps)

        if terminated:
            self._finalize()

        breakdown, vector = self.rewarder.compute(
            self.state,
            dt=rec.duration_s,
            terminal=terminated,
            in_table_zone=self._in_table_zone(),
            illegal=rec.illegal,
        )
        obs = self.encoder.encode(self.state)
        info = self._build_info(breakdown, vector, rec)
        return obs, float(breakdown.total), terminated, truncated, info

    # ========================================================= action legality

    def action_mask(self) -> np.ndarray:
        """Boolean mask of currently legal actions, shape ``(N_ACTIONS,)``.

        Strongly recommended for training: it removes the entire class of
        "score a cake you are not holding" transitions, which otherwise waste
        a large fraction of samples. ``WAIT`` is always legal, so the mask is
        never all-False.
        """
        return np.array(
            [self._is_legal(Action(a)) for a in range(N_ACTIONS)], dtype=bool
        )

    def legal_actions(self) -> List[Action]:
        """The legal actions right now, as enum members."""
        return [Action(a) for a in range(N_ACTIONS) if self._is_legal(Action(a))]

    def _is_legal(self, act: Action) -> bool:
        """Whether `act` may be selected in the current state."""
        st = self.state
        sp = ACTION_SPECS[act]
        kind = sp.kind

        if kind is ActionKind.WAIT:
            return True

        if kind is ActionKind.MOVE:
            assert sp.delta is not None
            target = (st.robot.cell[0] + sp.delta[0],
                      st.robot.cell[1] + sp.delta[1])
            return self.field.is_passable(target)

        if kind is ActionKind.INTAKE:
            if st.robot.free_slots(self.config.max_inventory) <= 0:
                return False
            if sp.piece is Piece.CARROT:
                return st.depot_carrots > 0
            # A cake intake may be attempted speculatively while the human
            # player is still working: the robot can legally drive over and
            # wait. If nothing has arrived by the time it gets there the
            # attempt fails and the time is lost -- which is the real tradeoff.
            return st.cakes_available > 0 or st.oven.pending_count > 0

        if kind is ActionKind.SCORE_PANTRY:
            assert sp.piece is not None and sp.level is not None
            # A full shelf makes the action illegal rather than merely
            # unproductive, so the mask steers the policy away from it instead
            # of letting it burn a whole travel leg to discover the problem.
            return st.robot.has(sp.piece) and st.pantry.can_add(sp.level)

        if kind is ActionKind.OVEN_DEPOSIT:
            assert sp.piece is not None
            return st.robot.has(sp.piece)

        if kind is ActionKind.PARK:
            return self.field.has_zone(Zone.OWN_TABLE)

        if kind is ActionKind.EXIT_KITCHEN:
            # Requires the robot to still be inside the kitchen complex.
            # Otherwise the action is a zero-distance no-op, which would be the
            # one free action in the space -- and nothing in this game should
            # be free.
            return (
                self._has_outside_target
                and not st.auto_leave_earned
                and st.phase(self.config.time) is Phase.AUTO
                and self.field.zone_at(st.robot.cell) in KITCHEN_COMPLEX
            )

        return False  # pragma: no cover - all kinds handled above

    # ======================================================= action execution

    def _execute_illegal(self, rec: ActionRecord) -> None:
        """Reject an action, optionally burning a little clock."""
        rec.illegal = True
        rec.success = False
        rec.note = "illegal action"
        if self.config.charge_time_for_illegal_actions:
            dt = min(self.config.time.wait_duration_s, self._time_left())
            rec.idle_s = dt
            self._advance_time(dt)
        self.state.robot.in_motion = False

    def _execute_wait(self, rec: ActionRecord) -> None:
        """Burn ``wait_duration_s``, doing nothing.

        WAIT is the only action that does not clear the parked flag, so a
        robot can park and then hold position through the buzzer.
        """
        dt = min(self.config.time.wait_duration_s, self._time_left())
        rec.idle_s = dt
        rec.truncated_by_buzzer = dt < self.config.time.wait_duration_s - 1e-9
        self._advance_time(dt)
        self.state.robot.in_motion = False

    def _execute_move(self, sp: ActionSpec, rec: ActionRecord) -> None:
        """Drive one grid cell."""
        st = self.state
        assert sp.delta is not None
        target = (st.robot.cell[0] + sp.delta[0], st.robot.cell[1] + sp.delta[1])
        distance = self.field.step_cost_ft(st.robot.cell, target)

        tcfg = self.config.time
        rolling = tcfg.momentum_carries_between_moves and st.robot.in_motion
        dt = self.timing.travel(distance, from_rest=not rolling)

        st.robot.parked = False
        remaining = self._time_left()
        if dt > remaining + 1e-9:
            # Not enough clock to complete the hop: the robot does not arrive.
            rec.travel_s = remaining
            rec.truncated_by_buzzer = True
            rec.success = False
            rec.note = "move cut off by buzzer"
            self._advance_time(remaining)
            st.robot.in_motion = False
            return

        rec.travel_s = dt
        self._advance_time(dt)
        st.robot.cell = target
        st.robot.in_motion = True
        self._check_auto_leave(rec)

    def _execute_macro(self, sp: ActionSpec, rec: ActionRecord) -> None:
        """Navigate to the action's target zone, then manipulate once."""
        st = self.state

        # ---- leg 1: travel -------------------------------------------------
        target_key = (_OUTSIDE_KITCHEN if sp.kind is ActionKind.EXIT_KITCHEN
                      else sp.target_zone)
        assert target_key is not None
        distance = self.field.distance_to_target(st.robot.cell, target_key)
        if not math.isfinite(distance):
            # Unreachable target: fail as a no-op rather than crashing, so a
            # hand-edited layout or an added obstacle cannot hard-fail a
            # training run mid-rollout. Time still passes.
            rec.success = False
            rec.note = f"target {target_key} unreachable"
            dt = min(self.config.time.wait_duration_s, self._time_left())
            rec.idle_s = dt
            self._advance_time(dt)
            st.robot.in_motion = False
            return

        travel = self.timing.travel(distance, from_rest=True)
        st.robot.parked = False

        remaining = self._time_left()
        if travel > remaining + 1e-9:
            rec.travel_s = remaining
            rec.truncated_by_buzzer = True
            rec.success = False
            rec.note = "travel cut off by buzzer"
            if self.config.allow_partial_actions and remaining > 0:
                self._advance_partial(target_key, distance, travel, remaining)
            self._advance_time(remaining)
            st.robot.in_motion = False
            self._check_auto_leave(rec)
            return

        rec.travel_s = travel
        self._advance_time(travel)
        path = self.field.path_to_target(st.robot.cell, target_key)
        if path:
            st.robot.cell = path[-1]
        st.robot.in_motion = False
        self._check_auto_leave(rec)

        # ---- leg 2: align + manipulate -------------------------------------
        # Align time is only charged when there is actually something to line
        # up on. EXIT_KITCHEN just needs the robot's frame across a line, so
        # it pays travel only.
        nominal_manipulate = self._manipulation_time(sp)
        align = (self.timing.sample(self.config.time.align_time_s, "align")
                 if nominal_manipulate > 0.0 else 0.0)
        manipulate = self.timing.sample(nominal_manipulate, "manipulate")
        need = align + manipulate
        remaining = self._time_left()
        if need > remaining + 1e-9:
            # Arrived but ran out of clock. The robot is in position -- which
            # is exactly what makes a last-instant park attempt worthwhile.
            rec.align_s = min(align, remaining)
            rec.manipulate_s = max(0.0, remaining - rec.align_s)
            rec.truncated_by_buzzer = True
            rec.success = False
            rec.note = "manipulation cut off by buzzer"
            self._advance_time(remaining)
            return

        rec.align_s = align
        rec.manipulate_s = manipulate
        self._advance_time(need)
        self._apply_interaction(sp, rec)

    def _advance_partial(
        self, target_key: Any, distance: float, travel: float, remaining: float
    ) -> None:
        """Move the robot part-way along its path when the buzzer intervenes.

        Uses a linear distance-vs-time approximation of the trapezoidal
        profile. Exact only for the cruise segment, but it is applied solely in
        the final fraction of a second of a match, where the alternative
        (freezing the robot in place) would be a much larger error -- the park
        bonus turns on which cell the robot ends in.
        """
        path = self.field.path_to_target(self.state.robot.cell, target_key)
        if not path or len(path) < 2 or travel <= 0.0:
            return
        reachable = distance * (remaining / travel)
        travelled = 0.0
        best = path[0]
        for a, b in zip(path, path[1:]):
            travelled += self.field.step_cost_ft(a, b)
            if travelled > reachable + 1e-9:
                break
            best = b
        self.state.robot.cell = best

    def _manipulation_time(self, sp: ActionSpec) -> float:
        """Manipulation seconds for a macro action, excluding travel/align."""
        tcfg = self.config.time
        if sp.kind is ActionKind.INTAKE:
            return (tcfg.intake_carrot_s if sp.piece is Piece.CARROT
                    else tcfg.intake_cake_s)
        if sp.kind is ActionKind.SCORE_PANTRY:
            assert sp.level is not None
            table = (tcfg.score_carrot_s if sp.piece is Piece.CARROT
                     else tcfg.score_cake_s)
            return table[sp.level]
        if sp.kind is ActionKind.OVEN_DEPOSIT:
            return tcfg.oven_deposit_s
        if sp.kind is ActionKind.PARK:
            return tcfg.park_settle_s
        if sp.kind is ActionKind.EXIT_KITCHEN:
            return 0.0  # the bonus is positional; nothing to manipulate
        return 0.0

    # ==================================================== interaction effects

    def _apply_interaction(self, sp: ActionSpec, rec: ActionRecord) -> None:
        """Resolve the game-piece effect of a completed macro action."""
        st = self.state
        scfg = self.config.scoring
        stoch = self.config.stochastic

        if sp.kind is ActionKind.INTAKE:
            assert sp.piece is not None
            if not self._roll(stoch.intake_success_prob):
                rec.success = False
                rec.note = "intake failed"
                return
            if sp.piece is Piece.CARROT:
                if st.depot_carrots <= 0:
                    rec.success = False
                    rec.note = "depot empty"
                    return
                st.depot_carrots -= 1
            else:
                if st.cakes_available <= 0:
                    # Drove over speculatively; the human player was not ready.
                    rec.success = False
                    rec.note = "no cake ready at depot"
                    return
                st.cakes_available -= 1
            if not st.robot.add(sp.piece, self.config.max_inventory):
                rec.success = False
                rec.note = "inventory full"
            return

        if sp.kind is ActionKind.SCORE_PANTRY:
            assert sp.piece is not None and sp.level is not None
            if not st.robot.has(sp.piece):
                rec.success = False
                rec.note = "piece no longer held"
                return
            if not st.pantry.can_add(sp.level):
                # The shelf filled up between selection and arrival. Cannot
                # happen with a single robot, but it will once opponents or
                # alliance partners can also score, so handle it now.
                rec.success = False
                rec.note = f"shelf {sp.level} is full"
                return
            if not self._roll(stoch.score_success_prob.get(sp.level, 1.0)):
                # A missed shot loses the time but not the piece: the robot
                # keeps it and can retry. Change here to model dropped pieces.
                rec.success = False
                rec.note = f"missed level {sp.level}"
                return
            st.robot.remove(sp.piece)
            st.pantry.add(sp.level, sp.piece)
            category = (ScoreCategory.PANTRY_CARROT if sp.piece is Piece.CARROT
                        else ScoreCategory.PANTRY_CAKE)
            rec.points += st.award(
                pantry_points(sp.piece, sp.level, scfg), category,
                level=sp.level, piece=sp.piece,
            )
            return

        if sp.kind is ActionKind.OVEN_DEPOSIT:
            assert sp.piece is not None
            if not st.robot.has(sp.piece):
                rec.success = False
                rec.note = "piece no longer held"
                return
            if not self._roll(stoch.oven_success_prob):
                rec.success = False
                rec.note = "oven deposit failed"
                return
            st.robot.remove(sp.piece)
            rec.points += st.award(
                scfg.oven_points, ScoreCategory.OVEN, piece=sp.piece
            )
            if sp.piece is Piece.CARROT:
                st.oven.carrots_deposited += 1
                st.oven.carrots_toward_next += 1
                self._maybe_trigger_exchange()
            else:
                st.oven.cakes_deposited += 1
            return

        if sp.kind is ActionKind.PARK:
            if not self._roll(stoch.park_success_prob):
                rec.success = False
                rec.note = "park attempt failed"
                return
            st.robot.parked = True
            rec.note = "parked (scored at buzzer if still in table zone)"
            return

        if sp.kind is ActionKind.EXIT_KITCHEN:
            # Points, if any, were already awarded by _check_auto_leave during
            # the travel leg. Nothing else to do.
            return

    def _maybe_trigger_exchange(self) -> None:
        """Convert oven carrots into a pending human-player cake exchange."""
        st = self.state
        scfg = self.config.scoring
        stoch = self.config.stochastic
        while st.oven.carrots_toward_next >= scfg.oven_carrots_per_cake:
            st.oven.carrots_toward_next -= scfg.oven_carrots_per_cake
            if not self._roll(stoch.hp_exchange_success_prob):
                st.oven.failed_exchanges += 1
                continue
            st.oven.pending_exchange_times.append(
                st.t + self.timing.hp_exchange_delay()
            )

    def _check_auto_leave(self, rec: ActionRecord) -> None:
        """Award the one-time autonomous leave-the-kitchen bonus if earned.

        The robot is modelled as a point, so "fully leaving the kitchen zone"
        becomes "the robot's cell is outside the kitchen complex" -- and the
        complex, not the bare ``OWN_KITCHEN`` label, is the right test, since
        the pantry and oven sit inside the kitchen.
        """
        st = self.state
        if st.auto_leave_earned:
            return
        if st.phase(self.config.time) is not Phase.AUTO:
            return
        if self.field.zone_at(st.robot.cell) in KITCHEN_COMPLEX:
            return
        st.auto_leave_earned = True
        rec.points += st.award(
            self.config.scoring.auto_leave_kitchen_points,
            ScoreCategory.AUTO_LEAVE,
        )

    # ============================================================ clock & rng

    def _time_left(self) -> float:
        """Seconds remaining in the match."""
        return self.state.time_remaining(self.config.time)

    def _advance_time(self, dt: float) -> None:
        """Advance the clock by `dt`, clamping at the buzzer.

        Also completes any human-player exchanges that mature in the interval.
        This is the single funnel through which time passes, which is what
        guarantees the ledger accounts for every second.
        """
        if dt <= 0.0:
            self._process_exchanges()
            return
        self.state.t = min(
            self.config.time.match_duration_s, self.state.t + dt
        )
        self._process_exchanges()

    def _process_exchanges(self) -> None:
        """Release cakes whose human-player exchange has completed."""
        oven = self.state.oven
        if not oven.pending_exchange_times:
            return
        due = [t for t in oven.pending_exchange_times if t <= self.state.t + 1e-9]
        if not due:
            return
        oven.pending_exchange_times = [
            t for t in oven.pending_exchange_times if t > self.state.t + 1e-9
        ]
        self.state.cakes_available += len(due)

    def _roll(self, probability: float) -> bool:
        """Bernoulli trial. Short-circuits at 0.0 and 1.0 to stay deterministic."""
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        return bool(self._rng.random() < probability)

    # ============================================================== finishing

    def _in_table_zone(self) -> bool:
        """True if the robot currently occupies its own table zone."""
        return self.field.zone_at(self.state.robot.cell) is Zone.OWN_TABLE

    def _finalize(self) -> None:
        """Score the endgame park and harvest haul. Called once, at the buzzer.

        Both conditions must hold: a PARK action must have completed (and not
        been invalidated by a later action), and the robot must actually be in
        its own table zone.
        """
        parked = self.state.robot.parked and self._in_table_zone()
        finalize_endgame(self.state, self.config.scoring, parked=parked)

    # =================================================================== info

    def _build_info(
        self,
        breakdown: RewardBreakdown,
        vector: np.ndarray,
        rec: Optional[ActionRecord],
    ) -> Dict[str, Any]:
        """Assemble the ``info`` dict for a step or reset."""
        st = self.state
        cfg = self.config
        in_table = self._in_table_zone()
        rps = evaluate_ranking_points(st, cfg.scoring, cfg.time)
        progress = rp_progress(
            st, cfg.scoring, cfg.time, in_table_zone=in_table
        )
        return {
            # --- clock -------------------------------------------------------
            "t": st.t,
            "phase": st.phase(cfg.time).name,
            "time_remaining": st.time_remaining(cfg.time),
            "time_cost": rec.duration_s if rec else 0.0,
            "time_split": {
                "travel": rec.travel_s if rec else 0.0,
                "align": rec.align_s if rec else 0.0,
                "manipulate": rec.manipulate_s if rec else 0.0,
                "idle": rec.idle_s if rec else 0.0,
            },
            # --- outcome -----------------------------------------------------
            "raw_score": st.raw_score,
            "score_by_category": {
                k.value: v for k, v in st.points_by_category().items()
            },
            "ranking_points": rps.as_dict(),
            "ranking_point_total": rps.total,
            "rp_progress": progress.as_dict(),
            "endgame_points": (
                endgame_points(st, cfg.scoring, cfg.time) if st.finalized
                else projected_endgame_points(
                    st, cfg.scoring, cfg.time, in_table_zone=in_table
                )
            ),
            # --- robot -------------------------------------------------------
            "cell": st.robot.cell,
            "zone": self.field.zone_at(st.robot.cell).name,
            "inventory": {"carrots": st.robot.carrots, "cakes": st.robot.cakes},
            "parked": st.robot.parked,
            "shelves": {
                lvl: {"carrots": s.carrots, "cakes": s.cakes,
                      "free": s.free_slots, "full": s.is_full}
                for lvl, s in st.pantry.shelves.items()
            },
            "pantry_full": st.pantry.is_full(),
            "cakes_available": st.cakes_available,
            "pending_exchanges": st.oven.pending_count,
            "depot_carrots": st.depot_carrots,
            # --- action feedback ---------------------------------------------
            "action_success": rec.success if rec else True,
            "action_illegal": rec.illegal if rec else False,
            "action_note": rec.note if rec else "",
            "action_mask": self.action_mask(),
            # --- reward ------------------------------------------------------
            "reward_breakdown": breakdown.as_dict(),
            "reward_vector": vector,
            "reward_objectives": REWARD_OBJECTIVES,
        }

    # ========================================================= time analysis

    def time_allocation_summary(self) -> Dict[str, Any]:
        """Attribute every second of the episode, and rate each activity.

        This is the environment's primary analytical output. It answers, for a
        completed rollout: where did the 150 seconds go, and what did each
        category of spending buy?

        Returns
        -------
        dict
            ``by_kind``
                Per :class:`~harvest_havoc.actions.ActionKind`: ``count``,
                ``seconds``, ``points``, ``points_per_second``.
            ``by_component``
                Total ``travel`` / ``align`` / ``manipulate`` / ``idle``
                seconds across the whole match.
            ``totals``
                ``steps``, ``accounted_s``, ``match_s``, ``unused_s``,
                ``raw_score``, ``overall_points_per_second``.
            ``by_phase``
                Seconds and points spent in each match phase.
            ``failures``
                Counts of illegal, failed, and buzzer-truncated actions.

        Notes
        -----
        Park and harvest-haul points are awarded at the buzzer rather than by
        an action, so ``sum(by_kind[...].points)`` can be less than
        ``raw_score``. The difference is exactly the endgame bonus, and is
        reported as ``totals["unattributed_points"]``.
        """
        by_kind: Dict[str, Dict[str, float]] = {}
        components = {"travel": 0.0, "align": 0.0, "manipulate": 0.0, "idle": 0.0}
        by_phase: Dict[str, Dict[str, float]] = {
            p.name: {"seconds": 0.0, "points": 0.0, "steps": 0} for p in Phase
        }
        failures = {"illegal": 0, "failed": 0, "truncated_by_buzzer": 0}

        for rec in self.time_ledger:
            kind = ACTION_SPECS[Action(rec.action)].kind.value
            bucket = by_kind.setdefault(
                kind, {"count": 0, "seconds": 0.0, "points": 0.0}
            )
            bucket["count"] += 1
            bucket["seconds"] += rec.duration_s
            bucket["points"] += rec.points

            components["travel"] += rec.travel_s
            components["align"] += rec.align_s
            components["manipulate"] += rec.manipulate_s
            components["idle"] += rec.idle_s

            ph = by_phase[rec.phase.name]
            ph["seconds"] += rec.duration_s
            ph["points"] += rec.points
            ph["steps"] += 1

            failures["illegal"] += int(rec.illegal)
            failures["failed"] += int(not rec.success)
            failures["truncated_by_buzzer"] += int(rec.truncated_by_buzzer)

        for bucket in by_kind.values():
            bucket["points_per_second"] = (
                bucket["points"] / bucket["seconds"]
                if bucket["seconds"] > 1e-9 else 0.0
            )

        accounted = sum(r.duration_s for r in self.time_ledger)
        match_s = self.config.time.match_duration_s
        attributed_points = sum(r.points for r in self.time_ledger)
        return {
            "by_kind": by_kind,
            "by_component": components,
            "by_phase": by_phase,
            "failures": failures,
            "totals": {
                "steps": len(self.time_ledger),
                "accounted_s": accounted,
                "match_s": match_s,
                "unused_s": max(0.0, match_s - accounted),
                "raw_score": self.state.raw_score,
                "unattributed_points": self.state.raw_score - attributed_points,
                "overall_points_per_second": (
                    self.state.raw_score / accounted if accounted > 1e-9 else 0.0
                ),
            },
        }

    def nominal_duration(
        self, action: int, from_cell: Optional[Cell] = None
    ) -> DurationBreakdown:
        """Noise-free duration of `action`, split by component.

        Every action in the space has a time cost, and this is how you see it.
        Travel is measured from `from_cell` (default: the robot's current
        cell), so the same action costs different amounts from different
        places -- which is the entire point of the spatial model.

        Parameters
        ----------
        action:
            An :class:`~harvest_havoc.actions.Action` value.
        from_cell:
            Where the robot is assumed to be. Defaults to its actual cell.

        Returns
        -------
        DurationBreakdown
            ``travel`` / ``align`` / ``manipulate`` / ``wait`` seconds, plus
            ``.total``. Travel is ``inf`` if the target is unreachable.
        """
        from .timing import traverse_time

        sp = action_spec(Action(int(action)))
        cell = from_cell if from_cell is not None else self.state.robot.cell
        out = DurationBreakdown()

        if sp.kind is ActionKind.WAIT:
            out.wait = self.config.time.wait_duration_s
            return out

        if sp.is_move:
            assert sp.delta is not None
            target = (cell[0] + sp.delta[0], cell[1] + sp.delta[1])
            if self.field.is_passable(target):
                out.travel = traverse_time(
                    self.field.step_cost_ft(cell, target), self.config.time
                )
            return out

        target_key = (_OUTSIDE_KITCHEN if sp.kind is ActionKind.EXIT_KITCHEN
                      else sp.target_zone)
        distance = self.field.distance_to_target(cell, target_key)
        out.travel = (traverse_time(distance, self.config.time)
                      if math.isfinite(distance) else math.inf)
        out.manipulate = self._manipulation_time(sp)
        out.align = self.config.time.align_time_s if out.manipulate > 0 else 0.0
        return out

    def action_time_table(
        self, from_cell: Optional[Cell] = None, samples: int = 2000
    ) -> List[Dict[str, Any]]:
        """Nominal and sampled duration of **every** action, as rows.

        One row per action in the space, so you can confirm at a glance that
        nothing is free and see how much variance the probabilistic model adds.

        Parameters
        ----------
        from_cell:
            Position to measure travel from. Defaults to the robot's cell.
        samples:
            Draws used to estimate the sampled mean/spread per action.

        Returns
        -------
        list[dict]
            Keys: ``action``, ``name``, ``kind``, ``travel``, ``align``,
            ``manipulate``, ``wait``, ``nominal_total``, ``mean``, ``p05``,
            ``p50``, ``p95``, ``legal``.
        """
        rows: List[Dict[str, Any]] = []
        for act in Action:
            sp = action_spec(act)
            parts = self.nominal_duration(int(act), from_cell)
            row: Dict[str, Any] = {
                "action": int(act),
                "name": act.name,
                "kind": sp.kind.value,
                **parts.as_dict(),
                "nominal_total": parts.total,
                "legal": bool(self._is_legal(act)),
            }
            if math.isfinite(parts.total):
                stats = {
                    comp: self.timing.describe(value, comp, samples=samples)
                    for comp, value in parts.as_dict().items() if value > 0
                }
                row["mean"] = sum(s["mean"] for s in stats.values())
                # Component draws are independent, so quantiles do not add;
                # report the sum of per-component quantiles as an indication
                # of spread rather than a true joint quantile.
                row["p05"] = sum(s["p05"] for s in stats.values())
                row["p50"] = sum(s["p50"] for s in stats.values())
                row["p95"] = sum(s["p95"] for s in stats.values())
            else:
                row.update(mean=math.inf, p05=math.inf, p50=math.inf,
                           p95=math.inf)
            rows.append(row)
        return rows

    def match_result(self) -> Dict[str, Any]:
        """Compact end-of-match summary: score, RPs, and the score breakdown."""
        cfg = self.config
        rps = evaluate_ranking_points(self.state, cfg.scoring, cfg.time)
        return {
            "raw_score": self.state.raw_score,
            "ranking_points": rps.as_dict(),
            "ranking_point_total": rps.total,
            "endgame_points": endgame_points(self.state, cfg.scoring, cfg.time),
            "score_by_category": {
                k.value: v for k, v in self.state.points_by_category().items()
            },
            "shelves": {
                lvl: {"carrots": s.carrots, "cakes": s.cakes,
                      "free": s.free_slots}
                for lvl, s in self.state.pantry.shelves.items()
            },
            "pantry_used": self.state.pantry.total_pieces(),
            "pantry_capacity": self.state.pantry.total_capacity(),
            "steps": len(self.time_ledger),
            "finalized": self.state.finalized,
        }

    # ================================================================= render

    def render(self) -> Optional[str]:
        """Render the field and scoreboard.

        Returns the string for ``render_mode="ansi"``, prints it for
        ``"human"``, and returns ``None`` if no render mode is set.
        """
        from .render import render_ascii

        if self.render_mode is None:
            return None
        text = render_ascii(self)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        """No resources to release; present for API completeness."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"HarvestHavocEnv(t={self.state.t:.1f}/"
            f"{self.config.time.match_duration_s:.0f}s, "
            f"score={self.state.raw_score}, steps={self._step_count})"
        )


__all__ = ["HarvestHavocEnv"]

"""
Scripted baseline policies and a rollout helper.

These are not learning agents. They exist for three reasons:

1. **Validating the simulation.** A hand-written cycling policy should produce
   a plausible number of cycles and a plausible score. If it does not, the time
   model is wrong.
2. **Establishing a floor.** Any learned policy that cannot beat
   :class:`CycleAndParkPolicy` has not learned anything.
3. **Probing the question directly.** Sweeping
   ``CycleAndParkPolicy(level=1|2|3)`` and comparing points-per-second is
   already a partial answer to "which shelf is worth the elevator time?", and
   it costs nothing to run.

Unlike an RL agent, these policies read :attr:`HarvestHavocEnv.state` directly
rather than the observation vector. That is intentional -- they are reference
strategies, not learners, and privileged access keeps them short and readable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .actions import Action
from .env import HarvestHavocEnv

#: Score-carrot action for each pantry level.
_SCORE_CARROT = {
    1: Action.SCORE_CARROT_L1,
    2: Action.SCORE_CARROT_L2,
    3: Action.SCORE_CARROT_L3,
}
#: Score-cake action for each pantry level.
_SCORE_CAKE = {
    1: Action.SCORE_CAKE_L1,
    2: Action.SCORE_CAKE_L2,
    3: Action.SCORE_CAKE_L3,
}


class ScriptedPolicy:
    """Base class for a hand-written strategy.

    Subclasses implement :meth:`act`. Returning an illegal action is safe --
    the environment rejects it and burns a little clock -- but every policy
    here filters against :meth:`HarvestHavocEnv.legal_actions` anyway.
    """

    name: str = "scripted"

    def reset(self, env: HarvestHavocEnv) -> None:
        """Called once at the start of each episode."""

    def act(self, env: HarvestHavocEnv) -> Action:
        """Choose an action for the current state."""
        raise NotImplementedError


class RandomPolicy(ScriptedPolicy):
    """Uniformly random over the currently legal actions.

    The absolute performance floor, and a useful smoke test: a random policy
    must never crash the environment or leave the clock unaccounted for.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, env: HarvestHavocEnv) -> Action:
        legal = env.legal_actions()
        return legal[int(self.rng.integers(0, len(legal)))]


class _HopperCycler(ScriptedPolicy):
    """Mixin providing fill-then-empty hopper batching.

    Naively "intake whenever there is a free slot" is a trap: it sends the
    robot back to the farm after *every single* score, so it pays the full
    round-trip travel per game piece instead of per three. Batching is worth
    roughly a third of the match, so every scripted baseline uses this.

    Subclasses read :attr:`_unloading` and must call :meth:`_update_phase`
    once at the top of ``act``.
    """

    def reset(self, env: HarvestHavocEnv) -> None:
        self._unloading = False

    def _update_phase(self, env: HarvestHavocEnv) -> None:
        """Flip between filling the hopper and emptying it."""
        held = env.state.robot.held
        if held >= env.config.max_inventory:
            self._unloading = True
        elif held == 0:
            self._unloading = False

    def _should_intake(self, env: HarvestHavocEnv) -> bool:
        """True if the robot should be heading to the farm right now."""
        return (
            not getattr(self, "_unloading", False)
            and env.state.robot.free_slots(env.config.max_inventory) > 0
        )


class CycleAndParkPolicy(_HopperCycler):
    """Fill the robot with carrots, dump them all on one shelf, repeat.

    The classic raw-score strategy, parameterised by shelf so the three
    variants can be compared head to head. Sweeping ``level`` over 1/2/3 with
    everything else held fixed is the cleanest available measurement of
    "is the extra elevator time worth the extra points?".

    Parameters
    ----------
    level:
        Pantry shelf to score every carrot on.
    park_lead_s:
        Seconds before the buzzer at which to abandon cycling and drive to the
        table zone. Must be generous enough to cover the drive plus the settle
        time, or the park is lost. ``None`` disables parking entirely, which
        is the right ablation for measuring the park's opportunity cost.
    hold_for_haul:
        If True, stop scoring once heading to park so whatever is aboard
        counts toward the harvest haul. If False, keep cycling and accept a
        bare 2-point park.
    """

    def __init__(
        self,
        level: int = 3,
        park_lead_s: Optional[float] = 8.0,
        hold_for_haul: bool = True,
    ) -> None:
        self.level = level
        self.park_lead_s = park_lead_s
        self.hold_for_haul = hold_for_haul
        self.name = f"cycle_L{level}" + ("" if park_lead_s else "_nopark")

    def act(self, env: HarvestHavocEnv) -> Action:
        st = env.state
        cfg = env.config
        self._update_phase(env)
        legal = set(env.legal_actions())
        t_left = st.time_remaining(cfg.time)

        # Free 2 points at the very start of autonomous.
        if Action.EXIT_KITCHEN in legal:
            return Action.EXIT_KITCHEN

        # Endgame: park, then hold still. WAIT is the only action that does
        # not clear the parked flag, so holding is mandatory once parked.
        parking = self.park_lead_s is not None and t_left <= self.park_lead_s
        if parking:
            if st.robot.parked:
                return Action.WAIT
            if Action.PARK in legal:
                return Action.PARK
            if self.hold_for_haul:
                return Action.WAIT

        # Cakes are worth 2-3x a carrot; never leave one aboard mid-match.
        # Preferred shelf first, then any shelf with room.
        if st.robot.cakes > 0:
            act = self._best_available(env, _SCORE_CAKE, legal)
            if act is not None:
                return act

        if self._should_intake(env) and not parking:
            if st.cakes_available > 0 and Action.INTAKE_CAKE in legal:
                return Action.INTAKE_CAKE
            if Action.INTAKE_CARROT in legal:
                return Action.INTAKE_CARROT

        act = self._best_available(env, _SCORE_CARROT, legal)
        if act is not None:
            return act

        # Every shelf is full. The oven is the only scoring left -- 2 points a
        # piece, and it also feeds the cake pipeline, though with a full pantry
        # those cakes are only good for the endgame harvest haul.
        if st.robot.cakes > 0 and Action.OVEN_DEPOSIT_CAKE in legal:
            return Action.OVEN_DEPOSIT_CAKE
        if Action.OVEN_DEPOSIT_CARROT in legal:
            return Action.OVEN_DEPOSIT_CARROT
        return Action.WAIT

    def _best_available(self, env, action_map, legal):
        """Preferred shelf if it has room, else the highest-value one that does."""
        if action_map[self.level] in legal:
            return action_map[self.level]
        for lvl in sorted(env.state.pantry.levels, reverse=True):
            if action_map[lvl] in legal:
                return action_map[lvl]
        return None


class RankingPointRushPolicy(_HopperCycler):
    """Heuristic that chases all three ranking points, then raw score.

    Priority order:

    1. Take the autonomous leave-kitchen bonus.
    2. Park (and hold) once the endgame deadline arrives.
    3. Place a held cake on the highest-value shelf that still lacks one
       (Baked Up).
    4. Feed the oven when more cakes are needed than are in the pipeline.
    5. Bring every shelf up to the Stocked Up threshold -- **while reserving
       one slot per shelf for a cake**.
    6. Otherwise dump carrots on the highest shelf with room.

    The slot reservation is the crux. With ``shelf_capacity == 3`` and
    ``stocked_up_threshold == 3``, a shelf filled with three carrots satisfies
    Stocked Up and **permanently forfeits Baked Up** -- pieces cannot be
    removed. So this policy parks at most ``capacity - baked_up_threshold``
    carrots on each shelf and waits for a cake to finish the set. That costs
    real time (three oven carrots and a depot round trip per cake) and is
    exactly the tradeoff the learner should be resolving.

    This is a *reasonable* strategy, not an optimal one: it does not reason
    about whether those cake round trips are worth more than the pantry points
    they displace, nor about when to give up on a ranking point.

    Parameters
    ----------
    park_lead_s:
        Seconds before the buzzer at which to break off and park.
    chase_dinner_rp:
        Whether to hoard cakes for the harvest haul (the only route to 10
        endgame points under the default ``park_and_haul`` reading).
    reserve_cake_slots:
        Keep one slot per shelf free for a cake. Setting this False reproduces
        the trap: Stocked Up arrives sooner, Baked Up becomes impossible.
    """

    def __init__(
        self,
        park_lead_s: float = 10.0,
        chase_dinner_rp: bool = True,
        reserve_cake_slots: bool = True,
    ) -> None:
        self.park_lead_s = park_lead_s
        self.chase_dinner_rp = chase_dinner_rp
        self.reserve_cake_slots = reserve_cake_slots
        self.name = (
            "rp_rush"
            + ("" if chase_dinner_rp else "_no_dinner")
            + ("" if reserve_cake_slots else "_no_reserve")
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _shelves_missing_cake(env: HarvestHavocEnv) -> List[int]:
        """Shelf levels short of their cake quota, highest value first.

        A shelf that is full of carrots is excluded -- it can never take a
        cake now, so counting it would make the policy chase an impossible
        target for the rest of the match.
        """
        st = env.state
        thr = env.config.scoring.baked_up_threshold
        return sorted(
            (lvl for lvl in st.pantry.levels
             if st.pantry.shelves[lvl].cakes < thr
             and not st.pantry.shelves[lvl].is_full),
            reverse=True,
        )

    def _carrot_quota(self, env: HarvestHavocEnv) -> int:
        """Carrots allowed on one shelf before its cake slot must be kept free."""
        scfg = env.config.scoring
        if not self.reserve_cake_slots:
            return scfg.shelf_capacity
        return max(0, scfg.shelf_capacity - scfg.baked_up_threshold)

    def _shelves_wanting_carrots(self, env: HarvestHavocEnv) -> List[int]:
        """Shelves that should take another carrot, emptiest first."""
        st = env.state
        quota = self._carrot_quota(env)
        thr = env.config.scoring.stocked_up_threshold
        wanting = [
            lvl for lvl in st.pantry.levels
            if st.pantry.shelves[lvl].total < thr
            and st.pantry.shelves[lvl].carrots < quota
            and not st.pantry.shelves[lvl].is_full
        ]
        return sorted(wanting, key=lambda lvl: st.pantry.shelves[lvl].total)

    def _hoard_window_s(self, env: HarvestHavocEnv) -> float:
        """Seconds before the buzzer at which harvest-haul cakes become useful."""
        return self.park_lead_s + env.config.time.endgame_duration_s

    def _cake_demand(self, env: HarvestHavocEnv) -> int:
        """How many more cakes the plan needs beyond those already coming.

        The harvest-haul component is **time-gated**. A cake collected for the
        haul is dead weight until the endgame -- it cannot be placed once the
        shelves are full, so acquiring one early strands an inventory slot for
        the rest of the match. Only start wanting them near the hoard window.
        """
        st = env.state
        need = len(self._shelves_missing_cake(env))
        if self.chase_dinner_rp:
            t_left = st.time_remaining(env.config.time)
            if t_left <= self._hoard_window_s(env):
                need += env.config.scoring.haul_max_pieces
        supply = (st.robot.cakes + st.cakes_available + st.oven.pending_count)
        return max(0, need - supply)

    def _stocking_shortfall(self, env: HarvestHavocEnv) -> int:
        """Carrots still wanted on shelves before the oven gets any."""
        return len(self._shelves_wanting_carrots(env))

    # -- policy --------------------------------------------------------------

    def act(self, env: HarvestHavocEnv) -> Action:
        st = env.state
        cfg = env.config
        self._update_phase(env)
        legal = set(env.legal_actions())
        t_left = st.time_remaining(cfg.time)

        # 1. Free autonomous points.
        if Action.EXIT_KITCHEN in legal:
            return Action.EXIT_KITCHEN

        # 2. Endgame deadline: park and hold.
        if t_left <= self.park_lead_s:
            if st.robot.parked:
                return Action.WAIT
            if Action.PARK in legal:
                return Action.PARK
            return Action.WAIT

        # 3. Baked Up: a held cake belongs on a shelf that has none. Highest
        #    shelf first, since a cake there is worth 12 rather than 8.
        missing = self._shelves_missing_cake(env)
        if st.robot.cakes > 0 and missing:
            act = _SCORE_CAKE[missing[0]]
            if act in legal:
                return act

        # 4. Hoard the last cakes for the harvest haul once endgame is near.
        #    Three cakes aboard a parked robot is 9 haul points, which is the
        #    only route to the Dinner RP under the default reading.
        hoarding = (self.chase_dinner_rp and not missing
                    and t_left <= self._hoard_window_s(env))
        if hoarding and st.robot.cakes >= cfg.scoring.haul_max_pieces:
            return Action.WAIT

        # 5. Feed the oven once the shelves have taken all the carrots their
        #    quota allows. Carrots onto shelves are cheap points; carrots into
        #    the oven are an investment in cakes. Doing the investment first
        #    starves the cheap ranking point.
        if (self._stocking_shortfall(env) == 0
                and self._cake_demand(env) > 0
                and st.robot.carrots > 0
                and Action.OVEN_DEPOSIT_CARROT in legal):
            return Action.OVEN_DEPOSIT_CARROT

        # 6. Restock the hopper (batched). Cakes first when any are waiting.
        if self._should_intake(env):
            if st.cakes_available > 0 and Action.INTAKE_CAKE in legal:
                return Action.INTAKE_CAKE
            if Action.INTAKE_CARROT in legal:
                return Action.INTAKE_CARROT

        # 7. Carrots onto shelves, respecting the reserved cake slots.
        for lvl in self._shelves_wanting_carrots(env):
            if _SCORE_CARROT[lvl] in legal:
                return _SCORE_CARROT[lvl]

        # 8. Nothing productive left for a carrot in the pantry -- the shelves
        #    are either full or holding their cake slots open. Bank 2 points in
        #    the oven, which also produces another cake.
        if st.robot.carrots > 0 and Action.OVEN_DEPOSIT_CARROT in legal:
            return Action.OVEN_DEPOSIT_CARROT
        for lvl in sorted(st.pantry.levels, reverse=True):
            if _SCORE_CARROT[lvl] in legal:
                return _SCORE_CARROT[lvl]

        # 9. Holding cakes with nowhere to put them and no haul to save them
        #    for. Two points in the oven beats idling for the next minute --
        #    idling is the one thing that is unambiguously wrong in a game
        #    whose only real currency is time.
        if (st.robot.cakes > 0 and not hoarding
                and Action.OVEN_DEPOSIT_CAKE in legal):
            return Action.OVEN_DEPOSIT_CAKE
        return Action.WAIT


class CakeEconomyPolicy(_HopperCycler):
    """Convert everything into carrot cakes and place them as high as possible.

    Worth testing because the shelf cap changes the arithmetic completely.
    Three carrots into the oven score 6 points *and* yield a cake worth 12 on
    level 3 -- so 18 points for three carrots, using **one** pantry slot.
    Putting those same three carrots straight onto level 3 scores 15 and burns
    **three** slots. With only nine slots in the pantry, cakes dominate on
    both points-per-carrot and points-per-slot.

    The catch is time: each cake costs an oven trip, a human-player wait, and
    a depot pickup. Whether that pipeline out-earns simply filling the shelves
    with carrots is a genuine question, and this policy is the honest test of
    the aggressive end of it.

    Parameters
    ----------
    park_lead_s:
        Seconds before the buzzer at which to break off and park.
    fill_leftovers:
        Once every shelf is full, put spare carrots in the oven for 2 points
        each rather than idling.
    """

    def __init__(
        self, park_lead_s: float = 10.0, fill_leftovers: bool = True
    ) -> None:
        self.park_lead_s = park_lead_s
        self.fill_leftovers = fill_leftovers
        self.name = "cake_economy"

    def act(self, env: HarvestHavocEnv) -> Action:
        st = env.state
        cfg = env.config
        self._update_phase(env)
        legal = set(env.legal_actions())
        t_left = st.time_remaining(cfg.time)

        if Action.EXIT_KITCHEN in legal:
            return Action.EXIT_KITCHEN

        if t_left <= self.park_lead_s:
            if st.robot.parked:
                return Action.WAIT
            if Action.PARK in legal:
                return Action.PARK
            return Action.WAIT

        # A cake in hand goes on the highest shelf with room -- 12 points.
        if st.robot.cakes > 0:
            for lvl in sorted(st.pantry.levels, reverse=True):
                if _SCORE_CAKE[lvl] in legal:
                    return _SCORE_CAKE[lvl]

        # Collect finished cakes the moment they appear.
        if (st.cakes_available > 0
                and st.robot.free_slots(cfg.max_inventory) > 0
                and Action.INTAKE_CAKE in legal):
            return Action.INTAKE_CAKE

        # Carrots exist only to become cakes.
        if st.robot.carrots > 0 and not st.pantry.is_full():
            if Action.OVEN_DEPOSIT_CARROT in legal:
                return Action.OVEN_DEPOSIT_CARROT

        if self._should_intake(env) and Action.INTAKE_CARROT in legal:
            return Action.INTAKE_CARROT

        if self.fill_leftovers:
            if Action.OVEN_DEPOSIT_CARROT in legal:
                return Action.OVEN_DEPOSIT_CARROT
            for lvl in sorted(st.pantry.levels, reverse=True):
                if _SCORE_CARROT[lvl] in legal:
                    return _SCORE_CARROT[lvl]
        return Action.WAIT


# =============================================================================
# Rollout driver
# =============================================================================

def rollout(
    env: HarvestHavocEnv,
    policy: ScriptedPolicy,
    *,
    seed: Optional[int] = None,
    render_every: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one full match with `policy` and return a result summary.

    Parameters
    ----------
    env:
        Environment to drive. Reset internally.
    policy:
        Any :class:`ScriptedPolicy`.
    seed:
        Passed through to ``env.reset``.
    render_every:
        Print a frame every N steps. ``None`` renders nothing.

    Returns
    -------
    dict
        ``env.match_result()`` plus ``total_reward``, ``reward_vector_sum``,
        ``steps``, ``policy``, and the full ``time_allocation`` summary.
    """
    env.reset(seed=seed)
    policy.reset(env)

    total_reward = 0.0
    vector_sum = np.zeros(5, dtype=np.float64)
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy.act(env)
        _obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        vector_sum += np.asarray(info["reward_vector"], dtype=np.float64)
        steps += 1
        if render_every and steps % render_every == 0:
            print(env.render() or "")

    result = env.match_result()
    result.update(
        policy=policy.name,
        total_reward=total_reward,
        reward_vector_sum=vector_sum,
        steps=steps,
        truncated=truncated,
        time_allocation=env.time_allocation_summary(),
    )
    return result


def compare_policies(
    env: HarvestHavocEnv,
    policies: Sequence[ScriptedPolicy],
    *,
    episodes: int = 1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Run each policy for `episodes` matches and return per-policy averages.

    With the default deterministic stochastic config, one episode is enough;
    raise `episodes` once you enable failure rates or timing noise.
    """
    rows: List[Dict[str, Any]] = []
    for policy in policies:
        scores: List[float] = []
        rps: List[float] = []
        rewards: List[float] = []
        last: Dict[str, Any] = {}
        for ep in range(episodes):
            last = rollout(env, policy, seed=seed + ep)
            scores.append(last["raw_score"])
            rps.append(last["ranking_point_total"])
            rewards.append(last["total_reward"])
        rows.append(
            {
                "policy": policy.name,
                "mean_raw_score": float(np.mean(scores)),
                "mean_ranking_points": float(np.mean(rps)),
                "mean_total_reward": float(np.mean(rewards)),
                "episodes": episodes,
                "last_ranking_points": last.get("ranking_points", {}),
                "last_shelves": last.get("shelves", {}),
                "last_time_allocation": last.get("time_allocation", {}),
            }
        )
    return rows


__all__ = [
    "ScriptedPolicy",
    "RandomPolicy",
    "CycleAndParkPolicy",
    "RankingPointRushPolicy",
    "CakeEconomyPolicy",
    "rollout",
    "compare_policies",
]

"""
Multi-objective reward.

Raw score alone is the wrong training signal for this game: ranking points
decide seeding, and two of the three RPs are *threshold* conditions that a
score-greedy policy will step right over (it is locally optimal to dump every
cake on level 3, which forfeits Baked Up outright). So the reward has three
parts:

1. **Raw score delta** -- the dense, honest base signal.
2. **Ranking-point shaping** -- a potential-difference term over continuous
   progress toward each RP, which turns three step functions into three ramps.
3. **Ranking-point achievement** -- a lump sum the moment an RP flips true.

The scalar returned by ``step()`` is the weighted sum, so any standard RL
algorithm works unmodified. But the *unweighted component vector* also ships in
``info["reward_vector"]`` and the labelled, weighted split in
``info["reward_breakdown"]``, so you can re-weight objectives offline, fit a
Pareto front, or drop in a genuine multi-objective learner without re-running
a single simulation.

On the shaping term
-------------------
The shaping is a potential difference ``Phi(s') - Phi(s)``. Under
Ng/Harada/Russell (1999) this is policy-invariant *provided* ``Phi`` is zero at
terminal states; otherwise the episode total picks up a
``Phi(s_T) - Phi(s_0)`` bias. Both behaviours are available via
``RewardConfig.strict_potential_based_shaping``, defaulting to the biased-but-
better-behaved version because the terminal spike from zeroing ``Phi`` tends to
swamp the endgame decisions this project exists to study.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, Tuple

import numpy as np

from .config import EnvConfig
from .scoring import (
    RPProgress,
    RankingPoints,
    evaluate_ranking_points,
    rp_progress,
)
from .state import MatchState

#: Names of the entries in ``info["reward_vector"]``, in order. These are the
#: *unweighted* objectives; the scalar reward is their weighted sum plus the
#: optional time penalty.
REWARD_OBJECTIVES: Tuple[str, ...] = (
    "raw_score_delta",
    "stocked_up_progress_delta",
    "baked_up_progress_delta",
    "dinner_progress_delta",
    "ranking_points_gained",
)


@dataclass
class RewardBreakdown:
    """Weighted reward contributions for one step."""

    raw_score: float = 0.0
    shaping: float = 0.0
    achievement: float = 0.0
    time_penalty: float = 0.0
    illegal_penalty: float = 0.0

    @property
    def total(self) -> float:
        """The scalar reward returned by ``step()``."""
        return (self.raw_score + self.shaping + self.achievement
                + self.time_penalty + self.illegal_penalty)

    def as_dict(self) -> Dict[str, float]:
        """Dict view, for logging into ``info``."""
        return {
            "raw_score": self.raw_score,
            "shaping": self.shaping,
            "achievement": self.achievement,
            "time_penalty": self.time_penalty,
            "illegal_penalty": self.illegal_penalty,
            "total": self.total,
        }


@dataclass
class _Snapshot:
    """The pieces of state the reward needs to remember between steps."""

    raw_score: int = 0
    potential: float = 0.0
    progress: RPProgress = _field(default_factory=RPProgress)
    ranking_points: RankingPoints = _field(default_factory=RankingPoints)


class RewardCalculator:
    """Stateful reward function. One instance per environment.

    Usage::

        calc = RewardCalculator(config)
        calc.reset(state, in_table_zone=False)
        ...
        breakdown, vector = calc.compute(state, dt=1.4, terminal=False,
                                         in_table_zone=False, illegal=False)
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self._prev = _Snapshot()

    # ------------------------------------------------------------- potential

    def _dinner_gate(self, t: float) -> float:
        """Ramp weight for the Dinner RP potential, in ``[0, 1]``.

        Zero until ``dinner_shaping_ramp_s`` before endgame, then linear to 1.0
        at the endgame whistle. Prevents the shaping from paying the agent to
        squat in the table zone holding cakes for the entire match.
        """
        ramp = self.config.reward.dinner_shaping_ramp_s
        if ramp <= 0.0:
            return 1.0
        start = self.config.time.endgame_start_s - ramp
        return float(np.clip((t - start) / ramp, 0.0, 1.0))

    def potential(self, state: MatchState, *, in_table_zone: bool) -> float:
        """Shaping potential ``Phi(s)``, in reward units.

        A pure function of state (the clock is part of the state, which is what
        makes the Dinner RP time-gate legitimate rather than a hack).
        """
        rcfg = self.config.reward
        prog = rp_progress(
            state, self.config.scoring, self.config.time,
            in_table_zone=in_table_zone,
        )
        return rcfg.w_rp_shaping * (
            rcfg.shaping_scale_stocked_up * prog.stocked_up
            + rcfg.shaping_scale_baked_up * prog.baked_up
            + rcfg.shaping_scale_dinner_rp * prog.dinner_rp
            * self._dinner_gate(state.t)
        )

    # ----------------------------------------------------------------- driver

    def reset(self, state: MatchState, *, in_table_zone: bool) -> None:
        """Prime the calculator at the start of an episode."""
        self._prev = _Snapshot(
            raw_score=state.raw_score,
            potential=self.potential(state, in_table_zone=in_table_zone),
            progress=rp_progress(
                state, self.config.scoring, self.config.time,
                in_table_zone=in_table_zone,
            ),
            ranking_points=evaluate_ranking_points(
                state, self.config.scoring, self.config.time
            ),
        )

    def compute(
        self,
        state: MatchState,
        *,
        dt: float,
        terminal: bool,
        in_table_zone: bool,
        illegal: bool = False,
    ) -> Tuple[RewardBreakdown, np.ndarray]:
        """Reward for the transition that just landed in `state`.

        Parameters
        ----------
        state:
            Post-transition match state.
        dt:
            Seconds consumed by the transition. Drives the optional time
            penalty and is reported for time-allocation analysis.
        terminal:
            True on the final step of the episode, after endgame finalisation.
        in_table_zone:
            Whether the robot occupies its own table zone right now.
        illegal:
            True if the requested action was rejected as illegal.

        Returns
        -------
        (RewardBreakdown, np.ndarray)
            The weighted split, and the unweighted objective vector described
            by :data:`REWARD_OBJECTIVES`.
        """
        rcfg = self.config.reward
        prev = self._prev

        progress = rp_progress(
            state, self.config.scoring, self.config.time,
            in_table_zone=in_table_zone,
        )
        rps = evaluate_ranking_points(
            state, self.config.scoring, self.config.time
        )

        score_delta = float(state.raw_score - prev.raw_score)

        phi_next = 0.0 if (terminal and rcfg.strict_potential_based_shaping) \
            else self.potential(state, in_table_zone=in_table_zone)
        shaping = phi_next - prev.potential

        rps_gained = rps.total - prev.ranking_points.total
        achievement = 0.0
        if rps.stocked_up and not prev.ranking_points.stocked_up:
            achievement += rcfg.bonus_stocked_up
        if rps.baked_up and not prev.ranking_points.baked_up:
            achievement += rcfg.bonus_baked_up
        if rps.dinner_rp and not prev.ranking_points.dinner_rp:
            achievement += rcfg.bonus_dinner_rp

        breakdown = RewardBreakdown(
            raw_score=rcfg.w_raw_score * score_delta,
            shaping=shaping,
            achievement=achievement,
            time_penalty=-rcfg.w_time_penalty * dt,
            illegal_penalty=(-rcfg.illegal_action_penalty if illegal else 0.0),
        )

        vector = np.array(
            [
                score_delta,
                progress.stocked_up - prev.progress.stocked_up,
                progress.baked_up - prev.progress.baked_up,
                progress.dinner_rp - prev.progress.dinner_rp,
                float(rps_gained),
            ],
            dtype=np.float64,
        )

        self._prev = _Snapshot(
            raw_score=state.raw_score,
            potential=phi_next,
            progress=progress,
            ranking_points=rps,
        )
        return breakdown, vector


__all__ = ["RewardCalculator", "RewardBreakdown", "REWARD_OBJECTIVES"]

"""
The probabilistic duration model -- how long every action actually takes.

This module is the single place where time is generated. Nothing else in the
package samples a duration, so if you want to change how long things take, or
how variable they are, you change this file or :class:`TimeConfig` and nothing
else.

Why durations are sampled, not fixed
------------------------------------
A real robot action does not take a constant time. It has a **floor** set by
physics (you cannot drive 14 ft in 0.4 s), most attempts land near that floor,
and the tail is long -- a bobbled game piece, a bad approach angle, a re-grip.
Crucially the tail is *one-sided*: things go wrong and cost extra time far more
often than they go unexpectedly right and save it.

So every nominal duration is multiplied by a draw from a **right-skewed,
mean-one** distribution:

* ``LOGNORMAL`` (default) -- ``exp(N(mu, sigma))`` with ``mu = -sigma^2/2`` so
  the mean is exactly 1. The classic model for task completion times; heavy
  tail, hard floor near zero.
* ``GAMMA`` -- shape ``1/cv^2``, scale ``cv^2``. Also mean 1 and right-skewed,
  but a lighter tail. Use when lognormal's outliers feel too dramatic.
* ``DETERMINISTIC`` -- the multiplier is always exactly 1.0.

Because the mean is pinned at 1.0, enabling noise adds *variance without bias*:
a noisy match is directly comparable to a deterministic one, and a strategy
that looks good on average does not quietly get slower when you turn noise on.

Every draw is clamped to :attr:`StochasticConfig.time_noise_clip` so a tail
event cannot consume the entire match.

Per-component variance
----------------------
Noise is applied per *component*, not per action, because the components have
genuinely different variability:

===============  ====  =============================================
component        cv    why
===============  ====  =============================================
``travel``       0.08  driving a known path is repeatable
``align``        0.25  the least repeatable part of any cycle
``manipulate``   0.15  intake/placement, moderately repeatable
``wait``         0.00  a decision, not a physical action
===============  ====  =============================================

Scale them all at once with ``StochasticConfig.time_noise_scale``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import StochasticConfig, TimeConfig


class DurationModel(str, Enum):
    """Distribution family for the duration multiplier."""

    DETERMINISTIC = "deterministic"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"


#: The duration components every action is decomposed into.
COMPONENTS: Tuple[str, ...] = ("travel", "align", "manipulate", "wait")


@dataclass
class DurationBreakdown:
    """A nominal (noise-free) action duration, split by component."""

    travel: float = 0.0
    align: float = 0.0
    manipulate: float = 0.0
    wait: float = 0.0

    @property
    def total(self) -> float:
        """Total nominal seconds."""
        return self.travel + self.align + self.manipulate + self.wait

    def as_dict(self) -> Dict[str, float]:
        """Dict view keyed by component name."""
        return {"travel": self.travel, "align": self.align,
                "manipulate": self.manipulate, "wait": self.wait}


# =============================================================================
# Deterministic kinematics
# =============================================================================

def traverse_time(distance_ft: float, cfg: TimeConfig) -> float:
    """Nominal time to drive `distance_ft`, starting and ending at rest.

    Trapezoidal velocity profile: accelerate at ``a`` up to ``v``, cruise,
    decelerate symmetrically. For a path of length ``d``::

        d_ramp = v^2 / a
        t = 2 * sqrt(d / a)     if d <  d_ramp   (triangular, never cruises)
        t = d / v + v / a       if d >= d_ramp   (trapezoidal)

    Returns
    -------
    float
        Seconds. Zero for non-positive distances.
    """
    if distance_ft <= 0.0:
        return 0.0
    v = cfg.max_velocity_ft_s
    a = cfg.max_accel_ft_s2
    ramp_distance = (v * v) / a
    if distance_ft >= ramp_distance:
        return distance_ft / v + v / a
    return 2.0 * math.sqrt(distance_ft / a)


def cruise_time(distance_ft: float, cfg: TimeConfig) -> float:
    """Nominal time to cover `distance_ft` while already at cruise speed."""
    if distance_ft <= 0.0:
        return 0.0
    return distance_ft / cfg.max_velocity_ft_s


# =============================================================================
# The sampler
# =============================================================================

class TimingModel:
    """Samples action durations from the configured probabilistic model.

    Parameters
    ----------
    time_config:
        Nominal durations and robot kinematics.
    stochastic_config:
        Distribution family, per-component variability, and clipping.
    rng:
        Source of randomness. The environment passes its own generator so a
        single ``reset(seed=...)`` makes an entire match reproducible.

    Examples
    --------
    >>> from harvest_havoc.config import TimeConfig, StochasticConfig
    >>> model = TimingModel(TimeConfig(), StochasticConfig())
    >>> samples = [model.sample(2.0, "manipulate") for _ in range(1000)]
    >>> 1.9 < sum(samples) / len(samples) < 2.1   # mean is preserved
    True
    """

    def __init__(
        self,
        time_config: TimeConfig,
        stochastic_config: StochasticConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.time = time_config
        self.stochastic = stochastic_config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.model = DurationModel(stochastic_config.duration_model)

    def set_rng(self, rng: np.random.Generator) -> None:
        """Point the sampler at a new generator (called on ``env.reset``)."""
        self.rng = rng

    # ------------------------------------------------------------ multiplier

    def multiplier(self, cv: float) -> float:
        """Draw one mean-one duration multiplier with the given spread.

        Returns exactly 1.0 when ``cv`` is zero or the model is deterministic.
        """
        if cv <= 0.0 or self.model is DurationModel.DETERMINISTIC:
            return 1.0

        if self.model is DurationModel.LOGNORMAL:
            # sigma chosen so that std/mean == cv, mu so that the mean is 1.
            sigma = math.sqrt(math.log1p(cv * cv))
            mu = -0.5 * sigma * sigma
            factor = float(self.rng.lognormal(mean=mu, sigma=sigma))
        else:  # GAMMA
            shape = 1.0 / (cv * cv)
            factor = float(self.rng.gamma(shape=shape, scale=1.0 / shape))

        lo, hi = self.stochastic.time_noise_clip
        return min(max(factor, lo), hi)

    def sample(self, nominal_s: float, component: str) -> float:
        """Sample an actual duration for a nominal time and component.

        Parameters
        ----------
        nominal_s:
            The noise-free duration from :class:`TimeConfig`.
        component:
            One of :data:`COMPONENTS`.

        Returns
        -------
        float
            Seconds actually consumed. Never negative.
        """
        if nominal_s <= 0.0:
            return 0.0
        return nominal_s * self.multiplier(self.stochastic.cv_for(component))

    # ------------------------------------------------------ derived samplers

    def travel(self, distance_ft: float, *, from_rest: bool = True) -> float:
        """Sample the time to drive `distance_ft`.

        Parameters
        ----------
        distance_ft:
            Path length.
        from_rest:
            True for a rest-to-rest move (the trapezoidal profile). False when
            the robot is already rolling, which is charged at cruise speed --
            this is what stops a chain of single-cell MOVE actions from paying
            a full accel/decel cycle per cell.
        """
        nominal = (traverse_time(distance_ft, self.time) if from_rest
                   else cruise_time(distance_ft, self.time))
        return self.sample(nominal, "travel")

    def hp_exchange_delay(self) -> float:
        """Sample the human-player carrot-to-cake turnaround, in seconds."""
        mean = self.stochastic.hp_exchange_delay_s
        cv = self.stochastic.hp_exchange_delay_cv
        if cv <= 0.0 or self.model is DurationModel.DETERMINISTIC:
            return float(mean)
        return float(mean) * self.multiplier(cv)

    # ------------------------------------------------------------ inspection

    def describe(self, nominal_s: float, component: str,
                 samples: int = 4000) -> Dict[str, float]:
        """Empirical statistics for a duration, for reporting and tests.

        Returns a dict with ``nominal``, ``mean``, ``std``, ``cv``, ``p05``,
        ``p50``, ``p95``. Uses an isolated generator so it never disturbs the
        environment's own random stream.
        """
        if nominal_s <= 0.0:
            return {"nominal": 0.0, "mean": 0.0, "std": 0.0, "cv": 0.0,
                    "p05": 0.0, "p50": 0.0, "p95": 0.0}
        saved, self.rng = self.rng, np.random.default_rng(0)
        try:
            draws = np.array(
                [self.sample(nominal_s, component) for _ in range(samples)]
            )
        finally:
            self.rng = saved
        mean = float(draws.mean())
        std = float(draws.std())
        return {
            "nominal": float(nominal_s),
            "mean": mean,
            "std": std,
            "cv": std / mean if mean > 0 else 0.0,
            "p05": float(np.percentile(draws, 5)),
            "p50": float(np.percentile(draws, 50)),
            "p95": float(np.percentile(draws, 95)),
        }


__all__ = [
    "DurationModel",
    "DurationBreakdown",
    "TimingModel",
    "COMPONENTS",
    "traverse_time",
    "cruise_time",
]

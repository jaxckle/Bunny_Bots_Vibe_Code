"""
Harvest Havoc -- a single-robot simulation of the 2026 Bunnybots offseason game.

Purpose
-------
This package exists to answer one question: **given a fixed 150-second match,
which sequence and timing of actions maximises the combination of raw score and
ranking points?** Everything is built around that. In particular:

* Time is the first-class quantity. One ``step`` advances the clock by exactly
  as long as the chosen action takes -- there is no fixed control period -- and
  every action's duration is decomposed into travel / align / manipulate / idle
  and logged. :meth:`~harvest_havoc.env.HarvestHavocEnv.time_allocation_summary`
  accounts for all 150 seconds and reports points-per-second by category.
* Travel time comes from a real trapezoidal velocity profile over shortest
  paths on a discrete field grid, so it responds correctly to layout and
  drivetrain changes instead of being a hand-tuned constant.
* Durations are **sampled, not fixed**. Every nominal time is multiplied by a
  right-skewed, mean-one draw (see :mod:`harvest_havoc.timing`), because real
  actions have a physical floor and a long tail. Every action in the space has
  a time cost; inspect them all with
  :meth:`~harvest_havoc.env.HarvestHavocEnv.action_time_table`.
* The reward is genuinely multi-objective: raw score plus dense progress toward
  each of the three ranking points plus a lump sum when one is earned, with the
  unweighted component vector exposed for offline re-weighting. Retune it with
  four numbers via :meth:`~harvest_havoc.config.RewardConfig.from_priorities`.
* The pantry holds at most three pieces per shelf, so nine slots decide the
  match. Which piece takes each slot is the central strategic question.

Quick start
-----------
    >>> from harvest_havoc import HarvestHavocEnv, CycleAndParkPolicy, rollout
    >>> env = HarvestHavocEnv()
    >>> result = rollout(env, CycleAndParkPolicy(level=3))
    >>> result["raw_score"] > 0
    True

Module map
----------
==================  =========================================================
:mod:`config`       every tunable number, including ``GRID_CELL_SIZE_FT``
:mod:`zones`        the :class:`~harvest_havoc.zones.Zone` enum and layout
:mod:`field`        grid and shortest-path navigation
:mod:`timing`       the probabilistic duration model -- all time comes from here
:mod:`state`        mutable match state and the time-ledger record
:mod:`actions`      the hybrid macro + micro action space
:mod:`scoring`      point values, endgame finalisation, ranking points
:mod:`observation`  the named, flat observation vector
:mod:`reward`       multi-objective reward and shaping
:mod:`env`          :class:`~harvest_havoc.env.HarvestHavocEnv` itself
:mod:`render`       ASCII field map, scoreboard, and ledger tables
:mod:`baselines`    scripted reference policies and a rollout driver
==================  =========================================================

Not yet modelled (deliberately, for this first version): opponents, alliance
partners, defence, penalties, and the human player as an explicit agent. See
the README for the extension points each of those hooks into.
"""

from .actions import (
    ACTION_SPECS,
    Action,
    ActionKind,
    ActionSpec,
    MACRO_ACTIONS,
    MICRO_ACTIONS,
    N_ACTIONS,
)
from .baselines import (
    CakeEconomyPolicy,
    CycleAndParkPolicy,
    RandomPolicy,
    RankingPointRushPolicy,
    ScriptedPolicy,
    compare_policies,
    rollout,
)
from .config import (
    GRID_CELL_SIZE_FT,
    MATCH_DURATION_S,
    EnvConfig,
    FieldConfig,
    RewardConfig,
    ScoringConfig,
    StochasticConfig,
    TimeConfig,
)
from .env import HarvestHavocEnv
from .field import Field
from .timing import (
    COMPONENTS,
    DurationBreakdown,
    DurationModel,
    TimingModel,
    cruise_time,
    traverse_time,
)
from .observation import ObservationEncoder
from .render import (
    render_ascii,
    render_field,
    render_scoreboard,
    render_time_allocation,
    render_time_ledger,
)
from .reward import REWARD_OBJECTIVES, RewardBreakdown, RewardCalculator
from .scoring import (
    DinnerRPMode,
    RankingPoints,
    RPProgress,
    evaluate_ranking_points,
    max_pantry_points,
    rp_progress,
)
from .spaces import GYMNASIUM_AVAILABLE
from .state import (
    ActionRecord,
    MatchState,
    Phase,
    Piece,
    RobotState,
    ScoreCategory,
)
from .zones import DEFAULT_LAYOUT, KITCHEN_COMPLEX, Zone, ZoneRect, mirror_layout

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # environment
    "HarvestHavocEnv",
    "GYMNASIUM_AVAILABLE",
    # configuration
    "EnvConfig",
    "FieldConfig",
    "TimeConfig",
    "ScoringConfig",
    "StochasticConfig",
    "RewardConfig",
    "GRID_CELL_SIZE_FT",
    "MATCH_DURATION_S",
    # field and zones
    "Field",
    "Zone",
    "ZoneRect",
    "DEFAULT_LAYOUT",
    "KITCHEN_COMPLEX",
    "mirror_layout",
    "traverse_time",
    "cruise_time",
    # timing model
    "TimingModel",
    "DurationModel",
    "DurationBreakdown",
    "COMPONENTS",
    # actions
    "Action",
    "ActionKind",
    "ActionSpec",
    "ACTION_SPECS",
    "N_ACTIONS",
    "MACRO_ACTIONS",
    "MICRO_ACTIONS",
    # state
    "MatchState",
    "RobotState",
    "Piece",
    "Phase",
    "ScoreCategory",
    "ActionRecord",
    # scoring
    "DinnerRPMode",
    "RankingPoints",
    "RPProgress",
    "evaluate_ranking_points",
    "rp_progress",
    "max_pantry_points",
    # observation and reward
    "ObservationEncoder",
    "RewardCalculator",
    "RewardBreakdown",
    "REWARD_OBJECTIVES",
    # rendering
    "render_ascii",
    "render_field",
    "render_scoreboard",
    "render_time_ledger",
    "render_time_allocation",
    # baselines
    "ScriptedPolicy",
    "RandomPolicy",
    "CycleAndParkPolicy",
    "RankingPointRushPolicy",
    "CakeEconomyPolicy",
    "rollout",
    "compare_policies",
]

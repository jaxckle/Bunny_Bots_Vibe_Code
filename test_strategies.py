"""
========================================================================
   HARVEST HAVOC -- STRATEGY TEST BENCH
========================================================================

Run it:

    python test_strategies.py                 # compare every strategy
    python test_strategies.py times           # what does each action cost?
    python test_strategies.py noise           # show the duration distribution
    python test_strategies.py detail          # deep dive on one strategy
    python test_strategies.py priorities      # how reward priorities change things
    python test_strategies.py all             # everything

EVERYTHING YOU NEED TO EDIT IS IN THE BLOCK MARKED "EDIT HERE" BELOW.
Change a number, save, re-run. No other file needs to be touched.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import numpy as np

from harvest_havoc import (
    Action,
    CakeEconomyPolicy,
    CycleAndParkPolicy,
    EnvConfig,
    HarvestHavocEnv,
    RandomPolicy,
    RankingPointRushPolicy,
    RewardConfig,
    ScoringConfig,
    StochasticConfig,
    TimeConfig,
    rollout,
)
from harvest_havoc.render import (
    render_action_times,
    render_duration_distribution,
    render_time_allocation,
    render_time_ledger,
)

# =========================================================================
# ============================  EDIT HERE  ================================
# =========================================================================

# ---- 1. HOW MANY MATCHES PER STRATEGY -----------------------------------
# Action durations are randomly sampled, so a single match is one roll of
# the dice. Use 20-50 to compare strategies fairly; use 1 for a quick look.
EPISODES = 25
SEED = 0


# ---- 2. WHICH STRATEGIES TO TEST ----------------------------------------
# Add, remove or re-parameterise freely. Every entry is just a policy object.
#
#   CycleAndParkPolicy(level, park_lead_s=8.0, hold_for_haul=True)
#       Fill up on carrots, dump them on `level`, repeat. park_lead_s is how
#       many seconds before the buzzer to break off and park (None = never).
#
#   RankingPointRushPolicy(park_lead_s=10.0, chase_dinner_rp=True,
#                          reserve_cake_slots=True)
#       Chase all three ranking points. reserve_cake_slots keeps one slot per
#       shelf free for a cake -- set False to see the trap of filling shelves
#       with carrots (Stocked Up arrives sooner, Baked Up becomes impossible).
#
#   CakeEconomyPolicy(park_lead_s=10.0)
#       Turn every carrot into a cake via the oven. Best points-per-slot,
#       worst points-per-second. The aggressive extreme.
#
STRATEGIES = [
    RandomPolicy(seed=0),
    CycleAndParkPolicy(level=1),
    CycleAndParkPolicy(level=2),
    CycleAndParkPolicy(level=3),
    CycleAndParkPolicy(level=3, park_lead_s=None),
    RankingPointRushPolicy(),
    RankingPointRushPolicy(chase_dinner_rp=False),
    RankingPointRushPolicy(reserve_cake_slots=False),
    CakeEconomyPolicy(),
]


# ---- 3. REWARD PRIORITIES -----------------------------------------------
# Four numbers. Only their RATIOS matter. Raise one to make the agent care
# more about it; set one to 0.0 to ignore that objective entirely.
#
# These affect the "reward" column only -- raw score and ranking points are
# facts about the match and do not change. This is the dial you would tune
# before training an RL agent.
PRIORITY_SCORE = 1.0      # value of one raw point
PRIORITY_STOCKED_UP = 1.0      # value of the Stocked Up RP
PRIORITY_BAKED_UP = 1.0      # value of the Baked Up RP
PRIORITY_DINNER_RP = 0.7      # value of the Dinner RP

# Or ignore the four numbers above and use a named preset instead.
# Options: None | "balanced" | "score" | "rp" | "seeding"
#          | "stocked" | "baked" | "dinner"
REWARD_PRESET = None


# ---- 4. GAME RULES ------------------------------------------------------
SHELF_CAPACITY = 3        # max pieces per shelf (carrots + cakes combined)
MATCH_SECONDS = 150.0     # 2:30
STOCKED_UP_NEEDS = 3        # pieces on every shelf
BAKED_UP_NEEDS = 1        # cakes on every shelf
DINNER_RP_NEEDS = 10       # endgame points
DINNER_RP_MODE = "park_and_haul"   # or "all_in_endgame"


# ---- 5. ROBOT SPEED AND ACTION TIMES ------------------------------------
# Seconds. These are the numbers that decide which strategy wins, so they
# are the ones worth measuring off your real robot.
MAX_VELOCITY_FT_S = 14.0    # drivetrain free speed
MAX_ACCEL_FT_S2 = 12.0    # drivetrain acceleration

ALIGN_TIME = 0.50    # squaring up on a target
INTAKE_CARROT_TIME = 1.00
INTAKE_CAKE_TIME = 1.30
SCORE_CARROT_TIMES = {1: 1.40, 2: 1.90, 3: 2.60}   # elevator travel
SCORE_CAKE_TIMES = {1: 1.70, 2: 2.20, 3: 3.00}
OVEN_DEPOSIT_TIME = 1.20
PARK_SETTLE_TIME = 0.60


# ---- 6. RANDOMNESS ------------------------------------------------------
# Action times are SAMPLED, not fixed. Each nominal time above is multiplied
# by a right-skewed, mean-one random factor: most attempts land near nominal,
# and the tail runs long because things go wrong more often than they go
# unexpectedly right.
DURATION_MODEL = "lognormal"   # "lognormal" | "gamma" | "deterministic"
NOISE_SCALE = 1.0           # 0.0 = fixed times, 1.0 = default, 2.0 = messy

TRAVEL_NOISE_CV = 0.08   # driving is repeatable
ALIGN_NOISE_CV = 0.25   # aligning is the least repeatable part of a cycle
MANIPULATE_NOISE_CV = 0.15

# How reliably the robot completes each attempt. Failures still cost time.
INTAKE_SUCCESS = 1.00
SCORE_SUCCESS = {1: 1.00, 2: 1.00, 3: 1.00}
OVEN_SUCCESS = 1.00
PARK_SUCCESS = 1.00

# Human player carrot-to-cake turnaround.
HP_DELAY_MEAN_S = 7.5
HP_DELAY_CV = 0.35

# ---- 7. WHICH STRATEGY TO DEEP-DIVE ON (for `detail`) -------------------
DETAIL_STRATEGY_INDEX = 3     # index into STRATEGIES above

# =========================================================================
# ==========================  END EDIT HERE  ==============================
# =========================================================================


def build_config() -> EnvConfig:
    """Assemble an EnvConfig from the EDIT HERE block."""
    reward = (RewardConfig.preset(REWARD_PRESET) if REWARD_PRESET
              else RewardConfig.from_priorities(
                  score=PRIORITY_SCORE,
                  stocked_up=PRIORITY_STOCKED_UP,
                  baked_up=PRIORITY_BAKED_UP,
                  dinner_rp=PRIORITY_DINNER_RP,
              ))
    return EnvConfig(
        time=TimeConfig(
            match_duration_s=MATCH_SECONDS,
            max_velocity_ft_s=MAX_VELOCITY_FT_S,
            max_accel_ft_s2=MAX_ACCEL_FT_S2,
            align_time_s=ALIGN_TIME,
            intake_carrot_s=INTAKE_CARROT_TIME,
            intake_cake_s=INTAKE_CAKE_TIME,
            score_carrot_s=dict(SCORE_CARROT_TIMES),
            score_cake_s=dict(SCORE_CAKE_TIMES),
            oven_deposit_s=OVEN_DEPOSIT_TIME,
            park_settle_s=PARK_SETTLE_TIME,
        ),
        scoring=ScoringConfig(
            shelf_capacity=SHELF_CAPACITY,
            stocked_up_threshold=STOCKED_UP_NEEDS,
            baked_up_threshold=BAKED_UP_NEEDS,
            dinner_rp_threshold=DINNER_RP_NEEDS,
            dinner_rp_mode=DINNER_RP_MODE,
        ),
        stochastic=StochasticConfig(
            duration_model=DURATION_MODEL,
            time_noise_scale=NOISE_SCALE,
            travel_noise_cv=TRAVEL_NOISE_CV,
            align_noise_cv=ALIGN_NOISE_CV,
            manipulate_noise_cv=MANIPULATE_NOISE_CV,
            intake_success_prob=INTAKE_SUCCESS,
            score_success_prob=dict(SCORE_SUCCESS),
            oven_success_prob=OVEN_SUCCESS,
            park_success_prob=PARK_SUCCESS,
            hp_exchange_delay_s=HP_DELAY_MEAN_S,
            hp_exchange_delay_cv=HP_DELAY_CV,
        ),
        reward=reward,
    )


def _rule(title: str) -> None:
    print(f"\n{'=' * 100}\n  {title}\n{'=' * 100}")


def run_one(policy, config: EnvConfig, episodes: int, seed: int) -> Dict[str, Any]:
    """Run `episodes` matches of one strategy and aggregate the results."""
    rows: List[Dict[str, Any]] = []
    for i in range(episodes):
        env = HarvestHavocEnv(config)
        rows.append(rollout(env, policy, seed=seed + i))

    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows], dtype=float)

    rp_rate = {
        name: float(np.mean([r["ranking_points"][name] for r in rows]))
        for name in ("stocked_up", "baked_up", "dinner_rp")
    }
    return {
        "policy": policy.name,
        "episodes": episodes,
        "score_mean": float(col("raw_score").mean()),
        "score_std": float(col("raw_score").std()),
        "score_min": float(col("raw_score").min()),
        "score_max": float(col("raw_score").max()),
        "rp_mean": float(col("ranking_point_total").mean()),
        "endgame_mean": float(col("endgame_points").mean()),
        "reward_mean": float(col("total_reward").mean()),
        "steps_mean": float(col("steps").mean()),
        "rp_rate": rp_rate,
        "last": rows[-1],
        "rows": rows,
    }


# =========================================================================
# Sections
# =========================================================================

def compare() -> List[Dict[str, Any]]:
    """The main table: every strategy, side by side."""
    config = build_config()
    _rule(f"STRATEGY COMPARISON   ({EPISODES} matches each, seed {SEED})")

    preset = REWARD_PRESET or (
        f"score={PRIORITY_SCORE} stocked={PRIORITY_STOCKED_UP} "
        f"baked={PRIORITY_BAKED_UP} dinner={PRIORITY_DINNER_RP}"
    )
    print(f"reward priorities : {preset}")
    print(f"shelf capacity    : {SHELF_CAPACITY} pieces/shelf "
          f"({SHELF_CAPACITY * 3} slots in the whole pantry)")
    print(f"duration model    : {DURATION_MODEL}, noise scale {NOISE_SCALE}\n")

    header = (
        f"{'strategy':<26}{'score':>13}{'range':>11}{'RP':>6}"
        f"{'stock':>7}{'bake':>7}{'dinner':>8}{'endgm':>7}"
        f"{'steps':>7}{'reward':>9}"
    )
    print(header)
    print("-" * len(header))

    results = [run_one(p, config, EPISODES, SEED) for p in STRATEGIES]
    for r in sorted(results, key=lambda x: -x["score_mean"]):
        rr = r["rp_rate"]
        print(
            f"{r['policy']:<26}"
            f"{r['score_mean']:>8.1f}±{r['score_std']:<4.1f}"
            f"{int(r['score_min']):>5}-{int(r['score_max']):<5}"
            f"{r['rp_mean']:>6.2f}"
            f"{rr['stocked_up'] * 100:>6.0f}%{rr['baked_up'] * 100:>6.0f}%"
            f"{rr['dinner_rp'] * 100:>7.0f}%"
            f"{r['endgame_mean']:>7.1f}{r['steps_mean']:>7.0f}"
            f"{r['reward_mean']:>9.1f}"
        )

    print(
        "\n  score   mean raw points ± std dev over the sampled matches\n"
        "  range   worst and best single match\n"
        "  stock / bake / dinner   how often each ranking point was earned\n"
        "  reward  the training signal, under your priority weights"
    )
    best_score = max(results, key=lambda r: r["score_mean"])
    best_rp = max(results, key=lambda r: r["rp_mean"])
    best_reward = max(results, key=lambda r: r["reward_mean"])
    print(
        f"\n  best raw score  : {best_score['policy']} "
        f"({best_score['score_mean']:.1f})"
        f"\n  best RP total   : {best_rp['policy']} "
        f"({best_rp['rp_mean']:.2f} RP)"
        f"\n  best reward     : {best_reward['policy']} "
        f"({best_reward['reward_mean']:.1f}) <- what an RL agent would chase"
    )
    return results


def show_action_times() -> None:
    """Every action's time cost, from several field positions."""
    from harvest_havoc.zones import Zone

    config = build_config()
    env = HarvestHavocEnv(config)
    env.reset(seed=SEED)
    # Give the robot a full hopper so scoring actions read as legal.
    env.state.robot.carrots = 2
    env.state.robot.cakes = 1
    env.state.cakes_available = 1

    for zone in (Zone.OWN_KITCHEN, Zone.OWN_FARM, Zone.OWN_PANTRY):
        cell = env.field.zone_centroid_cell(zone)
        _rule(f"ACTION TIMES FROM {zone.name}")
        print(render_action_times(env, from_cell=cell, samples=1500))


def show_noise() -> None:
    """Prove the durations are sampled from a curve, not fixed."""
    config = build_config()
    env = HarvestHavocEnv(config)
    env.reset(seed=SEED)

    _rule("DURATION DISTRIBUTION")
    for nominal, component in ((2.60, "manipulate"), (0.50, "align")):
        print(render_duration_distribution(env, nominal, component))
        print()

    _rule("SAME ACTION, TWENTY ATTEMPTS")
    print("SCORE_CARROT_L3 from inside the pantry, repeated:\n")
    times = []
    for _ in range(20):
        env.reset(seed=None)
        env.state.robot.cell = env.field.zone_centroid_cell(
            __import__("harvest_havoc").Zone.OWN_PANTRY
        )
        env.state.robot.carrots = 1
        env.step(int(Action.SCORE_CARROT_L3))
        times.append(env.time_ledger[-1].duration_s)
    print("  " + "  ".join(f"{t:.2f}" for t in times))
    print(
        f"\n  min {min(times):.2f}s   mean {np.mean(times):.2f}s   "
        f"max {max(times):.2f}s   spread {max(times) - min(times):.2f}s"
    )
    print("\n  Same action, same place, different durations every time.")
    print("  Set NOISE_SCALE = 0.0 in the EDIT HERE block to make these fixed.")


def show_detail() -> None:
    """Deep dive: one strategy's ledger and time allocation."""
    config = build_config()
    policy = STRATEGIES[DETAIL_STRATEGY_INDEX]
    env = HarvestHavocEnv(config)
    result = rollout(env, policy, seed=SEED)

    _rule(f"DETAIL: {policy.name}   (single match, seed {SEED})")
    print(f"raw score       : {result['raw_score']}")
    print(f"ranking points  : {result['ranking_point_total']}  "
          f"{result['ranking_points']}")
    print(f"endgame points  : {result['endgame_points']}")
    print(f"pantry          : {result['pantry_used']}/"
          f"{result['pantry_capacity']} slots used")
    for lvl, s in result["shelves"].items():
        print(f"   shelf {lvl}      : {s['carrots']} carrots, {s['cakes']} cakes, "
              f"{s['free']} free")
    print(f"score breakdown : {result['score_by_category']}")

    print()
    print(render_time_ledger(env, limit=30))
    print()
    print(render_time_allocation(env))


def show_priorities() -> None:
    """How the reward priorities change which strategy looks best."""
    _rule("REWARD PRIORITIES -- same matches, different objectives")
    print(
        "Raw score and ranking points are facts about the match. The reward is\n"
        "what an RL agent optimises, and it depends entirely on your weights.\n"
        "Each column below is a different setting of the four priority numbers.\n"
    )

    presets = ["score", "balanced", "seeding", "rp", "baked", "dinner"]
    base = build_config()

    table: Dict[str, Dict[str, float]] = {}
    for preset in presets:
        cfg = EnvConfig(
            time=base.time, scoring=base.scoring, stochastic=base.stochastic,
            reward=RewardConfig.preset(preset),
        )
        for policy in STRATEGIES:
            r = run_one(policy, cfg, max(5, EPISODES // 3), SEED)
            table.setdefault(policy.name, {})[preset] = r["reward_mean"]

    header = f"{'strategy':<26}" + "".join(f"{p:>12}" for p in presets)
    print(header)
    print("-" * len(header))
    for name, row in table.items():
        print(f"{name:<26}" + "".join(f"{row[p]:>12.1f}" for p in presets))

    print("\nWinner under each priority setting:")
    for preset in presets:
        best = max(table.items(), key=lambda kv: kv[1][preset])
        print(f"  {preset:<10} -> {best[0]}")
    print(
        "\nIf the winner changes between columns, your priority weights are\n"
        "doing real work -- and choosing them is a strategic decision, not a\n"
        "tuning detail."
    )


# =========================================================================

_SECTIONS = {
    "compare": compare,
    "times": show_action_times,
    "noise": show_noise,
    "detail": show_detail,
    "priorities": show_priorities,
}


def main(argv: List[str]) -> int:
    """Run the requested sections (default: just the comparison table)."""
    args = argv[1:] or ["compare"]
    if "all" in args:
        args = list(_SECTIONS)
    unknown = [a for a in args if a not in _SECTIONS]
    if unknown:
        print(f"unknown section(s): {', '.join(unknown)}")
        print(f"available: {', '.join(_SECTIONS)}, all")
        return 2
    for name in args:
        _SECTIONS[name]()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

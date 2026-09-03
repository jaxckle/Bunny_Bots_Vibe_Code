"""
Harvest Havoc environment tour.

Run with::

    python demo.py            # everything
    python demo.py field      # just the field map
    python demo.py compare    # just the baseline comparison
    python demo.py ledger     # just the time ledger + allocation report
    python demo.py obs        # just the observation schema
    python demo.py sweep      # config sweeps: shelf choice, drivetrain, park

Nothing here is required by the environment; it exists so the simulation can be
inspected and sanity-checked by eye before any learning code is written.
"""

from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np

from harvest_havoc import (
    GYMNASIUM_AVAILABLE,
    CakeEconomyPolicy,
    CycleAndParkPolicy,
    EnvConfig,
    HarvestHavocEnv,
    RandomPolicy,
    RankingPointRushPolicy,
    ScoringConfig,
    ScriptedPolicy,
    TimeConfig,
    compare_policies,
    rollout,
)
from harvest_havoc.render import (
    render_field,
    render_legend,
    render_scoreboard,
    render_time_allocation,
    render_time_ledger,
)


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# =============================================================================

def show_field() -> None:
    """Print the discretised field, the legend, and the key travel distances."""
    _rule("FIELD")
    env = HarvestHavocEnv()
    fld = env.field
    print(
        f"{fld.config.field_length_ft:.0f} x {fld.config.field_width_ft:.0f} ft "
        f"-> {fld.n_cols} x {fld.n_rows} cells at "
        f"{fld.cell_size:.2f} ft/cell  ({fld.n_cols * fld.n_rows} cells)"
    )
    print()
    print(render_field(env))
    print(render_legend(env))

    from harvest_havoc.field import traverse_time
    from harvest_havoc.zones import Zone

    print("\nTravel from the starting cell (rest to rest):")
    for zone in (Zone.OWN_FARM, Zone.OWN_PANTRY, Zone.OWN_OVEN, Zone.OWN_TABLE):
        d = fld.distance_to_zone(env.start_cell, zone)
        print(f"  -> {zone.name:<12} {d:6.2f} ft   "
              f"{traverse_time(d, env.config.time):5.2f} s")

    farm = fld.cells_of_zone(Zone.OWN_FARM)
    print("\nKey round trips (the numbers that decide the whole strategy):")
    for zone in (Zone.OWN_PANTRY, Zone.OWN_OVEN):
        d = min(fld.distance_to_zone(c, zone) for c in farm)
        one_way = traverse_time(d, env.config.time)
        print(f"  farm <-> {zone.name:<12} {d:6.2f} ft one way   "
              f"{2 * one_way:5.2f} s round trip")


def show_observation_schema() -> None:
    """Print every observation slot with its value at t=0."""
    _rule("OBSERVATION SCHEMA")
    env = HarvestHavocEnv()
    obs, _ = env.reset(seed=0)
    print(f"{env.encoder.size} float32 slots; "
          f"gymnasium installed: {GYMNASIUM_AVAILABLE}\n")
    for i, name in enumerate(env.encoder.names):
        print(f"  [{i:>2}] {name:<32} = {obs[i]:6.3f}   "
              f"bounds [{env.encoder.low[i]:.1f}, {env.encoder.high[i]:.1f}]")


def show_first_steps(n: int = 8) -> None:
    """Step through the opening of a match, printing the full frame."""
    _rule(f"FIRST {n} STEPS (rp_rush policy)")
    env = HarvestHavocEnv(render_mode="ansi")
    policy = RankingPointRushPolicy()
    env.reset(seed=0)
    policy.reset(env)
    for _ in range(n):
        action = policy.act(env)
        _obs, reward, terminated, _trunc, info = env.step(int(action))
        print(
            f"\n>> {action.name:<22} dt={info['time_cost']:5.2f}s "
            f"(travel {info['time_split']['travel']:.2f} / "
            f"align {info['time_split']['align']:.2f} / "
            f"manip {info['time_split']['manipulate']:.2f})  "
            f"reward={reward:+7.2f}"
        )
        print(render_scoreboard(env))
        if terminated:
            break


def compare_baselines() -> List[Dict]:
    """Run every scripted baseline and print a score-vs-RP table."""
    _rule("BASELINE COMPARISON")
    policies: List[ScriptedPolicy] = [
        RandomPolicy(0),
        CycleAndParkPolicy(1),
        CycleAndParkPolicy(2),
        CycleAndParkPolicy(3),
        CycleAndParkPolicy(3, park_lead_s=None),
        RankingPointRushPolicy(chase_dinner_rp=False),
        RankingPointRushPolicy(chase_dinner_rp=True),
        RankingPointRushPolicy(reserve_cake_slots=False),
        CakeEconomyPolicy(),
    ]
    print(
        f"{'policy':<20}{'score':>7}{'RP':>4}{'endgame':>9}{'steps':>7}"
        f"{'pts/s':>8}{'reward':>10}  ranking points"
    )
    print("-" * 96)
    results = []
    for policy in policies:
        r = rollout(HarvestHavocEnv(), policy, seed=1)
        results.append(r)
        pps = r["time_allocation"]["totals"]["overall_points_per_second"]
        rps = ", ".join(k for k, v in r["ranking_points"].items() if v) or "-"
        print(
            f"{r['policy']:<20}{r['raw_score']:>7}{r['ranking_point_total']:>4}"
            f"{r['endgame_points']:>9}{r['steps']:>7}{pps:>8.3f}"
            f"{r['total_reward']:>10.1f}  {rps}"
        )
    print(
        "\nThese are single sampled matches. Run `python test_strategies.py` for "
        "the\nsame comparison averaged over many matches, which is the fair way "
        "to rank\nstrategies now that durations are drawn from a distribution."
    )
    return results


def show_ledger() -> None:
    """Print the time ledger and allocation report for one match."""
    _rule("TIME LEDGER (first 25 actions, cycle_L3)")
    env = HarvestHavocEnv()
    rollout(env, CycleAndParkPolicy(3), seed=1)
    print(render_time_ledger(env, limit=25))
    print()
    print(render_time_allocation(env))

    _rule("TIME ALLOCATION (rp_rush)")
    env2 = HarvestHavocEnv()
    rollout(env2, RankingPointRushPolicy(), seed=1)
    print(render_time_allocation(env2))


def sweep() -> None:
    """Vary one configuration knob at a time and report the effect."""
    _rule("SWEEP 1 -- how much does the shelf capacity change the game?")
    print(
        "With only 3 pieces per shelf the pantry holds 9 total and saturates\n"
        "early, so the choice of shelf stops mattering and the choice of\n"
        "PIECE starts. Raising the cap restores the old 'which shelf?' game.\n"
    )
    print(f"{'cap':>5}{'slots':>7}  " + "".join(f"{f'L{l} score':>11}" for l in (1, 2, 3))
          + f"{'spread':>9}")
    print("-" * 56)
    for cap in (3, 6, 12):
        scores = []
        for level in (1, 2, 3):
            cfg = EnvConfig(scoring=ScoringConfig(shelf_capacity=cap))
            r = rollout(HarvestHavocEnv(cfg), CycleAndParkPolicy(level), seed=1)
            scores.append(r["raw_score"])
        print(f"{cap:>5}{cap * 3:>7}  " + "".join(f"{s:>11}" for s in scores)
              + f"{max(scores) - min(scores):>9}")
    print(
        "\n'spread' is the score gap between preferring L1 and preferring L3.\n"
        "At cap 3 it is noise; as the cap grows the higher shelf pulls ahead."
    )

    _rule("SWEEP 1b -- carrot or cake in a pantry slot?")
    cfg = ScoringConfig()
    top = max(cfg.shelf_levels)
    per_cake = cfg.oven_carrots_per_cake
    direct = per_cake * cfg.carrot_points[top]
    via_oven = per_cake * cfg.oven_points + cfg.cake_points[top]
    print(f"  {per_cake} carrots straight onto L{top} : {direct:>3} points, "
          f"{per_cake} slots  ({direct / per_cake:.1f}/slot)")
    print(f"  {per_cake} carrots -> oven -> 1 cake  : {via_oven:>3} points, "
          f"1 slot   ({via_oven:.1f}/slot)")
    print(
        f"\n  Cakes win by {via_oven - direct} points AND use "
        f"{per_cake - 1} fewer slots. They only lose on time:\n"
        f"  each one costs an oven trip, a human-player wait, and a depot pickup."
    )

    _rule("SWEEP 2 -- how much is drivetrain speed worth?")
    print(f"{'v_max':>8}{'a_max':>8}{'score':>8}{'travel_s':>10}{'%travel':>9}")
    print("-" * 43)
    for v, a in ((7.0, 8.0), (10.0, 10.0), (14.0, 12.0), (18.0, 16.0)):
        cfg = EnvConfig(time=TimeConfig(max_velocity_ft_s=v, max_accel_ft_s2=a))
        env = HarvestHavocEnv(cfg)
        r = rollout(env, CycleAndParkPolicy(3), seed=1)
        travel = r["time_allocation"]["by_component"]["travel"]
        print(f"{v:>8.1f}{a:>8.1f}{r['raw_score']:>8}{travel:>10.1f}"
              f"{100 * travel / 150:>8.1f}%")

    _rule("SWEEP 3 -- when should the robot break off to park?")
    print(f"{'lead_s':>8}{'score':>8}{'endgame':>9}{'dinner_rp':>11}")
    print("-" * 36)
    for lead in (None, 4.0, 6.0, 8.0, 12.0, 20.0, 30.0):
        env = HarvestHavocEnv()
        r = rollout(env, CycleAndParkPolicy(3, park_lead_s=lead), seed=1)
        label = "never" if lead is None else f"{lead:.0f}"
        print(f"{label:>8}{r['raw_score']:>8}{r['endgame_points']:>9}"
              f"{str(r['ranking_points']['dinner_rp']):>11}")
    print(
        "\nParking early costs cycles and buys at most 11 points, so under the\n"
        "default rules the break-off time is a genuinely tight optimisation --\n"
        "which is exactly the question the learner is meant to answer."
    )


# =============================================================================

_SECTIONS = {
    "field": show_field,
    "obs": show_observation_schema,
    "steps": show_first_steps,
    "compare": compare_baselines,
    "ledger": show_ledger,
    "sweep": sweep,
}


def main(argv: List[str]) -> int:
    """Run the requested demo sections (all of them by default)."""
    requested = argv[1:] or ["field", "obs", "steps", "compare", "ledger", "sweep"]
    unknown = [a for a in requested if a not in _SECTIONS]
    if unknown:
        print(f"unknown section(s): {', '.join(unknown)}")
        print(f"available: {', '.join(_SECTIONS)}")
        return 2
    for name in requested:
        _SECTIONS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

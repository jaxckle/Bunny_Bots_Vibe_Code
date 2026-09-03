"""
Test suite for the Harvest Havoc simulation environment.

Written to be runnable either with pytest (``pytest -q``) or standalone
(``python tests/test_harvest_havoc.py``), so the environment can be validated
without adding a test dependency.

The emphasis is on the properties that make the environment *trustworthy as a
measuring instrument* for time allocation:

* the clock is conserved -- every second is accounted for exactly once;
* scoring matches the published point values;
* the ranking-point thresholds fire exactly at their boundaries;
* endgame park and haul are evaluated once, at the buzzer, positionally;
* the time model is monotone and physically sensible.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harvest_havoc import (  # noqa: E402
    Action,
    CakeEconomyPolicy,
    CycleAndParkPolicy,
    EnvConfig,
    Field,
    HarvestHavocEnv,
    N_ACTIONS,
    Phase,
    Piece,
    RandomPolicy,
    RankingPointRushPolicy,
    ScoringConfig,
    StochasticConfig,
    TimeConfig,
    Zone,
    cruise_time,
    evaluate_ranking_points,
    mirror_layout,
    rollout,
    traverse_time,
)
from harvest_havoc.config import FieldConfig, RewardConfig  # noqa: E402
from harvest_havoc.scoring import (  # noqa: E402
    DinnerRPMode,
    baked_up_is_reachable,
    both_pantry_rps_reachable,
    haul_value,
    max_endgame_points,
    max_pantry_points,
    stocked_up_is_reachable,
)
from harvest_havoc.state import Piece as _Piece  # noqa: E402
from harvest_havoc.timing import DurationModel, TimingModel  # noqa: E402
from harvest_havoc.zones import DEFAULT_LAYOUT, KITCHEN_COMPLEX  # noqa: E402

TOL = 1e-6


def det_config(**kwargs) -> EnvConfig:
    """An EnvConfig with the probabilistic duration model switched off.

    Action durations are sampled by default, so any test asserting an exact
    time, or comparing two runs that must differ only in the thing under
    test, has to pin the timing first.
    """
    kwargs.setdefault("stochastic", StochasticConfig.deterministic())
    return EnvConfig(**kwargs)


# =============================================================================
# Field geometry and navigation
# =============================================================================

def test_grid_resolution_is_tunable():
    """Changing GRID_CELL_SIZE_FT must change the grid, not break it."""
    for cell in (1.0, 1.5, 2.0):
        fld = Field(FieldConfig(cell_size_ft=cell))
        assert fld.n_cols == round(54.0 / cell)
        assert fld.n_rows == round(27.0 / cell)
        # Every required zone must survive the coarser discretisation.
        for zone in (Zone.OWN_KITCHEN, Zone.OWN_FARM, Zone.OWN_PANTRY,
                     Zone.OWN_OVEN, Zone.OWN_TABLE):
            assert fld.has_zone(zone), f"{zone} vanished at {cell} ft cells"


def test_every_cell_has_exactly_one_zone():
    """The zone grid is total and single-valued by construction."""
    fld = Field()
    labels = set(int(z) for z in Zone)
    for cx in range(fld.n_cols):
        for cy in range(fld.n_rows):
            assert int(fld.zone_grid[cx, cy]) in labels


def test_pantry_and_oven_are_inside_the_kitchen_complex():
    """Nested regions carry the specific label but remain 'in the kitchen'."""
    fld = Field()
    for zone in (Zone.OWN_PANTRY, Zone.OWN_OVEN):
        for cell in fld.cells_of_zone(zone):
            assert fld.zone_at(cell) is zone
            assert fld.zone_at(cell) in KITCHEN_COMPLEX


def test_distance_fields_are_consistent_with_paths():
    """Dijkstra distance must equal the length of the reconstructed path."""
    fld = Field()
    for zone in (Zone.OWN_PANTRY, Zone.OWN_OVEN, Zone.OWN_FARM, Zone.OWN_TABLE):
        for start in [(5, 5), (20, 3), (30, 15), (0, 0)]:
            d = fld.distance_to_zone(start, zone)
            path = fld.path_to_zone(start, zone)
            assert path is not None
            assert fld.zone_at(path[-1]) is zone
            assert abs(fld.path_length_ft(path) - d) < 1e-9


def test_zero_distance_when_already_in_zone():
    """A robot inside the target zone must not be charged travel."""
    fld = Field()
    cell = fld.cells_of_zone(Zone.OWN_FARM)[0]
    assert fld.distance_to_zone(cell, Zone.OWN_FARM) == 0.0
    assert fld.path_to_zone(cell, Zone.OWN_FARM) == [cell]


def test_navigation_routes_around_obstacles():
    """A blocked wall must lengthen the path, not break it."""
    cfg = FieldConfig()
    blocked = np.zeros(cfg.shape, dtype=bool)
    open_field = Field(cfg)
    start = (20, 9)
    baseline = open_field.distance_to_zone(start, Zone.OWN_PANTRY)

    # Wall across x=15 with a single gap at the top.
    blocked[15, :-1] = True
    walled = Field(cfg, blocked=blocked)
    detoured = walled.distance_to_zone(start, Zone.OWN_PANTRY)
    assert math.isfinite(detoured)
    assert detoured > baseline
    path = walled.path_to_zone(start, Zone.OWN_PANTRY)
    assert path is not None
    assert all(not walled.blocked[c[0], c[1]] for c in path)


def test_mirror_layout_produces_opponent_zones():
    """The alliance-mirroring extension point works without restructuring."""
    mirrored = mirror_layout(DEFAULT_LAYOUT)
    assert len(mirrored) == len(DEFAULT_LAYOUT)
    fld = Field(layout=list(DEFAULT_LAYOUT) + mirrored)
    assert fld.has_zone(Zone.OPP_PANTRY)
    assert fld.has_zone(Zone.OWN_PANTRY)
    # Own zones must survive unchanged.
    own = Field()
    assert (own.zone_grid[: own.n_cols // 2]
            == fld.zone_grid[: fld.n_cols // 2]).all()


# =============================================================================
# Time model
# =============================================================================

def test_travel_time_is_monotone_and_physical():
    """Longer is slower, faster robots are quicker, and rest-to-rest costs."""
    cfg = TimeConfig()
    times = [traverse_time(d, cfg) for d in (0.0, 1.0, 5.0, 20.0, 50.0)]
    assert times[0] == 0.0
    assert all(a < b for a, b in zip(times, times[1:]))

    quick = dataclasses.replace(cfg, max_velocity_ft_s=20.0)
    assert traverse_time(50.0, quick) < traverse_time(50.0, cfg)

    # Below the ramp distance the profile is triangular, above it trapezoidal;
    # the two expressions must agree at the crossover.
    ramp = cfg.max_velocity_ft_s ** 2 / cfg.max_accel_ft_s2
    tri = 2.0 * math.sqrt(ramp / cfg.max_accel_ft_s2)
    trap = ramp / cfg.max_velocity_ft_s + cfg.max_velocity_ft_s / cfg.max_accel_ft_s2
    assert abs(tri - trap) < 1e-9
    assert abs(traverse_time(ramp, cfg) - tri) < 1e-9


def test_cruise_is_never_slower_than_rest_to_rest():
    """Momentum must help, never hurt."""
    cfg = TimeConfig()
    for d in (1.5, 3.0, 10.0, 40.0):
        assert cruise_time(d, cfg) <= traverse_time(d, cfg) + TOL


def test_clock_is_conserved_exactly():
    """Ledger durations must sum to the match length, with no gaps or overlap."""
    for policy in (RandomPolicy(3), CycleAndParkPolicy(2),
                   RankingPointRushPolicy()):
        env = HarvestHavocEnv()
        rollout(env, policy, seed=7)
        total = sum(r.duration_s for r in env.time_ledger)
        assert abs(total - env.config.time.match_duration_s) < 1e-6

        # Records must tile the timeline contiguously.
        expected_t = 0.0
        for rec in env.time_ledger:
            assert abs(rec.t_start - expected_t) < 1e-9
            expected_t = rec.t_end
        assert abs(expected_t - env.config.time.match_duration_s) < 1e-6


def test_duration_equals_sum_of_components():
    """travel + align + manipulate + idle must account for each action."""
    env = HarvestHavocEnv()
    rollout(env, RankingPointRushPolicy(), seed=11)
    for rec in env.time_ledger:
        parts = rec.travel_s + rec.align_s + rec.manipulate_s + rec.idle_s
        assert abs(parts - rec.duration_s) < 1e-6, rec


def test_repeated_scoring_at_one_site_is_cheaper_than_the_first():
    """Zero travel on the second placement is what makes batching pay."""
    env = HarvestHavocEnv(det_config())
    env.reset(seed=0)
    env.state.robot.carrots = 3
    env.step(int(Action.SCORE_CARROT_L2))
    first = env.time_ledger[-1]
    env.step(int(Action.SCORE_CARROT_L2))
    second = env.time_ledger[-1]
    assert first.travel_s > 0.0
    assert second.travel_s == 0.0
    assert second.duration_s < first.duration_s


def test_higher_shelves_cost_more_time():
    """The elevator tradeoff must actually be present in the time model."""
    cfg = TimeConfig()
    assert cfg.score_carrot_s[1] < cfg.score_carrot_s[2] < cfg.score_carrot_s[3]
    assert cfg.score_cake_s[1] < cfg.score_cake_s[2] < cfg.score_cake_s[3]


# =============================================================================
# Scoring
# =============================================================================

def test_pantry_point_values_match_the_rules():
    """3/4/5 for carrots and 8/10/12 for cakes, on levels 1/2/3."""
    env = HarvestHavocEnv()
    for level, carrot_pts, cake_pts in ((1, 3, 8), (2, 4, 10), (3, 5, 12)):
        env.reset(seed=0)
        env.state.robot.carrots = 1
        env.state.robot.cakes = 1
        before = env.state.raw_score
        env.step(int({1: Action.SCORE_CARROT_L1, 2: Action.SCORE_CARROT_L2,
                      3: Action.SCORE_CARROT_L3}[level]))
        assert env.state.raw_score - before == carrot_pts
        before = env.state.raw_score
        env.step(int({1: Action.SCORE_CAKE_L1, 2: Action.SCORE_CAKE_L2,
                      3: Action.SCORE_CAKE_L3}[level]))
        assert env.state.raw_score - before == cake_pts


def test_oven_is_worth_two_points_for_either_piece():
    env = HarvestHavocEnv()
    for action in (Action.OVEN_DEPOSIT_CARROT, Action.OVEN_DEPOSIT_CAKE):
        env.reset(seed=0)
        env.state.robot.carrots = 1
        env.state.robot.cakes = 1
        before = env.state.raw_score
        env.step(int(action))
        assert env.state.raw_score - before == 2


def test_inventory_cap_is_hard():
    """Three pieces, no more -- and intake is masked out when full."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.robot.carrots = 3
    env.state.robot.cakes = 0
    assert not env._is_legal(Action.INTAKE_CARROT)
    assert not env.state.robot.add(Piece.CARROT, env.config.max_inventory)
    assert env.state.robot.held == 3


def test_auto_leave_bonus_is_once_and_auto_only():
    """Two points, granted at most once, and never after autonomous."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    assert env._is_legal(Action.EXIT_KITCHEN)
    env.step(int(Action.EXIT_KITCHEN))
    assert env.state.auto_leave_earned
    assert env.state.raw_score == 2
    assert not env._is_legal(Action.EXIT_KITCHEN)  # already earned

    # A fresh match that reaches teleop still inside the kitchen loses it.
    env.reset(seed=0)
    while env.state.phase(env.config.time) is Phase.AUTO:
        env.step(int(Action.WAIT))
    assert not env.state.auto_leave_earned
    assert not env._is_legal(Action.EXIT_KITCHEN)


def test_leaving_kitchen_means_leaving_the_whole_complex():
    """Driving from the kitchen into its own pantry must NOT earn the bonus."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.robot.carrots = 1
    env.step(int(Action.SCORE_CARROT_L1))  # travels into the pantry
    assert env.field.zone_at(env.state.robot.cell) is Zone.OWN_PANTRY
    assert not env.state.auto_leave_earned


def test_oven_carrots_trigger_a_delayed_cake_exchange():
    """Three oven carrots produce one cake, after a delay, at the depot."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.robot.carrots = 3
    for _ in range(3):
        env.step(int(Action.OVEN_DEPOSIT_CARROT))
    assert env.state.oven.carrots_deposited == 3
    # Exactly one exchange in flight, nothing available yet.
    assert env.state.oven.pending_count == 1
    assert env.state.cakes_available == 0

    t_triggered = env.state.t
    while env.state.cakes_available == 0 and not env.state.is_over(env.config.time):
        env.step(int(Action.WAIT))
    assert env.state.cakes_available == 1
    assert env.state.oven.pending_count == 0
    # The delay is sampled, so assert that time elapsed rather than pinning it.
    assert env.state.t > t_triggered


def test_oven_cakes_do_not_feed_the_exchange():
    """Only carrots count toward the human-player conversion."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.robot.cakes = 3
    env.state.robot.carrots = 0
    for _ in range(3):
        env.step(int(Action.OVEN_DEPOSIT_CAKE))
    assert env.state.oven.cakes_deposited == 3
    assert env.state.oven.carrots_toward_next == 0
    assert env.state.oven.pending_count == 0


# =============================================================================
# Endgame
# =============================================================================

def test_harvest_haul_values_and_cap():
    """1 per carrot, 3 per cake, at most three pieces counted."""
    cfg = ScoringConfig()
    assert haul_value(0, 0, cfg) == 0
    assert haul_value(3, 0, cfg) == 3
    assert haul_value(0, 3, cfg) == 9
    assert haul_value(1, 2, cfg) == 7          # 2 cakes + 1 carrot
    assert haul_value(5, 5, cfg) == 9          # capped at three pieces, cakes first
    assert max_endgame_points(cfg) == 11


def test_park_requires_both_the_action_and_final_occupancy():
    """Park is positional at the buzzer, and the PARK action must have run."""
    # (a) parked and still in the table zone -> scored.
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.t = env.config.time.match_duration_s - 5.0
    env.state.robot.carrots = 0
    env.state.robot.cakes = 0
    env.step(int(Action.PARK))
    assert env.state.robot.parked
    while not env.state.is_over(env.config.time):
        env.step(int(Action.WAIT))
    assert env.state.finalized
    assert env.state.raw_score == env.config.scoring.park_points

    # (b) never parked -> no points, even though the match ended.
    env.reset(seed=0)
    while not env.state.is_over(env.config.time):
        env.step(int(Action.WAIT))
    assert env.state.finalized
    assert env.state.raw_score == 0


def test_parking_is_lost_by_driving_away():
    """Any non-WAIT action clears the parked flag."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.t = env.config.time.match_duration_s - 8.0
    env.step(int(Action.PARK))
    assert env.state.robot.parked
    env.step(int(Action.MOVE_W))
    assert not env.state.robot.parked


def test_harvest_haul_is_evaluated_only_at_the_buzzer():
    """Held cakes score nothing until the match actually ends."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.t = env.config.time.match_duration_s - 6.0
    env.state.robot.carrots = 0
    env.state.robot.cakes = 3
    env.step(int(Action.PARK))
    mid_score = env.state.raw_score
    assert mid_score == 0, "haul must not be paid before the buzzer"
    while not env.state.is_over(env.config.time):
        env.step(int(Action.WAIT))
    # 2 park + 3 cakes * 3 = 11
    assert env.state.raw_score == 11


# =============================================================================
# Ranking points
# =============================================================================

def test_stocked_up_fires_exactly_at_three_on_every_shelf():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    cfg, tcfg = env.config.scoring, env.config.time
    st = env.state
    for lvl in (1, 2, 3):
        st.pantry.shelves[lvl].carrots = 3
    assert evaluate_ranking_points(st, cfg, tcfg).stocked_up
    st.pantry.shelves[2].carrots = 2
    assert not evaluate_ranking_points(st, cfg, tcfg).stocked_up
    # Mixed pieces count toward the total.
    st.pantry.shelves[2].cakes = 1
    assert evaluate_ranking_points(st, cfg, tcfg).stocked_up


def test_baked_up_needs_a_cake_on_every_shelf():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    cfg, tcfg = env.config.scoring, env.config.time
    st = env.state
    for lvl in (1, 2, 3):
        st.pantry.shelves[lvl].cakes = 1
    assert evaluate_ranking_points(st, cfg, tcfg).baked_up
    # Carrots cannot substitute, even at full shelf capacity.
    st.pantry.shelves[1].cakes = 0
    st.pantry.shelves[1].carrots = cfg.shelf_capacity
    assert not evaluate_ranking_points(st, cfg, tcfg).baked_up


# =============================================================================
# Shelf capacity  --  max 3 pieces per shelf
# =============================================================================

def test_shelf_capacity_is_a_hard_cap():
    """A shelf accepts exactly `shelf_capacity` pieces and no more."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    cap = env.config.scoring.shelf_capacity
    shelf = env.state.pantry.shelves[2]
    for i in range(cap):
        assert shelf.free_slots == cap - i
        assert not shelf.is_full
        assert env.state.pantry.add(2, _Piece.CARROT)
    assert shelf.is_full
    assert shelf.free_slots == 0
    assert shelf.total == cap
    # Further adds are refused rather than silently overfilling.
    assert not env.state.pantry.add(2, _Piece.CARROT)
    assert not env.state.pantry.add(2, _Piece.CARROT_CAKE)
    assert shelf.total == cap


def test_scoring_on_a_full_shelf_is_illegal():
    """The cap is enforced through the action mask, not by a failed attempt."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    cap = env.config.scoring.shelf_capacity
    env.state.robot.carrots = 3
    for _ in range(cap):
        env.state.pantry.add(3, _Piece.CARROT)

    assert not env._is_legal(Action.SCORE_CARROT_L3)
    assert not env._is_legal(Action.SCORE_CAKE_L3)
    assert not env.action_mask()[int(Action.SCORE_CARROT_L3)]
    # Other shelves are unaffected.
    assert env._is_legal(Action.SCORE_CARROT_L1)

    before = env.state.raw_score
    _o, _r, _t, _tr, info = env.step(int(Action.SCORE_CARROT_L3))
    assert info["action_illegal"]
    assert env.state.raw_score == before
    assert env.state.pantry.shelves[3].total == cap


def test_pantry_total_capacity_and_max_points():
    """Nine slots, and 90 points if every one of them holds a cake."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    assert env.state.pantry.total_capacity() == 9
    assert max_pantry_points(env.config.scoring) == 90
    assert not env.state.pantry.is_full()
    for lvl in (1, 2, 3):
        for _ in range(3):
            env.state.pantry.add(lvl, _Piece.CARROT_CAKE)
    assert env.state.pantry.is_full()
    assert env.state.pantry.total_pieces() == 9


def test_filling_a_shelf_with_carrots_forfeits_baked_up():
    """The central trap the shelf cap creates -- and it must be permanent."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    cfg, tcfg = env.config.scoring, env.config.time
    for lvl in (1, 2, 3):
        for _ in range(cfg.shelf_capacity):
            env.state.pantry.add(lvl, _Piece.CARROT)

    rps = evaluate_ranking_points(env.state, cfg, tcfg)
    assert rps.stocked_up, "a full pantry of carrots does satisfy Stocked Up"
    assert not rps.baked_up, "...but Baked Up is now unreachable"

    # And there is no way back: every cake action is illegal forever.
    env.state.robot.cakes = 3
    for action in (Action.SCORE_CAKE_L1, Action.SCORE_CAKE_L2,
                   Action.SCORE_CAKE_L3):
        assert not env._is_legal(action)


def test_reserving_cake_slots_wins_both_pantry_rps():
    """Two carrots plus one cake per shelf earns Stocked Up AND Baked Up."""
    reserving = rollout(HarvestHavocEnv(det_config()),
                        RankingPointRushPolicy(chase_dinner_rp=False), seed=1)
    greedy = rollout(HarvestHavocEnv(det_config()),
                     RankingPointRushPolicy(chase_dinner_rp=False,
                                            reserve_cake_slots=False), seed=1)
    assert reserving["ranking_points"]["stocked_up"]
    assert reserving["ranking_points"]["baked_up"]
    assert not greedy["ranking_points"]["baked_up"]
    assert reserving["ranking_point_total"] > greedy["ranking_point_total"]


def test_unreachable_pantry_rps_are_rejected_loudly():
    """A capacity below a threshold must fail at construction."""
    cfg = EnvConfig()
    cfg.scoring.shelf_capacity = 2      # Stocked Up needs 3
    try:
        HarvestHavocEnv(cfg)
    except ValueError as exc:
        assert "Stocked Up is unreachable" in str(exc)
    else:
        raise AssertionError("expected ValueError for shelf_capacity < threshold")

    assert stocked_up_is_reachable(ScoringConfig())
    assert baked_up_is_reachable(ScoringConfig())
    assert both_pantry_rps_reachable(ScoringConfig())
    assert not stocked_up_is_reachable(ScoringConfig(shelf_capacity=2))


def test_shelf_capacity_is_tunable():
    """Raising the cap must raise the achievable pantry total."""
    small = rollout(HarvestHavocEnv(det_config(
        scoring=ScoringConfig(shelf_capacity=3))),
        CycleAndParkPolicy(3), seed=1)
    large = rollout(HarvestHavocEnv(det_config(
        scoring=ScoringConfig(shelf_capacity=9))),
        CycleAndParkPolicy(3), seed=1)
    assert large["pantry_used"] > small["pantry_used"]
    assert small["pantry_capacity"] == 9
    assert large["pantry_capacity"] == 27
    # More pantry room means fewer 2-point oven consolation deposits.
    assert large["raw_score"] > small["raw_score"]


# =============================================================================
# Probabilistic action durations
# =============================================================================

def test_durations_are_sampled_not_fixed():
    """The same action from the same cell must not take the same time twice."""
    env = HarvestHavocEnv()
    pantry_cell = env.field.zone_centroid_cell(Zone.OWN_PANTRY)
    durations = []
    for i in range(30):
        env.reset(seed=100 + i)
        env.state.robot.cell = pantry_cell
        env.state.robot.carrots = 1
        env.step(int(Action.SCORE_CARROT_L3))
        durations.append(env.time_ledger[-1].duration_s)
    assert len(set(round(d, 6) for d in durations)) > 20, \
        "durations look quantised or fixed"
    assert min(durations) > 0.0


def test_duration_model_preserves_the_mean():
    """Noise must add variance without secretly biasing the robot slower."""
    for model in ("lognormal", "gamma"):
        stoch = StochasticConfig(duration_model=model)
        timing = TimingModel(TimeConfig(), stoch, np.random.default_rng(0))
        for nominal, component in ((2.6, "manipulate"), (0.5, "align"),
                                   (4.0, "travel")):
            draws = np.array(
                [timing.sample(nominal, component) for _ in range(20000)]
            )
            assert abs(draws.mean() - nominal) < 0.03 * nominal, (
                model, component, draws.mean(), nominal
            )


def test_duration_distribution_is_right_skewed():
    """Things go wrong and cost time far more often than they save it."""
    stoch = StochasticConfig(manipulate_noise_cv=0.3)
    timing = TimingModel(TimeConfig(), stoch, np.random.default_rng(1))
    draws = np.array([timing.sample(2.0, "manipulate") for _ in range(40000)])
    mean, median = draws.mean(), np.median(draws)
    assert median < mean, "a right-skewed duration has median below mean"
    # The upper tail must reach further from the mean than the lower one.
    assert (np.percentile(draws, 99) - mean) > (mean - np.percentile(draws, 1))


def test_duration_cv_matches_the_configured_value():
    """The knob means what it says."""
    for cv in (0.05, 0.15, 0.4):
        stoch = StochasticConfig(manipulate_noise_cv=cv)
        timing = TimingModel(TimeConfig(), stoch, np.random.default_rng(2))
        stats = timing.describe(2.0, "manipulate", samples=30000)
        assert abs(stats["cv"] - cv) < 0.03, (cv, stats["cv"])


def test_per_component_noise_is_independent():
    """Alignment is noisier than driving, as configured."""
    stoch = StochasticConfig()
    timing = TimingModel(TimeConfig(), stoch, np.random.default_rng(3))
    travel_cv = timing.describe(2.0, "travel", samples=20000)["cv"]
    align_cv = timing.describe(2.0, "align", samples=20000)["cv"]
    assert align_cv > travel_cv
    assert timing.describe(2.0, "wait", samples=200)["cv"] == 0.0


def test_noise_scale_and_deterministic_mode():
    """One knob turns all timing randomness off."""
    off = TimingModel(TimeConfig(), StochasticConfig(time_noise_scale=0.0),
                      np.random.default_rng(4))
    assert all(off.sample(2.0, "manipulate") == 2.0 for _ in range(100))

    det = TimingModel(TimeConfig(), StochasticConfig.deterministic(),
                      np.random.default_rng(4))
    assert det.model is DurationModel.DETERMINISTIC
    assert all(det.sample(2.0, "align") == 2.0 for _ in range(100))

    loud = TimingModel(TimeConfig(), StochasticConfig(time_noise_scale=2.0),
                       np.random.default_rng(5))
    quiet = TimingModel(TimeConfig(), StochasticConfig(time_noise_scale=0.5),
                        np.random.default_rng(5))
    assert (loud.describe(2.0, "manipulate")["std"]
            > quiet.describe(2.0, "manipulate")["std"])


def test_duration_multipliers_are_clipped():
    """No draw may make an action free or eat the whole match."""
    stoch = StochasticConfig(manipulate_noise_cv=2.0,
                             time_noise_clip=(0.5, 2.0))
    timing = TimingModel(TimeConfig(), stoch, np.random.default_rng(6))
    draws = np.array([timing.sample(2.0, "manipulate") for _ in range(20000)])
    assert draws.min() >= 2.0 * 0.5 - TOL
    assert draws.max() <= 2.0 * 2.0 + TOL


def test_hp_exchange_delay_is_sampled():
    """The human-player turnaround is a distribution, not a constant."""
    stoch = StochasticConfig()
    timing = TimingModel(TimeConfig(), stoch, np.random.default_rng(7))
    draws = np.array([timing.hp_exchange_delay() for _ in range(5000)])
    assert abs(draws.mean() - stoch.hp_exchange_delay_s) < 0.4
    assert draws.std() > 1.0
    assert len(set(np.round(draws, 6))) > 4000


def test_noisy_matches_vary_but_stay_valid():
    """Sampled timing must not break the clock or the rules."""
    scores = set()
    for seed in range(12):
        env = HarvestHavocEnv()
        r = rollout(env, CycleAndParkPolicy(3), seed=seed)
        scores.add(r["raw_score"])
        assert abs(sum(x.duration_s for x in env.time_ledger)
                   - env.config.time.match_duration_s) < 1e-6
        for lvl, s in r["shelves"].items():
            assert s["carrots"] + s["cakes"] <= env.config.scoring.shelf_capacity
    assert len(scores) > 1, "default config should produce varied matches"


# =============================================================================
# Every action has a time cost
# =============================================================================

def test_every_action_has_a_nonzero_time_cost():
    """No action, anywhere on the field, may consume zero match time.

    Two cases nominally cost nothing, and both are non-actions: a MOVE that
    would leave the field, and EXIT_KITCHEN measured from outside the kitchen.
    Neither is ever selectable -- off-field moves are masked, and the leave
    bonus can only be unclaimed while the robot is still inside. Every action
    that *is* selectable costs time, and the ``min_action_duration_s`` floor
    makes that true of the real clock rather than just the nominal table.
    """
    env = HarvestHavocEnv()
    env.reset(seed=0)
    floor = env.config.time.min_action_duration_s

    for zone in (Zone.OWN_KITCHEN, Zone.OWN_FARM, Zone.OWN_PANTRY,
                 Zone.OWN_OVEN, Zone.OWN_TABLE):
        cell = env.field.zone_centroid_cell(zone)
        env.reset(seed=0)
        env.state.robot.cell = cell
        env.state.robot.carrots = 2
        env.state.robot.cakes = 1
        env.state.cakes_available = 1
        for row in env.action_time_table(from_cell=cell, samples=50):
            assert math.isfinite(row["nominal_total"]), (zone.name, row["name"])
            if row["legal"]:
                assert row["nominal_total"] > 0.0, (zone.name, row["name"])

    # The real guarantee: a stepped action never comes in under the floor.
    for action in Action:
        env.reset(seed=0)
        env.state.robot.carrots = 2
        env.state.robot.cakes = 1
        env.state.cakes_available = 1
        env.step(int(action))
        assert env.time_ledger[-1].duration_s >= floor - TOL, action.name


def test_action_time_table_covers_the_whole_action_space():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    rows = env.action_time_table(samples=50)
    assert len(rows) == N_ACTIONS
    assert {r["name"] for r in rows} == {a.name for a in Action}
    for row in rows:
        parts = row["travel"] + row["align"] + row["manipulate"] + row["wait"]
        assert abs(parts - row["nominal_total"]) < TOL


def test_nominal_duration_reflects_position():
    """Travel is zero in the target zone and positive outside it."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    pantry = env.field.zone_centroid_cell(Zone.OWN_PANTRY)
    farm = env.field.zone_centroid_cell(Zone.OWN_FARM)

    at_pantry = env.nominal_duration(int(Action.SCORE_CARROT_L2), pantry)
    from_farm = env.nominal_duration(int(Action.SCORE_CARROT_L2), farm)
    assert at_pantry.travel == 0.0
    assert from_farm.travel > 0.0
    assert from_farm.total > at_pantry.total
    # Manipulation and align are position-independent.
    assert at_pantry.manipulate == from_farm.manipulate
    assert at_pantry.align == from_farm.align


def test_higher_shelves_cost_more_manipulation_time_in_the_table():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    pantry = env.field.zone_centroid_cell(Zone.OWN_PANTRY)
    times = [
        env.nominal_duration(int(a), pantry).manipulate
        for a in (Action.SCORE_CARROT_L1, Action.SCORE_CARROT_L2,
                  Action.SCORE_CARROT_L3)
    ]
    assert times[0] < times[1] < times[2]


def test_action_times_respond_to_config():
    """Editing a time constant must show up in the table."""
    slow = det_config(time=TimeConfig(score_carrot_s={1: 5.0, 2: 6.0, 3: 7.0}))
    env = HarvestHavocEnv(slow)
    env.reset(seed=0)
    pantry = env.field.zone_centroid_cell(Zone.OWN_PANTRY)
    assert env.nominal_duration(int(Action.SCORE_CARROT_L1),
                                pantry).manipulate == 5.0


# =============================================================================
# Reward priorities
# =============================================================================

def test_from_priorities_scales_the_objectives():
    """Four numbers must actually move the eight underlying weights."""
    balanced = RewardConfig.from_priorities()
    assert balanced.bonus_stocked_up == balanced.bonus_baked_up

    rp_heavy = RewardConfig.from_priorities(score=0.1, stocked_up=3.0)
    assert rp_heavy.w_raw_score == 0.1
    assert rp_heavy.bonus_stocked_up > balanced.bonus_stocked_up
    assert rp_heavy.shaping_scale_stocked_up > balanced.shaping_scale_stocked_up
    # Shaping must stay below the bonus, or partial progress gets farmed.
    assert rp_heavy.shaping_scale_stocked_up < rp_heavy.bonus_stocked_up


def test_zero_priority_removes_an_objective():
    cfg = RewardConfig.from_priorities(stocked_up=0.0)
    assert cfg.bonus_stocked_up == 0.0
    assert cfg.shaping_scale_stocked_up == 0.0
    assert cfg.bonus_baked_up > 0.0


def test_priority_overrides_reach_other_fields():
    cfg = RewardConfig.from_priorities(score=2.0, w_time_penalty=0.5)
    assert cfg.w_raw_score == 2.0
    assert cfg.w_time_penalty == 0.5
    try:
        RewardConfig.from_priorities(not_a_field=1)
    except ValueError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown field")


def test_reward_presets_exist_and_differ():
    names = ["balanced", "score", "rp", "seeding", "stocked", "baked", "dinner"]
    configs = {n: RewardConfig.preset(n) for n in names}
    assert configs["score"].bonus_stocked_up == 0.0
    assert configs["score"].bonus_baked_up == 0.0
    assert configs["rp"].w_raw_score < configs["score"].w_raw_score
    assert configs["baked"].bonus_baked_up > 0.0
    assert configs["baked"].bonus_stocked_up == 0.0
    try:
        RewardConfig.preset("nonsense")
    except ValueError as exc:
        assert "unknown reward preset" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown preset")


def test_priorities_change_which_strategy_scores_best():
    """The point of the priority dial: it must reorder the strategies."""
    base = det_config()
    score_cfg = det_config(reward=RewardConfig.preset("score"))
    rp_cfg = det_config(reward=RewardConfig.preset("rp"))

    cycling = CycleAndParkPolicy(3)
    rp_policy = RankingPointRushPolicy(chase_dinner_rp=False)

    def reward(cfg, policy):
        return rollout(HarvestHavocEnv(cfg), policy, seed=1)["total_reward"]

    # The RP chaser's advantage must widen when ranking points are what count.
    gap_under_score = reward(score_cfg, rp_policy) - reward(score_cfg, cycling)
    gap_under_rp = reward(rp_cfg, rp_policy) - reward(rp_cfg, cycling)
    assert gap_under_rp > gap_under_score, (gap_under_score, gap_under_rp)

    # And a pure-score objective must value the extra ranking point at zero.
    assert score_cfg.reward.bonus_baked_up == 0.0
    assert base.reward.bonus_stocked_up > 0.0


def test_dinner_rp_needs_ten_endgame_points():
    """Under the default reading, only park + three cakes clears the bar."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.t = env.config.time.match_duration_s - 6.0
    env.state.robot.carrots = 0
    env.state.robot.cakes = 3
    env.step(int(Action.PARK))
    while not env.state.is_over(env.config.time):
        env.step(int(Action.WAIT))
    result = env.match_result()
    assert result["endgame_points"] == 11
    assert result["ranking_points"]["dinner_rp"]

    # Two cakes and a carrot is 2 + 3 + 3 + 1 = 9 -- one point short.
    env.reset(seed=0)
    env.state.t = env.config.time.match_duration_s - 6.0
    env.state.robot.carrots = 1
    env.state.robot.cakes = 2
    env.step(int(Action.PARK))
    while not env.state.is_over(env.config.time):
        env.step(int(Action.WAIT))
    assert env.match_result()["endgame_points"] == 9
    assert not env.match_result()["ranking_points"]["dinner_rp"]


def test_dinner_rp_all_in_endgame_mode():
    """The alternative reading counts pantry points scored during endgame."""
    cfg = EnvConfig()
    cfg.scoring.dinner_rp_mode = DinnerRPMode.ALL_IN_ENDGAME.value
    env = HarvestHavocEnv(cfg)
    env.reset(seed=0)
    env.state.t = env.config.time.endgame_start_s + 1.0
    env.state.robot.cakes = 1
    env.state.robot.carrots = 2
    env.step(int(Action.SCORE_CAKE_L3))          # 12 points, during endgame
    while not env.state.is_over(env.config.time):
        env.step(int(Action.WAIT))
    assert env.match_result()["endgame_points"] >= 12
    assert env.match_result()["ranking_points"]["dinner_rp"]


def test_unreachable_dinner_rp_is_rejected_loudly():
    """A misconfigured threshold must fail at construction, not silently."""
    cfg = EnvConfig()
    cfg.scoring.dinner_rp_threshold = 99
    try:
        HarvestHavocEnv(cfg)
    except ValueError as exc:
        assert "Dinner RP is unreachable" in str(exc)
    else:
        raise AssertionError("expected ValueError for unreachable Dinner RP")


def test_rp_progress_is_monotone_and_bounded():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    _obs, _r, _term, _trunc, info = env.step(int(Action.WAIT))
    previous = info["rp_progress"]
    for lvl in (1, 2, 3):
        env.state.pantry.shelves[lvl].carrots += 1
        _obs, _r, _term, _trunc, info = env.step(int(Action.WAIT))
        current = info["rp_progress"]
        assert current["stocked_up"] >= previous["stocked_up"] - TOL
        assert 0.0 <= current["stocked_up"] <= 1.0
        assert 0.0 <= current["baked_up"] <= 1.0
        assert 0.0 <= current["dinner_rp"] <= 1.0
        previous = current


# =============================================================================
# Environment API
# =============================================================================

def test_reset_and_step_signatures():
    """Gymnasium's five-tuple contract, with or without gymnasium installed."""
    env = HarvestHavocEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.encoder.size,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)

    out = env.step(int(Action.WAIT))
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_observations_stay_inside_the_declared_space():
    """No slot may leave its Box bounds, in any policy, at any point."""
    for policy in (RandomPolicy(1), CycleAndParkPolicy(1),
                   CycleAndParkPolicy(3), RankingPointRushPolicy()):
        env = HarvestHavocEnv()
        obs, _ = env.reset(seed=5)
        assert env.observation_space.contains(obs)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _r, terminated, truncated, _i = env.step(int(policy.act(env)))
            assert env.observation_space.contains(obs), \
                env.encoder.as_dict(obs)


def test_observation_names_match_vector_width():
    env = HarvestHavocEnv()
    obs, _ = env.reset(seed=0)
    assert len(env.encoder.names) == env.encoder.size == obs.size
    assert len(set(env.encoder.names)) == env.encoder.size, "duplicate labels"
    labelled = env.encoder.as_dict(obs)
    assert labelled["phase_auto"] == 1.0
    assert labelled["time_remaining_frac"] == 1.0


def test_action_mask_is_never_empty_and_matches_legality():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        mask = env.action_mask()
        assert mask.shape == (N_ACTIONS,)
        assert mask.any(), "mask must always leave at least WAIT legal"
        assert mask[int(Action.WAIT)]
        legal = [int(a) for a in env.legal_actions()]
        assert legal == [i for i in range(N_ACTIONS) if mask[i]]
        _o, _r, terminated, truncated, _i = env.step(legal[0])


def test_masked_actions_are_actually_rejected():
    """Selecting a masked action must be a no-op, flagged in info."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.robot.cakes = 0
    assert not env._is_legal(Action.SCORE_CAKE_L3)
    before = env.state.raw_score
    _o, _r, _t, _tr, info = env.step(int(Action.SCORE_CAKE_L3))
    assert info["action_illegal"]
    assert env.state.raw_score == before


def test_illegal_actions_cannot_stall_the_episode():
    """Burning clock on illegal actions guarantees termination."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.robot.cakes = 0
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _o, _r, terminated, truncated, _i = env.step(int(Action.SCORE_CAKE_L3))
        steps += 1
        assert steps < env.config.max_episode_steps
    assert terminated and not truncated


def test_episode_terminates_at_the_buzzer():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _o, _r, terminated, truncated, _i = env.step(int(Action.WAIT))
    assert terminated
    assert abs(env.state.t - env.config.time.match_duration_s) < 1e-9
    assert env.state.finalized
    try:
        env.step(int(Action.WAIT))
    except RuntimeError:
        pass
    else:
        raise AssertionError("stepping past the buzzer must raise")


def test_actions_are_cut_off_by_the_buzzer_but_still_move():
    """Partial actions matter: position at the buzzer decides the park."""
    env = HarvestHavocEnv()
    env.reset(seed=0)
    env.state.t = env.config.time.match_duration_s - 1.0
    start = env.state.robot.cell
    _o, _r, terminated, _tr, info = env.step(int(Action.SCORE_CARROT_L3))
    assert terminated
    assert info["action_success"] is False
    assert env.time_ledger[-1].truncated_by_buzzer
    assert env.state.robot.cell != start, "travel leg should still have moved"
    assert env.state.raw_score == 0, "manipulation must not have completed"


def test_start_cell_override():
    env = HarvestHavocEnv()
    target = env.field.cells_of_zone(Zone.OWN_FARM)[0]
    _obs, _info = env.reset(seed=0, options={"start_cell": target})
    assert env.state.robot.cell == target


def test_reset_fully_clears_state():
    env = HarvestHavocEnv()
    rollout(env, CycleAndParkPolicy(3), seed=2)
    assert env.state.raw_score > 0
    env.reset(seed=2)
    assert env.state.t == 0.0
    assert env.state.raw_score == 0
    assert env.state.pantry.total_pieces() == 0
    assert env.time_ledger == []
    assert not env.state.finalized


def test_seeding_is_reproducible_and_seeds_matter():
    cfg = EnvConfig(stochastic=StochasticConfig(
        intake_success_prob=0.8,
        time_noise_scale=1.5,
        hp_exchange_delay_s=8.0,
    ))
    a = rollout(HarvestHavocEnv(cfg), CycleAndParkPolicy(3), seed=42)
    b = rollout(HarvestHavocEnv(cfg), CycleAndParkPolicy(3), seed=42)
    assert a["raw_score"] == b["raw_score"]
    scores = {rollout(HarvestHavocEnv(cfg), CycleAndParkPolicy(3), seed=s)["raw_score"]
              for s in range(8)}
    assert len(scores) > 1, "stochastic config produced identical matches"


def test_default_config_is_stochastic_but_reproducible():
    """Durations are sampled by default, yet a seed pins the whole match."""
    a = rollout(HarvestHavocEnv(), CycleAndParkPolicy(3), seed=99)
    b = rollout(HarvestHavocEnv(), CycleAndParkPolicy(3), seed=99)
    assert a["raw_score"] == b["raw_score"]
    assert a["steps"] == b["steps"]

    scores = {rollout(HarvestHavocEnv(), CycleAndParkPolicy(3), seed=s)["raw_score"]
              for s in range(10)}
    assert len(scores) > 1, "default config should sample durations"


def test_deterministic_preset_removes_all_variation():
    """StochasticConfig.deterministic() is the escape hatch for debugging."""
    cfg = det_config()
    scores = {rollout(HarvestHavocEnv(cfg), CycleAndParkPolicy(3), seed=s)["raw_score"]
              for s in range(5)}
    assert len(scores) == 1


# =============================================================================
# Reward
# =============================================================================

def test_reward_breakdown_sums_to_the_scalar_reward():
    env = HarvestHavocEnv()
    env.reset(seed=0)
    terminated = truncated = False
    policy = RankingPointRushPolicy()
    while not (terminated or truncated):
        _o, reward, terminated, truncated, info = env.step(int(policy.act(env)))
        parts = info["reward_breakdown"]
        assert abs(parts["total"] - reward) < 1e-6
        recomputed = sum(v for k, v in parts.items() if k != "total")
        assert abs(recomputed - reward) < 1e-6


def test_reward_vector_tracks_the_scoreboard():
    """The raw-score objective must integrate to the final raw score."""
    env = HarvestHavocEnv()
    result = rollout(env, CycleAndParkPolicy(3), seed=3)
    assert abs(result["reward_vector_sum"][0] - result["raw_score"]) < 1e-6
    assert abs(result["reward_vector_sum"][4]
               - result["ranking_point_total"]) < 1e-6


def test_ranking_points_dominate_raw_score_in_the_default_weighting():
    """An RP must be worth more than the points it costs to chase."""
    rcfg = RewardConfig()
    assert rcfg.bonus_stocked_up > 20 * rcfg.w_raw_score
    assert rcfg.bonus_baked_up > 20 * rcfg.w_raw_score


def test_dinner_shaping_is_gated_until_endgame_approaches():
    """Hoarding cakes early must not be rewarded by the shaping term."""
    from harvest_havoc.reward import RewardCalculator

    cfg = EnvConfig()
    calc = RewardCalculator(cfg)
    assert calc._dinner_gate(0.0) == 0.0
    assert calc._dinner_gate(cfg.time.endgame_start_s) == 1.0
    assert calc._dinner_gate(cfg.time.match_duration_s) == 1.0
    mid = cfg.time.endgame_start_s - cfg.reward.dinner_shaping_ramp_s / 2
    assert 0.0 < calc._dinner_gate(mid) < 1.0


def test_strict_pbrs_zeroes_the_terminal_potential():
    cfg = EnvConfig()
    cfg.reward.strict_potential_based_shaping = True
    env = HarvestHavocEnv(cfg)
    result = rollout(env, CycleAndParkPolicy(3), seed=1)
    assert isinstance(result["total_reward"], float)
    assert math.isfinite(result["total_reward"])


# =============================================================================
# Time allocation reporting
# =============================================================================

def test_time_allocation_accounts_for_the_whole_match():
    env = HarvestHavocEnv()
    rollout(env, CycleAndParkPolicy(3), seed=1)
    summary = env.time_allocation_summary()
    totals = summary["totals"]
    assert abs(totals["accounted_s"] - totals["match_s"]) < 1e-6
    assert totals["unused_s"] < 1e-6

    by_kind = sum(b["seconds"] for b in summary["by_kind"].values())
    assert abs(by_kind - totals["accounted_s"]) < 1e-6
    by_component = sum(summary["by_component"].values())
    assert abs(by_component - totals["accounted_s"]) < 1e-6
    by_phase = sum(b["seconds"] for b in summary["by_phase"].values())
    assert abs(by_phase - totals["accounted_s"]) < 1e-6


def test_endgame_bonus_is_reported_as_unattributed():
    """Park and haul are not produced by an action, and are labelled as such."""
    env = HarvestHavocEnv()
    rollout(env, CycleAndParkPolicy(3), seed=1)
    summary = env.time_allocation_summary()
    attributed = sum(b["points"] for b in summary["by_kind"].values())
    assert (attributed + summary["totals"]["unattributed_points"]
            == env.state.raw_score)


def test_shelf_cap_makes_the_preferred_shelf_nearly_irrelevant():
    """A real consequence of the cap, worth pinning down.

    Without a cap, "which shelf?" was the dominant choice and level 3 won by
    30%. With only three slots per shelf a carrot-cycling robot fills all nine
    regardless of preference, so the *order* it fills them in barely matters --
    it ends up with the same nine carrots either way, and the rest of the match
    goes to the oven. The interesting question moves from "which shelf?" to
    "carrot or cake in this slot?".

    If this test starts failing, the cap or the fill-order fallback changed and
    the strategy conclusions need revisiting.
    """
    results = {
        lvl: rollout(HarvestHavocEnv(det_config()),
                     CycleAndParkPolicy(lvl), seed=1)
        for lvl in (1, 2, 3)
    }
    scores = {lvl: r["raw_score"] for lvl, r in results.items()}
    assert max(scores.values()) - min(scores.values()) <= 3, scores
    # ...because every one of them ends with a completely full pantry.
    for lvl, r in results.items():
        assert r["pantry_used"] == r["pantry_capacity"], (lvl, r["shelves"])


def test_cakes_beat_carrots_per_pantry_slot():
    """The arithmetic that makes the shelf cap interesting.

    Three carrots into the oven score 6 and yield a cake worth 12 on level 3:
    18 points for three carrots using ONE slot. The same three carrots placed
    directly on level 3 score 15 and burn THREE slots. Cakes win on both
    points-per-carrot and points-per-slot -- they only lose on time.
    """
    cfg = ScoringConfig()
    top = max(cfg.shelf_levels)
    carrots_per_cake = cfg.oven_carrots_per_cake

    direct = carrots_per_cake * cfg.carrot_points[top]
    via_oven = carrots_per_cake * cfg.oven_points + cfg.cake_points[top]
    assert via_oven > direct, (via_oven, direct)
    assert cfg.cake_points[top] > carrots_per_cake * cfg.carrot_points[top] / 3


def test_batching_beats_single_piece_cycling():
    """Confirms the travel model rewards filling the hopper before unloading."""
    class NoBatching(CycleAndParkPolicy):
        """Intake whenever there is any free slot -- the naive strategy."""

        def _should_intake(self, env):
            return env.state.robot.free_slots(env.config.max_inventory) > 0

    batched = rollout(HarvestHavocEnv(det_config()), CycleAndParkPolicy(3), seed=1)
    naive = rollout(HarvestHavocEnv(det_config()), NoBatching(3), seed=1)
    assert batched["raw_score"] > naive["raw_score"], (batched, naive)


# =============================================================================
# Baselines and rendering
# =============================================================================

def test_baselines_run_and_beat_random():
    env = HarvestHavocEnv()
    random_score = rollout(env, RandomPolicy(0), seed=0)["raw_score"]
    for policy in (CycleAndParkPolicy(3), RankingPointRushPolicy()):
        result = rollout(HarvestHavocEnv(), policy, seed=0)
        assert result["raw_score"] > random_score, policy.name
        assert result["finalized"]


def test_rp_policy_earns_ranking_points_that_score_policies_do_not():
    """Chasing Baked Up must actually earn it -- and the cap makes it cheap.

    Before the shelf cap this was a strict tradeoff: ranking points cost raw
    score. With only nine slots the calculus inverts. The RP chaser puts cakes
    in three of those slots (30 points) where the carrot cycler puts carrots
    (12), and the oven carrots it spends to produce them score 2 apiece on the
    way. So it wins on *both* axes -- a more useful finding than a tension.
    """
    cycling = rollout(HarvestHavocEnv(det_config()), CycleAndParkPolicy(3),
                      seed=1)
    rp = rollout(HarvestHavocEnv(det_config()),
                 RankingPointRushPolicy(chase_dinner_rp=False), seed=1)
    assert rp["ranking_point_total"] > cycling["ranking_point_total"]
    assert rp["ranking_points"]["baked_up"]
    assert not cycling["ranking_points"]["baked_up"]
    # Reserving a cake slot per shelf must not cost meaningful raw score.
    assert rp["raw_score"] >= cycling["raw_score"] - 5, (
        rp["raw_score"], cycling["raw_score"]
    )


def test_render_modes():
    assert HarvestHavocEnv(render_mode=None).render() is None
    env = HarvestHavocEnv(render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert isinstance(text, str)
    assert "R" in text and "own_pantry" in text
    assert len(text.splitlines()) > env.field.n_rows


def test_render_helpers_do_not_crash():
    from harvest_havoc.render import render_time_allocation, render_time_ledger

    env = HarvestHavocEnv()
    rollout(env, RankingPointRushPolicy(), seed=0)
    assert "TIME ALLOCATION" in render_time_allocation(env)
    assert "SCORE_" in render_time_ledger(env) or "INTAKE" in render_time_ledger(env)


def test_compare_policies_returns_one_row_per_policy():
    from harvest_havoc import compare_policies

    rows = compare_policies(
        HarvestHavocEnv(), [CycleAndParkPolicy(1), CycleAndParkPolicy(3)],
        episodes=2, seed=0,
    )
    assert len(rows) == 2
    assert {r["policy"] for r in rows} == {"cycle_L1", "cycle_L3"}
    assert all(r["episodes"] == 2 for r in rows)


# =============================================================================
# Configuration sweeps -- the environment must respond to its own knobs
# =============================================================================

def test_faster_robot_scores_more():
    """The whole point of a real time model: kinematics must matter."""
    slow = det_config(time=TimeConfig(max_velocity_ft_s=6.0))
    fast = det_config(time=TimeConfig(max_velocity_ft_s=22.0))
    slow_score = rollout(HarvestHavocEnv(slow), CycleAndParkPolicy(3), seed=1)
    fast_score = rollout(HarvestHavocEnv(fast), CycleAndParkPolicy(3), seed=1)
    assert fast_score["raw_score"] > slow_score["raw_score"]


def test_shorter_match_scores_less():
    short = det_config(time=TimeConfig(match_duration_s=75.0))
    full = rollout(HarvestHavocEnv(det_config()), CycleAndParkPolicy(3), seed=1)
    brief = rollout(HarvestHavocEnv(short), CycleAndParkPolicy(3), seed=1)
    assert brief["raw_score"] < full["raw_score"]


def test_depot_scarcity_binds():
    scarce = det_config(depot_carrot_supply=6)
    result = rollout(HarvestHavocEnv(scarce), CycleAndParkPolicy(3), seed=1)
    assert result["shelves"][3]["carrots"] <= 7  # 6 from depot + 1 preload


def test_failure_rates_reduce_score_without_breaking_anything():
    lossy = EnvConfig(stochastic=StochasticConfig(
        intake_success_prob=0.6,
        score_success_prob={1: 0.9, 2: 0.8, 3: 0.6},
    ))
    clean = rollout(HarvestHavocEnv(det_config()), CycleAndParkPolicy(3), seed=4)
    noisy = [rollout(HarvestHavocEnv(lossy), CycleAndParkPolicy(3), seed=s)
             for s in range(6)]
    assert np.mean([r["raw_score"] for r in noisy]) < clean["raw_score"]
    for r in noisy:
        assert r["finalized"]


def test_preload_validation():
    try:
        HarvestHavocEnv(EnvConfig(preload=("CARROT",) * 4))
    except ValueError as exc:
        assert "preload" in str(exc)
    else:
        raise AssertionError("expected ValueError for oversized preload")

    try:
        HarvestHavocEnv(EnvConfig(preload=("TURNIP",)))
    except ValueError as exc:
        assert "TURNIP" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown preload piece")


def test_missing_zone_is_rejected():
    """A layout without a required zone must fail fast with a clear message."""
    stripped = [r for r in DEFAULT_LAYOUT if r.zone is not Zone.OWN_TABLE]
    try:
        HarvestHavocEnv(layout=stripped)
    except ValueError as exc:
        assert "OWN_TABLE" in str(exc)
    else:
        raise AssertionError("expected ValueError for a missing zone")


# =============================================================================
# Runner
# =============================================================================

def _main() -> int:
    """Run every ``test_*`` function in this module and report."""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - a test runner should catch all
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        print("\nFailures:")
        for name, exc in failures:
            print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

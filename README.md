# Harvest Havoc — RL Simulation Environment

A single-robot simulation of **Harvest Havoc**, the 2026 Bunnybots offseason game,
built to answer one question:

> Given a fixed 2:30 match, which **sequence and timing of actions** yields the best
> combination of raw score and ranking points?

This repository contains **only the environment** — no learning algorithm yet, by
design. Everything here is shaped around making time allocation measurable.

---

## Quick start

```bash
python test_strategies.py             # >>> START HERE <<< compare strategies
python test_strategies.py all         # + action times, noise, deep dive, priorities
python demo.py                        # guided tour of the environment
python tests/test_harvest_havoc.py    # 90 tests, no test framework required
```

**[`test_strategies.py`](test_strategies.py) is the file to edit.** Everything
tunable — which strategies to run, reward priorities, shelf capacity, robot
speed, every action's time, how random those times are — sits in one block
marked `EDIT HERE` at the top. Change a number, save, re-run.

```
python test_strategies.py             compare every strategy (the main table)
python test_strategies.py times       what does each action cost, from where?
python test_strategies.py noise       show that durations are sampled, not fixed
python test_strategies.py detail      full time ledger for one strategy
python test_strategies.py priorities  how reward weights reorder the strategies
```

```python
from harvest_havoc import HarvestHavocEnv, CycleAndParkPolicy, rollout

env = HarvestHavocEnv()
result = rollout(env, CycleAndParkPolicy(level=3))
print(result["raw_score"], result["ranking_points"])
print(env.time_allocation_summary())
```

**Requirements:** Python 3.9+ and `numpy`. `gymnasium` is *optional* — if installed,
the env subclasses `gymnasium.Env` and uses real `gymnasium.spaces`; if not, a
duck-typed shim in [`spaces.py`](harvest_havoc/spaces.py) provides the same surface.
Check `harvest_havoc.GYMNASIUM_AVAILABLE`.

---

## The central design decision: time is the state variable

The environment is **event-driven, not fixed-tick**. One `step()` advances the match
clock by exactly as long as the chosen action takes — 0.5 s for a `WAIT`, ~6.3 s for
"drive to the pantry and place a carrot on level 3". There is no control period.

This matters because it makes the agent's decision variable *how to spend time*
rather than *what to do this 20 ms*, and it makes episode length a direct readout of
how many actions the 150-second budget bought.

Four consequences run through the whole codebase:

1. **Time costs are data, not code.** Every duration lives in
   [`TimeConfig`](harvest_havoc/config.py) so you can sweep them, fit them to real
   robot logs, or ask "how much faster does our L3 scorer need to be before L3 beats
   L1?" without touching logic.

2. **Travel time is physical, not tabulated.** Distances come from a multi-source
   Dijkstra over the field grid; durations come from a **trapezoidal velocity
   profile** with real `v_max` and `a_max`:

   ```
   d_ramp = v²/a
   t = 2·√(d/a)        if d < d_ramp   (triangular — never reaches cruise)
   t = d/v + v/a       if d ≥ d_ramp   (trapezoidal)
   ```

   So the model responds correctly to layout and drivetrain changes. It also
   reproduces a real result: at these distances the robot is **acceleration-limited**,
   so doubling top speed from 7 → 14 ft/s buys only ~10 s of the match.

3. **Durations are sampled, not tabulated.** See the next section.

4. **Every second is audited.** Each action appends an
   [`ActionRecord`](harvest_havoc/state.py) to `env.time_ledger`, splitting its
   duration into `travel / align / manipulate / idle` and tagging the points it
   produced. `env.time_allocation_summary()` accounts for all 150 seconds and reports
   points-per-second by category. A test asserts the ledger tiles the timeline
   contiguously with no gaps or overlaps.

---

## Action durations are probabilistic

`python test_strategies.py noise`

Nominal times live in [`TimeConfig`](harvest_havoc/config.py), but nothing ever uses
one directly. [`timing.py`](harvest_havoc/timing.py) multiplies each by a
**right-skewed, mean-one** random factor before it reaches the clock:

```
DURATION DISTRIBUTION   component=manipulate, nominal=2.60s, model=lognormal
  1.85- 2.05s |############
  2.05- 2.25s |#############################
  2.25- 2.45s |#############################################
  2.45- 2.65s |#################################################### <- nominal
  2.65- 2.85s |############################################
  2.85- 3.05s |#############################
  3.05- 3.25s |#################
  3.25- 3.45s |#########
  3.45- 3.65s |####
  3.65- 3.84s |##
mean=2.594s (nominal 2.600s)  cv=0.150  p05=2.02  p50=2.56  p95=3.29
```

The skew is the point. A real action has a **floor** set by physics, most attempts
land near it, and the tail is long and **one-sided** — bobbles, re-grips and bad
approach angles only ever cost more time, never less. A symmetric Gaussian would
wrongly imply that luck saves as much as it costs.

The same action, twenty times, from the same cell:

```
3.02  3.18  3.14  3.63  2.54  3.00  2.76  2.66  2.88  3.54
3.62  3.12  3.21  2.64  3.09  2.87  3.25  3.22  2.93  3.80
min 2.54s   mean 3.11s   max 3.80s
```

Two properties worth knowing:

- **The mean is pinned to the nominal time** (`mu = -sigma^2/2` for the lognormal).
  Turning noise on adds variance *without bias*, so a noisy run stays comparable to
  a deterministic one and a good-on-average strategy does not quietly get slower.
- **Variance is per component**, because they differ genuinely:

  | component | cv | why |
  |---|---|---|
  | `travel` | 0.08 | driving a known path is repeatable |
  | `align` | 0.25 | the least repeatable part of any cycle |
  | `manipulate` | 0.15 | intake/placement, moderately repeatable |
  | `wait` | 0.00 | a decision, not a physical action |

Knobs: `DURATION_MODEL` (`"lognormal"` / `"gamma"` / `"deterministic"`) and
`NOISE_SCALE` (one number scaling all four cvs; `0.0` = fixed times). Draws are
clamped so a tail event cannot eat the match. Everything is seeded, so
`reset(seed=n)` reproduces a match exactly.

**Every action has a time cost.** `python test_strategies.py times` prints all 21,
from any field position:

```
action                kind            travel  align  manip  wait  nominal   mean    p05    p95
MOVE_N                move              0.71   0.00   0.00  0.00     0.71   0.71   0.62   0.80
INTAKE_CARROT         intake            0.00   0.50   1.00  0.00     1.50   1.50   1.11   1.97
SCORE_CARROT_L3       score_pantry      2.59   0.50   2.60  0.00     5.69   5.67   4.64   6.91
SCORE_CAKE_L3         score_pantry      2.59   0.50   3.00  0.00     6.09   6.07   4.95   7.41
OVEN_DEPOSIT_CARROT   oven_deposit      2.45   0.50   1.20  0.00     4.15   4.14   3.42   5.00
PARK                  park              2.12   0.50   0.60  0.00     3.22   3.21   2.66   3.87
```

A `min_action_duration_s` floor guarantees no action is ever free, so the step count
is always a meaningful budget.

---

## Reward priorities are four numbers

`python test_strategies.py priorities`

The full weight set is still available, but you should not have to tune eight
numbers to say "care more about ranking points". Edit these in
[`test_strategies.py`](test_strategies.py):

```python
PRIORITY_SCORE      = 1.0   # value of one raw point
PRIORITY_STOCKED_UP = 1.0   # value of the Stocked Up RP
PRIORITY_BAKED_UP   = 1.0   # value of the Baked Up RP
PRIORITY_DINNER_RP  = 0.7   # value of the Dinner RP
```

Only the *ratios* matter; `0.0` removes an objective entirely. In code:

```python
RewardConfig.from_priorities(score=0.1, stocked_up=3, baked_up=3, dinner_rp=0)
RewardConfig.preset("seeding")   # balanced | score | rp | seeding
                                 # stocked | baked | dinner
```

Each priority sets both the terminal RP bonus and the dense shaping scale (at 40% of
the bonus — shaping only needs to point toward a threshold; making it comparable to
the bonus lets a policy farm partial progress and never close the RP out).

The `priorities` section runs every strategy under every preset and names the winner
per column, so you can see whether your weights are doing real work:

```
strategy                 score   balanced   seeding      rp     baked    dinner
cycle_L3                  81.4      126.2      54.2    50.1       8.1      12.2
rp_rush                   81.9      168.1      95.3    91.2      50.2      11.3
rp_rush_no_dinner         84.5      172.3      97.9    93.6      50.4      13.9
rp_rush_no_reserve        79.8      127.0      57.5    53.5       8.0      15.5
cake_economy              80.1       95.5      24.2    20.2      16.0      10.9
```

---

## Game model

### Match structure
| Phase | Window | Notes |
|---|---|---|
| Autonomous | 0 – 15 s | Leave-kitchen bonus available |
| Teleop | 15 – 130 s | |
| Endgame | 130 – 150 s | Final 20 s; park + harvest haul |

### Scoring
| Action | Points |
|---|---|
| Carrot on pantry L1 / L2 / L3 | 3 / 4 / 5 |
| Carrot cake on pantry L1 / L2 / L3 | 8 / 10 / 12 |
| Either piece into the oven | 2 |
| Fully leave the kitchen during auto (once) | 2 |
| Park in own table zone at the buzzer | 2 |
| Harvest haul, per held piece (max 3) | 1 per carrot, 3 per cake |

Three oven-scored **carrots** trigger a human-player exchange producing one new
carrot cake at the farm depot after a sampled delay (mean 7.5 s, cv 0.35). Cakes put
into the oven score 2 and are consumed — they do *not* feed the exchange, which makes
it a deliberately bad trade the agent should learn to avoid.

### Two hard capacity limits

| Limit | Value |
|---|---|
| Robot inventory | **3 pieces** (carrots + cakes combined) |
| Each pantry shelf | **3 pieces** (carrots + cakes combined) |

The shelf cap is the most consequential rule in the game. **The whole pantry holds
only nine pieces**, so the question stops being "how many pieces can we score?" and
becomes "which nine, and in what order?".

It creates a trap that is easy to walk into and impossible to undo. Pieces can never
be removed, so **filling a shelf with three carrots permanently forfeits Baked Up on
that shelf**. Stocked Up arrives sooner and the second ranking point is gone for the
rest of the match. A test pins this down, and
`RankingPointRushPolicy(reserve_cake_slots=False)` demonstrates it costing a whole RP.

The cap also inverts the scoring arithmetic. Three carrots into the oven score 6 and
yield a cake worth 12 on level 3 — **18 points using one slot**. The same three
carrots placed directly on level 3 score 15 and burn **three slots**. Cakes win on
both points-per-carrot and points-per-slot; they only lose on time.

### Ranking points
| RP | Condition |
|---|---|
| **Stocked Up** | ≥3 pieces (any type) on *every* shelf |
| **Baked Up** | ≥1 carrot cake on *every* shelf |
| **Dinner RP** | ≥10 endgame points |

---

## Documented interpretations

Five points in the rules as given were ambiguous. Rather than silently picking one,
each is an explicit, switchable decision:

**0. Does "max 3 carrots or carrot cakes per shelf" mean 3 total, or 3 of each?**
Modelled as **3 total** (`ScoringConfig.shelf_capacity = 3`), matching the identical
phrasing in the Stocked Up condition — "at least three carrots or carrot cakes on
every shelf" — which reads as a count of pieces regardless of type. That also makes
Stocked Up equivalent to "the pantry is full", which is clean. If it actually means
3 carrots *and* 3 cakes, set `SHELF_CAPACITY = 6`; the env validates that every
ranking point stays reachable and raises if not.

**1. What counts as an "endgame point" for the Dinner RP?**
`ScoringConfig.dinner_rp_mode` selects between:
- `"park_and_haul"` *(default)* — park + harvest haul only. Maximum is `2 + 3×3 = 11`
  against a threshold of 10, so the RP effectively demands **parking while holding
  three cakes**. This matches how FRC manuals normally scope the phrase.
- `"all_in_endgame"` — every point earned at `t ≥ 130 s`, including pantry scoring.
  Far easier, and it produces a very different optimal allocation.

The env raises at construction if the threshold is unreachable, so a
misconfiguration surfaces immediately rather than as a flat learning curve.

**2. What does "fully leaving the kitchen zone" mean when the pantry and oven are
inside the kitchen?** Each cell carries exactly one zone label (as specified), so
pantry/oven cells are labelled `OWN_PANTRY`/`OWN_OVEN`, not `OWN_KITCHEN`. The bonus
therefore tests membership in `KITCHEN_COMPLEX = {kitchen, pantry, oven}`, so driving
from the kitchen into your own pantry does **not** earn it. The robot is a point mass,
so "fully leaving" becomes "the robot's cell is outside the complex".

**3. Is parking an action or a position?** Both. A `PARK` action must complete (paying
`align + park_settle_s`), *and* the robot must actually occupy its own table zone at
the buzzer. Any non-`WAIT` action clears the parked flag, since driving away or
reaching out to score breaks the parked configuration.

**4. Field geometry.** The real field drawings were not available. The layout in
[`zones.py`](harvest_havoc/zones.py) is a plausible, internally consistent 54×27 ft
arrangement chosen so the travel tradeoffs are interesting. **It is plain data** —
replace `DEFAULT_LAYOUT` with surveyed coordinates and nothing else changes.

```
kkkkkkkkkk.TTTTT....................    k  own kitchen
kkkkkkkkkk.TTTTT....................    P  own pantry (3 shelves)
PPkkkkkkkk.TTTTT....................    O  own oven
PPkkkkkkkk..........................    F  own farm / depot
PPkkkkkRkk..........................    T  own table (endgame park)
PPkkkkkkkkFFFFFFFF..................    .  neutral
kkkkkkkkkkFFFFFFFF..................    R  robot
OOkkkkkkkkFFFFFFFF..................
OOkkkkkkkkFFFFFFFF..................    36 × 18 cells @ 1.5 ft
```

---

## Spatial model

`GRID_CELL_SIZE_FT` sits at the top of [`config.py`](harvest_havoc/config.py) and
defaults to **1.5 ft**, giving a 36×18 grid. Values of 1.0–2.0 ft all work; a test
asserts every required zone survives each resolution.

Zone labels use a reserved numbering scheme so the field can grow without
invalidating a trained policy's one-hot encoding:

| Range | Meaning |
|---|---|
| `0` | neutral |
| `1–9` | own alliance (`OWN_KITCHEN`, `OWN_FARM`, `OWN_PANTRY`, `OWN_OVEN`, `OWN_TABLE`) |
| `11–19` | opponent alliance — declared now, unpopulated in v1 |
| `21–29` | reserved for contested/shared zones |

`mirror_layout()` generates the opponent half by reflection, and
`is_own()` / `is_opponent()` / `counterpart()` let game logic be written once per
alliance. An optional `blocked` cell mask is already honoured by navigation.

---

## Action space (hybrid: 21 discrete actions)

| Index | Actions |
|---|---|
| 0–7 | `MOVE_N/S/E/W/NE/NW/SE/SW` — one grid cell |
| 8–9 | `INTAKE_CARROT`, `INTAKE_CAKE` |
| 10–15 | `SCORE_CARROT_L1..L3`, `SCORE_CAKE_L1..L3` |
| 16–17 | `OVEN_DEPOSIT_CARROT`, `OVEN_DEPOSIT_CAKE` |
| 18–19 | `PARK`, `EXIT_KITCHEN` |
| 20 | `WAIT` |

**Micro** actions (`MOVE_*`) step one cell, charged at cruise speed if the robot was
already rolling, otherwise a full accel/decel trapezoid. They exist so positioning is
expressible and travel cost is auditable.

**Macro** actions are *options*: shortest path into the required zone over the same
grid, then align, then **one** manipulation. Both levels share one time model, so a
macro costs exactly the micro actions it replaces plus manipulation overhead —
nothing is double counted, and a macro-only policy replays as a cell-by-cell
trajectory.

Emptying a full robot onto L2 is three `SCORE_CARROT_L2` actions; the 2nd and 3rd are
cheap because travel is already zero. That is what makes hopper batching pay, and it
is worth roughly a third of the match (a test asserts batching beats naive cycling).

`env.action_mask()` returns a boolean legality mask — strongly recommended for
training. `WAIT` is always legal, so the mask is never empty. Illegal actions are
rejected but still burn `wait_duration_s`, guaranteeing episodes terminate.

**Buzzer handling.** An action that cannot finish is cut off. With
`allow_partial_actions` (default), the travel leg still moves the robot as far as
time allows even though the manipulation does not happen — which matters, because
park is evaluated positionally and creeping into the table zone at the buzzer is a
legal play.

---

## Observation (42 named float32 slots)

Every slot has a label in `env.encoder.names`; `env.encoder.as_dict(obs)` zips a
vector back into a readable dict. Run `python demo.py obs` to print the schema.

Contents: clock (remaining fraction, phase one-hot, phase-remaining fraction) ·
robot position + zone one-hot · inventory · per-shelf carrot/cake/**free-slot**
counts ·
**RP progress (3 continuous values)** · oven pipeline and depot state · flags · score.

The clock appears three ways deliberately: an agent deciding "one more cycle, or leave
for the table zone now?" needs absolute urgency, which rules are in force, *and*
distance to the next rule change.

RP progress is a deterministic function of shelf counts — redundant on purpose. It
hands the network the threshold structure instead of making it discover three step
functions.

---

## Reward (multi-objective)

Raw score alone is the wrong signal: it is locally optimal to dump every cake on L3,
which forfeits **Baked Up** outright. So:

```
reward = w_raw_score · Δscore              # dense, honest base
       + [Φ(s') − Φ(s)]                    # RP progress shaping
       + Σ bonus_rp for each RP newly won  # threshold lump sum
       − w_time_penalty · Δt               # optional, default 0
```

`step()` returns the weighted scalar so any standard RL algorithm works unmodified.
But `info` also carries:

- `info["reward_vector"]` — the **unweighted** 5-objective vector
  (`raw_score_delta`, `stocked_up_progress_delta`, `baked_up_progress_delta`,
  `dinner_progress_delta`, `ranking_points_gained`)
- `info["reward_breakdown"]` — the labelled weighted split

so you can re-weight objectives offline, fit a Pareto front, or drop in a true
multi-objective learner **without re-simulating anything**.

Two shaping details worth knowing:

- **Averaging, not minimum.** `stocked_up`/`baked_up` progress averages per-shelf
  progress. The minimum would be flat over most of the state space; averaging gives a
  gradient for filling the 2nd and 3rd shelves before the 1st is done.
- **The Dinner RP potential is time-gated.** It ramps in over the 30 s before endgame.
  Without the gate, dense shaping would pay the agent to squat in the table zone
  hoarding cakes from `t=0` — a terrible strategy.

`RewardConfig.strict_potential_based_shaping` forces `Φ(terminal)=0` for strict
policy-invariance (Ng et al. 1999). It defaults to `False` because the terminal spike
tends to swamp exactly the endgame decisions this project exists to study.

---

## Baseline results

`python test_strategies.py` — 25 sampled matches each, default config:

| strategy | score | RP | stock | bake | dinner | steps | earned |
|---|---|---|---|---|---|---|---|
| rp_rush_no_dinner | **83.8 ± 1.3** | **2.00** | 100% | 100% | 0% | 54 | stocked + baked |
| rp_rush | 81.5 ± 2.0 | **2.00** | 100% | 100% | 0% | 53 | stocked + baked |
| cycle_L1 | 81.3 ± 1.4 | 1.00 | 100% | 0% | 0% | 66 | stocked |
| cycle_L3 | 81.3 ± 1.4 | 1.00 | 100% | 0% | 0% | 66 | stocked |
| cycle_L2 | 81.2 ± 1.4 | 1.00 | 100% | 0% | 0% | 66 | stocked |
| cycle_L3_nopark | 80.0 ± 0.0 | 1.00 | 100% | 0% | 0% | 62 | stocked |
| cake_economy | 80.0 ± 0.9 | 0.00 | 0% | 0% | 0% | 49 | — |
| rp_rush_no_reserve | 79.8 ± 1.7 | 1.00 | 100% | 0% | 0% | 69 | stocked |
| random | 39.2 ± 5.1 | 0.08 | 8% | 0% | 0% | 131 | — |

Four findings, all of them consequences of the shelf cap:

- **Reserving a cake slot per shelf is nearly free and worth a whole RP.**
  `rp_rush` beats `rp_rush_no_reserve` on *both* score and RPs. Chasing Baked Up is
  not a sacrifice — it is strictly better, because a cake in a slot beats a carrot in
  that slot by 7–8 points and the oven carrots it costs score 2 apiece anyway.
- **Which shelf you prefer barely matters any more.** `cycle_L1`, `L2` and `L3` land
  within 0.1 points of each other, because a carrot-cycling robot fills all nine
  slots regardless of preference and then spends the rest of the match at the oven.
  Before the cap, L3 beat L1 by 30%. The interesting question moved from *which
  shelf* to *carrot or cake*.
- **The pantry saturates well before the buzzer.** Every non-random strategy ends
  with a completely full pantry, so ~half the match is spent on 2-point oven
  deposits. That is a strong hint that the real optimum invests much earlier in
  cakes.
- **The Dinner RP looks unreachable.** No strategy earned it. It needs 10 endgame
  points, and the only route under the default reading is parking with three cakes
  (2 + 9 = 11) — 9 oven carrots plus three depot round trips, all inside the last
  30 seconds. `rp_rush` with `chase_dinner_rp=True` scores *worse* than with it off.
  Worth verifying against the real manual before optimising for it.

Two findings from the environment sweeps (`python demo.py sweep`) are unchanged:

- The robot is **acceleration-limited**, not top-speed-limited, at these distances.
- The **park break-off time** is a tight optimum around 4–6 s of lead.

---

## Module map

| Module | Responsibility |
|---|---|
| [`config.py`](harvest_havoc/config.py) | **Every tunable number**, incl. `GRID_CELL_SIZE_FT` |
| [`zones.py`](harvest_havoc/zones.py) | `Zone` enum + declarative field layout |
| [`field.py`](harvest_havoc/field.py) | Grid + Dijkstra navigation |
| [`timing.py`](harvest_havoc/timing.py) | **Probabilistic duration model — all time comes from here** |
| [`state.py`](harvest_havoc/state.py) | Mutable match state + `ActionRecord` ledger entry |
| [`actions.py`](harvest_havoc/actions.py) | Hybrid macro + micro action table |
| [`scoring.py`](harvest_havoc/scoring.py) | Point values, endgame finalisation, ranking points |
| [`observation.py`](harvest_havoc/observation.py) | Named flat observation vector |
| [`reward.py`](harvest_havoc/reward.py) | Multi-objective reward and shaping |
| [`env.py`](harvest_havoc/env.py) | `HarvestHavocEnv` |
| [`render.py`](harvest_havoc/render.py) | ASCII map, scoreboard, ledger tables |
| [`baselines.py`](harvest_havoc/baselines.py) | Scripted policies + rollout driver |
| [`test_strategies.py`](test_strategies.py) | **The strategy test bench — edit this one** |

State is deliberately dumb — it holds no rules — so it is trivial to snapshot, diff,
log, and unit-test. Scoring holds no geometry; the env passes occupancy in.

---

## Not modelled (deliberately, in v1)

Opponents · alliance partners · defence · penalties · the human player as an explicit
agent · robot collisions · piece dropping on a missed shot (a miss costs time, keeps
the piece).

Each has a hook: opponent zones are already numbered and `mirror_layout()` generates
them; `Field(blocked=...)` supports obstacles; the human player is a probabilistic
delay in `OvenState.pending_exchange_times` that could become a scheduled agent.

---

## Suggested next steps

1. **Sweep the baselines harder** — `park_lead_s`, shelf mix, and oven investment are
   all one-line changes, and the answers bound whatever the learner finds.
2. **Train with action masking.** The mask removes an enormous class of wasted
   transitions. Maskable PPO is the natural first algorithm.
3. **Start macro-only.** Mask out indices 0–7 for a ~13-action space and horizons of
   ~50 steps; re-enable micro moves once macro-level allocation is solved.
4. **Use the reward vector.** Train once, then re-weight offline to trace the
   score-vs-RP Pareto front rather than re-running per weighting.
5. **Add failure rates.** Timing noise is already on by default; layer in
   `StochasticConfig.realistic()` (intake 93%, L3 placement 90%) to test whether an
   allocation is *robust* or merely lucky. Use `StochasticConfig.deterministic()`
   when debugging, so a single number is reproducible.

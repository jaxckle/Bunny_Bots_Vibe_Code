"""
Plain-text rendering: field map, scoreboard, and time-ledger tables.

Everything here is presentation only -- no game logic. The ASCII field is
deliberately readable at a glance in a terminal, because the fastest way to
catch a layout mistake is to look at it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .actions import ACTION_SPECS, Action
from .scoring import (
    endgame_points,
    evaluate_ranking_points,
    projected_endgame_points,
    rp_progress,
)
from .state import Phase
from .zones import ZONE_GLYPH, Zone

if TYPE_CHECKING:  # pragma: no cover
    from .env import HarvestHavocEnv

#: Glyph drawn at the robot's cell.
ROBOT_GLYPH = "R"
#: Glyph drawn on blocked cells.
BLOCKED_GLYPH = "#"


def render_field(env: "HarvestHavocEnv") -> str:
    """ASCII map of the field with the robot's position marked.

    Rows are printed with +y at the top so the picture matches a field drawing
    viewed from the own-alliance driver station.
    """
    fld = env.field
    robot = env.state.robot.cell
    lines: List[str] = []
    for cy in range(fld.n_rows - 1, -1, -1):
        row = []
        for cx in range(fld.n_cols):
            if (cx, cy) == robot:
                row.append(ROBOT_GLYPH)
            elif fld.blocked[cx, cy]:
                row.append(BLOCKED_GLYPH)
            else:
                row.append(ZONE_GLYPH.get(fld.zone_at((cx, cy)), "?"))
        lines.append("".join(row))
    return "\n".join(lines)


def render_legend(env: "HarvestHavocEnv") -> str:
    """One-line legend covering only the zones present in this layout."""
    parts = [f"{ROBOT_GLYPH}=robot"]
    for zone in Zone:
        if env.field.has_zone(zone):
            parts.append(f"{ZONE_GLYPH.get(zone, '?')}={zone.name.lower()}")
    return "  ".join(parts)


def _bar(fraction: float, width: int = 10) -> str:
    """Small text progress bar for a value in ``[0, 1]``."""
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render_scoreboard(env: "HarvestHavocEnv") -> str:
    """Clock, inventory, pantry contents, ranking-point progress, and score."""
    st = env.state
    cfg = env.config
    in_table = env.field.zone_at(st.robot.cell) is Zone.OWN_TABLE
    rps = evaluate_ranking_points(st, cfg.scoring, cfg.time)
    prog = rp_progress(st, cfg.scoring, cfg.time, in_table_zone=in_table)
    eg = (endgame_points(st, cfg.scoring, cfg.time) if st.finalized
          else projected_endgame_points(
              st, cfg.scoring, cfg.time, in_table_zone=in_table))

    lines: List[str] = []
    lines.append(
        f"t = {st.t:6.2f}s / {cfg.time.match_duration_s:.0f}s   "
        f"phase = {st.phase(cfg.time).name:<8} "
        f"({st.phase_time_remaining(cfg.time):5.2f}s left in phase)"
    )
    lines.append(
        f"robot  cell={st.robot.cell}  zone={env.field.zone_at(st.robot.cell).name:<12} "
        f"carrots={st.robot.carrots} cakes={st.robot.cakes} "
        f"{'PARKED' if st.robot.parked else ''}"
    )

    shelf_bits = []
    for lvl in st.pantry.levels:
        s = st.pantry.shelves[lvl]
        flag = "FULL" if s.is_full else f"{s.free_slots} free"
        shelf_bits.append(
            f"L{lvl}: {s.carrots}c/{s.cakes}k {s.total}/{s.capacity} [{flag}]"
        )
    lines.append("pantry " + "  ".join(shelf_bits))

    lines.append(
        f"oven   deposited={st.oven.carrots_deposited}carrot/"
        f"{st.oven.cakes_deposited}cake  "
        f"toward_next={st.oven.carrots_toward_next}/"
        f"{cfg.scoring.oven_carrots_per_cake}  "
        f"pending={st.oven.pending_count}  ready_at_depot={st.cakes_available}"
    )

    lines.append(
        f"RP     stocked_up {_bar(prog.stocked_up)} "
        f"{'YES' if rps.stocked_up else ' no'}   "
        f"baked_up {_bar(prog.baked_up)} {'YES' if rps.baked_up else ' no'}   "
        f"dinner {_bar(prog.dinner_rp)} {'YES' if rps.dinner_rp else ' no'} "
        f"({eg}/{cfg.scoring.dinner_rp_threshold} eg pts)"
    )

    cats = st.points_by_category()
    cat_bits = "  ".join(
        f"{k.value}={v}" for k, v in cats.items() if v
    ) or "none yet"
    lines.append(f"score  {st.raw_score} raw   ({cat_bits})")
    lines.append(f"       {rps.total} RP earned")
    return "\n".join(lines)


def render_ascii(env: "HarvestHavocEnv") -> str:
    """Full single-frame render: map, legend, and scoreboard."""
    return "\n".join(
        [
            render_field(env),
            render_legend(env),
            "-" * 60,
            render_scoreboard(env),
        ]
    )


def render_time_ledger(
    env: "HarvestHavocEnv", limit: Optional[int] = None
) -> str:
    """Tabulate the per-action time ledger.

    Parameters
    ----------
    env:
        Environment whose ``time_ledger`` to print.
    limit:
        Show only the first `limit` rows. ``None`` shows all.
    """
    header = (
        f"{'#':>4} {'t_start':>8} {'dur':>6} {'trv':>5} {'aln':>5} "
        f"{'man':>5} {'idle':>5} {'pts':>4} {'phase':<8} {'action':<22} note"
    )
    rows = [header, "-" * len(header)]
    records = env.time_ledger[:limit] if limit else env.time_ledger
    for rec in records:
        flag = ""
        if rec.illegal:
            flag = "ILLEGAL"
        elif rec.truncated_by_buzzer:
            flag = "BUZZER"
        elif not rec.success:
            flag = "FAIL"
        note = f"{flag} {rec.note}".strip()
        rows.append(
            f"{rec.step:>4} {rec.t_start:>8.2f} {rec.duration_s:>6.2f} "
            f"{rec.travel_s:>5.2f} {rec.align_s:>5.2f} {rec.manipulate_s:>5.2f} "
            f"{rec.idle_s:>5.2f} {rec.points:>4} {rec.phase.name:<8} "
            f"{rec.action_name:<22} {note}"
        )
    return "\n".join(rows)


def render_time_allocation(env: "HarvestHavocEnv") -> str:
    """Human-readable form of :meth:`HarvestHavocEnv.time_allocation_summary`.

    This is the table to look at when asking "was that a good use of 150
    seconds?".
    """
    summary = env.time_allocation_summary()
    lines: List[str] = ["TIME ALLOCATION", "=" * 66]

    lines.append(
        f"{'action kind':<16}{'count':>7}{'seconds':>10}{'% match':>9}"
        f"{'points':>8}{'pts/sec':>10}"
    )
    lines.append("-" * 66)
    match_s = summary["totals"]["match_s"]
    for kind, b in sorted(
        summary["by_kind"].items(), key=lambda kv: -kv[1]["seconds"]
    ):
        lines.append(
            f"{kind:<16}{int(b['count']):>7}{b['seconds']:>10.2f}"
            f"{100.0 * b['seconds'] / match_s:>8.1f}%"
            f"{int(b['points']):>8}{b['points_per_second']:>10.3f}"
        )

    comp = summary["by_component"]
    lines.append("-" * 66)
    lines.append(
        "time split      "
        + "  ".join(f"{k}={v:.2f}s" for k, v in comp.items())
    )

    lines.append("-" * 66)
    for phase_name, b in summary["by_phase"].items():
        lines.append(
            f"{phase_name:<16}{int(b['steps']):>7}{b['seconds']:>10.2f}"
            f"{100.0 * b['seconds'] / match_s:>8.1f}%"
            f"{int(b['points']):>8}"
        )

    tot = summary["totals"]
    fail = summary["failures"]
    lines.append("-" * 66)
    lines.append(
        f"steps={tot['steps']}  accounted={tot['accounted_s']:.2f}s  "
        f"unused={tot['unused_s']:.2f}s"
    )
    lines.append(
        f"raw_score={tot['raw_score']}  "
        f"endgame_bonus={tot['unattributed_points']}  "
        f"overall={tot['overall_points_per_second']:.3f} pts/sec"
    )
    lines.append(
        f"illegal={fail['illegal']}  failed={fail['failed']}  "
        f"buzzer_cut={fail['truncated_by_buzzer']}"
    )
    return "\n".join(lines)


def render_action_times(
    env: "HarvestHavocEnv", from_cell=None, samples: int = 2000
) -> str:
    """Table of the time cost of **every** action in the space.

    Confirms at a glance that no action is free, and shows how much spread the
    probabilistic duration model adds. Travel is measured from `from_cell`
    (default: the robot's current cell), so the same table looks different
    from the farm than from the pantry -- which is the whole point.

    The ``p05``/``p95`` columns are sums of per-component quantiles, so read
    them as an indication of spread rather than exact joint quantiles.
    """
    rows = env.action_time_table(from_cell=from_cell, samples=samples)
    cell = from_cell if from_cell is not None else env.state.robot.cell
    zone = env.field.zone_at(cell)

    out = [
        f"ACTION TIME COSTS   (from cell {cell}, zone {zone.name}, "
        f"model={env.timing.model.value}, "
        f"noise_scale={env.config.stochastic.time_noise_scale})",
        "=" * 92,
        f"{'action':<22}{'kind':<14}{'travel':>8}{'align':>7}{'manip':>7}"
        f"{'wait':>6}{'nominal':>9}{'mean':>7}{'p05':>7}{'p95':>7}{'legal':>7}",
        "-" * 92,
    ]

    def fmt(value: float) -> str:
        return "  inf" if value == float("inf") else f"{value:.2f}"

    for r in rows:
        out.append(
            f"{r['name']:<22}{r['kind']:<14}{fmt(r['travel']):>8}"
            f"{fmt(r['align']):>7}{fmt(r['manipulate']):>7}{fmt(r['wait']):>6}"
            f"{fmt(r['nominal_total']):>9}{fmt(r['mean']):>7}"
            f"{fmt(r['p05']):>7}{fmt(r['p95']):>7}"
            f"{('yes' if r['legal'] else 'no'):>7}"
        )
    out.append("-" * 92)
    out.append(
        "Every action costs time. Macro actions cost travel + align + "
        "manipulate;\nmoves cost travel only; WAIT costs its fixed duration. "
        "Travel is zero when\nthe robot is already in the target zone -- which "
        "is what makes batching pay."
    )
    return "\n".join(out)


def render_duration_distribution(
    env: "HarvestHavocEnv", nominal_s: float = 2.0,
    component: str = "manipulate", width: int = 52, samples: int = 20000,
) -> str:
    """ASCII histogram of the sampled duration for one nominal time.

    Makes the right-skew visible: the mode sits at or just below the nominal
    time and the tail runs long, because bobbles cost time and never save it.
    """
    import numpy as np

    saved = env.timing.rng
    env.timing.rng = np.random.default_rng(12345)
    try:
        draws = np.array(
            [env.timing.sample(nominal_s, component) for _ in range(samples)]
        )
    finally:
        env.timing.rng = saved

    stats = env.timing.describe(nominal_s, component)
    lo, hi = float(draws.min()), float(draws.max())
    bins = 18
    counts, edges = np.histogram(draws, bins=bins, range=(lo, hi))
    peak = max(1, counts.max())

    out = [
        f"DURATION DISTRIBUTION   component={component}, "
        f"nominal={nominal_s:.2f}s, model={env.timing.model.value}",
        "-" * (width + 22),
    ]
    for count, left, right in zip(counts, edges, edges[1:]):
        bar = "#" * int(round(width * count / peak))
        marker = " <- nominal" if left <= nominal_s < right else ""
        out.append(f"{left:6.2f}-{right:5.2f}s |{bar}{marker}")
    out.append("-" * (width + 22))
    out.append(
        f"mean={stats['mean']:.3f}s (nominal {stats['nominal']:.3f}s)  "
        f"cv={stats['cv']:.3f}  p05={stats['p05']:.2f}  "
        f"p50={stats['p50']:.2f}  p95={stats['p95']:.2f}"
    )
    out.append(
        "The mean is pinned to the nominal time, so turning noise on adds "
        "variance\nwithout secretly making the robot slower."
    )
    return "\n".join(out)


__all__ = [
    "render_ascii",
    "render_field",
    "render_legend",
    "render_scoreboard",
    "render_time_ledger",
    "render_time_allocation",
    "render_action_times",
    "render_duration_distribution",
]

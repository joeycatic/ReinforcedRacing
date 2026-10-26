"""Render the README learning curve as a dependency-free, theme-aware SVG."""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

WIDTH = 760
LEFT, RIGHT = 64, 120
PLOT_W = WIDTH - LEFT - RIGHT
PANEL_H = 170
TOP_A = 64
TOP_B = TOP_A + PANEL_H + 84
HEIGHT = TOP_B + PANEL_H + 56
SMOOTH = 15

STYLE = """
svg { --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #8a8984;
  --grid: #e6e5e1; --shade: #f1f0ec; --s1: #2a78d6; --s2: #eb6834; }
@media (prefers-color-scheme: dark) {
  svg { --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #8f8e88;
    --grid: #33332f; --shade: #242422; --s1: #3987e5; --s2: #d95926; }
}
text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: var(--ink-2); }
.title { font-size: 15px; font-weight: 600; fill: var(--ink); }
.sub { font-size: 12px; }
.tick { font-size: 11px; fill: var(--muted); }
.label { font-size: 12px; font-weight: 600; fill: var(--ink); }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--muted); stroke-width: 1; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--surface); stroke-width: 2; }
"""


def load_training(run_dir: Path) -> tuple[list[tuple[int, float]], int]:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    switch = next(row["step"] for row in rows if row["stage"] != rows[0]["stage"])
    points = []
    for i, row in enumerate(rows):
        window = rows[max(0, i - SMOOTH + 1) : i + 1]
        points.append(
            (row["step"], sum(r["rollout/episodic_return"] for r in window) / len(window))
        )
    return points, switch


def load_snapshots(summary: Path) -> list[tuple[int, float, float]]:
    return [
        (
            snap["global_step"],
            100 * snap["showcase"]["summary"]["lap_completion_rate"],
            100 * snap["held_out"]["summary"]["lap_completion_rate"],
        )
        for snap in json.loads(summary.read_text(encoding="utf-8"))
    ]


def render(
    training: list[tuple[int, float]], switch: int, snapshots: list[tuple[int, float, float]]
) -> str:
    max_step = max(step for step, _ in training)
    max_return = 50 * (int(max(v for _, v in training)) // 50 + 1)

    def x(step: float) -> float:
        return LEFT + PLOT_W * step / max_step

    def y(value: float, top: int, high: float) -> float:
        return top + PANEL_H * (1 - value / high)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        'aria-label="PPO learning curve: training return and lap completion by training step">',
        f"<style>{STYLE}</style>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="8" fill="var(--surface)"/>',
    ]

    def panel(top: int, high: float, ticks: list[float], suffix: str, title: str, sub: str) -> None:
        out.append(f'<text class="title" x="{LEFT}" y="{top - 34}">{escape(title)}</text>')
        out.append(f'<text class="sub" x="{LEFT}" y="{top - 16}">{escape(sub)}</text>')
        out.append(
            f'<rect x="{x(0):.1f}" y="{top}" width="{x(switch) - x(0):.1f}" '
            f'height="{PANEL_H}" fill="var(--shade)"/>'
        )
        for tick in ticks:
            ty = y(tick, top, high)
            out.append(
                f'<line class="grid" x1="{LEFT}" x2="{LEFT + PLOT_W}" y1="{ty:.1f}" y2="{ty:.1f}"/>'
            )
            out.append(
                f'<text class="tick" x="{LEFT - 8}" y="{ty + 4:.1f}" text-anchor="end">'
                f"{tick:g}{suffix}</text>"
            )
        base = top + PANEL_H
        out.append(f'<line class="axis" x1="{LEFT}" x2="{LEFT + PLOT_W}" y1="{base}" y2="{base}"/>')
        for million in range(0, max_step // 1_000_000 + 1):
            out.append(
                f'<text class="tick" x="{x(million * 1_000_000):.1f}" y="{base + 18}" '
                f'text-anchor="middle">{million}M</text>'
            )

    panel(
        TOP_A,
        max_return,
        list(range(0, max_return + 1, 50)),
        "",
        "Training return",
        f"Mean episodic return per rollout, {SMOOTH}-update moving average. "
        "Shaded: single-track curriculum.",
    )
    out.append(
        f'<text class="tick" x="{x(switch / 2):.1f}" y="{TOP_A + 16}" text-anchor="middle">'
        "one track</text>"
    )
    out.append(f'<text class="tick" x="{x(switch) + 8:.1f}" y="{TOP_A + 16}">200-track pool</text>')
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x(s):.1f},{y(v, TOP_A, max_return):.1f}"
        for i, (s, v) in enumerate(training)
    )
    out.append(f'<path class="line" stroke="var(--s1)" d="{path}"/>')

    panel(
        TOP_B,
        100,
        [0, 25, 50, 75, 100],
        "%",
        "Lap completion by snapshot",
        "Stochastic policy. Showcase: 50 trials × 3 tracks. Held-out: 5 trials × 50 unseen tracks.",
    )
    series = [("Showcase tracks", "var(--s1)", 1), ("Held-out tracks", "var(--s2)", 2)]
    for name, color, column in series:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(s[0]):.1f},{y(s[column], TOP_B, 100):.1f}"
            for i, s in enumerate(snapshots)
        )
        out.append(f'<path class="line" stroke="{color}" d="{path}"/>')
        for snap in snapshots:
            out.append(
                f'<circle class="dot" fill="{color}" cx="{x(snap[0]):.1f}" '
                f'cy="{y(snap[column], TOP_B, 100):.1f}" r="5">'
                f"<title>{escape(name)} · {snap[0]:,} steps · {snap[column]:.1f}%</title></circle>"
            )
        last = snapshots[-1]
        offset = -6 if column == 1 else 14
        out.append(
            f'<text class="label" x="{x(last[0]) + 12:.1f}" '
            f'y="{y(last[column], TOP_B, 100) + offset:.1f}">'
            f'{last[column]:.1f}% <tspan class="sub" style="font-weight:400">'
            f"{escape(name.split()[0].lower())}</tspan></text>"
        )
    out.append(
        f'<text class="tick" x="{LEFT + PLOT_W}" y="{HEIGHT - 12}" text-anchor="end">'
        "Agent steps (seed 2)</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/training/m3-seed2"))
    parser.add_argument(
        "--evaluation", type=Path, default=Path("artifacts/evaluation/m4/summary.json")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/images/learning-curve.svg"))
    args = parser.parse_args()
    training, switch = load_training(args.run_dir)
    args.out.write_text(render(training, switch, load_snapshots(args.evaluation)), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

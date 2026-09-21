#!/usr/bin/env python3
"""
Extended PMJ capture map.

Top row    : propagating capture (orthodromic | antidromic)
Bottom row : junctional capture  (orthodromic | antidromic)

Both rows use exactly the same plot style: green = captured, red = not captured,
circles = coarse c_PMJ sweep, diamonds = fine (refined) sweep, a grey vertical tick
marks the estimated capture threshold (geometric mean of the last "no" and first
"yes" c_PMJ scale in that row). The x-axis is c_PMJ in multiples of the base value
implied by the JSON (c_pmj / c_pmj_scale, currently 5 mS).

Input  : <script dir>/outputs/exp4_pmj_results.json
Output : <script dir>/outputs/exp4_pmj_capture_map_extended.pdf  (vector, bold serif text)

Usage:
    python pmj_capture_extended.py
    python pmj_capture_extended.py --json path/to/results.json --out fig.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_JSON = HERE / "outputs" / "exp4_pmj_results.json"
DEFAULT_OUT = HERE / "outputs" / "exp4_pmj_capture_map_extended.pdf"

GREEN, RED = "#2ca02c", "#d62728"

# Bold serif text everywhere (titles, labels, ticks, and math text such as c_PMJ, 10^-3).
# STIXGeneral ships with matplotlib; the others are used first if installed.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "mathtext.fontset": "stix",
    "mathtext.default": "bf",
    "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable in PDF
})

# junction_capture_fraction is 0.0 or 1.0 in the current data; a junction counts
# as "captured" if at least this fraction of junctions were captured.
JUNCTION_CAPTURE_THRESHOLD = 0.5

DIRECTIONS = ("orthodromic", "antidromic")

# (title keyword, function record -> bool)
CAPTURE_KINDS = (
    ("propagating", lambda r: bool(r["propagating_capture"])),
    ("junctional", lambda r: r["junction_capture_fraction"] >= JUNCTION_CAPTURE_THRESHOLD),
)


def load_results(path):
    """json.load accepts the NaN literals that appear in pmj_delay_ms."""
    with open(path) as f:
        return json.load(f)


def threshold_between(scales, captured):
    """Geometric mean of the last 'no' before the first 'yes' (None if not bracketed)."""
    for i, ok in enumerate(captured):
        if ok:
            return None if i == 0 else float(np.sqrt(scales[i - 1] * scales[i]))
    return None


def draw_panel(ax, records, direction, n_values, is_captured, kind, base_c):
    for row, n in enumerate(n_values):
        rows = sorted(
            (r for r in records if r["direction"] == direction and r["n_pmj"] == n),
            key=lambda r: r["c_pmj_scale"],
        )
        if not rows:
            continue
        scales = np.array([r["c_pmj_scale"] for r in rows])
        flags = [is_captured(r) for r in rows]

        # capture threshold tick (drawn first so markers sit on top)
        thr = threshold_between(scales, flags)
        if thr is not None:
            ax.plot([thr, thr], [row - 0.4, row + 0.4], color="0.4", lw=1.2, zorder=2)

        for r, ok in zip(rows, flags):
            fine = r["phase"] == "fine"
            ax.scatter(
                r["c_pmj_scale"], row,
                marker="D" if fine else "o",
                s=32 if fine else 70,
                c=GREEN if ok else RED,
                edgecolors="black", linewidths=0.5, zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlim(10 ** -4.2, 10 ** 0.2)
    ax.set_ylim(-0.5, len(n_values) - 0.5)
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels(n_values)
    ax.set_xlabel(rf"$c_{{PMJ}}$ (× {base_c:g} mS)")
    ax.set_ylabel(r"$n_{PMJ}$")
    ax.grid(True, which="both", color="0.9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{direction.capitalize()}: {kind} capture "
        "(green=yes, red=no; circle=coarse, diamond=fine)",
        fontsize=11,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="results JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output image")
    args = ap.parse_args()

    data = load_results(args.json)
    records = data["results"]
    n_values = sorted(data.get("n_pmj_values") or {r["n_pmj"] for r in records})
    # c_PMJ (mS) that corresponds to a scale factor of 1.0
    base_c = records[0]["c_pmj"] / records[0]["c_pmj_scale"]

    fig, axes = plt.subplots(
        len(CAPTURE_KINDS), len(DIRECTIONS), figsize=(14, 4.5 * len(CAPTURE_KINDS)),
        constrained_layout=True,
    )
    for i, (kind, is_captured) in enumerate(CAPTURE_KINDS):
        for j, direction in enumerate(DIRECTIONS):
            draw_panel(axes[i, j], records, direction, n_values, is_captured, kind, base_c)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

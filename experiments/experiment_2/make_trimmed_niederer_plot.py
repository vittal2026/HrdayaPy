#!/usr/bin/env python3
"""
make_trimmed_niederer_plot.py
===================================================================
Produces the trimmed (single-panel, P1-P9 only) Niederer comparison
figure from the already-saved exp2_niederer_results.json -- no solver
calls, no re-running any simulations.

Run this FROM the experiment_2_an directory (the one that CONTAINS
outputs_niederer/), e.g.:

    cd "C:\\Users\\User\\Desktop\\HrdayaPy manuscript\\experiments\\experiment_2_an"
    py make_trimmed_niederer_plot.py

If you run it from *inside* outputs_niederer instead, either move this
file there and change OUT_DIR below to Path(".") , or keep OUT_DIR as
is but cd back out one level first. The path is relative to wherever
you launch `py` from -- not to where this file lives on disk.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs_niederer")
results_path = OUT_DIR / "exp2_niederer_results.json"

POINT_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]


def make_comparison_plot(all_results: dict, out_path: Path):
    dx_levels = sorted(all_results.keys(), reverse=True)
    colors = {0.1: "C3", 0.2: "C2", 0.5: "C0"}

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(POINT_ORDER))
    width = 0.8 / (len(dx_levels) + 1)
    for i, dx_m in enumerate(dx_levels):
        comparison = all_results[dx_m]["comparison"]
        ref_mean = [comparison[p]["ref"]["mean_ms"] for p in POINT_ORDER]
        ref_sd   = [comparison[p]["ref"]["sd_ms"]   for p in POINT_ORDER]
        ours     = [comparison[p]["ours_ms"]        for p in POINT_ORDER]
        c = colors.get(dx_m, f"C{i+4}")
        offset = (i - (len(dx_levels)-1)/2) * width
        ax.errorbar(x + offset, ref_mean, yerr=ref_sd, fmt="s", color=c,
                    alpha=0.5, capsize=2, label=f"literature, dx={dx_m}mm")
        ax.plot(x + offset, ours, marker="_", linestyle="none", color="black",
                 markersize=14, markeredgewidth=2, zorder=5, label=f"ours, dx={dx_m}mm")
    ax.set_xticks(x); ax.set_xticklabels(POINT_ORDER)
    ax.set_ylabel("Activation time (ms)")
    ax.set_title("Point-wise activation times, P1-P9")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not results_path.exists():
        raise SystemExit(
            f"Could not find {results_path.resolve()}\n"
            f"This script expects to be run from the directory that CONTAINS "
            f"'outputs_niederer/', not from inside it. Check your current "
            f"directory with `import os; os.getcwd()` if unsure."
        )

    with open(results_path) as f:
        saved = json.load(f)

    all_results = {}
    for level in saved["levels"]:
        dx_m = float(level["dx_m"])
        all_results[dx_m] = dict(
            point_times=level["point_times"],
            comparison=level["comparison"],
        )

    out_path = OUT_DIR / "exp2_niederer_comparison_trimmed.png"
    make_comparison_plot(all_results, out_path)
    print(f"Saved trimmed figure to {out_path.resolve()}")


if __name__ == "__main__":
    main()

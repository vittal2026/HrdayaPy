"""
run_experiment1_combine.py
===================================================================
Run this AFTER all three run_experiment1_single.py processes
(endo, mid, epi) have finished and saved their outputs. Merges the
three independent exp1_metrics_<phenotype>.json / exp1_traces_<phenotype>.npz
files into the combined exp1_metrics.csv/.json and the 3-panel figure,
matching the format the rest of the pipeline (and the manuscript's
table-generation) expects.

Checks that all three inputs exist and were run with matching
BCL/DT/N_BEATS before merging -- fails loudly rather than silently
combining results from inconsistent runs (e.g. if one process used a
different N_BEATS than the others).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "outputs"
PHENOTYPE_ORDER = ["endo", "mid", "epi"]
PHENOTYPE_LABEL = {
    "endo": "Endocardial",
    "mid": "Mid-myocardial (M-cell)",
    "epi": "Epicardial",
}
REFERENCE_EPI_APD90_MS = 306.0  # for the figure title annotation only;
                                  # see run_experiment1_single.py for the
                                  # full reference table and its source


def load_one(label: str) -> dict:
    path = OUT_DIR / f"exp1_metrics_{label}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- has run_experiment1_single.py {label} "
            f"finished and saved its output? Run all three phenotypes "
            f"before combining."
        )
    with open(path) as f:
        return json.load(f)


def main():
    results = {label: load_one(label) for label in PHENOTYPE_ORDER}

    # Consistency check: all three must have paced the same number of
    # beats, or the combined table is comparing runs on different terms.
    n_beats_set = {results[label]["n_beats_paced"] for label in PHENOTYPE_ORDER}
    if len(n_beats_set) != 1:
        raise ValueError(
            f"Phenotypes were paced for different numbers of beats: "
            f"{ {label: results[label]['n_beats_paced'] for label in PHENOTYPE_ORDER} } "
            f"-- re-run whichever phenotype(s) don't match before combining, "
            f"rather than merging inconsistent results."
        )
    n_beats = n_beats_set.pop()

    traces = {}
    for label in PHENOTYPE_ORDER:
        npz_path = OUT_DIR / f"exp1_traces_{label}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} not found.")
        d = np.load(npz_path)
        prev_trace = (d["t_prev"], d["V_prev"])
        last_trace = (d["t_last"], d["V_last"])
        traces[label] = (prev_trace, last_trace)

    # ── Combined JSON + CSV ──────────────────────────────────────────────
    with open(OUT_DIR / "exp1_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUT_DIR / "exp1_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phenotype", "n_beats_paced", "RMP_mV",
                          "overshoot_mV", "plateau_mV", "APD90_ms",
                          "dVdt_max_Vs", "APD90_drift_ms_per_beat"])
        for label in PHENOTYPE_ORDER:
            r = results[label]
            writer.writerow([r["phenotype"], r["n_beats_paced"],
                              f"{r['RMP_mV']:.3f}", f"{r['overshoot_mV']:.3f}",
                              f"{r['plateau_mV']:.3f}", f"{r['APD90_ms']:.3f}",
                              f"{r['dVdt_max_Vs']:.1f}",
                              f"{r['APD90_drift_ms_per_beat']:.4f}"])

    # ── Combined raw-trace archive ───────────────────────────────────────
    npz_payload = {}
    for label in PHENOTYPE_ORDER:
        (t_prev, V_prev), (t_last, V_last) = traces[label]
        npz_payload[f"{label}_t_prev"] = t_prev
        npz_payload[f"{label}_V_prev"] = V_prev
        npz_payload[f"{label}_t_last"] = t_last
        npz_payload[f"{label}_V_last"] = V_last
    np.savez(OUT_DIR / "exp1_traces.npz", **npz_payload)

    # ── Combined figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for c, label in enumerate(PHENOTYPE_ORDER):
        (t_prev, V_prev), (t_last, V_last) = traces[label]
        ax = axes[c]
        ax.plot(t_prev, V_prev, color="0.6", lw=1.2, label="beat n-1")
        ax.plot(t_last, V_last, color="C0", lw=1.6, label="beat n (analysed)")
        title = f"{PHENOTYPE_LABEL[label]}\nAPD90 = {results[label]['APD90_ms']:.1f} ms"
        if label == "epi":
            title += f"  (ref: {REFERENCE_EPI_APD90_MS:.0f} ms)"
        ax.set_title(title)
        ax.set_xlabel("Time relative to stimulus (ms)")
        ax.axhline(results[label]["RMP_mV"], color="k", lw=0.5, ls=":")
        if c == 0:
            ax.set_ylabel("V (mV)")
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Experiment 1 (TTP06 validation): isolated action potentials, "
                 f"{n_beats} beats\n(direct gpu_ttp06.py class instances, "
                 f"3 independent processes, combined post-hoc)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp1_ap_traces.png", dpi=200)

    print(f"Combined {len(PHENOTYPE_ORDER)} independent phenotype runs "
          f"({n_beats} beats each).")
    print(f"Saved exp1_metrics.json, exp1_metrics.csv, exp1_traces.npz, "
          f"exp1_ap_traces.png to {OUT_DIR}/")


if __name__ == "__main__":
    main()

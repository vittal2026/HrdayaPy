"""
run_experiment1_single.py
===================================================================
Paces ONE phenotype (endo / mid / epi) and saves its own result
immediately on completion. Launch three of these as independent OS
processes to parallelise across CPU cores -- each is fully
self-contained and doesn't wait on, or share state with, the others.

Usage (run in three separate terminals, or as background jobs):
    py run_experiment1_single.py endo
    py run_experiment1_single.py mid
    py run_experiment1_single.py epi

Once all three have finished, run run_experiment1_combine.py to merge
their outputs into the combined table/figure.

Why per-process, not per-thread: this is explicit forward-Euler ODE
integration -- step N depends on step N-1, so there is no parallelism
available WITHIN one phenotype's timeline, only ACROSS the three
independent phenotypes. Separate OS processes get genuine, GIL-free
parallelism across CPU cores for that; threads would not (Python's GIL
serialises pure-Python/PyTorch-CPU-eager code across threads on a
single process).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hrdayapy.simulation.functions.ionic_models.gpu_ttp06 import (
    EndoTTP06, MidTTP06, EpiTTP06,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")   # GPU not expected to help at N=1 -- see
                                # run_experiment1.py's original device note
DTYPE = torch.float64

BCL = 1000.0    # ms -- ten Tusscher & Panfilov (2006), 1 Hz pacing
DT = 0.02       # ms -- their reported single-cell timestep, exactly
STIM_DUR = 1.0  # ms
STIM_AMP_IONIC = 52.0  # pA/pF
N_BEATS = 100
DRIFT_WINDOW_BEATS = 10
V0 = -86.709    # mV, standard CellML TTP06 resting initial condition
PLATEAU_OFFSET_MS = 100.0  # see run_experiment1.py's note: this is an
                            # assumed proxy, not the paper's exact method

N_STEPS_PER_BEAT = int(round(BCL / DT))
STIM_STEPS = int(round(STIM_DUR / DT))

PHENOTYPE_CLASSES = {"endo": EndoTTP06, "mid": MidTTP06, "epi": EpiTTP06}
PHENOTYPE_LABEL = {
    "endo": "Endocardial",
    "mid": "Mid-myocardial (M-cell)",
    "epi": "Epicardial",
}

# Only epicardial has a published reference in ten Tusscher & Panfilov
# (2006)'s main text -- do not fabricate endo/mid comparators.
REFERENCE_EPI = dict(
    RMP_mV=-85.8, plateau_mV=24.9, dVdt_max_Vs=289.0, APD90_ms=306.0,
    source="ten Tusscher & Panfilov 2006, Results text (Fig. 2, steady-state 1Hz epicardial AP)",
)


def apd90_and_upstroke(t, V):
    dVdt = np.diff(V) / np.diff(t)
    i_up = int(np.argmax(dVdt))
    dvdt_max = float(dVdt[i_up])
    t_up = t[i_up]

    rmp = float(V[0])
    peak = float(V.max())
    i_peak = int(V.argmax())
    target = rmp + 0.1 * (peak - rmp)

    apd90 = None
    for k in range(i_peak, len(V)):
        if V[k] <= target:
            apd90 = float(t[k] - t_up)
            break
    if apd90 is None:
        apd90 = float(t[-1] - t_up)

    plateau_idx = int(np.searchsorted(t, t_up + PLATEAU_OFFSET_MS))
    plateau_mV = float(V[plateau_idx]) if plateau_idx < len(V) else float("nan")

    return dict(RMP_mV=rmp, overshoot_mV=peak, plateau_mV=plateau_mV,
                APD90_ms=apd90, dVdt_max_Vs=dvdt_max)


def compare_to_reference(metrics: dict, reference: dict) -> dict:
    comparison = {}
    for key in ["RMP_mV", "plateau_mV", "dVdt_max_Vs", "APD90_ms"]:
        if key not in reference:
            continue
        sim_val = metrics[key]
        ref_val = reference[key]
        gap = sim_val - ref_val
        pct = 100.0 * gap / ref_val if ref_val != 0 else float("nan")
        comparison[key] = dict(simulated=sim_val, reference=ref_val,
                                gap=gap, gap_pct=pct)
    return comparison


def calibrate(phenotype_key: str, n_calib_steps: int = 2000) -> float:
    cls = PHENOTYPE_CLASSES[phenotype_key]
    model = cls()
    state = model.init_state_tensor(1, DEVICE, DTYPE)
    V = torch.full((1,), V0, dtype=DTYPE, device=DEVICE)
    t0 = time.time()
    for k in range(n_calib_steps):
        I_stim = STIM_AMP_IONIC if k < STIM_STEPS else 0.0
        I_ion, state = model.step(V, state, DT, I_stim)
        V = V + DT * (I_stim - I_ion)
    return n_calib_steps / (time.time() - t0)


def pace_phenotype(phenotype_key: str, n_beats: int, progress_every: int = 10):
    if n_beats <= DRIFT_WINDOW_BEATS:
        raise ValueError(
            f"N_BEATS ({n_beats}) must be > DRIFT_WINDOW_BEATS "
            f"({DRIFT_WINDOW_BEATS}) to compute residual drift. "
            f"If you're doing a quick smoke test, lower "
            f"DRIFT_WINDOW_BEATS too, not just N_BEATS."
        )
    cls = PHENOTYPE_CLASSES[phenotype_key]
    model = cls()
    state = model.init_state_tensor(1, DEVICE, DTYPE)
    V = torch.full((1,), V0, dtype=DTYPE, device=DEVICE)

    apd90_history = []
    prev_beat_trace = None
    last_beat_trace = None
    t0 = time.time()

    for beat in range(1, n_beats + 1):
        Vs = np.empty(N_STEPS_PER_BEAT, dtype=np.float64)
        for k in range(N_STEPS_PER_BEAT):
            I_stim = STIM_AMP_IONIC if k < STIM_STEPS else 0.0
            I_ion, state = model.step(V, state, DT, I_stim)
            V = V + DT * (I_stim - I_ion)
            Vs[k] = V.item()
        t = np.arange(N_STEPS_PER_BEAT) * DT
        m = apd90_and_upstroke(t, Vs)
        apd90_history.append(m["APD90_ms"])

        if beat == n_beats - 1:
            prev_beat_trace = (t.copy(), Vs.copy())
        if beat == n_beats:
            last_beat_trace = (t.copy(), Vs.copy())

        if beat % progress_every == 0 or beat == n_beats:
            elapsed = time.time() - t0
            eta = elapsed / beat * (n_beats - beat)
            print(f"  [{phenotype_key}] beat {beat}/{n_beats}  "
                  f"APD90={m['APD90_ms']:.2f}ms  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    drift = (apd90_history[-1] - apd90_history[-1 - DRIFT_WINDOW_BEATS]) / DRIFT_WINDOW_BEATS
    t_last, V_last = last_beat_trace
    final_metrics = apd90_and_upstroke(t_last, V_last)
    final_metrics["APD90_drift_ms_per_beat"] = drift
    return final_metrics, prev_beat_trace, last_beat_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phenotype", choices=["endo", "mid", "epi"])
    args = parser.parse_args()
    label = args.phenotype

    t_start = time.time()
    print(f"=== Phenotype: {PHENOTYPE_LABEL[label]} "
          f"({PHENOTYPE_CLASSES[label].__name__}) on {DEVICE} ===")

    print("Calibrating ...")
    rate = calibrate(label, n_calib_steps=2000)
    total_steps = N_STEPS_PER_BEAT * N_BEATS
    est_seconds = total_steps / rate
    print(f"  Measured rate: {rate:.1f} steps/s")
    print(f"  ESTIMATED RUNTIME for this phenotype alone: "
          f"{est_seconds/60:.1f} min ({est_seconds/3600:.2f} h)")

    m, prev_trace, last_trace = pace_phenotype(label, N_BEATS)

    line = (f"{label:>4s}: beat={N_BEATS:3d}  "
            f"RMP={m['RMP_mV']:7.2f} mV  "
            f"overshoot={m['overshoot_mV']:6.2f} mV  "
            f"plateau={m['plateau_mV']:6.2f} mV  "
            f"APD90={m['APD90_ms']:6.2f} ms  "
            f"dV/dt_max={m['dVdt_max_Vs']:8.1f} V/s  "
            f"residual drift={m['APD90_drift_ms_per_beat']:+.4f} ms/beat")

    result = dict(phenotype=label, n_beats_paced=N_BEATS, **m)
    if label == "epi":
        cmp = compare_to_reference(m, REFERENCE_EPI)
        result["reference_comparison"] = cmp
        line += "\n      vs. ten Tusscher & Panfilov (2006), epicardial:"
        for key, c in cmp.items():
            line += (f"\n        {key:12s} sim={c['simulated']:8.2f}  "
                     f"ref={c['reference']:8.2f}  "
                     f"gap={c['gap']:+7.2f} ({c['gap_pct']:+.2f}%)")
    print(line)

    # ── Save THIS phenotype's result immediately -- does not wait on,
    # or depend on, the other two processes. ────────────────────────────
    with open(OUT_DIR / f"exp1_metrics_{label}.json", "w") as f:
        json.dump(result, f, indent=2)

    t_prev, V_prev = prev_trace
    t_last, V_last = last_trace
    np.savez(OUT_DIR / f"exp1_traces_{label}.npz",
              t_prev=t_prev, V_prev=V_prev, t_last=t_last, V_last=V_last)

    print(f"\nSaved {OUT_DIR / f'exp1_metrics_{label}.json'} and "
          f"{OUT_DIR / f'exp1_traces_{label}.npz'}")
    print(f"Total runtime for this phenotype: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()

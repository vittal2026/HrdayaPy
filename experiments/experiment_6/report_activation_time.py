"""
report_activation_time.py
==========================
Total ventricular activation time (beat 1) from a saved activation_maps.npz,
computed as last_activation - first_activation over myocardial nodes that
actually activated on that beat.

CHANGE from previous version: results are now also saved to a JSON metrics
file (in addition to the existing printout), so the numbers going into the
paper's timing table/figure caption come from a saved artifact rather than
being copied off the terminal by hand. The activation-time calculation
itself is unchanged.

Usage
-----
    python report_activation_time.py path/to/P001_activation_maps.npz
    python report_activation_time.py path/to/P001_activation_maps.npz --event 1
    python report_activation_time.py path/to/P001_activation_maps.npz --no-save
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import hrdayapy as hp

# =============================================================================
# Inputs -- edit to match run_simulation.py for this patient (same convention
# as 12_lead_ECG.py / run_simulation.py: everything defaults off PATIENT_ID,
# no path needs to be typed on the command line for the standard case).
# =============================================================================

PATIENT_ID = "P001"
EVENT = 1

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "outputs" / PATIENT_ID
OUT        = str(OUT_DIR)

DEFAULT_ACTIVATION_MAPS_PATH = f"{OUT}/{PATIENT_ID}_activation_maps.npz"


def report(activation_maps_path: str, event: int = 1,
           save_path: str | None = None) -> dict:
    maps = hp.simulation.load_activation_maps(activation_maps_path)

    a1 = maps.activation_map(event=event)     # (N_myo,), NaN = never activated
    n_never = int(np.isnan(a1).sum())

    if n_never == maps.N_myo:
        print(f"No nodes activated on beat {event} -- check act_threshold "
              f"({maps.act_threshold:g} mV) and that this is the right beat.")
        metrics = {
            "activation_maps_path": str(activation_maps_path),
            "event": event,
            "act_threshold_mV": float(maps.act_threshold),
            "N_myo": int(maps.N_myo),
            "n_never_activated": n_never,
            "first_activation_ms": None,
            "last_activation_ms": None,
            "total_activation_time_ms": None,
            "status": "no_activation",
        }
        if save_path:
            _save_json(metrics, save_path)
        return metrics

    first_activation = float(np.nanmin(a1))
    last_activation  = float(np.nanmax(a1))
    total_activation_time = last_activation - first_activation

    print(f"Activation maps : {activation_maps_path}")
    print(f"Beat            : {event}")
    print(f"Act threshold   : {maps.act_threshold:g} mV")
    print(f"N_myo           : {maps.N_myo:,}")
    print(f"Never activated : {n_never:,} / {maps.N_myo:,} "
          f"({100 * n_never / maps.N_myo:.1f}%)")
    print()
    print(f"First activation : {first_activation:.2f} ms")
    print(f"Last activation  : {last_activation:.2f} ms")
    print(f"Total activation time (beat {event}): {total_activation_time:.2f} ms")

    if n_never:
        print()
        print(f"NOTE: {n_never:,} node(s) never crossed threshold on beat "
              f"{event} and were excluded from first/last -- worth reporting "
              f"separately as conduction block rather than folding into "
              f"the headline number.")

    metrics = {
        "activation_maps_path": str(activation_maps_path),
        "event": event,
        "act_threshold_mV": float(maps.act_threshold),
        "N_myo": int(maps.N_myo),
        "n_never_activated": n_never,
        "pct_never_activated": 100 * n_never / maps.N_myo,
        "first_activation_ms": first_activation,
        "last_activation_ms": last_activation,
        "total_activation_time_ms": total_activation_time,
        "status": "ok",
    }

    if save_path:
        _save_json(metrics, save_path)

    return metrics


def _save_json(metrics: dict, save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {save_path}")


def _default_save_path(event: int) -> str:
    """outputs/<PATIENT_ID>/<PATIENT_ID>_activation_metrics.json, matching
    the naming convention used everywhere else in this experiment
    (e.g. P001_bspm.npz)."""
    suffix = f"_event{event}" if event != 1 else ""
    return f"{OUT}/{PATIENT_ID}_activation_metrics{suffix}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Report (and save) total ventricular activation time "
                    "from a saved activation_maps.npz")
    parser.add_argument("activation_maps_path", type=str, nargs="?",
                         default=DEFAULT_ACTIVATION_MAPS_PATH,
                         help=f"Path to activation_maps.npz "
                              f"(default: {DEFAULT_ACTIVATION_MAPS_PATH})")
    parser.add_argument("event", type=int, nargs="?", default=EVENT,
                         help=f"Beat/event index (default: {EVENT})")
    parser.add_argument("--save-path", type=str, default=None,
                         help="Override output JSON path "
                              "(default: outputs/<PATIENT_ID>/<PATIENT_ID>_activation_metrics.json)")
    parser.add_argument("--no-save", action="store_true",
                         help="Print only, do not write a metrics JSON")
    args = parser.parse_args()

    if args.no_save:
        save_path = None
    else:
        save_path = args.save_path or _default_save_path(args.event)

    report(args.activation_maps_path, event=args.event, save_path=save_path)

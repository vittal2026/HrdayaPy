from __future__ import annotations

import argparse
import json
from pathlib import Path

from rotated_fiber_values import (
    DX_M, DT_MS, T, CONFIG_ORDER, FIBRE_CONFIGS, REL_ERROR_TOLERANCE,
    predicted_v45, print_rotated_fiber_values,
)
from rotated_fiber_common import (
    run_rotated_fiber, fit_cv_along_x, check_planarity,
    validate_eikonal_pointwise,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", type=str, nargs="+", default=CONFIG_ORDER,
                    choices=CONFIG_ORDER,
                    help="Which fibre configurations to run (default: all three -- "
                         "all three are needed to validate the formula; a subset is "
                         "only useful for re-checking one config in isolation).")
    p.add_argument("--out-dir", type=str, default="outputs_rotated_fiber",
                    help="Output directory (default: outputs_rotated_fiber).")
    p.add_argument("--results-name", type=str, default="exp5_rotated_fiber_results.json",
                    help="Filename for the results JSON, written under --out-dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_rotated_fiber_values()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"# Experiment 5 -- single resolution: dx_m={DX_M} mm, dt={DT_MS} ms, T={T} ms")
    print("#" * 70)

    cv_results: dict = {}
    runs: dict = {}

    for config_name in args.configs:
        theta_deg = FIBRE_CONFIGS[config_name]["theta_deg"]
        print(f"\n--- config={config_name} (theta={theta_deg} deg) ---")
        run = run_rotated_fiber(DX_M, DT_MS, T, config_name, out_dir)
        runs[config_name] = run
        cv = fit_cv_along_x(run["myo_coords_mm"], run["myo_act"])
        print(f"  measured v = {cv['v_mm_per_ms']:.5f} mm/ms  "
              f"(slope={cv['slope_ms_per_mm']:.5f} ms/mm, r2={cv['r2']:.6f}, n={cv['n_nodes']})")

        if config_name == "rotated45":


            planarity = check_planarity(run["myo_coords_mm"], run["myo_act"], DX_M)
            cv["planarity"] = planarity
            if not planarity["interior_flat"]:
                print("  WARNING: check_planarity() flagged the interior band as not flat -- "
                      "the measured v45 below may be contaminated by the z-wall boundary layer. "
                      "See rotated_fiber_values.py's BOUNDARY-LAYER WARNING before trusting the "
                      "pass/fail verdict.")

        cv["theta_deg"] = theta_deg
        cv_results[config_name] = cv

    result = dict(dx_m=DX_M, dt_ms=DT_MS, T=T, configs=cv_results)

    if all(c in cv_results for c in ("longitudinal", "transverse", "rotated45")):
        v_l = cv_results["longitudinal"]["v_mm_per_ms"]
        v_t = cv_results["transverse"]["v_mm_per_ms"]


        v45_predicted_naive = predicted_v45(v_l, v_t)
        v45_measured = cv_results["rotated45"]["v_mm_per_ms"]


        rotated_run = runs["rotated45"]
        eikonal = validate_eikonal_pointwise(
            rotated_run["myo_coords_mm"], rotated_run["myo_act"], DX_M,
            FIBRE_CONFIGS["rotated45"]["direction"], v_l, v_t,
            rel_error_tolerance=REL_ERROR_TOLERANCE,
        )
        passed = eikonal["passed"]

        result.update(
            v_l_mm_per_ms=v_l,
            v_t_mm_per_ms=v_t,
            v45_measured_mm_per_ms=v45_measured,
            v45_predicted_mm_per_ms=v45_predicted_naive,
            rel_error=eikonal["rms_rel_error"],
            passed=bool(passed),
            eikonal_pointwise=eikonal,
        )

        print("\n" + "=" * 70)
        print("RESULT -- does the discretisation reproduce the anisotropic eikonal relation?")
        print("-" * 70)
        print(f"  v_L (longitudinal, theta=0)  = {v_l:.5f} mm/ms   [ground truth input -- still a valid planar fit]")
        print(f"  v_T (transverse,   theta=90) = {v_t:.5f} mm/ms   [ground truth input -- still a valid planar fit]")
        print(f"  v_45 naive prediction sqrt((v_L^2+v_T^2)/2) = {v45_predicted_naive:.5f} mm/ms   "
              f"[n=x_hat SPECIAL CASE ONLY -- not trustworthy for rotated45, see below]")
        print(f"  v_45 global-fit measured (assumes n=x_hat)  = {v45_measured:.5f} mm/ms   "
              f"[reference only -- the fit's own n=x_hat assumption is violated here]")
        print(f"  pointwise eikonal check (measured local n, no n=x_hat assumption):")
        print(f"    rms relative error = {eikonal['rms_rel_error']*100:.3f}%  "
              f"(require < {eikonal['rel_error_tolerance']*100:.1f}%)")
        verdict = "PASS" if passed else "FAIL"
        conclusion = "correctly handles" if passed else "does NOT correctly handle"
        print(f"  [{verdict}] -> the discretisation {conclusion} the rotated (mixed-derivative) fibre term.")
        print("=" * 70)
    else:
        missing = sorted(set(CONFIG_ORDER) - set(cv_results))
        print(f"\nNOTE: cannot validate the formula -- missing config(s) {missing} "
              f"(re-run with --configs {' '.join(CONFIG_ORDER)}, or omit --configs entirely).")

    results_path = out_dir / args.results_name
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {results_path.resolve()}")


if __name__ == "__main__":
    main()

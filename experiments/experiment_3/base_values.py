"""
base_values.py
===============================================================================
SINGLE SOURCE OF TRUTH for the physical and discretisation parameters shared
by Experiments 2, 3, and 4 (conduction-velocity convergence, conductivity
scaling, and PMJ capture/delay). Import from this module instead of
hardcoding these values locally in each experiment's experiment_common.py.

WHY THIS FILE EXISTS
---------------------
Experiments 2/3/4 each previously carried their own copy-pasted
experiment_common.py, and two concrete inconsistencies were found between
them:

  1. REF_DX_P / REF_DX_M / REF_DT (the "fixed reference discretisation" used
     by Experiments 3 and 4) were left at 0.5 mm / 0.5 mm / 0.05 ms, while
     Experiment 2's *actual, reported* level-0 grid -- the one the manuscript
     Table cites -- was run at 0.3152 mm / 0.3152 mm / 0.05 ms via an
     unrelated CLI default override. This file fixes that by making
     REF_DX_P = REF_DX_M = 0.3152 mm the one and only definition, matching
     the grid empirically found (Experiment 2, dx: 0.5 -> 0.3152 mm) to
     bring the myocardial CV relative-change under the 5% criterion.

  2. SIGMA_P as used here (3.6 mS/mm) differs from the production hrdayapy
     config.py's CPL_SIGMA_P value quoted in the manuscript's Table 4
     (0.9 uS/mm) by an exact factor of 4000 -- consistent with the
     unit-conversion bug already flagged against config.py. Since config.py
     has since been deleted, THIS file's value (validated by producing a
     physiologically plausible Purkinje CV in Experiment 2: 249-258 cm/s)
     is treated as the ground truth going forward. If a recovered copy of
     config.py's production value is ever reconciled against this, update
     here and nowhere else.

ANISOTROPIC MYOCARDIAL CONDUCTIVITY (SIGMA_L / SIGMA_T / FIBRE_DIRECTION)
-------------------------------------------------------------------------
The myocardium is now solved with hrdayapy's ANISOTROPIC solver
(hrdayapy.anisotropic_simulation.compute_anisotropic_coupled), not the
isotropic hrdayapy.simulation.compute_coupled these experiments originally
used. That solver takes a fibre-aligned conductivity SIGMA_L and a
cross-fibre conductivity SIGMA_T (plus a per-voxel fibre_dir field)
instead of a single scalar SIGMA_M.

SIGMA_M (2.625e-4 mS/mm, the value this file used to define, reverse-
engineered for THIS geometry's isotropic solve to hit 73-79 cm/s
myocardial CV) is retired outright, not kept as a special "isotropic"
case -- SIGMA_L / SIGMA_T below are instead taken directly from
niederer_values.py's own literature-sourced pair (0.133 / 0.0176 mS/mm,
~7.56:1 ratio, see that module's docstring for provenance), by explicit
decision rather than by deriving them from the old SIGMA_M. This means
the myocardial CV Experiments 2/3 reported under the old isotropic
SIGMA_M is NOT reproduced numerically by this new anisotropic pair --
that tradeoff was accepted because Experiment 4 (the only experiment
currently being ported to the anisotropic solver) only sweeps c_pmj and
n_pmj, and doesn't depend on matching that old absolute CV value.

FIBRE_DIRECTION is uniform along Z, the slab's long axis (SLAB_Z_MM=50mm
in experiment_common.py) -- the same "fibre along the long axis" role
niederer_common.py's FIBRE_DIRECTION plays for its own (X-long) slab, and
consistent with this geometry's own CV fit already being taken along Z
(estimate_myocardial_cv). Component 0/1/2 <-> array axis 0/1/2, raw
array-index / physical-mm-aligned convention throughout -- same as
niederer_common.py and experiment_common.py's other geometry arrays
(S/Z/phi, P_ROOT/P_GRAZE/P_PMJ). NOTE: hrdayapy's anisotropic solver
applies its own internal Purkinje-tree axis convention to its returned
myocardial coordinates (a swap of array axes 0 and 1) -- callers must
undo this on myo_coords before using it in this raw frame, exactly as
niederer_common.run_niederer() does; see experiment_common.py's run_one/
run_directional for where that correction now lives.

USAGE
-----
In each experiment's experiment_common.py, replace the local hardcoded
parameter block with:

    from base_values import (
        SIGMA_P, SIGMA_L, SIGMA_T, FIBRE_DIRECTION, CM, A_M, R_P, A_P,
        C_PMJ, N_PMJ, ACT_THRESHOLD,
        REF_DT, REF_DX_P, REF_DX_M,
        STIM_LEN_MM, STIM_AMP, STIM_DUR,
        P_ROOT, P_GRAZE, P_PMJ, MYO_PAD_VOX,
        print_base_values,
    )

and call `print_base_values()` once near the top of each run_experimentN.py
`main()`, so the parameters actually in effect are echoed to the console /
log for every run -- making a future silent drift immediately visible
instead of requiring a source-file diff to catch.
===============================================================================
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# =============================================================================
P_ROOT  = np.array([1.25, 7.5, -45.0])   # proximal, elevated, stimulus site
P_GRAZE = np.array([1.25, 5.5,  -3.0])   # grazes 3 mm above the slab's top face
P_PMJ   = np.array([1.25,  1.25,   0.0])   # terminal PMJ, centre of the proximal (z=0) face

MYO_PAD_VOX = 1   # 1-voxel zero-padding border around the slab (required by
                  # marching_cubes for the internal animation surface mesh)

# =============================================================================
# Physical / material parameters
# -----------------------------------------------------------------------------
# SIGMA_P is the EXPERIMENT-VALIDATED value (Experiment 2 produced a
# physiologically plausible Purkinje CV -- 249-258 cm/s -- using this
# number). It is NOT currently reconciled against hrdayapy's production
# config.py (deleted; last known value showed an unexplained 4000x
# discrepancy -- see module docstring). Treat it as ground truth until
# that is resolved.
#
# SIGMA_L / SIGMA_T / FIBRE_DIRECTION are the anisotropic myocardial
# conductivity pair + fibre field, taken directly from niederer_values.py
# (literature-sourced, NOT derived from the old isotropic SIGMA_M -- see
# module docstring for why that value was retired rather than reused).
# =============================================================================
SIGMA_P = 0.6       # mS/mm   Purkinje axial conductivity
SIGMA_L = 0.133       # mS/mm   myocardial conductivity ALONG the fibre
                      #         (= niederer_values.SIGMA_L; see module docstring)
SIGMA_T = 0.0176      # mS/mm   myocardial conductivity ACROSS the fibre
                      #         (= niederer_values.SIGMA_T; ~7.56:1 vs SIGMA_L)
FIBRE_DIRECTION = np.array([0.0, 0.0, 1.0])   # uniform along Z, the slab's
                      # long axis (SLAB_Z_MM) -- see module docstring
CM      = 0.01        # uF/mm^2 specific membrane capacitance (PDE)
A_M     = 140        # mm^-1   myocardial surface-to-volume ratio
R_P     = 0.1125        # mm      Purkinje fibre radius


C_PMJ = 5e-0   # mS   lumped PMJ conductance (production default; swept
               #      directly in Experiment 4)
N_PMJ = 1      # nearest myocardial nodes coupled per PMJ (production
               #      default; swept directly in Experiment 4)

ACT_THRESHOLD = -40.0   # mV, upstroke crossing used for activation time
                          # (both domains, all three experiments)

# =============================================================================
# Reference discretisation
# -----------------------------------------------------------------------------
# This is Experiment 2's *actually reported* level-0 grid (see manuscript
# Table cv_convergence), NOT the untested 0.5 mm production target. 0.3152 mm
# was the coarsest grid at which the myocardial CV's relative change on
# further refinement fell under the pre-registered 5% criterion; Experiments
# 3 and 4 fix their discretisation here so all three experiments' "default
# operating point" numbers are directly comparable.
#
# If you deliberately want Experiment 3 or 4 to characterise behaviour at the
# production-affordable 0.5 mm grid instead (e.g. to compute a conductivity
# compensation factor per the earlier discussion), do that as an explicit,
# separately labelled additional run -- do not silently redefine these.
# =============================================================================
REF_DT   = 0.0500     # ms
REF_DX_P = 0.10    # mm
REF_DX_M = 0.10   # mm

# =============================================================================
# Stimulation protocol (root/His stimulus; shared by Exp 2, 3, 4)
# =============================================================================
STIM_LEN_MM = 2.0    # mm      radius of the root stimulus zone
STIM_AMP    = 12.0   # uA/mm^2 (matches config.py's av_node stimulus, prior
                       #          to the config.py deletion -- re-verify
                       #          against a recovered copy if possible)
STIM_DUR    = 1.0    # ms


def print_base_values() -> None:
    """Echo every shared parameter to stdout. Call once near the top of each
    experiment's main() so the parameters actually in effect for a given run
    are visible in the console output / log, not just in source code."""
    print("=" * 70)
    print("base_values.py -- shared parameters in effect for this run")
    print("-" * 70)
    print(f"  SIGMA_L       = {SIGMA_L:.6g} mS/mm  (fibre-aligned, myocardium)")
    print(f"  SIGMA_T       = {SIGMA_T:.6g} mS/mm  (cross-fibre, myocardium; "
          f"ratio {SIGMA_L/SIGMA_T:.3g}:1)")
    print(f"  FIBRE_DIRECTION = {FIBRE_DIRECTION.tolist()}  (uniform, along Z)")
    print(f"  SIGMA_P       = {SIGMA_P:.6g} mS/mm")
    print(f"  CM            = {CM:.6g} uF/mm^2")
    print(f"  A_M           = {A_M:.6g} mm^-1")
    print(f"  R_P           = {R_P:.6g} mm")
    print(f"  C_PMJ         = {C_PMJ:.6g} mS")
    print(f"  N_PMJ         = {N_PMJ}")
    print(f"  ACT_THRESHOLD = {ACT_THRESHOLD:.6g} mV")
    print(f"  REF_DT        = {REF_DT:.6g} ms")
    print(f"  REF_DX_P      = {REF_DX_P:.6g} mm")
    print(f"  REF_DX_M      = {REF_DX_M:.6g} mm")
    print(f"  STIM_LEN_MM   = {STIM_LEN_MM:.6g} mm")
    print(f"  STIM_AMP      = {STIM_AMP:.6g} uA/mm^2")
    print(f"  STIM_DUR      = {STIM_DUR:.6g} ms")
    print("=" * 70)


if __name__ == "__main__":
    # Running this file directly is a quick sanity check / manual audit.
    print_base_values()

"""
niederer_values.py
===============================================================================
Parameters for reproducing the Niederer et al. (2011) N-version benchmark
("Verification of cardiac tissue electrophysiology simulators using an
N-version benchmark", Phil. Trans. R. Soc. A 369(1954):4331-4351), used
here to validate the ANISOTROPIC solver against a literature ground truth
rather than against our own internal grid-refinement self-consistency
(that's what Experiments 2/3/4's base_values.py already does, for the
isotropic Purkinje-slab geometry -- this is a separate, independent check).

WHY THESE ARE NOT IN base_values.py
-------------------------------------------------------------------------
base_values.py's own docstring exists specifically to prevent silent
parameter drift between experiments that are supposed to share the same
physics. This benchmark deliberately does NOT share that physics -- it's
a different geometry (bare 20x7x3mm slab, no Purkinje-driven stimulus),
different conductivities (literature-sourced, not locally-tuned), and a
different validation criterion (absolute accuracy against a published
reference, not internal convergence). Merging the two files would risk
exactly the kind of cross-experiment silent drift base_values.py warns
against, just in the other direction.

GEOMETRY AND POINT CONVENTION
-------------------------------------------------------------------------
Cuboid, fibres along the long (Lx=20mm) axis. Points P1-P8 are the 8
corners, P9 is the domain CENTRE (not a 9th corner) -- this exact
convention (P1=(0,0,0), P8=(Lx,Ly,Lz) opposite corner, P9=centre) is
confirmed against a working reference implementation (fenics-beat's
Niederer benchmark demo). The stimulus is a small cube at the P1 corner.
Reported outputs, per the original paper: point-wise activation time at
P1-P9, and the activation-time curve along the P1->P8 diagonal line.

CONDUCTIVITY
-------------------------------------------------------------------------
SIGMA_L / SIGMA_T below (0.133 / 0.0176 mS/mm, ~7.56:1 ratio) are the
monodomain-equivalent transversely-isotropic conductivities commonly
cited as Niederer et al.'s own values (e.g. "conductivity of 0.133 mS/mm
in the fibre direction ... and 0.0176 mS/mm along the other axes ... as
in Niederer et al. 2011a", Whiteley-group high-order FEM paper, Frontiers
in Physiology 2015). NOTE: this is a DIFFERENT pair from base_values.py's
SIGMA_M (2.625e-4 mS/mm) -- that value was reverse-engineered for a
completely different geometry (Purkinje point-source) and would defeat
the purpose of a literature comparison if reused here.

STIMULUS AMPLITUDE / DURATION -- UNVERIFIED, CHECK BEFORE TREATING AS A
STRICT REPRODUCTION
-------------------------------------------------------------------------
Different re-implementations of this benchmark report the stimulus in
different native units (uA/mm^3, pA/pF, mA) depending on each code's own
current-injection convention, and I could not confidently cross-convert
between them to pin down a single authoritative number from secondary
sources alone. STIM_AMP/STIM_DUR below are a physiologically reasonable,
safely-suprathreshold placeholder (matching base_values.py's existing
STIM_AMP/STIM_DUR units and rough magnitude) -- the benchmark is
deliberately designed so activation timing away from the stimulus corner
is insensitive to the exact stimulus strength (only capture, not
propagation speed, depends on it), but if you need a strict
reproduction, verify these two against the paper's primary text/
supplementary material and update here.

REFERENCE ACTIVATION TIMES -- sourced from the paper's own supplementary
material (rsta20110139supp1.pdf), which tabulates each of the 11
participating codes' activation times at P1-P9 for every dx/dt
combination they tested (dx in {0.1, 0.2, 0.5} mm x dt in
{0.005, 0.01, 0.05} ms, roughly -- a few codes used slightly different
values, e.g. Cherry's dt=0.04 in place of 0.05).
-------------------------------------------------------------------------
We now run at the paper's own three tested dx levels (LEVELS below,
dx=0.1/0.2/0.5mm, all at dt=0.01ms), so each level has an EXACT matching
reference row -- mean +/- sample SD (ddof=1) across the 11 codes at that
dx/dt combination -- rather than needing to interpolate between two
bounding rows, which is what an earlier single-resolution design (a fixed
dx=0.15mm, between the paper's 0.1 and 0.2mm rows) required.

One data-cleaning note: Alan Benson (Leeds)'s P1 entry at dx=0.2mm is
-1.0, which is a "did not activate" sentinel (confirmed against a
working reference implementation that initialises activation times to
-1.0 before any threshold crossing is recorded), not a real activation
time of -1 ms -- excluded from that point's mean/SD (n=10 there, n=11
everywhere else).
===============================================================================
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# Geometry
# =============================================================================
LX_MM, LY_MM, LZ_MM = 20.0, 10.0, 10.0   # long axis (fibre direction) = X

MYO_PAD_VOX = 1   # 1-voxel zero-padding border, same purpose as
                  # experiment_common.py's MYO_PAD_VOX (marching_cubes needs
                  # for the internal animation surface mesh)

# CONVENTION (distance-from-P1 hierarchy, not fixed axis labels)
# -------------------------------------------------------------------------
# P1 is the stimulus corner (origin). Every OTHER point is geometrically
# one of 7 fixed "roles" a cuboid corner (or its centre) can play relative
# to P1 -- for edge lengths short < mid < long these roles fall in this
# ascending-distance order, unconditionally (single-edge distances are
# just the edge length; a face-diagonal distance is sqrt of the sum of its
# two component edges' squares, so it's always > either edge alone and <
# any diagonal with a larger component swapped in; the space diagonal is
# necessarily the largest of all):
#
#   short  <  mid  <  short_mid  <  centre  <  long  <  short_long
#     <  mid_long  <  space (opposite P1)
#
# ROLE_TO_POINT below is the ONLY place that says which point-name (P2..P9)
# gets which role. To reassign labels -- e.g. to make P2 the closer of
# {P2, P5} instead of P5 -- edit ONLY this dict (swap the two values);
# nothing else in corner_points_mm() needs to change. The dict's own
# iteration order still reflects the built-in ascending-distance order of
# the ROLES, so whatever labels you put in "short"/"mid"/etc. will inherit
# that ordering automatically.
ROLE_TO_POINT: dict[str, str] = {
    "short":      "P3",   # shortest single edge
    "mid":        "P5",   # middle single edge
    "short_mid":  "P7",   # face diagonal: short + mid edges
    "centre":     "P9",   # domain centre = (P1 + space-role point) / 2
    "long":       "P2",   # longest single edge
    "short_long": "P4",   # face diagonal: short + long edges
    "mid_long":   "P6",   # face diagonal: mid + long edges
    "space":      "P8",   # space diagonal, corner opposite P1
}


# The 8 corners + centre, in the slab's own (unpadded) physical mm frame --
# add MYO_PAD_VOX * dx_m when converting to the padded array's coordinate
# frame, same convention as experiment_common.py's estimate_myocardial_cv.
#
# Coordinates are computed from sorted(LX_MM, LY_MM, LZ_MM) (not hardcoded
# axis assignments like "P2 = Y edge"), so the ROLE_TO_POINT ascending-
# distance order above still holds even if LX_MM/LY_MM/LZ_MM are edited to
# a different relative ordering.
def corner_points_mm() -> dict[str, np.ndarray]:
    axis_lengths = {"x": LX_MM, "y": LY_MM, "z": LZ_MM}
    # which physical axis (x/y/z) plays the short/mid/long role, by length
    short_axis, mid_axis, long_axis = sorted(axis_lengths, key=axis_lengths.get)
    short, mid, long_ = (axis_lengths[short_axis], axis_lengths[mid_axis],
                          axis_lengths[long_axis])

    def pt(**on_axes: float) -> np.ndarray:
        """Point with the given axes set to their listed length, others 0."""
        coords = {"x": 0.0, "y": 0.0, "z": 0.0}
        coords.update(on_axes)
        return np.array([coords["x"], coords["y"], coords["z"]])

    origin = pt()
    space_pt = pt(**{short_axis: short, mid_axis: mid, long_axis: long_})
    role_points = {
        "short":      pt(**{short_axis: short}),
        "mid":        pt(**{mid_axis: mid}),
        "short_mid":  pt(**{short_axis: short, mid_axis: mid}),
        "centre":     (origin + space_pt) / 2.0,
        "long":       pt(**{long_axis: long_}),
        "short_long": pt(**{short_axis: short, long_axis: long_}),
        "mid_long":   pt(**{mid_axis: mid, long_axis: long_}),
        "space":      space_pt,
    }

    out = {"P1": origin}
    for role, label in ROLE_TO_POINT.items():
        out[label] = role_points[role]
    return out

DIAGONAL_FROM, DIAGONAL_TO = "P1", "P8"   # the paper's standard reported line

# Fibre direction, uniform, along the long (X) axis -- component 0 <-> array
# axis 0 here (this module's geometry builder works entirely in raw
# array-index / physical-mm-aligned convention, no axis swap, unlike the
# anatomical fibre pipeline in hrdayapy.fibres).
FIBRE_DIRECTION = np.array([1.0, 0.0, 0.0])

# =============================================================================
# Stimulus -- small cube at the P1 corner (direct myocardial / "ectopic"
# stimulus, no Purkinje involvement -- see niederer_common.py)
# =============================================================================
STIM_CUBE_SIDE_MM = 1.5   # cube side length, corner at P1=(0,0,0)

# See module docstring -- placeholder, not verified against the primary
# source. Units match base_values.py's STIM_AMP/STIM_DUR.
STIM_AMP = 0.36   # uA/mm^2
STIM_DUR = 2.0    # ms

# =============================================================================
# Physical / material parameters -- see module docstring for provenance
# =============================================================================
SIGMA_L = 0.133    # mS/mm   conductivity ALONG the fibre (long/X axis)
SIGMA_T = 0.0176   # mS/mm   conductivity ACROSS the fibre
CM      = 0.01     # uF/mm^2 specific membrane capacitance
A_M     = 140    # mm^-1   myocardial surface-to-volume ratio
ACT_THRESHOLD = -40.0   # mV, upstroke crossing used for activation time (same as base_values.py)

# =============================================================================
# Discretisation -- the paper's OWN three spatial resolutions (dx=0.1, 0.2,
# 0.5mm; Table 3), each run here at dt=0.01ms. This replaces an earlier
# single-resolution design (dx=0.15mm, compared against an interpolated
# envelope between the paper's 0.1/0.2mm rows) -- running at the paper's
# actual tested levels instead means each level compares against an EXACT
# matching reference row, not an interpolated bound.
#
# dt=0.01ms fixed across all three levels, not re-tuned per level to match
# the paper's own dx/dt pairing convention: the paper itself found "time
# discretization did not have a significant impact on the point-wise
# activation times" (comparing their own dt=0.05/0.01/0.005 sweep), so one
# dt across all three dx levels is a reasonable simplification, not a
# shortcut that trades away accuracy. Revisit if a level's results look
# dt-sensitive in practice.
#
# Cost note: dx=0.1mm has roughly (0.2/0.1)^3 = 8x the voxels of dx=0.2mm,
# and dx=0.5mm roughly (0.2/0.5)^3 ~= 1/16 as many -- the three levels are
# NOT comparable in compute cost, dx=0.1mm is by far the expensive one.
# =============================================================================
LEVELS = [
    dict(dx_m=0.1, dt=0.01),
    dict(dx_m=0.2, dt=0.01),
    dict(dx_m=0.5, dt=0.01),
]
T = 150.0     # ms  -- generous margin over the ~100-130 ms full-diagonal
              #        activation time reported in the original paper

# =============================================================================
# Reference activation times -- mean +/- sample SD (ddof=1) across the 11
# codes from the paper's own supplementary material (rsta20110139supp1.pdf),
# at dt=0.01 ms, for the paper's own three dx levels (LEVELS above). Keyed
# by dx_m directly (a float) now that levels are run at exact matches, not
# an interpolated envelope between two bounding rows.
#
# See module docstring for the -1.0-sentinel exclusion note (A.Benson, P1,
# dx=0.2mm: n=10 there, n=11 everywhere else). No sentinels present at
# dx=0.1mm or dx=0.5mm (n=11 for every point at those two levels).
#
# NOTE on dx=0.5mm specifically: the paper's own text describes this level
# as having "180-230%" error relative to the converged solution and being
# "extremely difficult to interpret... as physiologically meaningful" for
# most codes -- consistent with the SD values here being an order of
# magnitude larger than at 0.1/0.2mm (e.g. P8: 36.11 vs 3.22/8.17). A wide
# match against THIS row is expected and not a strong validation signal by
# itself; a narrow, precise match would be the more surprising result.
# =============================================================================
REFERENCE_ACTIVATION_TIMES_MS: dict[float, dict[str, dict[str, float]]] = {
    0.1: {
        "P1": dict(mean_ms=1.04,  sd_ms=0.38, n=11),
        "P2": dict(mean_ms=32.23, sd_ms=0.56, n=11),
        "P3": dict(mean_ms=8.95,  sd_ms=0.56, n=11),
        "P4": dict(mean_ms=33.70, sd_ms=1.00, n=11),
        "P5": dict(mean_ms=29.76, sd_ms=1.65, n=11),
        "P6": dict(mean_ms=43.97, sd_ms=2.83, n=11),
        "P7": dict(mean_ms=31.32, sd_ms=2.12, n=11),
        "P8": dict(mean_ms=44.93, sd_ms=3.22, n=11),
        "P9": dict(mean_ms=20.50, sd_ms=1.30, n=11),
    },
    0.2: {
        "P1": dict(mean_ms=1.05,  sd_ms=0.40, n=10),   # n=10: A.Benson -1.0 sentinel excluded
        "P2": dict(mean_ms=34.43, sd_ms=1.48, n=11),
        "P3": dict(mean_ms=11.29, sd_ms=1.60, n=11),
        "P4": dict(mean_ms=36.67, sd_ms=2.98, n=11),
        "P5": dict(mean_ms=38.56, sd_ms=5.00, n=11),
        "P6": dict(mean_ms=52.56, sd_ms=7.10, n=11),
        "P7": dict(mean_ms=40.52, sd_ms=6.08, n=11),
        "P8": dict(mean_ms=53.81, sd_ms=8.17, n=11),
        "P9": dict(mean_ms=24.58, sd_ms=3.76, n=11),
    },
    0.5: {
        "P1": dict(mean_ms=1.04,   sd_ms=0.38,  n=11),
        "P2": dict(mean_ms=47.73,  sd_ms=8.33,  n=11),
        "P3": dict(mean_ms=27.37,  sd_ms=11.04, n=11),
        "P4": dict(mean_ms=54.57,  sd_ms=11.97, n=11),
        "P5": dict(mean_ms=105.66, sd_ms=33.36, n=11),
        "P6": dict(mean_ms=117.34, sd_ms=34.83, n=11),
        "P7": dict(mean_ms=106.80, sd_ms=35.08, n=11),
        "P8": dict(mean_ms=117.86, sd_ms=36.11, n=11),
        "P9": dict(mean_ms=55.17,  sd_ms=17.40, n=11),
    },
}


def print_niederer_values() -> None:
    print("=" * 70)
    print("niederer_values.py -- benchmark parameters in effect for this run")
    print("-" * 70)
    print(f"  geometry      = {LX_MM} x {LY_MM} x {LZ_MM} mm (fibre along X)")
    print(f"  SIGMA_L       = {SIGMA_L:.6g} mS/mm")
    print(f"  SIGMA_T       = {SIGMA_T:.6g} mS/mm  (ratio {SIGMA_L/SIGMA_T:.3g}:1)")
    print(f"  CM            = {CM:.6g} uF/mm^2")
    print(f"  A_M           = {A_M:.6g} mm^-1")
    print(f"  ACT_THRESHOLD = {ACT_THRESHOLD:.6g} mV")
    print(f"  LEVELS        = {LEVELS}")
    print(f"  T             = {T:.6g} ms")
    print(f"  STIM_CUBE_SIDE_MM = {STIM_CUBE_SIDE_MM:.6g} mm @ P1 corner")
    print(f"  STIM_AMP      = {STIM_AMP:.6g} uA/mm^2   [UNVERIFIED -- see module docstring]")
    print(f"  STIM_DUR      = {STIM_DUR:.6g} ms        [UNVERIFIED -- see module docstring]")
    if REFERENCE_ACTIVATION_TIMES_MS:
        print("  Reference activation times loaded: "
              f"{list(REFERENCE_ACTIVATION_TIMES_MS.keys())} "
              "(11-code mean +/- SD, from rsta20110139supp1.pdf)")
    else:
        print("  REFERENCE_ACTIVATION_TIMES_MS is EMPTY -- comparison-to-literature "
              "step will be skipped.")
    print("=" * 70)


if __name__ == "__main__":
    print_niederer_values()

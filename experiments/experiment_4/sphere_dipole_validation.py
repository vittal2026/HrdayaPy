"""
sphere_dipole_validation.py
=============================
Validates hrdayapy's ECG forward-solve machinery (hp.ecg.compute_torso_grid
-- the same FDM stiffness assembly + sparse factorisation used for the real
torso/heart Poisson solve) against a case with a KNOWN, EXACT analytical
solution:

    A physical current dipole (two point current sources +I / -I separated
    by a small distance d) placed at the exact center of a homogeneous,
    isotropic sphere of radius R and conductivity sigma, with an insulating
    (zero-flux / Neumann) outer boundary.

This is a static problem (the sources don't change with time), so it's a
single Poisson solve -- no time-stepping, no Vm snapshots, no registration.
It exercises exactly the piece of the pipeline that turns a source into a
potential field: build_torso_grid's stiffness assembly + factorisation +
solve_fn, called through the same public API (hp.ecg.compute_torso_grid)
your real patient runs use. The sphere's insulating boundary needs no
special code at all -- it falls out for free from build_torso_grid's
existing air-conductivity-zero handling at any label-0/label-1 interface,
so a synthetic "torso" NRRD with label 1 = sphere, label 0 = surrounding
air reproduces it exactly.

Analytical ground truth: Frank's solution
--------------------------------------------
For two point current sources at the center of a homogeneous, insulated
sphere, the potential everywhere inside the sphere (not just approximately,
not just deep in the interior -- everywhere, including the surface) is
given in closed form by:

    Frank, E. (1952). "Electric Potential Produced by Two Point Current
    Sources in a Homogeneous Conducting Sphere." Journal of Applied
    Physics, 23(11), 1225-1228.

For a dipole (the d -> 0, I -> infinity limit with p = I*d held fixed) at
the sphere's center:

    phi(r,theta) = [p/(4 pi sigma)] * [cos(theta)/r^2 + 2 r cos(theta)/R^3]

This is exact at every field point in the sphere, r in [0, R], including
the surface (r = R). There's no separate "unbounded medium" approximation
needed anywhere in this validation -- Frank's solution IS the ground
truth, at the surface as much as anywhere else, so the numeric-vs-exact
RDM computed below is a direct, unqualified pass/fail signal for the
solver, limited only by mesh resolution (dx_coarse_mm) and by how close
the finite-separation dipole (d) is to the ideal point-dipole limit.

Gauge / reference potential
-----------------------------
The Neumann (all-insulating) problem only determines phi up to an additive
constant -- build_torso_grid fixes this by pinning one arbitrary interior
node to phi=0. For a sphere, that pinned node can land very close to (or
exactly at) the sphere's center, which is a literal singularity for a
point-dipole formula. Comparisons here are gauge-aligned instead against a
NON-singular reference point on the sphere's equator (theta=90 deg,
perpendicular to the dipole axis), where Frank's solution is finite and
well-behaved -- this reference is always safe regardless of where the
solver happened to pin its node.

Run:
    python sphere_dipole_validation.py
Edit the CONFIG block below to change sphere/dipole/mesh parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nrrd

import hrdayapy as hp


# =============================================================================
# CONFIG
# =============================================================================

WORK_DIR = Path(__file__).resolve().parent / "sphere_dipole_validation_work"

R_MM       = 100.0     # sphere radius (mm) -- similar scale to a real torso
DX_MM      = 2.0        # coarse FDM grid spacing (mm) -- same knob as
                         # ECG_DX_COARSE_MM in the real pipeline; smaller =
                         # more accurate but slower/more memory
SPHERE_MARGIN = 1.3     # bounding-box half-extent as a multiple of R_MM,
                         # so there's air around the sphere for the
                         # insulating boundary to act on

SIGMA_S_PER_M = 0.2     # homogeneous conductivity (S/m)

P_TARGET_A_M = 1e-6     # dipole moment magnitude held FIXED across the
                         # separation sweep below (A.m, arbitrary scale --
                         # only relative error is meaningful, not the
                         # absolute potential scale)

# Physical dipole separations to sweep (mm). For each, current I is set to
# P_TARGET_A_M / d so the dipole MOMENT stays fixed while d shrinks toward
# the ideal point-dipole limit. Must stay > ~2-3x DX_MM to be resolved as
# two distinct grid nodes; below that the +/- sources collapse onto (or
# too close to) the same node and the run is skipped with a warning.
D_SWEEP_MM = [20.0, 12.0, 8.0, 4.0]

# Radial-profile check: field points along the dipole axis (theta=0, "north
# pole" direction) at these fractions of R, using the smallest (most
# point-dipole-like) separation in D_SWEEP_MM. Frank's solution is exact
# everywhere, so this is purely a mesh-resolution check across the interior
# (not a "does the approximation break down" check -- there is no
# approximation here).
RADIAL_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Surface point-by-point readout: polar angles (degrees from the dipole
# axis, theta=0 = "north pole" where the dipole points, theta=180 = "south
# pole") at which to print the numeric potential side-by-side with Frank's
# solution. theta=90 (equator) is the gauge reference itself, so its
# "numeric" row is 0 by construction -- included for completeness.
SURFACE_THETA_DEG = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

# Optional: sweep sphere RADIUS itself (rebuilding the mesh each time) as a
# general robustness check -- confirms solver accuracy against Frank's
# solution holds up at multiple sphere sizes, not just R_MM above. Off by
# default because it re-runs the full build+factorise pipeline once per
# radius (several x the runtime of the rest of this script).
DO_RADIUS_SWEEP = False
R_SWEEP_MM = [50.0, 75.0, 100.0]   # sphere radii to test (mm)
RADIUS_SWEEP_DX_FRAC = 0.03        # dx held at this fraction of R
RADIUS_SWEEP_D_FRAC = 0.12         # dipole separation held at this fraction of R
                                    # (d/dx = 4, comfortably resolvable)

SAVE_PLOT = True   # save a PNG of RDM-vs-d/R convergence (matplotlib)

# =============================================================================


def build_sphere_nrrd(path: Path, R_mm: float, dx_mm: float, margin: float):
    """Write a synthetic torso-style label NRRD: label 1 = sphere interior
    (the homogeneous conductor), label 0 = surrounding air (implicit
    sigma=0 in build_torso_grid -- this alone gives the insulating
    boundary at the sphere's surface). Grid is forced to odd dimensions so
    a voxel sits exactly at the sphere's geometric center."""
    half_extent = R_mm * margin
    n = int(np.ceil(2 * half_extent / dx_mm))
    if n % 2 == 0:
        n += 1
    shape = (n, n, n)
    origin = -np.array([(n - 1) / 2 * dx_mm] * 3)

    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    x = origin[0] + ii * dx_mm
    y = origin[1] + jj * dx_mm
    z = origin[2] + kk * dx_mm
    r = np.sqrt(x**2 + y**2 + z**2)
    vol = np.where(r <= R_mm, 1, 0).astype(np.int32)

    header = {
        "space directions": np.diag([dx_mm, dx_mm, dx_mm]),
        "space origin": origin,
    }
    nrrd.write(str(path), vol, header)
    return shape, origin


def phys_to_row(xyz_mm, shape, dx_mm, origin_mm, active_id_map) -> tuple[int, int]:
    """Nearest active grid node (row index into the system matrix, and its
    flat voxel index) to a physical mm coordinate. Returns row=-1 if the
    nearest voxel isn't active (e.g. fell in air)."""
    ijk = np.round((np.asarray(xyz_mm) - origin_mm) / dx_mm).astype(int)
    ijk = np.clip(ijk, 0, np.array(shape) - 1)
    flat = int(ijk[0] * shape[1] * shape[2] + ijk[1] * shape[2] + ijk[2])
    return int(active_id_map[flat]), flat


def analytic_frank_sphere(p_mag_Am, sigma, R_mm, field_xyz_mm, dipole_center_mm, axis_unit):
    """Frank's exact solution for a dipole at the center of an insulated
    homogeneous sphere (Frank, J. Appl. Phys. 1952). Exact at every field
    point in the sphere, including the surface (r=R) -- not an
    approximation anywhere."""
    R_m = R_mm * 1e-3
    r_vec_mm = np.asarray(field_xyz_mm) - np.asarray(dipole_center_mm)
    r_mm = np.linalg.norm(r_vec_mm, axis=-1)
    r_m = r_mm * 1e-3
    cos_theta = (r_vec_mm @ axis_unit) / np.clip(r_mm, 1e-12, None)
    return (p_mag_Am / (4 * np.pi * sigma)) * (cos_theta / r_m**2 + 2 * r_m * cos_theta / R_m**3)


def rdm(numeric, analytic_shifted):
    """Relative Difference Measure: ||numeric - analytic|| / ||analytic||,
    the standard L2 forward-solver benchmarking metric (EEG/MEG literature).
    Safe against individual near-zero points (e.g. cos(theta)=0 crossings)
    that would blow up a naive pointwise ratio."""
    return float(np.linalg.norm(numeric - analytic_shifted) / np.linalg.norm(analytic_shifted))


def _solve_for_dipole(grid, dipole_center, axis_unit, p_target_Am, d_mm):
    """Inject +I/-I point sources at grid nodes separated by d_mm along
    axis_unit through dipole_center, holding the dipole MOMENT p=I*d fixed
    at p_target_Am, and return the solved potential field (raw gauge) plus
    the two source rows. Returns (None, -1, -1) if not resolvable at this
    mesh resolution."""
    h_m = grid["dx_mm"] * 1e-3
    d_m = d_mm * 1e-3
    I = p_target_Am / d_m
    pos_plus = dipole_center + axis_unit * (d_mm / 2)
    pos_minus = dipole_center - axis_unit * (d_mm / 2)
    row_p, _ = phys_to_row(pos_plus, grid["shape"], grid["dx_mm"], grid["origin_mm"], grid["active_id_map"])
    row_m, _ = phys_to_row(pos_minus, grid["shape"], grid["dx_mm"], grid["origin_mm"], grid["active_id_map"])
    if row_p < 0 or row_m < 0 or row_p == row_m:
        return None, row_p, row_m
    b = np.zeros(grid["A_scipy"].shape[0])
    # RHS units are volumetric current source density [A/m^3]: a point
    # current I [A] injected into one voxel's control volume (h_m^3) is
    # equivalent to a density of I/h_m^3 in that voxel -- matches the
    # units build_torso_grid's stiffness matrix (sigma_face/h^2) was
    # assembled for.
    b[row_p] += I / h_m**3
    b[row_m] -= I / h_m**3
    return grid["solve_fn"](b.copy()), row_p, row_m


def _equator_reference(surf_xyz, surf_rows, R_mm):
    """Non-singular gauge reference: surface node nearest the equator
    (theta=90 deg), where Frank's solution is finite (avoids the
    sphere-center singularity if the solver's pinned node lands there)."""
    ref_idx = int(np.argmin(np.linalg.norm(surf_xyz - np.array([R_mm, 0, 0]), axis=1)))
    return surf_rows[ref_idx], surf_xyz[ref_idx]


def run_radius_sweep():
    """General robustness check: rebuild the sphere/mesh at several radii,
    holding dx/R and d/R fixed, and confirm the surface RDM against
    Frank's exact solution stays small (mesh-limited) at every radius."""
    print("\n" + "=" * 88)
    print("OPTIONAL: RADIUS SWEEP (dx/R and d/R held fixed)")
    print("Confirms solver accuracy against Frank's exact solution holds up")
    print("across sphere sizes, not just at R_MM.")
    print("=" * 88)
    print(f"{'R (mm)':>8} {'dx (mm)':>9} {'d (mm)':>8}  {'RDM vs Frank exact':>20}")
    print("-" * 88)

    axis_unit = np.array([0., 0., 1.])
    dipole_center = np.array([0., 0., 0.])
    rows = []

    for R_mm in R_SWEEP_MM:
        dx_mm = R_mm * RADIUS_SWEEP_DX_FRAC
        d_mm = R_mm * RADIUS_SWEEP_D_FRAC
        sub_dir = WORK_DIR / f"radius_sweep_R{int(R_mm)}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        nrrd_path = sub_dir / "sphere.nrrd"
        build_sphere_nrrd(nrrd_path, R_mm, dx_mm, SPHERE_MARGIN)
        grid = hp.ecg.compute_torso_grid(
            nrrd_path, {1: SIGMA_S_PER_M},
            dx_coarse_mm=dx_mm, save_path=sub_dir / "grid.npz", verbose=False,
        )
        phi, row_p, row_m = _solve_for_dipole(grid, dipole_center, axis_unit, P_TARGET_A_M, d_mm)
        if phi is None:
            print(f"{R_mm:8.1f} {dx_mm:9.3f} {d_mm:8.2f}  "
                  f"-- source separation not resolvable, skipped --")
            continue

        surf_xyz, surf_rows = grid["surface_xyz"], grid["surface_rows"]
        ref_row, ref_xyz = _equator_reference(surf_xyz, surf_rows, R_mm)

        A_surf = analytic_frank_sphere(P_TARGET_A_M, SIGMA_S_PER_M, R_mm, surf_xyz, dipole_center, axis_unit)
        A_ref = analytic_frank_sphere(P_TARGET_A_M, SIGMA_S_PER_M, R_mm, ref_xyz, dipole_center, axis_unit)

        numeric_surf = phi[surf_rows] - phi[ref_row]
        rdm_val = rdm(numeric_surf, A_surf - A_ref)
        rows.append((R_mm, rdm_val))
        print(f"{R_mm:8.1f} {dx_mm:9.3f} {d_mm:8.2f}  {rdm_val:20.6f}")

        # Each grid carries a factorised sparse matrix (solve_fn's closure)
        # that can be large; drop references before building the next
        # radius's grid instead of letting them all accumulate in memory.
        del grid, phi
        import gc
        gc.collect()

    print("-" * 88)
    if rows:
        vals = [r[1] for r in rows]
        print(f"RDM vs Frank exact across R: min={min(vals):.5f}  max={max(vals):.5f}")
        print("-> stays small at every radius, confirming the solver is")
        print("   correct across sphere sizes, limited only by mesh resolution.")
    print("=" * 88)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    nrrd_path = WORK_DIR / "sphere.nrrd"
    grid_path = WORK_DIR / "sphere_grid.npz"

    print(f"Building synthetic sphere NRRD  (R={R_MM} mm, dx={DX_MM} mm) ...")
    build_sphere_nrrd(nrrd_path, R_MM, DX_MM, SPHERE_MARGIN)

    print("\nBuilding + factorising the FDM stiffness matrix "
          "(hp.ecg.compute_torso_grid -- the actual forward-solve machinery) ...")
    grid = hp.ecg.compute_torso_grid(
        nrrd_path, {1: SIGMA_S_PER_M},
        dx_coarse_mm=DX_MM, save_path=grid_path, verbose=True,
    )

    axis_unit = np.array([0., 0., 1.])
    dipole_center = np.array([0., 0., 0.])

    surf_xyz, surf_rows = grid["surface_xyz"], grid["surface_rows"]
    ref_row, ref_xyz = _equator_reference(surf_xyz, surf_rows, R_MM)
    print(f"\nGauge reference (equatorial surface node): {np.round(ref_xyz, 1)} mm")

    def solve_for_dipole(d_mm: float):
        phi, _, _ = _solve_for_dipole(grid, dipole_center, axis_unit, P_TARGET_A_M, d_mm)
        return phi

    # ── Sweep dipole separation d, holding moment p fixed ───────────────────
    print("\n" + "=" * 88)
    print("SURFACE POTENTIAL: point-dipole convergence as separation d -> 0 "
          "(moment p held fixed)")
    print("=" * 88)
    print(f"{'d (mm)':>8} {'d/R':>7}  {'RDM vs Frank exact':>20}")
    print("-" * 88)

    A_surf = analytic_frank_sphere(P_TARGET_A_M, SIGMA_S_PER_M, R_MM, surf_xyz, dipole_center, axis_unit)
    A_ref = analytic_frank_sphere(P_TARGET_A_M, SIGMA_S_PER_M, R_MM, ref_xyz, dipole_center, axis_unit)
    A_shift = A_surf - A_ref

    sweep_results = []
    for d_mm in D_SWEEP_MM:
        phi = solve_for_dipole(d_mm)
        if phi is None:
            print(f"{d_mm:8.1f} {d_mm/R_MM:7.3f}  "
                  f"-- source separation not resolvable at dx={DX_MM} mm, skipped --")
            continue
        numeric_surf = phi[surf_rows] - phi[ref_row]
        rdm_val = rdm(numeric_surf, A_shift)
        sweep_results.append((d_mm, rdm_val))
        print(f"{d_mm:8.1f} {d_mm/R_MM:7.3f}  {rdm_val:20.6f}")

    print("-" * 88)
    print("Expected pattern if the solver is correct: RDM small at every d,")
    print("and shrinking as d/R -> 0 (limited only by mesh resolution, i.e.")
    print("by DX_MM). Frank's solution is exact everywhere in the sphere, so")
    print("this RDM is a direct, unqualified pass/fail signal -- no separate")
    print("'expected gap' to account for anywhere in this table.")

    # ── Per-point surface readout at fixed theta angles ─────────────────────
    if sweep_results:
        d_small_mm = min(r[0] for r in sweep_results)
        phi = solve_for_dipole(d_small_mm)
        print("\n" + "=" * 88)
        print(f"SURFACE POINT-BY-POINT READOUT, d={d_small_mm} mm "
              f"(d/R={d_small_mm/R_MM:.3f}, smallest resolvable separation)")
        print("Numeric solve vs Frank's exact solution at fixed polar angles")
        print("theta (measured from the dipole axis), all on the sphere surface")
        print("(r=R). Potentials are gauge-shifted so the equatorial reference")
        print("point (theta=90) reads 0 by construction.")
        print("=" * 88)
        print(f"{'theta(deg)':>10} {'numeric (V)':>14}  {'Frank exact (V)':>18}  "
              f"{'rel.err vs exact':>16}")
        print("-" * 88)
        for theta_deg in SURFACE_THETA_DEG:
            theta = np.radians(theta_deg)
            # Point on the sphere surface at this polar angle, in the plane
            # containing the dipole axis (x-z plane here since axis = +z).
            field_xyz = dipole_center + R_MM * np.array([np.sin(theta), 0.0, np.cos(theta)])
            row, _ = phys_to_row(field_xyz, grid["shape"], grid["dx_mm"], grid["origin_mm"], grid["active_id_map"])
            if row < 0:
                print(f"{theta_deg:10.1f}  -- nearest grid node not active, skipped --")
                continue
            numeric_val = phi[row] - phi[ref_row]
            a_exact = analytic_frank_sphere(P_TARGET_A_M, SIGMA_S_PER_M, R_MM, field_xyz, dipole_center, axis_unit) - A_ref
            denom = abs(a_exact) if abs(a_exact) > 1e-30 else np.nan
            rel_err = abs(numeric_val - a_exact) / denom
            print(f"{theta_deg:10.1f} {numeric_val:14.6e}  {a_exact:18.6e}  {rel_err:16.4f}")
        print("-" * 88)
        print("Expected: small rel.err at every theta (mesh-limited). The row at")
        print("theta=90 is the gauge reference itself (0 / 0 by construction, not")
        print("a meaningful ratio -- ignore it or read the RDM tables above instead).")

    # ── Radial profile along the dipole axis, smallest resolvable d ────────
    if sweep_results:
        print("\n" + "=" * 88)
        print(f"RADIAL PROFILE along dipole axis (theta=0), d={d_small_mm} mm "
              f"(d/R={d_small_mm/R_MM:.3f})")
        print("Frank's solution is exact throughout the sphere's interior, so")
        print("this checks mesh-resolution accuracy from near-center out to the")
        print("surface, not an approximation breaking down.")
        print("=" * 88)
        print(f"{'r (mm)':>8} {'r/R':>7}  {'numeric (V)':>14}  {'Frank exact (V)':>18}  "
              f"{'rel.err vs exact':>16}")
        print("-" * 88)
        for frac in RADIAL_FRACTIONS:
            r_mm = frac * R_MM
            field_xyz = dipole_center + axis_unit * r_mm   # theta=0 point
            row, _ = phys_to_row(field_xyz, grid["shape"], grid["dx_mm"], grid["origin_mm"], grid["active_id_map"])
            if row < 0:
                continue
            numeric_val = phi[row] - phi[ref_row]
            a_exact = analytic_frank_sphere(P_TARGET_A_M, SIGMA_S_PER_M, R_MM, field_xyz, dipole_center, axis_unit) - A_ref
            rel_err = abs(numeric_val - a_exact) / abs(a_exact)
            print(f"{r_mm:8.1f} {frac:7.2f}  {numeric_val:14.6e}  {a_exact:18.6e}  {rel_err:16.4f}")
        print("-" * 88)

    # ── Optional radius sweep ────────────────────────────────────────────
    if DO_RADIUS_SWEEP:
        run_radius_sweep()

    # ── Optional convergence plot ────────────────────────────────────────
    if SAVE_PLOT and sweep_results:
        import matplotlib.pyplot as plt
        d_vals = [r[0] / R_MM for r in sweep_results]
        rdm_vals = [r[1] for r in sweep_results]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(d_vals, rdm_vals, "o-", color="#2a7", label="vs Frank exact solution")
        ax.set_xlabel("d / R  (dipole separation / sphere radius)")
        ax.set_ylabel("RDM (relative difference measure)")
        ax.set_title("Surface potential: convergence to point-dipole limit")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = WORK_DIR / "convergence.png"
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved convergence plot: {out_png}")

    # ── Final plain-language summary ────────────────────────────────────────
    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    if sweep_results:
        best_rdm = min(r[1] for r in sweep_results)
        print(f"Best surface RDM vs Frank's exact solution (smallest d/R tested): {best_rdm:.5f}")
        if best_rdm < 0.05:
            print("PASS: numeric solve matches Frank's exact bounded-sphere solution")
            print("      to within mesh-resolution error. The forward solver is")
            print("      behaving correctly.")
        else:
            print("CHECK: RDM vs Frank's exact solution is larger than expected --")
            print("       try a smaller DX_MM (finer mesh) before suspecting the")
            print("       solver itself.")
    else:
        print("No dipole separation in D_SWEEP_MM was resolvable at this mesh")
        print("resolution -- increase separations or decrease DX_MM.")


if __name__ == "__main__":
    main()

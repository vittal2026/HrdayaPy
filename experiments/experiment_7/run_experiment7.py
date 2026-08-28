"""
run_experiment7.py
===================
Experiment 7: ECG / body-surface potential (BSPM) forward solve, run on the
FIXED coupled Purkinje/myocardium monodomain solution produced by
Experiment 6 (run_experiment6.py) -- vm_snapshots is loaded from disk here,
never recomputed. This isolates the forward-solve machinery (the torso
Poisson solve, hp.ecg.compute_torso_grid / compute_bspm) as its own
experiment, exactly the way Experiment 5 isolates the same machinery
against Frank's analytical sphere solution -- see
experiments/experiment_5/sphere_dipole_validation.py, whose rdm() metric
and gauge-alignment convention this script reuses directly rather than
redefining.

Two parts:

  Part A (Stage 7).  Production-resolution registration + torso grid + BSPM,
      at dx_coarse_mm = 2.0 mm (ECG_DX_COARSE_MM, Table 4/Section 2.7) --
      unchanged from the original run_simulation.py's Stage 7.

  Part B (Stage 7b).  Torso-grid resolution convergence study. Holds
      vm_snapshots and the heart-torso registration FIXED and sweeps only
      dx_coarse_mm, refining by a factor Q per level so node count scales
      as Q**3 per level (Q applied per axis).

"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import hrdayapy as hp

if __name__ == "__main__":

    PATIENT_ID = "P001"

    SCRIPT_DIR = Path(__file__).resolve().parent
    OUT_DIR = SCRIPT_DIR / "outputs" / PATIENT_ID
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT = str(OUT_DIR)

    INPUTS = SCRIPT_DIR / "inputs"

    TORSO_NRRD_PATH = str(INPUTS / "torso.seg.nrrd")

    EXPERIMENT6_OUT_DIR = SCRIPT_DIR.parent / "experiment_6" / "outputs" / PATIENT_ID
    vm_snapshots_path = str(EXPERIMENT6_OUT_DIR / f"{PATIENT_ID}_vm_snapshots_ds2.npz")

    SIGMA_TORSO = {0: 0.0, 1: 0.2, 2: 0.9, 3: 0.05, 4: 0.05, 5: 0.7, 6: 0.7}
    TORSO_HEART_LABEL = 2

    DX_PRODUCTION_MM = 1.78

    CG_METRIC = "linf"
    CG_TOL_MV = 0.000001

    USE_SBM = True
    SBM_INTERFACE_WIDTH_MM = 10.0
    SBM_PSI_CUTOFF         = 1e-3

    GAUGE_PICK_JSON = f"{OUT}/{PATIENT_ID}_gauge_pick.json"

    USE_SAVED_GAUGE_PICK = True

    GAUGE_SNAP_WARN_DX_MULTIPLE = 1.5

    RUN_ECG_REGISTRATION   = False
    RUN_TORSO_GRID          = False

    RUN_BSPM                = False

    RUN_CONVERGENCE_SWEEP   = True

    registration_path = f"{OUT}/{PATIENT_ID}_registration.npz"

    if RUN_ECG_REGISTRATION:
        registration = hp.ecg.compute_registration(
            vm_snapshots_path, TORSO_NRRD_PATH,
            heart_label=TORSO_HEART_LABEL,
            save_path=registration_path,
        )
    else:
        registration = hp.ecg.load_registration(registration_path)

    print(f"  Registration: scale={registration['scale']:.4f}  dice={registration['dice']:.4f}")

    if RUN_TORSO_GRID:
        grid = hp.ecg.compute_torso_grid(
            TORSO_NRRD_PATH, SIGMA_TORSO, dx_coarse_mm=DX_PRODUCTION_MM,
            save_path=f"{OUT}/{PATIENT_ID}_torso_grid.npz",
            use_sbm=USE_SBM,
            sbm_interface_width_mm=SBM_INTERFACE_WIDTH_MM,
            sbm_psi_cutoff=SBM_PSI_CUTOFF,
        )
    else:
        grid = hp.ecg.load_torso_grid(
            f"{OUT}/{PATIENT_ID}_torso_grid.npz", TORSO_NRRD_PATH, SIGMA_TORSO,
            dx_coarse_mm=DX_PRODUCTION_MM,
        )
        actual_use_sbm = bool(grid.get("use_sbm", False))
        if actual_use_sbm != USE_SBM:
            raise RuntimeError(
                f"STALE CACHE DETECTED (production grid): requested "
                f"USE_SBM={USE_SBM} but "
                f"{OUT}/{PATIENT_ID}_torso_grid.npz was built with "
                f"use_sbm={actual_use_sbm}. Delete that file (and "
                f"{PATIENT_ID}_bspm.npz) and set RUN_TORSO_GRID=True / "
                f"RUN_BSPM=True to regenerate under the current setting."
            )

    if RUN_BSPM:
        bspm = hp.ecg.compute_bspm(
            vm_snapshots_path, grid, registration,
            save_path=f"{OUT}/{PATIENT_ID}_bspm.npz",
            cg_tol_mV=CG_TOL_MV,
            cg_metric=CG_METRIC,
            cg_max_iter=5000,
        )
    else:
        bspm = hp.ecg.load_bspm(f"{OUT}/{PATIENT_ID}_bspm.npz")["bspm_signal"]

    print(f"  Production BSPM ({DX_PRODUCTION_MM} mm): shape={np.asarray(bspm).shape}")
    print("  -> feeds the existing EASI-lead qualitative comparison (Section 3.9),")
    print("     computed elsewhere from this BSPM/registration pair.")

    Q_FACTOR   = 1.5
    N_LEVELS   = 5
    DX_BASE_MM = 4.0

    RMS_PASS_THRESHOLD_MV = 0.01

    LINF_PASS_THRESHOLD_MV = 1.0

    SIGNAL_FRAC_THRESHOLD = 0.10

    def rms(numeric: np.ndarray, reference: np.ndarray) -> float:
        diff = numeric - reference
        return float(np.linalg.norm(diff) / np.sqrt(diff.size))

    def pick_gauge_point_pyvista(surface_xyz: np.ndarray) -> int:
        """Opens a PyVista window over the surface point cloud for a SINGLE
        manual pick of the gauge/reference node. Same interaction pattern
        as EASI_leads.py's pick_four_points (click near a point, press F to
        confirm), reduced to one label instead of four. Returns the index
        into surface_xyz of the finalised pick.
        """
        import pyvista as pv
        from scipy.spatial import cKDTree

        tree = cKDTree(surface_xyz)
        state = {"candidate_idx": None, "finalized_idx": None}

        pv.set_plot_theme("dark")
        pl = pv.Plotter()

        cloud = pv.PolyData(surface_xyz.astype(np.float32))
        surf_mesh = cloud.reconstruct_surface()
        pl.add_mesh(surf_mesh, color="#e8b593", opacity=1.0, smooth_shading=True,
                    label="Simulated surface")

        def prompt_text() -> str:
            if state["finalized_idx"] is not None:
                return "Gauge point selected!\nClose this window to continue."
            return ("Click near a point for the GAUGE reference, then press F to confirm.\n"
                    "Pick anywhere on a smooth, well-resolved part of the surface, away\n"
                    "from sharp boundary features.   (U = undo)")

        def refresh_prompt():
            pl.add_text(prompt_text(), position="upper_left", font_size=12,
                        color="white", name="prompt")

        def on_pick(point):
            if state["finalized_idx"] is not None:
                return
            _, idx = tree.query(point, k=1)
            state["candidate_idx"] = int(idx)
            snapped = surface_xyz[idx]
            pl.add_point_labels(
                snapped.reshape(1, 3), ["candidate"], name="candidate_marker",
                point_color="yellow", text_color="yellow", point_size=16,
                render_points_as_spheres=True, shape=None, always_visible=True,
            )
            refresh_prompt()

        def finalize():
            if state["finalized_idx"] is not None or state["candidate_idx"] is None:
                return
            idx = state["candidate_idx"]
            state["finalized_idx"] = idx
            pt = surface_xyz[idx].reshape(1, 3)
            pl.add_point_labels(
                pt, ["GAUGE"], name="finalized_gauge",
                point_color="cyan", text_color="cyan",
                point_size=16, render_points_as_spheres=True,
                shape=None, always_visible=True, font_size=20, bold=True,
            )
            pl.remove_actor("candidate_marker")
            print(f"  [gauge] finalised -> surface node {idx} at "
                  f"({pt[0,0]:.1f}, {pt[0,1]:.1f}, {pt[0,2]:.1f}) mm")
            refresh_prompt()

        def undo():
            if state["finalized_idx"] is None:
                return
            state["finalized_idx"] = None
            pl.remove_actor("finalized_gauge")
            print("  [gauge] selection undone")
            refresh_prompt()

        pl.enable_point_picking(callback=on_pick, picker="point",
                                 left_clicking=False, show_message=False,
                                 show_point=False, tolerance=0.015)
        pl.add_key_event("f", finalize)
        pl.add_key_event("F", finalize)
        pl.add_key_event("u", undo)
        pl.add_key_event("U", undo)

        refresh_prompt()
        pl.add_axes()
        pl.show()

        if state["finalized_idx"] is None:
            raise RuntimeError(
                "Window closed before the gauge point was finalised. "
                "Re-run and press F after clicking a point."
            )
        return state["finalized_idx"]

    def save_gauge_pick(idx: int, surface_xyz: np.ndarray, save_path: str) -> None:
        """Persist the manually-picked gauge node's PHYSICAL coordinate (the
        only part that's portable across resolution levels -- see
        load_gauge_pick_xyz)."""
        import json
        record = {"xyz_mm": surface_xyz[idx].round(2).tolist()}
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  Saved gauge pick -> {save_path}")

    def load_gauge_pick_xyz(save_path: str) -> np.ndarray:
        """Reload the previously-picked gauge location as a physical (mm)
        coordinate. Do NOT reuse a saved node index across levels -- each
        level has its own surface point cloud (different node count and
        ordering); only the physical location is portable. Re-derive the
        index for whichever level's surface_xyz is in play via
        snap_gauge_to_surface below."""
        import json
        with open(save_path) as f:
            record = json.load(f)
        return np.asarray(record["xyz_mm"], dtype=np.float64)

    def snap_gauge_to_surface(
        gauge_xyz: np.ndarray, surface_xyz: np.ndarray,
        dx_mm: float | None = None, context: str = "",
        warn_dx_multiple: float = GAUGE_SNAP_WARN_DX_MULTIPLE,
    ) -> tuple[int, float]:
        """Nearest-neighbour snap the fixed physical gauge location onto
        THIS level's own surface point cloud -- the operation that makes
        the gauge choice portable across torso resolutions while staying
        pinned to the same anatomical spot. Mirrors EASI_leads.py's
        snap_xyz_to_surface. Prints (does not raise on) a warning if the
        snap distance exceeds warn_dx_multiple * dx_mm, since a large snap
        distance means this level's mesh has no close representative of the
        picked location -- exactly the kind of silent level-to-level drift
        that motivated fixing the physical location in the first place."""
        from scipy.spatial import cKDTree
        tree = cKDTree(surface_xyz)
        dist, idx = tree.query(gauge_xyz, k=1)
        idx, dist = int(idx), float(dist)
        flag = ""
        if dx_mm is not None and dist > warn_dx_multiple * dx_mm:
            flag = (f"  [warn] snap distance {dist:.2f} mm exceeds "
                    f"{warn_dx_multiple}x dx ({dx_mm:.2f} mm) -- gauge point "
                    f"may not have a comparable representative at this level")
        label_ctx = f" [{context}]" if context else ""
        print(f"    snap{label_ctx} [gauge]: -> node {idx}, "
              f"{dist:.2f} mm from picked location{flag}")
        return idx, dist

    def load_bspm_surface(bspm_npz_path: str, grid: dict) -> np.ndarray:
        bspm_signal = hp.ecg.load_bspm(bspm_npz_path)["bspm_signal"]
        bspm_signal = np.asarray(bspm_signal)
        n_surf = len(grid["surface_rows"])
        if bspm_signal.shape[-1] != n_surf and bspm_signal.shape[0] == n_surf:
            bspm_signal = bspm_signal.T
        assert bspm_signal.shape[-1] == n_surf, (
            f"bspm_signal's last axis ({bspm_signal.shape[-1]}) doesn't match "
            f"grid's surface node count ({n_surf}) -- indexing assumption is wrong."
        )
        return bspm_signal

    INTERP_K       = 8
    INTERP_POWER   = 2.0
    INTERP_EPS_MM  = 1e-6

    def build_interp_weights(
        surf_xyz_fine: np.ndarray, surf_xyz_coarse: np.ndarray,
        k: int = INTERP_K, power: float = INTERP_POWER,
    ):
        """k-NN inverse-distance-weighted interpolation from the fine
        surface point cloud onto the coarse surface node locations.

        Replaces plain nearest-neighbour matching (kept below as
        `restrict_to_common_nodes` for reference/fallback). NN matching
        snaps each coarse node to whichever fine node happens to be
        closest and then treats the fine value there as exact -- that
        introduces an O(h_fine) matching error of its own, on top of
        (and indistinguishable from) genuine PDE discretization error,
        which is exactly what was confounding the fitted convergence
        order p=0.80. IDW interpolation instead blends the k nearest
        fine-node values by inverse distance, which is smoother and
        does not have that O(h) floor from snapping to a single point.

        Returns a scipy.sparse CSR matrix W of shape (n_coarse, n_fine)
        with each row's k nonzeros summing to 1, so that
            phi_interp = W @ phi_fine
        interpolates the fine-grid field at the coarse node locations.
        Building this (cKDTree query, k=8, over ~1e5-3e5 points) is a
        one-time O(n_coarse * k) cost, same order as the old single-NN
        query -- still well under a second for the largest level pair.
        """
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix

        n_c = len(surf_xyz_coarse)
        n_f = len(surf_xyz_fine)
        k_eff = min(k, n_f)

        tree = cKDTree(surf_xyz_fine)
        dist, idx = tree.query(surf_xyz_coarse, k=k_eff, workers=-1)
        if k_eff == 1:
            dist = dist[:, None]
            idx = idx[:, None]

        w = 1.0 / np.maximum(dist, INTERP_EPS_MM) ** power
        exact_hit = dist[:, 0] <= INTERP_EPS_MM
        if np.any(exact_hit):
            w[exact_hit, :] = 0.0
            w[exact_hit, 0] = 1.0
        w /= w.sum(axis=1, keepdims=True)

        rows = np.repeat(np.arange(n_c), k_eff)
        cols = idx.reshape(-1)
        vals = w.reshape(-1)
        return csr_matrix((vals, (rows, cols)), shape=(n_c, n_f))

    def restrict_to_common_nodes(
        surf_xyz_fine: np.ndarray, surf_xyz_coarse: np.ndarray,
    ) -> np.ndarray:

        from scipy.spatial import cKDTree
        tree = cKDTree(surf_xyz_fine)
        _, idx = tree.query(surf_xyz_coarse, k=1)
        return idx.astype(int)

    def surface_rms_and_linf_all_frames(
        grid_coarse: dict, bspm_coarse: np.ndarray, ref_row_coarse: int,
        grid_fine: dict, bspm_fine: np.ndarray, ref_row_fine: int,
        signal_frac_threshold: float = SIGNAL_FRAC_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gauge-align both levels, restrict fine -> coarse node set, and
        return (rms_per_frame, linf_per_frame, ref_norm_per_frame,
        valid_mask) -- one value per saved frame, no frame pre-selection.
        Fully vectorised: bspm_coarse/bspm_fine already hold every frame,
        no re-solve needed.

        Both disagreement metrics (RMS = L-2 / sqrt(V), and L-infinity)
        are PRECISION measures (cross-level disagreement), not accuracy
        measures against a true solution -- see module docstring
        "PRECISION, NOT ACCURACY".

        valid_mask is True where ref_norm_per_frame (the gauge-aligned
        FINE-grid reference field's norm at that frame) is at least
        signal_frac_threshold of its own peak over the whole beat. This
        masking was originally added to protect RDM (a normalised metric)
        from a near-zero denominator at near-quiescent frames; RMS, like
        L-inf, is an ABSOLUTE metric with no such denominator, so it does
        not strictly need this masking. It's kept here for consistency
        with L-inf's worst-case frame selection -- see module docstring
        "NOTE (metric update)".
        """
        surf_rows_coarse = grid_coarse["surface_rows"]
        surf_rows_fine = grid_fine["surface_rows"]
        pos_ref_coarse = int(np.searchsorted(surf_rows_coarse, ref_row_coarse))
        pos_ref_fine = int(np.searchsorted(surf_rows_fine, ref_row_fine))

        phi_coarse = bspm_coarse - bspm_coarse[:, [pos_ref_coarse]]
        phi_fine_full = bspm_fine - bspm_fine[:, [pos_ref_fine]]

        import time as _time
        n_c, n_f = len(grid_coarse["surface_xyz"]), len(grid_fine["surface_xyz"])
        print(f"    interpolating {n_f} fine -> {n_c} coarse surface nodes "
              f"(k={INTERP_K}-NN IDW)...")
        _t0 = _time.time()
        W = build_interp_weights(grid_fine["surface_xyz"], grid_coarse["surface_xyz"])
        print(f"    weights built in {_time.time() - _t0:.2f} s "
              f"({W.nnz} nonzeros, {W.nnz / n_c:.1f} avg/row)")

        _t0 = _time.time()
        used_device = "CPU (scipy sparse)"
        try:
            import torch
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            W_coo = W.tocoo()
            W_t = torch.sparse_coo_tensor(
                np.vstack([W_coo.row, W_coo.col]), W_coo.data.astype(np.float32),
                size=W_coo.shape, device=dev,
            ).coalesce()
            phi_fine_t = torch.as_tensor(phi_fine_full.T.astype(np.float32), device=dev)
            phi_fine_interp = torch.sparse.mm(W_t, phi_fine_t).T.cpu().numpy()
            used_device = f"GPU ({dev})" if dev.type == "cuda" else "CPU (torch)"
        except ImportError:
            phi_fine_interp = (W @ phi_fine_full.T).T
        print(f"    interpolation applied in {_time.time() - _t0:.2f} s ({used_device})")

        diff = phi_coarse - phi_fine_interp
        abs_diff = np.abs(diff)
        n_nodes = diff.shape[1]
        ref_norm_per_frame = np.linalg.norm(phi_fine_interp, axis=1)
        rms_per_frame = np.linalg.norm(diff, axis=1) / np.sqrt(n_nodes)
        linf_per_frame = np.max(abs_diff, axis=1)

        valid_mask = ref_norm_per_frame >= signal_frac_threshold * ref_norm_per_frame.max()

        return rms_per_frame, linf_per_frame, ref_norm_per_frame, valid_mask

    def fit_convergence_order(dx_values: list[float], linf_values: list[float]) -> float:
        log_dx = np.log(np.asarray(dx_values))
        log_linf = np.log(np.asarray(linf_values))
        p, _ = np.polyfit(log_dx, log_linf, 1)
        return float(p)

    if RUN_CONVERGENCE_SWEEP:
        dx_levels = [DX_BASE_MM / (Q_FACTOR ** lvl) for lvl in range(N_LEVELS)]

        LEVELS_TO_RUN = [0, 1, 2, 3, 4]

        levels_to_process = list(range(N_LEVELS)) if LEVELS_TO_RUN is None else LEVELS_TO_RUN
        print(f"  Full sweep schedule: {dx_levels} mm (Q={Q_FACTOR}, {N_LEVELS} levels)")
        print(f"  Processing only levels: {levels_to_process} "
              f"(dx = {[round(dx_levels[l], 3) for l in levels_to_process]} mm)")

        DX_MISMATCH_TOL = 1e-3

        def _check_grid_not_stale(grid_lvl: dict, lvl: int, dx_mm: float,
                                   grid_path: str, bspm_path: str) -> None:
            actual_dx = _grid_dx_mm(grid_lvl)
            if abs(actual_dx - dx_mm) > DX_MISMATCH_TOL:
                raise RuntimeError(
                    f"STALE CACHE DETECTED at level {lvl}: requested "
                    f"dx_coarse_mm={dx_mm:.4f} mm but {grid_path} "
                    f"actually contains dx={actual_dx:.4f} mm. This "
                    f"file is almost certainly left over from a run "
                    f"with different Q_FACTOR/DX_BASE_MM/N_LEVELS, "
                    f"where level index {lvl} meant a different dx. "
                    f"Delete {grid_path} (and the matching {bspm_path}) "
                    f"and re-run so this level is recomputed at the "
                    f"correct spacing -- do NOT bump DX_MISMATCH_TOL to "
                    f"silence this."
                )
            actual_use_sbm = bool(grid_lvl.get("use_sbm", False))
            if actual_use_sbm != USE_SBM:
                raise RuntimeError(
                    f"STALE CACHE DETECTED at level {lvl}: requested "
                    f"USE_SBM={USE_SBM} but {grid_path} was built with "
                    f"use_sbm={actual_use_sbm}. Comparing a sharp-boundary "
                    f"level against an SBM level fits a convergence order "
                    f"across two different discretization METHODS, not "
                    f"across resolutions of one method -- not valid. "
                    f"Delete {grid_path} (and the matching {bspm_path}) "
                    f"and re-run so this level is regenerated under the "
                    f"current USE_SBM setting, and do the same for every "
                    f"other level in this comparison."
                )

        def _grid_dx_mm(grid_dict: dict) -> float:
            for key in ("dx_coarse_mm", "dx_mm", "dx"):
                if key in grid_dict:
                    return float(grid_dict[key])
            raise KeyError(
                "Loaded grid dict has none of the expected dx keys "
                "('dx_coarse_mm'/'dx_mm'/'dx') -- update _grid_dx_mm() to "
                "match hp.ecg.compute_torso_grid's actual saved schema "
                "before trusting any cache validation here."
            )

        def load_grid_surface_only(grid_path: str) -> dict:
            """Load ONLY the lightweight surface-geometry fields (and dx)
            directly from the cached .npz via np.load, bypassing
            hp.ecg.load_torso_grid entirely -- specifically to skip its
            "Re-assembling stiffness matrix" step. Safe to use ONLY when
            the BSPM for this level is ALSO already cached: at that point
            the stiffness matrix is never touched again by anything in
            this script (it was only ever needed to solve for the BSPM
            in the first place, via hp.ecg.compute_bspm), so reassembling
            it just to read surface_rows/surface_xyz back out is pure
            waste -- that reassembly was what was costing time on a run
            that was going to load a cached BSPM and skip solving anyway.

            ASSUMPTION (unverified without hp.ecg source): the on-disk
            grid .npz stores "surface_rows"/"surface_xyz" (and one of
            "dx_coarse_mm"/"dx_mm"/"dx") as top-level arrays under the
            same keys the in-memory grid dict uses elsewhere in this
            script. If that's wrong, this raises KeyError with the
            actual keys found, and the caller below falls back to the
            full hp.ecg.load_torso_grid path rather than silently
            producing a bad grid dict.
            """
            with np.load(grid_path, allow_pickle=True) as f:
                missing = [k for k in ("surface_rows", "surface_xyz") if k not in f.files]
                if missing:
                    raise KeyError(
                        f"{grid_path} is missing expected key(s) {missing} for "
                        f"the surface-only fast path; available keys: {f.files}"
                    )
                out = {"surface_rows": f["surface_rows"], "surface_xyz": f["surface_xyz"]}
                for dx_key in ("dx_coarse_mm", "dx_mm", "dx"):
                    if dx_key in f.files:
                        val = f[dx_key]
                        out[dx_key] = val.item() if val.shape == () else val
                        break
                else:
                    raise KeyError(
                        f"{grid_path} has none of the expected dx keys "
                        f"('dx_coarse_mm'/'dx_mm'/'dx') for stale-cache "
                        f"validation; available keys: {f.files}"
                    )

                out["use_sbm"] = bool(f["use_sbm"]) if "use_sbm" in f.files else False
                return out

        if USE_SAVED_GAUGE_PICK and Path(GAUGE_PICK_JSON).exists():
            gauge_xyz = load_gauge_pick_xyz(GAUGE_PICK_JSON)
            print(f"\n  Reusing saved gauge pick -> {GAUGE_PICK_JSON}")
            print(f"    gauge location: {np.round(gauge_xyz, 2)} mm")
        else:
            print("\n  Opening interactive picker for the gauge reference point "
                  "(production-resolution surface)...")
            gauge_idx = pick_gauge_point_pyvista(grid["surface_xyz"])
            save_gauge_pick(gauge_idx, grid["surface_xyz"], GAUGE_PICK_JSON)
            gauge_xyz = grid["surface_xyz"][gauge_idx]

        level_grids = {}
        level_bspm = {}
        level_ref = {}

        for lvl in levels_to_process:
            dx_mm = dx_levels[lvl]
            print(f"\n  --- Level {lvl}: dx_coarse_mm={dx_mm} ---")
            grid_path = f"{OUT}/{PATIENT_ID}_torso_grid_conv_lvl{lvl}.npz"
            bspm_path = f"{OUT}/{PATIENT_ID}_bspm_conv_lvl{lvl}.npz"
            grid_cached = Path(grid_path).exists()
            bspm_cached = Path(bspm_path).exists()

            if grid_cached and bspm_cached:

                try:
                    print(f"    loading cached grid (surface-only, "
                          f"skipping stiffness reassembly): {grid_path}")
                    grid_lvl = load_grid_surface_only(grid_path)
                except KeyError as e:
                    print(f"    [warn] surface-only fast path failed ({e}); "
                          f"falling back to full hp.ecg.load_torso_grid "
                          f"(will reassemble the stiffness matrix).")
                    grid_lvl = hp.ecg.load_torso_grid(
                        grid_path, TORSO_NRRD_PATH, SIGMA_TORSO, dx_coarse_mm=dx_mm,
                    )
                _check_grid_not_stale(grid_lvl, lvl, dx_mm, grid_path, bspm_path)
                print(f"    loading cached BSPM: {bspm_path}")
            elif grid_cached and not bspm_cached:

                print(f"    loading cached grid (full -- BSPM not yet "
                      f"cached, stiffness matrix needed to solve it): "
                      f"{grid_path}")
                grid_lvl = hp.ecg.load_torso_grid(
                    grid_path, TORSO_NRRD_PATH, SIGMA_TORSO, dx_coarse_mm=dx_mm,
                )
                _check_grid_not_stale(grid_lvl, lvl, dx_mm, grid_path, bspm_path)
                hp.ecg.compute_bspm(
                    vm_snapshots_path, grid_lvl, registration,
                    save_path=bspm_path,
                    cg_tol_mV=CG_TOL_MV,
                    cg_metric=CG_METRIC,
                    cg_max_iter=5000,
                )
            else:
                grid_lvl = hp.ecg.compute_torso_grid(
                    TORSO_NRRD_PATH, SIGMA_TORSO, dx_coarse_mm=dx_mm,
                    save_path=grid_path,
                    use_sbm=USE_SBM,
                    sbm_interface_width_mm=SBM_INTERFACE_WIDTH_MM,
                    sbm_psi_cutoff=SBM_PSI_CUTOFF,
                )
                hp.ecg.compute_bspm(
                    vm_snapshots_path, grid_lvl, registration,
                    save_path=bspm_path,
                    cg_tol_mV=CG_TOL_MV,
                    cg_metric=CG_METRIC,
                    cg_max_iter=5000,
                )

            bspm_lvl = load_bspm_surface(bspm_path, grid_lvl)

            ref_idx, ref_dist = snap_gauge_to_surface(
                gauge_xyz, grid_lvl["surface_xyz"],
                dx_mm=dx_mm, context=f"level {lvl}",
            )
            ref_row = grid_lvl["surface_rows"][ref_idx]
            ref_xyz = grid_lvl["surface_xyz"][ref_idx]

            level_grids[lvl] = grid_lvl
            level_bspm[lvl] = bspm_lvl
            level_ref[lvl] = ref_row

            print(f"    surface nodes: {len(grid_lvl['surface_rows'])}   "
                  f"gauge ref: {np.round(ref_xyz, 1)} mm  "
                  f"({ref_dist:.2f} mm from picked location)")

        print(f"\n  === Worst-case frame across all saved frames, per level pair ===")
        print(f"  (PRECISION metrics: cross-level disagreement, not error vs. a true")
        print(f"   solution -- see module docstring 'PRECISION, NOT ACCURACY')")
        print(f"  (frames with reference-field norm < {SIGNAL_FRAC_THRESHOLD:.0%} of the "
              f"beat's peak reference norm are excluded from worst-case selection --")
        print(f"   see module docstring, 'QUIESCENT-FRAME MASKING')")
        print(f"  Acceptance criterion: worst L-inf < {LINF_PASS_THRESHOLD_MV} mV "
              f"(PRIMARY, gates PASS/CHECK below). RMS<{RMS_PASS_THRESHOLD_MV} mV shown for")
        print(f"  context only -- see module docstring 'ACCEPTANCE CRITERION' for why "
              f"RMS is not gating here.")
        print(f"  {'level':>5} {'dx (mm)':>9} {'worst RMS':>12} {'@frame':>7} "
              f"{'worst L-inf':>13} {'@frame':>7} {'excluded':>10}")
        print("  " + "-" * 96)

        level_pairs = list(zip(levels_to_process[:-1], levels_to_process[1:]))

        results = []
        ref_norm_by_level_pair = {}
        for lvl_coarse, lvl in level_pairs:
            rms_arr, linf_arr, ref_norm_arr, valid_mask = surface_rms_and_linf_all_frames(
                level_grids[lvl_coarse], level_bspm[lvl_coarse], level_ref[lvl_coarse],
                level_grids[lvl], level_bspm[lvl], level_ref[lvl],
            )
            ref_norm_by_level_pair[lvl] = ref_norm_arr

            n_total = len(rms_arr)
            n_excluded = int((~valid_mask).sum())

            rms_valid = np.where(valid_mask, rms_arr, -np.inf)
            linf_valid = np.where(valid_mask, linf_arr, -np.inf)

            worst_rms = float(rms_valid.max())
            worst_rms_frame = int(rms_valid.argmax())
            worst_linf = float(linf_valid.max())
            worst_linf_frame = int(linf_valid.argmax())

            results.append(dict(
                level=lvl, level_coarse=lvl_coarse, dx_mm=dx_levels[lvl],
                dx_mm_coarse=dx_levels[lvl_coarse],
                rms=worst_rms, rms_frame=worst_rms_frame,
                linf=worst_linf, linf_frame=worst_linf_frame,
                n_excluded=n_excluded, n_total=n_total,
            ))
            flag = "PASS" if worst_linf < LINF_PASS_THRESHOLD_MV else "CHECK"
            print(f"  {lvl:>5} {dx_levels[lvl]:>9.3f} {worst_rms:>12.6f} {worst_rms_frame:>7d} "
                  f"{worst_linf:>13.6e} {worst_linf_frame:>7d} "
                  f"{n_excluded:>4d}/{n_total:<4d}  [{flag}]")
            if flag == "PASS":
                print(f"        -> Level {lvl} (dx={dx_levels[lvl]:.3f} mm): worst observed "
                      f"cross-level disagreement {worst_linf:.4f} mV < "
                      f"{LINF_PASS_THRESHOLD_MV} mV threshold. (Precision bound, "
                      f"not accuracy vs. a true solution.)")
            else:
                print(f"        -> Level {lvl} (dx={dx_levels[lvl]:.3f} mm): worst observed "
                      f"cross-level disagreement {worst_linf:.4f} mV still exceeds "
                      f"{LINF_PASS_THRESHOLD_MV} mV threshold -- NOT yet accepted as converged.")

        results_path = f"{OUT}/{PATIENT_ID}_ecg_convergence_results.npz"
        merged = {(r["level_coarse"], r["level"]): r for r in results}
        prior_ref_norms = {}
        if Path(results_path).exists():
            prior = np.load(results_path, allow_pickle=True)
            if "level_coarse" in prior.files and "level" in prior.files:
                for i in range(len(prior["level"])):
                    key = (int(prior["level_coarse"][i]), int(prior["level"][i]))
                    if key not in merged:
                        merged[key] = dict(
                            level=key[1], level_coarse=key[0],
                            dx_mm=float(prior["dx_mm"][i]),
                            dx_mm_coarse=float(prior["dx_mm_coarse"][i]),
                            rms=float(prior["rms"][i]), rms_frame=int(prior["rms_frame"][i]),
                            linf=float(prior["linf"][i]), linf_frame=int(prior["linf_frame"][i]),
                            n_excluded=int(prior["n_excluded"][i]), n_total=int(prior["n_total"][i]),
                        )
                for key in prior.files:
                    if key.startswith("ref_norm_lvl"):
                        prior_ref_norms[key] = prior[key]
            else:
                print("  [warn] existing results file predates level_coarse/level "
                      "tracking (from a full, non-restricted sweep) -- cannot merge "
                      "pair-by-pair; new results will be saved alongside under a "
                      "'_partial' filename instead of overwriting it.")
                results_path = f"{OUT}/{PATIENT_ID}_ecg_convergence_results_partial.npz"

        all_results = sorted(merged.values(), key=lambda r: r["level"])
        print(f"\n  Saving {len(all_results)} level-pair result(s) total "
              f"(this run computed {len(results)} new): "
              f"{[(r['level_coarse'], r['level']) for r in all_results]}")

        if len(all_results) >= 2:
            dxs = [r["dx_mm"] for r in all_results]
            linfs = [r["linf"] for r in all_results]
            p = fit_convergence_order(dxs, linfs)
            print(f"  Fitted L-inf convergence order p={p:.2f} across all "
                  f"{len(all_results)} saved pairs (expect ~1, not ~2 -- "
                  f"staircased Neumann BC + discontinuous sigma_T; see "
                  f"module docstring)")
            if len(all_results) < 3:
                print("  NOTE: fewer than 3 rate points -- threshold-crossing "
                      "precision result, not a validated asymptotic rate. "
                      "Same caveat as Section 3.2.1 (Experiment 2).")

        ref_norm_out = dict(prior_ref_norms)
        ref_norm_out.update({f"ref_norm_lvl{lvl}": arr for lvl, arr in
                              zip([r["level"] for r in results], ref_norm_by_level_pair.values())})

        np.savez(
            results_path,
            dx_levels=dx_levels,
            q_factor=Q_FACTOR,
            level=[r["level"] for r in all_results],
            level_coarse=[r["level_coarse"] for r in all_results],
            dx_mm=[r["dx_mm"] for r in all_results],
            dx_mm_coarse=[r["dx_mm_coarse"] for r in all_results],
            rms=[r["rms"] for r in all_results],
            rms_frame=[r["rms_frame"] for r in all_results],
            linf=[r["linf"] for r in all_results],
            linf_frame=[r["linf_frame"] for r in all_results],
            n_excluded=[r["n_excluded"] for r in all_results],
            n_total=[r["n_total"] for r in all_results],
            signal_frac_threshold=SIGNAL_FRAC_THRESHOLD,
            rms_pass_threshold_mv=RMS_PASS_THRESHOLD_MV,
            linf_pass_threshold_mv=LINF_PASS_THRESHOLD_MV,

            **ref_norm_out,
        )
        print(f"\n  Saved: {OUT}/{PATIENT_ID}_ecg_convergence_results.npz")

    print("\nDone (Experiment 7).")
    print(f"  Outputs in: {OUT}")

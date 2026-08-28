"""
run_experiment6.py
===================
Experiment 6, component (a): full-pipeline patient-specific activation-timing
benchmark.

Geometry -> coordinates -> Purkinje -> coupled Purkinje/myocardium monodomain
simulation -> activation maps -> total ventricular activation time, compared
against Durrer et al. (1970)'s in-situ-equivalent range.

This is Stages 1-6b of the original run_simulation.py, split out on its own
because this component's job is done once the coupled monodomain solution
and activation maps exist -- everything downstream of that (the ECG/BSPM
forward solve and its resolution-convergence study) is a separate, isolable
question, addressed in run_experiment7.py, which loads this script's saved
vm_snapshots as a FIXED input rather than recomputing it.

Same RUN_*/load-vs-compute pattern as the original script -- see that
docstring for the full explanation if this is your first time reading one
of these. First run on a new patient: leave the RUN_* flags below all True.

Interactive stages -- heads up
-------------------------------
RUN_LANDMARKS, RUN_THETA (first run only) always open a PyVista/matplotlib
window and need a display. The VISUALISE_*/ANIMATE_* flags are also
interactive by nature -- off by default so this runs headless end to end;
flip them on one at a time to look at a given stage's output.

Fixes applied relative to the original run_simulation.py
-----------------------------------------------------------
Two things in the original Stage 6 call were bugs, not intentional
parameter choices, and are fixed here:
  - `c_pmj: float = 0.005,` -- a type annotation used as a keyword
    argument, which is a SyntaxError in a function call. Fixed to
    `c_pmj=0.005,`.
  - `targer="purkinje"` -- typo for `target="purkinje"` in the second
    STIM_PROTOCOL entry. Fixed.

Discretisation / conductivity / PMJ-coupling correction (this revision)
-------------------------------------------------------------------------
The previous version of this script ran Stage 6 at parameters that were
neither the converged grid (Experiment 2) nor the validated production
conductivities/PMJ coupling (Experiment 3/4). Specifically, it used
dx_p=0.1987, voxel_size=0.2 (an UNTESTED spatial resolution -- not any of
Experiment 2's three convergence levels, despite pairing it with Level 2's
dt); sigma_M=1.31e-4 (the Experiment-3 0.5x-scale value, mislabelled in a
comment as the 2.0x point) paired with sigma_P=0.45 (which matches NONE of
Experiment 3's tested scale points, so the pair was never jointly validated
against CV proportional-to-sqrt(sigma)); and c_pmj=0.005 at n_pmj=1, which
the corrected Experiment 4 sweep (exp4_pmj_results.json) shows sits AT a
point that produces junction capture but NOT propagating capture for an
isolated PMJ (orthodromic, n_pmj=1, c_pmj_scale=0.1: propagating_capture
= false).

This revision replaces those four Stage 6 parameters with Experiment 2's
converged grid and Table 4's validated production defaults:
  - dt          : 0.0125       (unchanged -- already matched Level 2)
  - dx_p        : 0.1987   -> 0.140089   (Experiment 2, Level 2, converged)
  - voxel_size  : 0.2      -> 0.140089   (Experiment 2, Level 2, converged)
  - sigma_M     : 1.31e-4  -> 2.625e-4   (Table 4 / CPL_SIGMA_M, production)
  - sigma_P     : 0.45     -> 3.6        (Table 4 / CPL_SIGMA_P, production)
  - c_pmj       : 0.005    -> 0.05       (Table 4 / CPL_C_PMJ, production)
  - n_pmj       : 1        -> 4          (Table 4 / CPL_N_PMJ, production;
                                           per Table 8, this sits roughly
                                           160-250x above the orthodromic
                                           propagating-capture threshold,
                                           unlike n_pmj=1 above)
  - stim_region voxel_size (Stage 4, "point" method): 0.4 -> 0.140089, to
    match compute_coupled's voxel_size as the method's own comment already
    requires (it did not match either the old or the new value before this
    fix -- 0.4 vs. 0.2 -- so this was a pre-existing inconsistency, not
    something newly introduced here).

phi_endo_max/phi_epi_min (0.5/0.5, collapsing the mid-myocardial band to
zero width) and T (800 ms, one beat, vs. CPL_T=1600 ms/two beats in Table
4) are LEFT UNCHANGED here -- they were flagged by the same discrepancy
note in the prior version of this docstring and are a separate decision
from the grid/conductivity/PMJ fix above. Flag if you want those brought
to production defaults too.

IMPORTANT -- cache invalidation: RUN_GEOMETRY, RUN_SURFACE_LABEL, RUN_CHI,
RUN_PHI, RUN_LANDMARKS, RUN_PSI, RUN_THETA, RUN_STIM_REGION, and
RUN_PURKINJE are all still False below, i.e. they LOAD previously cached
.npy/.npz files from OUT_DIR rather than recomputing. Those caches were
generated under the OLD voxel_size (0.2 mm) and OLD dx_p (0.1987 mm).
Whether that matters depends on hrdayapy internals not visible from this
script alone (e.g. whether compute_geometry resamples onto voxel_size, or
onto a fixed resolution set at segmentation time in Section 2.2, and
whether the Purkinje network's node coordinates/act_times, which feed
directly into compute_coupled, depend on voxel_size at all). At minimum,
compute_stim_region_from_point's cached stim_region.npy was explicitly
generated at voxel_size=0.4 previously and should be regarded as suspect
now that compute_coupled's grid has changed twice over (0.2 -> 0.4
mismatch, now 0.140089 mm). Before trusting a full rerun, confirm which of
RUN_GEOMETRY / RUN_PHI / RUN_STIM_REGION / RUN_PURKINJE actually need to be
flipped back to True to regenerate at the new grid -- do not assume the
cached files are resolution-independent just because this script loads
them without complaint.
"""


import numpy as np
import hrdayapy as hp
from pathlib import Path
import torch

if __name__ == "__main__":
    # Windows multiprocessing (used by the parallel Purkinje growth below)
    # uses spawn, not fork -- child processes re-import this file, so
    # everything below MUST live inside this guard or every stage
    # (including interactive pickers) reruns once per worker process.

    # =============================================================================
    # Inputs -- edit these for your patient
    # =============================================================================

    PATIENT_ID = "P001"

    SCRIPT_DIR = Path(__file__).resolve().parent
    OUT_DIR = SCRIPT_DIR / "outputs" / PATIENT_ID
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT = str(OUT_DIR)

    INPUTS = SCRIPT_DIR / "inputs"

    MASK_PATH        = str(INPUTS / "real_muscle.seg.nrrd")                       # myocardium segmentation
    CUT_MASK_PATH    = str(INPUTS / "real_muscle_cut.seg.nrrd")                   # Slicer-exported cut, for visualisation only
    TORSO_NRRD_PATH  = str(INPUTS / "torso.seg.nrrd")                             # full torso segmentation, for the ECG stage
    GT_TORSO_MAT     = str(INPUTS / "ground_truth" / "geometries" / f"{PATIENT_ID}_torso.mat")   # ground-truth torso mesh, for comparison
    GT_TS_MAT        = str(INPUTS / "ground_truth" / "signals" / f"{PATIENT_ID}-ts.mat")        # ground-truth recorded BSPM, for comparison

    # Not used by this script yet, but sitting alongside the above in your
    # ground_truth/ folder if a later stage needs them:
    #   geometries\P001_EndoEpi.mat, geometries\P001_Epi.mat,
    #   geometries\P001_otherStructures.mat, signals\P001-ecg.mat


    # =============================================================================
    # RUN FLAGS -- True = compute + save, False = load from disk
    # First run on a new patient: leave these all True.
    # =============================================================================

    RUN_GEOMETRY          = False
    RUN_SURFACE_LABEL     = False
    REFINE_ANATOMY_WITH_PSI=False
    RUN_CHI               = False
    RUN_PHI               = False
    RUN_LANDMARKS         = False   # interactive PyVista picker when True
    RUN_PSI               = False
    RUN_THETA             = False   # interactive the first time; cached afterwards regardless of this flag
    RUN_STIM_REGION       = False
    RUN_PURKINJE          = False
    RUN_PURKINJE_VM_CHECK = False
    RUN_COUPLED           = False
    RUN_ACTIVATION_MAPS   = True


    # =============================================================================
    # INSPECTION FLAGS -- interactive PyVista views, off by default.
    # Turn one on at a time once its stage above has actually run/loaded.
    # =============================================================================

    VISUALISE_COORDINATES      = False   # 2x2 view: phi / psi / chi / theta on the cut mask
    VISUALISE_STIM_REGION      = False   # ectopic stimulus region highlighted against the full mask
    VISUALISE_PURKINJE         = False   # the grown Purkinje tree on the myocardium surface
    ANIMATE_VM_FIELD           = False   # Vm(t) animated on the cut-mask surface
    VISUALISE_ACTIVATION_MAPS  = False  # static activation/deactivation maps, side by side

    # Activation/deactivation map settings -- used by Stage 6b and by
    # VISUALISE_ACTIVATION_MAPS above. See hp.simulation.compute_activation_maps
    # / plot_activation_maps docstrings for the full parameter list.
    ACT_THRESHOLD_MV     = -40.0   # mV -- upstroke crossing = "activated"
    DEACT_THRESHOLD_MV   = -40.0   # mV -- downstroke crossing = "deactivated"
    ACTIVATION_MAP_MODE  = "cut"   # "full" (outer shell of S) or "cut" (shell of Z,
                                    # also exposes the interior cut face)
    ACTIVATION_MAP_EVENT = 1       # 1-based beat index to visualise (1 = first beat)


    # =============================================================================
    # Stage 1 -- Geometry (mask + anatomy labelling)
    # =============================================================================

    if RUN_GEOMETRY:
        S, anatomy = hp.coordinates.compute_geometry(
            MASK_PATH,
            save_path_S=f"{OUT}/{PATIENT_ID}_S.npy",
            save_path_anatomy=f"{OUT}/{PATIENT_ID}_anatomy.npz",
        )
    else:
        S, anatomy = hp.coordinates.load_geometry(
            f"{OUT}/{PATIENT_ID}_S.npy", f"{OUT}/{PATIENT_ID}_anatomy.npz",
        )

    Z = hp.coordinates.load_cut_mask(CUT_MASK_PATH)


    # =============================================================================
    # Stage 2 -- Surface labels + biventricular split + transmural coordinate
    # =============================================================================

    if RUN_SURFACE_LABEL:
        surface_label = hp.coordinates.compute_surface_label(
            anatomy, save_path=f"{OUT}/{PATIENT_ID}_surface_label.npy",
        )
    else:
        surface_label = hp.coordinates.load_surface_label(f"{OUT}/{PATIENT_ID}_surface_label.npy")


    if RUN_PSI:
        psi = hp.coordinates.compute_psi(
            S, landmarks["apex_voxels"], landmarks["basal_voxels"],
            save_path=f"{OUT}/{PATIENT_ID}_psi.npy",
        )
    else:
        psi = hp.coordinates.load_psi(f"{OUT}/{PATIENT_ID}_psi.npy")

    # Now that psi exists, redo just the LV/RV endocardial split with the
    # psi-based basal cut (ported from test_psi_basal_cut.py -- see
    # coordinates.refine_geometry_with_psi / mesh_labelling.
    # relabel_lv_rv_with_psi). The Stage-1 split baked into `anatomy` ran
    # before psi existed, so it fell back to naive connected components +
    # a hull-volume heuristic, which can merge LV/RV endocardium into one
    # component (or pick the wrong one as LV) wherever they're
    # topologically connected through an open/cut base. This reuses the
    # cached surface_label instead of repeating the expensive ray-tracing
    # step, so it's cheap -- like the other compute_* stages it always
    # reruns and overwrites its save path.
    if REFINE_ANATOMY_WITH_PSI:
        anatomy = hp.coordinates.refine_geometry_with_psi(
            S, anatomy, psi,
            save_path_anatomy=f"{OUT}/{PATIENT_ID}_anatomy_psi_refined.npz",
        )

    if RUN_CHI:
        chi, chi_labels = hp.coordinates.compute_chi(
            S, anatomy["lv_endo_voxels"], anatomy["rv_endo_voxels"],
            save_path_chi=f"{OUT}/{PATIENT_ID}_chi.npy",
            save_path_chi_labels=f"{OUT}/{PATIENT_ID}_chi_labels.npy",
        )
    else:
        chi, chi_labels = hp.coordinates.load_chi(
            f"{OUT}/{PATIENT_ID}_chi.npy", f"{OUT}/{PATIENT_ID}_chi_labels.npy",
        )

    if RUN_PHI:
        phi = hp.coordinates.compute_phi(
            S, surface_label, save_path=f"{OUT}/{PATIENT_ID}_phi.npy",
        )
        phi_rv = hp.coordinates.compute_phi(
            S, anatomy["surface_label_rv"], save_path=f"{OUT}/{PATIENT_ID}_phi_rv.npy",
        )
    else:
        phi = hp.coordinates.load_phi(f"{OUT}/{PATIENT_ID}_phi.npy")
        phi_rv = hp.coordinates.load_phi(f"{OUT}/{PATIENT_ID}_phi_rv.npy")


    # =============================================================================
    # Stage 3 -- Landmarks + apicobasal + rotational coordinates
    # =============================================================================

    if RUN_LANDMARKS:
        landmarks = hp.coordinates.compute_landmarks(
            S, save_path=f"{OUT}/{PATIENT_ID}_landmarks.npz",
        )
    else:
        landmarks = hp.coordinates.load_landmarks(f"{OUT}/{PATIENT_ID}_landmarks.npz")


    if RUN_THETA:
        theta = hp.coordinates.compute_theta(
            S, chi_labels, psi, anatomy["long_axis"],
            ridge_save_path=f"{OUT}/{PATIENT_ID}_septal_ridge.npz",
            anterior_vertex_save_path=f"{OUT}/{PATIENT_ID}_anterior_vertex.npy",
            save_path=f"{OUT}/{PATIENT_ID}_theta.npy",
        )
    else:
        theta = hp.coordinates.load_theta(f"{OUT}/{PATIENT_ID}_theta.npy")

    # --- Inspection: the four coordinate fields, 2x2 linked-camera view -------
    if VISUALISE_COORDINATES:
        hp.coordinates.visualise_coordinates(
            Z, phi=phi, psi=psi, chi=chi, theta=theta, Z_theta=S,
            screenshot_path=f"{OUT}/{PATIENT_ID}_coordinates.png",
        )


    # =============================================================================
    # Stage 4 -- Ectopic stimulus region
    # =============================================================================
    # TOY THRESHOLDS -- these four target/tolerance pairs pick which voxels
    # count as "the ectopic focus" purely by coordinate value. They are a
    # starting guess, not tuned to this patient -- expect to adjust them
    # once you've looked at the region with VISUALISE_STIM_REGION below.
    #
    # Two independent ways to define the same kind of output (an (Nx,Ny,Nz)
    # bool mask) -- pick whichever is easier for this case:
    #   "uvc"   -- target windows in (psi, phi, chi, theta) UVC space, as
    #              below. Fully scripted, no display needed.
    #   "point" -- interactively click a point on the myocardium surface and
    #              grow a small ball around it. Opens a PyVista window --
    #              needs a real display, not for headless/batch runs.
    STIM_REGION_METHOD = "point"   # "uvc" or "point"

    if RUN_STIM_REGION:
        if STIM_REGION_METHOD == "uvc":
            stim_region = hp.coordinates.compute_stim_region(
                psi=psi, psi_target=0.95, psi_tol=0.05,
                phi=phi, phi_target=0.9, phi_tol=0.05,
                chi=chi, chi_target=0.0, chi_tol=0.5,        # 0.0 = LV
                theta=theta, theta_target_deg=180.0, theta_tol_deg=10.0,
                save_path=f"{OUT}/{PATIENT_ID}_stim_region.npy",
            )
        elif STIM_REGION_METHOD == "point":
            stim_region = hp.coordinates.compute_stim_region_from_point(
                S, save_path=f"{OUT}/{PATIENT_ID}_stim_region.npy",
                radius_mm=1.0,      # radius of the grown ball around the click
                voxel_size=0.4,     # must match compute_coupled's voxel_size below
            )
        else:
            raise ValueError(f"Unknown STIM_REGION_METHOD: {STIM_REGION_METHOD!r}")
    else:
        # Same file format either way -- load_stim_region reads whichever
        # method was used to save it, no need to branch on load.
        stim_region = hp.coordinates.load_stim_region(f"{OUT}/{PATIENT_ID}_stim_region.npy")

    print(f"  stim_region: {stim_region.sum()} voxels flagged"
          f"{'  <-- EMPTY: ectopic stimulus will never fire!' if not stim_region.any() else ''}")

    # --- Inspection: ectopic region highlighted against the full mask --------
    if VISUALISE_STIM_REGION:
        hp.coordinates.plot_stimulus_region(S, stim_region)


    # =============================================================================
    # Stage 5 -- Purkinje network
    # =============================================================================

    if RUN_PURKINJE:
        # A single connected tree grown over the whole biventricular mask can
        # only ever occupy the chamber its root happens to land in -- LV and RV
        # endocardium are two separate surfaces, and local_opt() only bifurcates
        # along paths that stay inside the valid (near-surface) Purkinje region,
        # so it can never bridge across the septum. So we still grow LV and RV
        # as two separate trees and stitch them together with a short His stub
        # via merge_purkinje_networks -- but instead of picking each chamber's
        # root by hand on that chamber's own (cut-in-half) surface, we pick the
        # AV node ONCE on the whole biventricular surface and let the algorithm
        # find the two closest points on each chamber's own valid Purkinje
        # region (endocardial + within surface_depth_vox of the surface -- the
        # same "allowed region" each tree is grown in) and use those as the
        # LV/RV roots. This also sidesteps the "root too far from any
        # candidate" failure mode, the same way picking on each chamber's own
        # surface used to.
        _res_cm = float(anatomy["spacing_mm"][0]) / 10.0  # cm/voxel, from Stage 1 geometry

        S_lv = S & (chi_labels == 1)
        S_rv = S & (chi_labels == 2)

        # "Allowed Purkinje region" criteria -- shared between the root finder
        # below and the two compute_network() calls that grow each tree, so the
        # roots the picker finds are guaranteed valid candidates for the trees
        # actually grown from them.
        _PKN_DEPTH        = 0.1   # transmural coordinate ceiling (endocardial)
        _PKN_MAX_HEIGHT   = 0.97   # apicobasal coordinate ceiling
        _PKN_SURFACE_VOX  = 7     # max distance (voxels) from the TRUE myocardium
                                   # surface S -- see myocardium_mask below; measuring
                                   # against S_lv/S_rv instead would let nearly the
                                   # whole septum qualify as "near surface"

        print("Pick the AV node once, on the whole biventricular myocardium surface...")
        av_node, lv_root, rv_root = hp.purkinje.pick_av_node_and_biventricular_roots(
            S, S_lv, S_rv, phi, psi,
            depth=_PKN_DEPTH,
            max_height=_PKN_MAX_HEIGHT,
            surface_depth_vox=_PKN_SURFACE_VOX,
        )

        # LV and RV are two fully independent tree-growth problems, so grow
        # them at the same time in two worker processes (two CPU cores) --
        # wall-clock roughly halves vs. growing LV then RV one after another.
        # This is CPU parallelism, not GPU: the growth loop is a sequential
        # adaptive search (each new terminal depends on the tree built so
        # far), so there's nothing to vectorise onto the GPU here -- see
        # compute_biventricular_networks's docstring for the full reasoning.
        # Set PARALLEL_PURKINJE_GROWTH = False below to fall back to growing
        # them one after another (e.g. to get live tqdm progress bars back).
        PARALLEL_PURKINJE_GROWTH = False

        (lv_nodes, lv_elements, lv_act), (rv_nodes, rv_elements, rv_act) = \
            hp.purkinje.compute_biventricular_networks(
                lv_root, S_lv,
                rv_root, S_rv,
                phi, psi,
                n_term=750,
                resolution=_res_cm,
                depth=_PKN_DEPTH,
                max_height=_PKN_MAX_HEIGHT,
                surface_depth_vox=_PKN_SURFACE_VOX,
                # Measure "near surface" against the whole, unsplit
                # myocardium (S), not against S_lv/S_rv individually --
                # otherwise the artificial LV/RV split cut through the
                # (thin) septum gets mistaken for a real surface across
                # nearly its whole thickness, and the tree ends up
                # disproportionately filling the septum instead of
                # fanning out across the free wall. See
                # compute_biventricular_networks's myocardium_mask
                # docstring for the full explanation.
                myocardium_mask=S,
                min_seg_length=10,
                max_seg_length=50,
                lv_save_path=f"{OUT}/{PATIENT_ID}_pkn_lv.npz",
                rv_save_path=f"{OUT}/{PATIENT_ID}_pkn_rv.npz",
                parallel=PARALLEL_PURKINJE_GROWTH,
            )

        # Zero-length His stub sitting at the picked AV node itself (rather
        # than the midpoint of the two chamber roots, now that both roots are
        # derived from that single click): node 0 = His root (what the merged
        # tree's overall activation time-zero will be), node 1 = the split
        # point. merge_purkinje_networks wires node 1 to the closest node in
        # each chamber tree (i.e. lv_nodes[0], rv_nodes[0]).
        his_split_pt = np.array(av_node, dtype=float)
        his_nodes = np.vstack([his_split_pt, his_split_pt])
        his_elem  = np.array([[0, 1]])

        nodes, elements, act_times = hp.purkinje.merge_purkinje_networks(
            his_nodes, his_elem,
            lv_nodes, lv_elements,
            rv_nodes, rv_elements,
            lv_root=lv_nodes[0], rv_root=rv_nodes[0],
            resolution=_res_cm,
        )
        hp.purkinje.save_purkinje(f"{OUT}/{PATIENT_ID}_pkn.npz", nodes, elements, act_times)
    else:
        nodes, elements, act_times = hp.purkinje.load_network(f"{OUT}/{PATIENT_ID}_pkn.npz")


    if RUN_PURKINJE_VM_CHECK:
        pkn_vm = hp.purkinje.compute_vm(
            nodes, elements, act_times, save_path=f"{OUT}/{PATIENT_ID}_pkn_vm.npz",
        )

    # --- Inspection: the grown Purkinje tree on the myocardium surface --------
    if VISUALISE_PURKINJE:
        hp.purkinje.visualise_purkinje(
            nodes, elements, act_times, voxel_mat=S,
            screenshot_path=f"{OUT}/{PATIENT_ID}_purkinje.png",
        )


    # =============================================================================
    # Stage 6 -- Coupled Purkinje + myocardium monodomain simulation
    # =============================================================================
    # Pacing protocol: a single ectopic focus (the region computed above),
    # first beat at t=10 ms, repeating every 800 ms (i.e. beats at
    # 10, 810, 1610 ms across the 1700 ms run). amp/dur below match the
    # library's own default pulse strength/duration -- adjust alongside the
    # stim_region thresholds once you've inspected both.

    STIM_PROTOCOL = [
        dict(name="ectopic", target="ectopic", amp=0.0, dur=2.0,
             onset=100.0, period=1000.0),
        dict(name="purkinje", targer="purkinje", amp = 12.0, dur = 2.0,
             onset = 100.0, period = 800.0)
    ]

    vm_snapshots_path = f"{OUT}/{PATIENT_ID}_vm_snapshots.npz"

    from hrdayapy.simulation.functions.purkinje_myocardium_pipeline import run_coupled

    if RUN_COUPLED:
        with torch.no_grad():
            results = run_coupled(
                nodes=nodes, elements=elements, activation_times=act_times,
                S=S, Z=Z, phi=phi,
                stim_protocol=STIM_PROTOCOL,
                ectopic_region=stim_region,
                T=800.0,
                vm_save_dt=1.0,
                out_npz=f"{OUT}/{PATIENT_ID}_coupled.npz",       # was save_path
                out_vm_npz=vm_snapshots_path,                     # was vm_save_path
                phi_endo_max=0.5,
                phi_epi_min=0.5,
                sigma_P=3.6,
                sigma_M=2.625e-4,
                c_pmj=0.05,
                n_pmj=4,
                R_P=0.01,
                dt=0.0125,
                dx_p=0.140089,
                voxel_size=0.140089,
                dtype=torch.float32,   # the actual point of this change
            )
    else:
        results = hp.simulation.load_coupled(f"{OUT}/{PATIENT_ID}_coupled.npz")
        vm_snapshots = hp.simulation.load_vm_snapshots(vm_snapshots_path)

    # --- Inspection: Vm(t) animated on the cut-mask surface -------------------
    if ANIMATE_VM_FIELD:
        hp.simulation.animate_coupled(f"{OUT}/{PATIENT_ID}_coupled.npz")

        hp.simulation.plot_activation_snapshot_grid(
            f"{OUT}/{PATIENT_ID}_coupled.npz",
            times_ms=np.array([0, 14, 28, 42, 56, 70, 84, 98])+100,
            grid_shape=(2, 4),
            screenshot_path=f"{OUT}/{PATIENT_ID}_activation_snapshots.png",
        )



    # =============================================================================
    # Stage 6b -- Static activation (depolarisation) / deactivation
    #             (repolarisation) maps, one column per beat
    # =============================================================================

    activation_maps_path = f"{OUT}/{PATIENT_ID}_activation_maps.npz"

    if RUN_ACTIVATION_MAPS:
        activation_maps = hp.simulation.compute_activation_maps(
            vm_snapshots, S,
            act_threshold=ACT_THRESHOLD_MV,
            deact_threshold=DEACT_THRESHOLD_MV,
            save_path=activation_maps_path,
        )
    else:
        activation_maps = hp.simulation.load_activation_maps(activation_maps_path)

    print(f"  Activation maps: {activation_maps.n_beats_activated} beat(s) detected "
          f"(activation threshold {ACT_THRESHOLD_MV} mV / "
          f"deactivation threshold {DEACT_THRESHOLD_MV} mV)")

    # --- Inspection: activation vs deactivation map, side by side -------------
    if VISUALISE_ACTIVATION_MAPS:
        hp.simulation.plot_activation_maps(
            activation_maps, S=S, Z=Z,
            mask_mode=ACTIVATION_MAP_MODE, event=ACTIVATION_MAP_EVENT,
            screenshot_path=f"{OUT}/{PATIENT_ID}_activation_maps_beat{ACTIVATION_MAP_EVENT}.png",
        )

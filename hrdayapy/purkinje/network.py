"""
purkinje/network.py
=====================
Compute/load pair for the Purkinje conduction network (Step 5).

The root location is the AV node point picked during the coordinates
stage: landmarks["av_node_point"] from hrdayapy.coordinates.compute_landmarks
/ load_landmarks.
"""

from __future__ import annotations

from pathlib import Path

from .functions import create_purkinje, save_purkinje, load_purkinje


def compute_network(
    root,
    S,
    phi,
    psi,
    *,
    n_term: int = 650,
    n_con_max: int = 20,
    theta_max: float = 63.0,
    cond_vel: float = 340.0,
    resolution: float = 0.25,
    depth: float = 0.1,
    max_height: float = 0.9,
    pkn_diff: int = 5,
    min_seg_length: float = 0.0,
    max_seg_length: float = float("inf"),
    characteristic_length_mm: float = 10.0,
    verbose: bool = True,
    surface_depth_vox = None,
    surface_reference_mask = None,
    save_path=None,
):
    """
    Grow the fractal Purkinje tree from an AV-node root.

    Parameters
    ----------
    root       : (3,) or None -- AV node position in voxel coordinates,
                 e.g. tuple(landmarks["av_node_point"]). Pass None to pick
                 it interactively instead -- create_purkinje will open a
                 PyVista window on the myocardium surface for you to click
                 the AV node by hand.
    S          : (Nx,Ny,Nz) bool -- myocardium mask
    phi, psi   : (Nx,Ny,Nz) float -- transmural / apicobasal coordinates
    n_term     : target number of terminal branches
    cond_vel   : conduction velocity along the tree (mm/ms)
    characteristic_length_mm : characteristic length of the domain l_d, in
                 mm (Berg et al., Eq. 1) -- scale this to your mesh's actual
                 size (the reference paper uses ~10mm for typical
                 ventricular meshes, ~30mm for larger patient-specific
                 geometries). Controls how far apart early terminals are
                 spaced; too large relative to the mesh makes the initial
                 candidate search need far more rejected attempts (or, in
                 the worst case, exhaust the candidate pool) before the
                 threshold shrinks enough to succeed.
    surface_reference_mask : (Nx,Ny,Nz) bool or None -- pass the whole,
                 unsplit myocardium mask here when S is a chamber-
                 restricted sub-mask (e.g. growing the LV alone via
                 `S_full & (chi_labels == 1)`), so surface_depth_vox is
                 measured against the real endocardium/epicardium
                 instead of against the artificial LV/RV split boundary
                 running through the septum -- see create_purkinje's own
                 docstring for why that distinction matters (it's what
                 causes the tree to over-fill the septum otherwise).
                 Default None measures against S itself.
    (see create_purkinje's own docstring for the rest of the growth
    parameters -- defaults here match the ones create_purkinje ships with)
    save_path  : if given, the tree is written here as .npz

    Returns
    -------
    nodes    : (N, 3) float  -- node positions in voxel coordinates
    elements : (E, 2) int    -- branch endpoint indices into nodes
    act_times: (N,) float    -- local activation time (ms) at each node
    """
    nodes, elements, act_times = create_purkinje(
        root=(tuple(float(c) for c in root) if root is not None else None),
        voxel_mat=S, transmural=phi, apicobasal=psi,
        n_term=n_term, n_con_max=n_con_max, theta_max=theta_max,
        cond_vel=cond_vel, resolution=resolution, depth=depth,
        max_height=max_height, pkn_diff=pkn_diff,
        min_seg_length=min_seg_length, max_seg_length=max_seg_length,
        surface_depth_vox=surface_depth_vox,
        surface_reference_mask=surface_reference_mask,
        verbose=verbose,
    )
    if save_path is not None:
        save_purkinje(str(save_path), nodes, elements, act_times)
    return nodes, elements, act_times


def load_network(path):
    """Load a previously saved Purkinje tree. No recomputation."""
    path = Path(path).with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"No saved Purkinje tree at {path}")
    return load_purkinje(str(path))


# =============================================================================
# Parallel LV + RV growth
# =============================================================================
# create_purkinje()'s growth loop is a sequential, adaptive constructive-
# optimization search: each new terminal's bifurcation point depends on the
# *current* state of the whole tree built so far (nearest-segment search,
# local_opt against existing segments added one at a time), so there's no
# batched array op here to hand to the GPU without rewriting the algorithm
# itself. What IS embarrassingly parallel is that LV and RV are two fully
# independent instances of that same sequential problem -- neither reads
# the other's tree while growing -- so compute_biventricular_networks()
# below runs them as two separate OS processes on two CPU cores instead.
#
# Progress bars: each worker gets its own tqdm bar PINNED to its own
# terminal row (tqdm's `position` argument -- LV always row 0, RV always
# row 1), and both workers share one lock (tqdm.set_lock, handed to each
# worker via the pool's initializer) so their cursor-movement writes don't
# interleave and corrupt each other's row. Only the two bars themselves are
# shown -- the per-stage text prints ("[1/5] Defining Purkinje region...",
# etc.) are suppressed in each worker (verbose=False, show_progress=True)
# because plain, non-positioned text from two processes has no such
# coordination and would still garble the terminal.

def _init_tqdm_lock(lock) -> None:
    """ProcessPoolExecutor initializer: share one tqdm lock across workers
    so LV's and RV's position-pinned progress bars don't stomp on each
    other's terminal row."""
    from tqdm import tqdm
    tqdm.set_lock(lock)


def _grow_chamber_worker(payload: dict):
    """
    Module-level (picklable) worker: grows one chamber's tree in its own
    process. Must stay top-level (not a closure/lambda) so it can be
    pickled and sent to the worker under the "spawn" start method.
    """
    import os
    from tqdm import tqdm
    from .functions import create_purkinje, save_purkinje

    label         = payload["label"]
    root          = payload["root"]
    S             = payload["S"]
    phi           = payload["phi"]
    psi           = payload["psi"]
    kwargs        = payload["kwargs"]
    save_path     = payload["save_path"]
    tqdm_position = payload["tqdm_position"]

    # tqdm.write, not print: it routes through the shared lock and moves
    # the cursor below the pinned bars first, so this line doesn't land
    # in the middle of either progress row.
    tqdm.write(f"[{label}] growing in worker process (pid={os.getpid()})...")
    nodes, elements, act_times = create_purkinje(
        root=tuple(float(c) for c in root),
        voxel_mat=S, transmural=phi, apicobasal=psi,
        verbose=False,          # per-stage text from 2 processes garbles the terminal
        show_progress=True,     # ...but the position-pinned bars alone render cleanly
        tqdm_position=tqdm_position,
        **kwargs,
    )
    if save_path is not None:
        save_purkinje(str(save_path), nodes, elements, act_times)
    tqdm.write(f"[{label}] done: {len(nodes)} nodes, {len(elements)} elements")
    return nodes, elements, act_times


def compute_biventricular_networks(
    lv_root, S_lv,
    rv_root, S_rv,
    phi, psi,
    *,
    n_term: int = 650,
    n_con_max: int = 20,
    theta_max: float = 63.0,
    cond_vel: float = 340.0,
    resolution: float = 0.25,
    depth: float = 0.1,
    max_height: float = 0.9,
    pkn_diff: int = 5,
    min_seg_length: float = 0.0,
    max_seg_length: float = float("inf"),
    surface_depth_vox=None,
    myocardium_mask=None,
    lv_save_path=None,
    rv_save_path=None,
    parallel: bool = True,
    verbose: bool = True,
):
    """
    Grow the LV and RV Purkinje trees, optionally at the same time.

    parallel=True (default) runs LV and RV growth concurrently in two
    worker processes (concurrent.futures.ProcessPoolExecutor, "spawn"
    context so this also works correctly on Windows) -- wall-clock time
    drops from time(LV) + time(RV) to roughly max(time(LV), time(RV)).
    This is two-CPU-core parallelism, not GPU: see the module-level note
    above for why the growth algorithm itself doesn't vectorize onto a
    GPU. (The monodomain solve in Step 6 is a separate stage and already
    has a GPU path -- see simulation/functions/ionic_models/gpu_ttp06.py
    -- this function only concerns tree *generation*.)

    In parallel mode you'll see two progress bars, one per chamber,
    each pinned to its own terminal row ("LV" on row 0, "RV" on row 1)
    and each running through its own two phases -- "Terminals added"
    while the tree is grown, then "Nodes activated" for the final BFS
    activation-time pass -- independently and at whatever pace that
    chamber happens to progress. The per-stage text prints you'd
    normally see from create_purkinje (verbose=True) are suppressed in
    each worker, since unpinned text from two processes has no such
    coordination and would garble the terminal; only the two bars print.

    parallel=False runs them one after another in this process (original
    behaviour) -- useful for debugging, or if you'd rather see the full
    per-stage text output (verbose applies here).

    IMPORTANT (Windows): because this spawns new processes, the *calling
    script* must guard its top-level code with
    `if __name__ == "__main__":`, or each worker process will re-execute
    the whole script from scratch when it's re-imported. See
    run_simulation.py for the pattern.

    Parameters
    ----------
    lv_root, rv_root : tuple of float
        Chamber roots, e.g. from pick_av_node_and_biventricular_roots().
    S_lv, S_rv : np.ndarray
        Chamber-restricted myocardium masks.
    phi, psi : np.ndarray
        Transmural / apicobasal coordinates (shared by both chambers).
    (see create_purkinje's docstring for n_term..surface_depth_vox)
    myocardium_mask : np.ndarray or None
        The whole, unsplit myocardium mask (e.g. plain `S`, before
        splitting into S_lv/S_rv). When surface_depth_vox is set, both
        chambers measure "near surface" against this instead of against
        their own S_lv/S_rv -- otherwise the artificial LV/RV split cut
        through the septum gets mistaken for a real surface and the tree
        ends up over-filling the septum. See create_purkinje's
        `surface_reference_mask` docstring for the full explanation.
        Default None falls back to each chamber measuring against its
        own S_lv/S_rv (only correct if surface_depth_vox is unset, or if
        S_lv/S_rv aren't actually sub-masks of a larger shared mask).
    lv_save_path, rv_save_path : path-like or None
        Where to save each chamber's tree (.npz). Written from inside
        the worker process itself.
    parallel : bool
        See above.
    verbose : bool
        Only takes effect when parallel=False -- workers in parallel
        mode always run with verbose=False, show_progress=True (see
        above); each still prints one tqdm.write start line and one
        done line alongside its progress bar.

    Returns
    -------
    (lv_nodes, lv_elements, lv_act), (rv_nodes, rv_elements, rv_act)
    """
    kwargs = dict(
        n_term=n_term, n_con_max=n_con_max, theta_max=theta_max,
        cond_vel=cond_vel, resolution=resolution, depth=depth,
        max_height=max_height, pkn_diff=pkn_diff,
        min_seg_length=min_seg_length, max_seg_length=max_seg_length,
        surface_depth_vox=surface_depth_vox,
        surface_reference_mask=myocardium_mask,
    )

    if not parallel:
        lv_nodes, lv_elements, lv_act = create_purkinje(
            root=tuple(float(c) for c in lv_root), voxel_mat=S_lv,
            transmural=phi, apicobasal=psi, verbose=verbose, **kwargs,
        )
        if lv_save_path is not None:
            save_purkinje(str(lv_save_path), lv_nodes, lv_elements, lv_act)

        rv_nodes, rv_elements, rv_act = create_purkinje(
            root=tuple(float(c) for c in rv_root), voxel_mat=S_rv,
            transmural=phi, apicobasal=psi, verbose=verbose, **kwargs,
        )
        if rv_save_path is not None:
            save_purkinje(str(rv_save_path), rv_nodes, rv_elements, rv_act)

        return (lv_nodes, lv_elements, lv_act), (rv_nodes, rv_elements, rv_act)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    lv_payload = dict(label="LV", root=lv_root, S=S_lv, phi=phi, psi=psi,
                       kwargs=kwargs, save_path=lv_save_path, tqdm_position=0)
    rv_payload = dict(label="RV", root=rv_root, S=S_rv, phi=phi, psi=psi,
                       kwargs=kwargs, save_path=rv_save_path, tqdm_position=1)

    ctx  = mp.get_context("spawn")
    lock = ctx.RLock()   # shared so LV's and RV's bars don't corrupt each other's row

    with ProcessPoolExecutor(
        max_workers=2, mp_context=ctx,
        initializer=_init_tqdm_lock, initargs=(lock,),
    ) as ex:
        fut_lv = ex.submit(_grow_chamber_worker, lv_payload)
        fut_rv = ex.submit(_grow_chamber_worker, rv_payload)
        lv = fut_lv.result()
        rv = fut_rv.result()

    # Move the cursor past the two pinned bar rows before any further
    # printing from the parent process.
    print("\n\n", end="")

    return lv, rv

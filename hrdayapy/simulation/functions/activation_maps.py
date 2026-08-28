"""
activation_maps.py
===================
Static activation (depolarisation) and deactivation (repolarisation) maps
derived from the volumetric Vm snapshots (vm_snapshots.npz, see
load_vm_snapshots.py).

For every myocardial node, the membrane voltage time-series is scanned for
threshold crossings:

  activation   (depolarisation) : Vm rises through `act_threshold`   (mV)
  deactivation (repolarisation) : Vm falls through `deact_threshold` (mV)

Multi-beat protocols produce more than one crossing per node -- every
crossing is kept (not just the first), so a node's "activation map" is
really a (N_myo, n_events) table: column 0 is the 1st-beat activation time,
column 1 the 2nd-beat activation time, etc. A node that a given beat never
reaches (conduction block, refractory tissue, ...) simply gets NaN in that
column -- this is what downstream plotting renders as gray (see
plot_activation_maps.py).

NPZ schema (written by save_activation_maps / compute_activation_maps)
------------------------------------------------------------------------------
  node_ids            (N_myo,)                  int32    global solver index
  coords_mm           (N_myo, 3)                float32  physical xyz [mm]
  vox_idx             (N_myo, 3)                int32    voxel ijk in the mask
  grid_shape          (3,)                      int32    (Nx, Ny, Nz) of S
  activation_times    (N_myo, n_act_events)      float32  ms, NaN = no event
  deactivation_times  (N_myo, n_deact_events)    float32  ms, NaN = no event
  n_activations       (N_myo,)                   int32    events found per node
  n_deactivations     (N_myo,)                   int32    events found per node
  act_threshold       scalar                     float32  mV
  deact_threshold     scalar                     float32  mV
  source_vm_path      scalar                     str      provenance only

Typical usage
-------------
    from simulation.functions.activation_maps import (
        compute_activation_maps, load_activation_maps,
    )

    maps = compute_activation_maps(vm_snapshots, S,
                                    act_threshold=-40.0, deact_threshold=-40.0,
                                    save_path="simulation/outputs/activation_maps.npz")

    a1 = maps.activation_map(event=1)     # (N_myo,) first-beat LAT, NaN where never activated
    d1 = maps.deactivation_map(event=1)   # (N_myo,) first-beat repolarisation time
    print(maps.n_events)                  # how many beats were actually detected
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# =============================================================================
# Threshold-crossing detection (vectorised over all nodes at once)
# =============================================================================

def _interp_crossings(Vm: np.ndarray, t: np.ndarray, edge_mask: np.ndarray,
                       threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate the crossing time for every True entry of
    `edge_mask` (shape (n_frames-1, N_myo), True at frame i means Vm crosses
    `threshold` somewhere between frame i and frame i+1).

    Returns
    -------
    idx_node : (K,) int   -- node index for every crossing found
    t_cross  : (K,) float32 -- interpolated crossing time [ms]
    """
    idx_frame, idx_node = np.nonzero(edge_mask)
    if idx_frame.size == 0:
        return idx_node, np.zeros(0, dtype=np.float32)

    V_i   = Vm[idx_frame,     idx_node]
    V_ip1 = Vm[idx_frame + 1, idx_node]
    t_i   = t[idx_frame]
    t_ip1 = t[idx_frame + 1]

    denom = V_ip1 - V_i
    denom = np.where(denom == 0, 1e-9, denom)      # guard: shouldn't happen
    frac  = np.clip((threshold - V_i) / denom, 0.0, 1.0)
    t_cross = t_i + frac * (t_ip1 - t_i)

    return idx_node, t_cross.astype(np.float32)


def _pad_events(idx_node: np.ndarray, t_cross: np.ndarray,
                 N_myo: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn a flat (idx_node, t_cross) crossing list into a dense
    (N_myo, max_events) array, NaN-padded, plus a per-node event count.

    Crossings for the same node are assumed already in increasing time
    order (true here because `_interp_crossings` scans frames in order).
    """
    if idx_node.size == 0:
        return np.zeros((N_myo, 0), dtype=np.float32), np.zeros(N_myo, dtype=np.int32)

    order  = np.argsort(idx_node, kind="stable")
    idx_s  = idx_node[order]
    t_s    = t_cross[order]

    counts = np.bincount(idx_s, minlength=N_myo)
    max_events = int(counts.max())

    starts    = np.cumsum(counts) - counts
    event_pos = np.arange(len(idx_s)) - np.repeat(starts, counts)

    out = np.full((N_myo, max_events), np.nan, dtype=np.float32)
    out[idx_s, event_pos] = t_s
    return out, counts.astype(np.int32)


def _apply_min_gap(times_padded: np.ndarray, min_gap_ms: float | None) -> np.ndarray:
    """
    Optional debounce: drop crossings that follow the previous *kept*
    crossing (for the same node) by less than `min_gap_ms`. Guards against
    numerical chatter around the threshold being mistaken for extra beats.

    Only rows with more than one detected event are touched, so this is
    cheap in the common case (most nodes activate 0 or 1 times per beat).
    """
    if min_gap_ms is None or times_padded.shape[1] <= 1:
        return times_padded

    kept_lists = []
    max_events = 0
    for row in times_padded:
        vals = row[~np.isnan(row)]
        kept = []
        last = -np.inf
        for v in vals:
            if v - last >= min_gap_ms:
                kept.append(v)
                last = v
        kept_lists.append(kept)
        max_events = max(max_events, len(kept))

    out = np.full((times_padded.shape[0], max_events), np.nan, dtype=np.float32)
    for i, kept in enumerate(kept_lists):
        if kept:
            out[i, :len(kept)] = kept
    return out


# =============================================================================
# Container
# =============================================================================

@dataclass
class ActivationMaps:
    """
    Container for per-node activation (depolarisation) and deactivation
    (repolarisation) timing tables, one column per beat/event.

    Attributes
    ----------
    node_ids           : (N_myo,)               Global solver indices.
    coords_mm          : (N_myo, 3)              Physical xyz [mm].
    vox_idx            : (N_myo, 3)              Voxel ijk in the mask array.
    grid_shape         : (3,) or None            (Nx, Ny, Nz) of the mask
                         used at compute time (None if not provided).
    activation_times   : (N_myo, n_act_events)   ms, NaN = no such event.
    deactivation_times : (N_myo, n_deact_events) ms, NaN = no such event.
    n_activations      : (N_myo,)                events actually found.
    n_deactivations    : (N_myo,)                events actually found.
    act_threshold      : float                   mV, upstroke crossing.
    deact_threshold    : float                   mV, downstroke crossing.
    source_vm_path     : str                     provenance (vm_snapshots.npz).
    path               : Path                    where this object was saved.
    """

    node_ids:            np.ndarray
    coords_mm:           np.ndarray
    vox_idx:              np.ndarray
    activation_times:    np.ndarray
    deactivation_times:  np.ndarray
    n_activations:       np.ndarray
    n_deactivations:     np.ndarray
    act_threshold:       float
    deact_threshold:     float
    grid_shape:          np.ndarray | None = None
    source_vm_path:      str = ""
    path:                Path = field(default_factory=lambda: Path("."))

    # ------------------------------------------------------------------
    @property
    def N_myo(self) -> int:
        return int(self.activation_times.shape[0])

    @property
    def n_beats_activated(self) -> int:
        """Max number of activation events found on any single node."""
        return int(self.activation_times.shape[1])

    @property
    def n_beats_deactivated(self) -> int:
        """Max number of deactivation events found on any single node."""
        return int(self.deactivation_times.shape[1])

    @property
    def n_events(self) -> int:
        """Convenience: the larger of the two event counts above."""
        return max(self.n_beats_activated, self.n_beats_deactivated)

    def __repr__(self) -> str:
        return (
            f"ActivationMaps("
            f"N_myo={self.N_myo:,}, "
            f"activation_events={self.n_beats_activated}, "
            f"deactivation_events={self.n_beats_deactivated}, "
            f"act_threshold={self.act_threshold:g} mV, "
            f"deact_threshold={self.deact_threshold:g} mV, "
            f"never_activated={(self.n_activations == 0).sum():,})"
        )

    # ------------------------------------------------------------------
    # Per-event maps
    # ------------------------------------------------------------------

    def activation_map(self, event: int = 1) -> np.ndarray:
        """
        Activation (depolarisation) time for every node, for the `event`-th
        beat (1-based: event=1 is the first beat). NaN where that node was
        never activated on that beat.
        """
        idx = event - 1
        if idx < 0 or idx >= self.activation_times.shape[1]:
            return np.full(self.N_myo, np.nan, dtype=np.float32)
        return self.activation_times[:, idx].copy()

    def deactivation_map(self, event: int = 1) -> np.ndarray:
        """Deactivation (repolarisation) time for the `event`-th beat."""
        idx = event - 1
        if idx < 0 or idx >= self.deactivation_times.shape[1]:
            return np.full(self.N_myo, np.nan, dtype=np.float32)
        return self.deactivation_times[:, idx].copy()

    def duration_map(self, event: int = 1) -> np.ndarray:
        """
        Above-threshold duration for the `event`-th beat
        (deactivation_map - activation_map); NaN if either is missing.
        """
        return self.deactivation_map(event) - self.activation_map(event)

    def apd_map(self, event: int = 1) -> np.ndarray:
        """
        Action-potential-duration map for the `event`-th beat: per-node
        deactivation_map - activation_map [ms]. NaN at any node missing
        either event (never activated, or activated but not yet
        repolarised at the time the recording ends). Alias of
        `duration_map`, named for readability at call sites that are
        specifically about APD rather than generic "duration".
        """
        return self.duration_map(event)

    # ------------------------------------------------------------------
    # Volumetric reconstruction
    # ------------------------------------------------------------------

    def field_on_grid(self, values: np.ndarray) -> np.ndarray:
        """
        Scatter a (N_myo,) per-node array back onto the (Nx,Ny,Nz) voxel
        grid it came from (background = NaN). Requires `grid_shape`, i.e.
        that the mask `S` was passed to `compute_activation_maps`.
        """
        if self.grid_shape is None:
            raise ValueError(
                "grid_shape is not set on this ActivationMaps object -- "
                "pass S to compute_activation_maps() to enable this.")
        grid = np.full(tuple(int(x) for x in self.grid_shape), np.nan, dtype=np.float32)
        vi = self.vox_idx
        grid[vi[:, 0], vi[:, 1], vi[:, 2]] = values
        return grid

    def activation_grid(self, event: int = 1) -> np.ndarray:
        return self.field_on_grid(self.activation_map(event))

    def deactivation_grid(self, event: int = 1) -> np.ndarray:
        return self.field_on_grid(self.deactivation_map(event))

    def apd_grid(self, event: int = 1) -> np.ndarray:
        return self.field_on_grid(self.apd_map(event))


# =============================================================================
# Compute
# =============================================================================

def compute_activation_maps(
    vm_snapshots,
    S: np.ndarray | None = None,
    *,
    act_threshold:   float = -40.0,
    deact_threshold: float = -40.0,
    min_event_gap_ms: float | None = None,
    save_path: str | Path | None = None,
) -> ActivationMaps:
    """
    Detect activation and deactivation events for every myocardial node from
    a loaded VmSnapshots object.

    Parameters
    ----------
    vm_snapshots : VmSnapshots
        As returned by load_vm_snapshots().
    S : (Nx,Ny,Nz) ndarray, optional
        The full binary myocardium mask used to build the simulation grid
        (e.g. cfg.PATH_S / hp.coordinates.load_geometry()). If given, its
        shape is stored as `grid_shape` (and node coverage is sanity
        checked against it) so activation_grid()/deactivation_grid() and
        plot_activation_maps() can scatter values back onto the full
        volume. If omitted, `grid_shape` is inferred from the bounding box
        of the node coordinates, which is fine for computing the maps
        themselves but may not exactly match the true simulation grid.
    act_threshold, deact_threshold : float
        Membrane voltage [mV] that marks the upstroke ("activated") and
        downstroke ("deactivated") crossings respectively. Defaults
        (-40 mV both) sit well above resting potential and below peak for
        the TTP06 ionic model used elsewhere in this package.
    min_event_gap_ms : float, optional
        Debounce: drop any crossing that follows the previous kept
        crossing (same node, same kind) by less than this many ms. Use
        this if numerical chatter around the threshold is producing
        spurious extra beats. None (default) disables debouncing.
    save_path : str or Path, optional
        If given, the result is written to this NPZ path immediately
        (equivalent to calling save_activation_maps() yourself).

    Returns
    -------
    ActivationMaps
    """
    Vm = vm_snapshots.Vm
    t  = vm_snapshots.time_vm
    N_myo = vm_snapshots.N_myo

    if vm_snapshots.n_frames < 2:
        raise ValueError(
            "vm_snapshots has fewer than 2 frames -- cannot detect "
            "threshold crossings. Re-run the simulation with a smaller "
            "vm_save_dt / longer T.")

    # ── Activation (upstroke) ────────────────────────────────────────────
    above_act = Vm >= act_threshold
    rising    = np.diff(above_act.astype(np.int8), axis=0) == 1
    idx_a, t_a = _interp_crossings(Vm, t, rising, act_threshold)
    activation_times, n_activations = _pad_events(idx_a, t_a, N_myo)
    activation_times = _apply_min_gap(activation_times, min_event_gap_ms)
    n_activations = (~np.isnan(activation_times)).sum(axis=1).astype(np.int32)

    # ── Deactivation (downstroke) ────────────────────────────────────────
    above_deact = Vm >= deact_threshold
    falling     = np.diff(above_deact.astype(np.int8), axis=0) == -1
    idx_d, t_d = _interp_crossings(Vm, t, falling, deact_threshold)
    deactivation_times, n_deactivations = _pad_events(idx_d, t_d, N_myo)
    deactivation_times = _apply_min_gap(deactivation_times, min_event_gap_ms)
    n_deactivations = (~np.isnan(deactivation_times)).sum(axis=1).astype(np.int32)

    # ── Grid shape / sanity check against S ──────────────────────────────
    grid_shape = None
    if S is not None:
        S = np.asarray(S)
        grid_shape = np.array(S.shape, dtype=np.int32)
        vi = vm_snapshots.vox_idx
        in_mask = S[vi[:, 0], vi[:, 1], vi[:, 2]] != 0
        if not in_mask.all():
            n_out = int((~in_mask).sum())
            print(f"[compute_activation_maps] WARNING: {n_out:,} myocardial "
                  f"nodes fall outside the provided S mask -- make sure S "
                  f"matches the mask the simulation actually used.")
    else:
        vi = vm_snapshots.vox_idx
        grid_shape = (vi.max(axis=0) + 1).astype(np.int32)
        print("[compute_activation_maps] no S mask passed -- grid_shape "
              "inferred from the node bounding box. Pass S for an exact "
              "match with the simulation grid (needed for the 'full' vs "
              "'cut' views in plot_activation_maps()).")

    maps = ActivationMaps(
        node_ids           = vm_snapshots.node_ids.astype(np.int32),
        coords_mm          = vm_snapshots.coords_mm.astype(np.float32),
        vox_idx            = vm_snapshots.vox_idx.astype(np.int32),
        grid_shape         = grid_shape,
        activation_times   = activation_times,
        deactivation_times = deactivation_times,
        n_activations      = n_activations,
        n_deactivations    = n_deactivations,
        act_threshold      = float(act_threshold),
        deact_threshold    = float(deact_threshold),
        source_vm_path     = str(vm_snapshots.path),
        path               = Path(save_path) if save_path is not None else Path("."),
    )

    n_never_act = int((n_activations == 0).sum())
    print(f"[compute_activation_maps] {maps!r}")
    if n_never_act:
        print(f"  {n_never_act:,} / {N_myo:,} nodes never crossed "
              f"{act_threshold:g} mV (will render gray).")

    if save_path is not None:
        save_activation_maps(maps, save_path)

    return maps


# =============================================================================
# Save / load
# =============================================================================

def save_activation_maps(maps: ActivationMaps, path: str | Path) -> None:
    """Write an ActivationMaps object to a compressed NPZ."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        node_ids           = maps.node_ids.astype(np.int32),
        coords_mm          = maps.coords_mm.astype(np.float32),
        vox_idx            = maps.vox_idx.astype(np.int32),
        activation_times   = maps.activation_times.astype(np.float32),
        deactivation_times = maps.deactivation_times.astype(np.float32),
        n_activations      = maps.n_activations.astype(np.int32),
        n_deactivations    = maps.n_deactivations.astype(np.int32),
        act_threshold      = np.float32(maps.act_threshold),
        deact_threshold    = np.float32(maps.deact_threshold),
        source_vm_path     = str(maps.source_vm_path),
    )
    if maps.grid_shape is not None:
        payload["grid_shape"] = np.asarray(maps.grid_shape, dtype=np.int32)

    np.savez_compressed(path, **payload)
    size_mb = path.stat().st_size / 1e6
    print(f"[save_activation_maps] {path}  ({size_mb:.1f} MB)")
    maps.path = path


def load_activation_maps(path: str | Path) -> ActivationMaps:
    """
    Load an activation_maps.npz file produced by compute_activation_maps().

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    KeyError           if a required array is missing (wrong/old file).
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"activation_maps file not found: {path}")

    d = np.load(str(path), allow_pickle=False)

    required = {"node_ids", "coords_mm", "vox_idx", "activation_times",
                "deactivation_times", "n_activations", "n_deactivations",
                "act_threshold", "deact_threshold"}
    missing = required - set(d.files)
    if missing:
        raise KeyError(
            f"activation_maps NPZ is missing keys: {missing}\n"
            f"  (found: {d.files})\n"
            f"  Re-run compute_activation_maps() to regenerate the file.")

    maps = ActivationMaps(
        node_ids           = d["node_ids"],
        coords_mm          = d["coords_mm"],
        vox_idx            = d["vox_idx"],
        activation_times   = d["activation_times"],
        deactivation_times = d["deactivation_times"],
        n_activations      = d["n_activations"],
        n_deactivations    = d["n_deactivations"],
        act_threshold      = float(d["act_threshold"]),
        deact_threshold    = float(d["deact_threshold"]),
        grid_shape         = d["grid_shape"] if "grid_shape" in d.files else None,
        source_vm_path     = str(d["source_vm_path"]) if "source_vm_path" in d.files else "",
        path               = path,
    )
    print(f"[load_activation_maps] {maps!r}")
    return maps

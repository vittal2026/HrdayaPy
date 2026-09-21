"""
load_vm_snapshots.py
====================
Utility for loading and querying the Vm volumetric snapshot file produced by
the coupled Purkinje–myocardium monodomain solver.

NPZ schema (written by _save_vm_snapshots in purkinje_myocardium_pipeline.py)
------------------------------------------------------------------------------
  node_ids   (N_myo,)             int32    Global solver index  (Np + local_id)
  coords_mm  (N_myo, 3)           float32  Physical xyz [mm]
  vox_idx    (N_myo, 3)           int32    Voxel ijk in original mask
  Vm         (n_frames, N_myo)    float32  Membrane voltage [mV]
  time_vm    (n_frames,)          float32  Simulation time [ms] of each frame
  dt_solver  scalar               float32  Solver step [ms]
  vm_save_dt scalar               float32  Snapshot interval [ms]

Typical usage
-------------
    from simulation.functions.load_vm_snapshots import load_vm_snapshots

    snap = load_vm_snapshots("simulation/outputs/vm_snapshots.npz")
    # snap.Vm           → (n_frames, N_myo) float32
    # snap.coords_mm    → (N_myo, 3)        float32
    # snap.time_vm      → (n_frames,)       float32

    # Vm at a single node over time
    v = snap.vm_at_node(42)                     # (n_frames,)

    # Closest node to a physical point
    node_id = snap.nearest_node([35.0, 20.0, 80.0])
    v = snap.vm_at_node(node_id)

    # Full spatial field at a given time
    field, t_actual = snap.field_at_time(150.0)  # (N_myo,), float

    # Nodes within 5 mm of a point
    ids = snap.nodes_in_sphere([35.0, 20.0, 80.0], radius_mm=5.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VmSnapshots:
    """
    Container for the vm_snapshots.npz data.

    Attributes
    ----------
    node_ids   : (N_myo,)          Global solver indices (Np + local_id).
    coords_mm  : (N_myo, 3)        Physical xyz coordinates [mm].
    vox_idx    : (N_myo, 3)        Voxel ijk indices in the mask array.
    Vm         : (n_frames, N_myo) Membrane voltage [mV].
    time_vm    : (n_frames,)       Simulation time of each frame [ms].
    dt_solver  : float             Solver time step [ms].
    vm_save_dt : float             Actual snapshot sampling interval [ms].
    path       : Path              Source file path.
    """

    node_ids:   np.ndarray
    coords_mm:  np.ndarray
    vox_idx:    np.ndarray
    Vm:         np.ndarray
    time_vm:    np.ndarray
    dt_solver:  float
    vm_save_dt: float
    path:       Path = field(default_factory=lambda: Path("."))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.Vm.shape[0])

    @property
    def N_myo(self) -> int:
        return int(self.Vm.shape[1])

    def __repr__(self) -> str:
        return (
            f"VmSnapshots("
            f"N_myo={self.N_myo:,}, "
            f"n_frames={self.n_frames}, "
            f"t=[{self.time_vm[0]:.2f}…{self.time_vm[-1]:.2f}] ms, "
            f"vm_save_dt={self.vm_save_dt:.4g} ms, "
            f"Vm=[{self.Vm.min():.1f}, {self.Vm.max():.1f}] mV)"
        )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def vm_at_node(self, local_id: int) -> np.ndarray:
        """
        Return the Vm time-series for myocardial node `local_id`.

        Parameters
        ----------
        local_id : int
            Local index into the myocardium array (0 … N_myo-1).
            To convert a global solver index g: local_id = g − node_ids[0].

        Returns
        -------
        (n_frames,) float32 array in mV.
        """
        if not (0 <= local_id < self.N_myo):
            raise IndexError(
                f"local_id {local_id} out of range [0, {self.N_myo})")
        return self.Vm[:, local_id]

    def nearest_node(self, point_mm: list[float] | np.ndarray) -> int:
        """
        Return the local index of the myocardial node closest to `point_mm`.

        Parameters
        ----------
        point_mm : array-like, shape (3,)
            Physical coordinate [mm] (x, y, z).

        Returns
        -------
        int  — local node index suitable for `vm_at_node`.
        """
        pt = np.asarray(point_mm, dtype=np.float32)
        dists = np.linalg.norm(self.coords_mm - pt[None, :], axis=1)
        return int(dists.argmin())

    def field_at_time(self, t_ms: float) -> tuple[np.ndarray, float]:
        """
        Return the Vm spatial field at the frame closest to simulation time
        `t_ms`.

        Parameters
        ----------
        t_ms : float
            Target simulation time [ms].

        Returns
        -------
        field    : (N_myo,) float32  — Vm values [mV] at that frame.
        t_actual : float             — Actual time [ms] of the returned frame.
        """
        idx = int(np.argmin(np.abs(self.time_vm - t_ms)))
        return self.Vm[idx], float(self.time_vm[idx])

    def nodes_in_sphere(
        self,
        centre_mm: list[float] | np.ndarray,
        radius_mm: float,
    ) -> np.ndarray:
        """
        Return local indices of all myocardial nodes within `radius_mm` of
        `centre_mm`.

        Parameters
        ----------
        centre_mm : array-like (3,)   Centre point [mm].
        radius_mm : float             Search radius [mm].

        Returns
        -------
        (K,) int32 array of local node indices.
        """
        c = np.asarray(centre_mm, dtype=np.float32)
        dists = np.linalg.norm(self.coords_mm - c[None, :], axis=1)
        return np.where(dists <= radius_mm)[0].astype(np.int32)

    def mean_vm_in_sphere(
        self,
        centre_mm: list[float] | np.ndarray,
        radius_mm: float,
    ) -> np.ndarray:
        """
        Mean Vm time-series over all nodes within `radius_mm` of `centre_mm`.

        Returns
        -------
        (n_frames,) float32, or zeros if no nodes fall inside the sphere.
        """
        ids = self.nodes_in_sphere(centre_mm, radius_mm)
        if ids.size == 0:
            return np.zeros(self.n_frames, dtype=np.float32)
        return self.Vm[:, ids].mean(axis=1)


# =============================================================================
# Loader
# =============================================================================

def load_vm_snapshots(path: str | Path) -> VmSnapshots:
    """
    Load a vm_snapshots.npz file produced by the coupled monodomain solver.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ file (with or without the .npz extension).

    Returns
    -------
    VmSnapshots
        Dataclass with arrays and convenience query methods.

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    KeyError           if a required array is missing (wrong/old file).
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"vm_snapshots file not found: {path}")

    d = np.load(str(path))

    required = {"node_ids", "coords_mm", "vox_idx", "Vm", "time_vm",
                "dt_solver", "vm_save_dt"}
    missing = required - set(d.files)
    if missing:
        raise KeyError(
            f"vm_snapshots NPZ is missing keys: {missing}\n"
            f"  (found: {d.files})\n"
            f"  Re-run the simulation to regenerate the file."
        )

    snap = VmSnapshots(
        node_ids   = d["node_ids"],
        coords_mm  = d["coords_mm"],
        vox_idx    = d["vox_idx"],
        Vm         = d["Vm"],
        time_vm    = d["time_vm"],
        dt_solver  = float(d["dt_solver"]),
        vm_save_dt = float(d["vm_save_dt"]),
        path       = path,
    )

    print(f"[load_vm_snapshots] {snap}")
    return snap

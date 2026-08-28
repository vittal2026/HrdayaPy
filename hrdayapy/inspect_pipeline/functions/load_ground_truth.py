"""
inspect_pipeline/functions/load_ground_truth.py
=================================================
Loaders for the Bratislava / EDGAR 2026 ECGI Challenge ground-truth files
(same .mat schema used by Bratislava_dataset/pick_torso_ecg.py).

    P###_torso.mat  -> struct 'torso'  { node (N,3), face (M,3) 0-based,
                                          leadlinks (128,) 0-based node idx }
    P###-ts.mat     -> struct 'ts'     { potvals (128, n_samples), ... }

Kept dependency-light (numpy + scipy.io only) so the registration step
(Step 13) doesn't need PyVista just to load points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio


def _load_struct(mat_path: Path, field_name: str | None = None):
    """Load a scalar MATLAB struct (shape (1,1) struct array) from a .mat file."""
    data = sio.loadmat(str(mat_path))
    if field_name is None:
        field_name = next(k for k in data.keys() if not k.startswith("__"))
    return data[field_name][0, 0]


def load_gt_torso_mesh(torso_mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the dataset's torso surface mesh.

    Returns
    -------
    nodes     : (N, 3) float64  mm, physical coordinates
    faces     : (M, 3) int64    0-based triangle indices into `nodes`
    leadlinks : (128,)  int64   0-based node indices of the recording electrodes
    """
    torso_mat_path = Path(torso_mat_path)
    if not torso_mat_path.exists():
        raise FileNotFoundError(
            f"Ground-truth torso mesh not found:\n    {torso_mat_path}\n"
            f"Set GT_DATA_DIR / GT_PATIENT_ID in config.py."
        )
    t = _load_struct(torso_mat_path, "torso")
    nodes = np.asarray(t["node"], dtype=np.float64)
    faces = np.asarray(t["face"], dtype=np.int64)          # already 0-based
    leadlinks = np.asarray(t["leadlinks"], dtype=np.int64).ravel()
    return nodes, faces, leadlinks


def load_gt_bspm(ts_mat_path: Path) -> np.ndarray:
    """
    Load the dataset's recorded body-surface potentials.

    Returns
    -------
    potvals : (128, n_samples) float64  one row per electrode, same order
              as torso.leadlinks.
    """
    ts_mat_path = Path(ts_mat_path)
    if not ts_mat_path.exists():
        raise FileNotFoundError(
            f"Ground-truth BSPM recording not found:\n    {ts_mat_path}\n"
            f"Set GT_DATA_DIR / GT_PATIENT_ID in config.py."
        )
    ts = _load_struct(ts_mat_path, "ts")
    return np.asarray(ts["potvals"], dtype=np.float64)

"""
ionic_models/region_map.py
===========================
Utilities for building the per-node region label array from a
multi-label segmentation mask.

The segmentation mask is a 3-D integer array of the same spatial shape
as the binary myocardium mask S, where each voxel holds an integer region
label.  Non-tissue voxels (outside S) are ignored — only tissue voxels
contribute nodes.

Conventional label scheme (not enforced — user may use any integers):

    0   Infarct / scar      → SilentCell
    1   Endocardium         → e.g. TTP06_Endo
    2   Mid-myocardium      → e.g. TTP06_MCell
    3   Epicardium          → e.g. TTP06_Epi
    4   Border zone         → e.g. a remodelled variant
    …   (user-defined)

If no segmentation mask is available, ``make_transmural_region_map``
reproduces the old three-layer behaviour: it derives labels from the
transmural coordinate phi using the same PHI_ENDO_MAX / PHI_EPI_MIN
thresholds as before.

Public API
----------
    region_map = region_map_from_segmentation(seg_mask, vox_idx)
    region_map = make_transmural_region_map(phi_nodes)
    validate_region_map(region_map, model_registry)   # optional sanity check
"""

from __future__ import annotations

import numpy as np
import warnings


# ── Conventional label constants ──────────────────────────────────────────────
LABEL_INFARCT  = 0
LABEL_ENDO     = 1
LABEL_MCELL    = 2
LABEL_EPI      = 3
LABEL_BORDER   = 4   # peri-infarct border zone — user assigns a model

# Transmural thresholds (used only by make_transmural_region_map)
PHI_ENDO_MAX = 0.35
PHI_EPI_MIN  = 0.65


def region_map_from_segmentation(
    seg_mask: np.ndarray,
    vox_idx:  np.ndarray,
) -> np.ndarray:
    """
    Extract per-node region labels from a multi-label segmentation mask.

    Parameters
    ----------
    seg_mask : (Nx, Ny, Nz) int array
        Multi-label segmentation mask, same spatial shape as the binary
        myocardium mask S.  Label values are arbitrary integers; the
        dispatcher maps them to model instances via the model_registry.
    vox_idx  : (N, 3) int array
        Tissue node voxel coordinates — the output of ``np.argwhere(S)``.
        These are the same coordinates used to build the Laplacian.

    Returns
    -------
    region_map : (N,) int32 array
        One integer region label per node, in the same order as vox_idx.
    """
    region_map = seg_mask[
        vox_idx[:, 0],
        vox_idx[:, 1],
        vox_idx[:, 2],
    ].astype(np.int32)
    return region_map


def make_transmural_region_map(
    phi_nodes: np.ndarray,
) -> np.ndarray:
    """
    Derive a three-layer region map from the transmural coordinate phi.

    This reproduces the old assign_phenotypes() behaviour using the new
    label convention (LABEL_ENDO / LABEL_MCELL / LABEL_EPI), so existing
    pipelines that do not have a segmentation mask can still use the
    dispatcher framework without change.

    Parameters
    ----------
    phi_nodes : (N,) float  transmural coordinate, 0 = endo, 1 = epi

    Returns
    -------
    region_map : (N,) int32
    """
    region_map = np.full(phi_nodes.shape, LABEL_MCELL, dtype=np.int32)
    region_map[phi_nodes <  PHI_ENDO_MAX] = LABEL_ENDO
    region_map[phi_nodes >= PHI_EPI_MIN]  = LABEL_EPI

    n_endo  = int((region_map == LABEL_ENDO).sum())
    n_mcell = int((region_map == LABEL_MCELL).sum())
    n_epi   = int((region_map == LABEL_EPI).sum())
    total   = len(region_map)
    print(
        f"[region_map] Transmural 3-layer split:  "
        f"endo={n_endo} ({100*n_endo/total:.1f}%)  "
        f"mcell={n_mcell} ({100*n_mcell/total:.1f}%)  "
        f"epi={n_epi} ({100*n_epi/total:.1f}%)"
    )
    return region_map


def validate_region_map(
    region_map:      np.ndarray,
    model_registry:  dict,
    warn_unlabelled: bool = True,
) -> None:
    """
    Check that every region label in region_map has a registered model.

    Labels absent from the registry will be silently handled by SilentCell
    (zero ionic current).  This function makes that assignment explicit by
    printing a warning so the user can decide whether it is intentional.

    Parameters
    ----------
    region_map      : (N,) int array
    model_registry  : dict[int, RegionIonicModel]
    warn_unlabelled : if True, emit a warning for unmapped labels
    """
    unique = np.unique(region_map)
    unmapped = [int(u) for u in unique if int(u) not in model_registry]

    if unmapped and warn_unlabelled:
        warnings.warn(
            f"[region_map] The following region labels have no entry in "
            f"model_registry and will use SilentCell (zero ionic current): "
            f"{unmapped}.  If this is intentional (e.g. infarct core), "
            f"suppress this warning with warn_unlabelled=False.",
            UserWarning,
            stacklevel=2,
        )

    mapped = [int(u) for u in unique if int(u) in model_registry]
    print(
        f"[region_map] Validation:  "
        f"mapped={mapped}  "
        f"silent (no model)={unmapped}"
    )

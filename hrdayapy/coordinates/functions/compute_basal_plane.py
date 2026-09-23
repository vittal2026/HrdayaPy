import numpy as np
import pyvista as pv
from scipy import ndimage as ndi

from .mesh_viz_utils import mask_to_pv_surface


def compute_basal_plane(
    S,
    axis,
    plot=False,
    voxel_size: float = 0.4,
    target_mm: float = 1.0,
):
    """
    Compute the basal plane index along the apico-basal axis.

    Parameters
    ----------
    S : np.ndarray (bool or {0,1})
        Binary myocardium mask.
    axis : int
        Apico-basal axis index (0, 1, or 2).
    voxel_size, target_mm : only used when plot=True -- passed to
        mesh_viz_utils.mask_to_pv_surface for the sanity-check surface.
        target_mm (default 1.0 mm) keeps that surface's reconstruction
        cost roughly constant regardless of voxel_size, instead of it
        silently getting denser (and slower) at a finer mesh.
    plot : bool, optional
        If True, visualize the basal plane location.

    Returns
    -------
    base_idx : int
        Slice index corresponding to the basal plane along `axis`.

    Notes
    -----
    The basal plane is identified as the boundary of the largest contiguous
    region along the apico-basal axis where two ventricular cavities are present.
    Between the start and end of this region, the slice with the larger myocardial
    perimeter is selected.

    """

    S = S.astype(bool)

    # -------------------------------------------------
    # Step 1: f(i) = 1 if slice i has >= 2 cavities
    # -------------------------------------------------

    def f(binary_vol, dim):
        """Return a 0/1 vector: slice i has value 1 iff it contains >= 3
        connected components (2 cavity lumens + 1 myocardium wall).

        Parameters
        ----------
        binary_vol : np.ndarray (bool)
            ``True`` where tissue is *absent* (cavity / background).
        dim : int
            Axis along which to take slices.
        """
        assert dim in (0, 1, 2)

        n_slices = binary_vol.shape[dim]
        out = np.zeros(n_slices, dtype=int)
        structure = np.ones((3, 3), dtype=int)

        for i in range(n_slices):
            slc = np.take(binary_vol, i, axis=dim)
            _, n_cc = ndi.label(slc, structure=structure)
            out[i] = 1 if n_cc == 3 else 0

        return out

    binary = S < 1   # True where there is no tissue (cavity / air)
    vec = f(binary, dim=axis)
    

    # -------------------------------------------------
    # Step 2: find contiguous islands where f == 1
    # -------------------------------------------------
    # indices where value changes (0→1 or 1→0)
    vec = np.asarray(vec).astype(int)

    # 1) Pad with zeros to catch islands touching the ends
    padded = np.r_[0, vec, 0]

    # 2) Find transitions
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]      # 0 → 1
    ends   = np.where(diff == -1)[0]     # 1 → 0

    # 3) Lengths of each 1-island
    lengths = ends - starts

    # 4) Index of the largest island
    i = np.argmax(lengths)

    # 5) Transitions in ORIGINAL vec indexing
    start_idx = starts[i]
    end_idx   = ends[i]

    transitions = np.array([start_idx, end_idx])

    slc0 = np.take(binary, start_idx, axis=axis)
    slc1 = np.take(binary, end_idx,   axis=axis)

    perim0 = np.count_nonzero(slc0 ^ ndi.binary_erosion(slc0))
    perim1 = np.count_nonzero(slc1 ^ ndi.binary_erosion(slc1))

    # -------------------------------------------------
    # Step 3: pick slice with larger myocardium perimeter
    # -------------------------------------------------
    base_idx = start_idx if perim0 >= perim1 else end_idx
        
    # -------------------------------------------------
    # Optional visualization
    # -------------------------------------------------
    if plot:
        # Centroid
        coords = np.argwhere(S)
        centroid = coords.mean(axis=0)

        
        # Plane
        normal = np.zeros(3)
        normal[axis] = 1.0

        plane_center = centroid.copy()
        plane_center[axis] = base_idx

        plane = pv.Plane(
            center=plane_center,
            direction=normal,
            i_size=max(S.shape),
            j_size=max(S.shape)
        )

        # Surface (Taubin-smoothed; replaces the old per-voxel-cell
        # ImageData()+threshold()+Laplacian .smooth() -- see
        # mesh_viz_utils.mask_to_pv_surface docstring)
        surface = mask_to_pv_surface(S, voxel_size=voxel_size, target_mm=target_mm,
                                      smooth_iter=20, pass_band=0.1)

        plotter = pv.Plotter()
        plotter.add_mesh(surface, color="lightgray", opacity=0.3)
        plotter.add_mesh(plane, color="red", opacity=0.4)
        plotter.add_axes()
        plotter.show()

    return base_idx

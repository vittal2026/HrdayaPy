]import numpy as np
import pyvista as pv
from scipy.ndimage import label

from .mesh_viz_utils import mask_to_pv_surface


def find_apico_basal_axis(
    S,
    min_area=100,
    plot=False,
    voxel_size: float = 0.4,
    target_mm: float = 1.0,
):
    """
    Determine the apico-basal axis from a binary myocardium mask.

    Parameters
    ----------
    S : np.ndarray (bool or {0,1})
        Binary myocardium volume.
    min_area : int, optional
        Minimum cavity area (in pixels) to be counted as a lumen.
    plot : bool, optional
        If True, visualize the detected axis as a line through the centroid.

    Returns
    -------
    axis : int
        Apico-basal axis index (0=x, 1=y, 2=z).
    """

    S = S.astype(bool)

    # -------------------------------------------------
    # Helper: count slices with >=2 cavities
    # -------------------------------------------------
    def count_two_cavity_slices(axis):
        score = 0
        for i in range(S.shape[axis]):
            sl = np.take(S, i, axis=axis)
            if sl.sum() == 0:
                continue

            bg = ~sl
            labels, _ = label(bg)
            sizes = np.bincount(labels.ravel())[1:]

            if np.sum(sizes > min_area) >= 2:
                score += 1

        return score

    # -------------------------------------------------
    # Score each axis
    # -------------------------------------------------
    scores = {ax: count_two_cavity_slices(ax) for ax in (0, 1, 2)}
    axis = max(scores, key=scores.get)

    # -------------------------------------------------
    # Optional visualization
    # -------------------------------------------------
    if plot:
        # Centroid in voxel coordinates
        coords = np.argwhere(S)
        centroid = coords.mean(axis=0)

        # Axis direction vector
        direction = np.zeros(3)
        direction[axis] = 1.0

        # Line endpoints
        length = max(S.shape) * 0.6
        p0 = centroid - length * direction
        p1 = centroid + length * direction

        # PyVista volume (for context) -- Taubin-smoothed; replaces the old
        # per-voxel-cell ImageData()+threshold()+Laplacian .smooth() (see
        # mesh_viz_utils.mask_to_pv_surface docstring)
        surface = mask_to_pv_surface(S, voxel_size=voxel_size, target_mm=target_mm,
                                      smooth_iter=20, pass_band=0.1)

        axis_line = pv.Line(p0, p1)

        plotter = pv.Plotter()
        plotter.add_mesh(surface, color="lightgray", opacity=0.3)
        plotter.add_mesh(axis_line, color="red", line_width=5)
        plotter.add_point_labels(
            [centroid],
            ["centroid"],
            point_size=12,
            font_size=14
        )
        plotter.add_axes()
        plotter.show()

    return axis

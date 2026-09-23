import nrrd
import numpy as np
import pyvista as pv
from scipy.ndimage import zoom

from ._nrrd_spacing import resolve_original_spacing
from .mesh_viz_utils import mask_to_pv_surface


def load_muscle_mask(
    seg_path,
    target_spacing=None,
    original_spacing=None,
    target_shape=None,
    pad_width=0,
    plot=False,
    threshold=0,
    plot_voxel_size: float = 0.4,
    plot_target_mm: float = 1.0,
):
    """
    Load a segmentation NRRD and return a binary myocardium mask, correctly
    resampled to a known physical voxel size.

    Parameters
    ----------
    seg_path : str
        Path to the .nrrd segmentation file.
    target_spacing : float or (3,) sequence of float, optional
        Desired output spacing in mm/voxel (isotropic if a single float).
        This is the physically-correct way to control resolution: the
        function resamples so the OUTPUT grid actually has this spacing,
        regardless of the file's native resolution. Preferred over
        target_shape for anything new.
    original_spacing : None | "nrrd" | float | (3,) sequence of float, optional
        States what the NRRD's native spacing (mm/voxel) actually is, so
        the resample factor (original_spacing / target_spacing) can be
        computed correctly. Three forms:

          * None (default)   -- no information assumed; the loaded array
                                 is used as-is and simply labelled with
                                 target_spacing (zoom factor = 1, i.e. NO
                                 interpolation happens). This reproduces
                                 the old behaviour and is only correct if
                                 the file already happens to be sampled at
                                 target_spacing -- use this only if you
                                 know that to be true.
          * "nrrd"            -- read the true spacing from the NRRD
                                 header itself (`space directions` or
                                 `spacings` field) and resample from that.
                                 Use this whenever the file's header is
                                 trustworthy -- the common case.
          * float or 3-tuple  -- you state the original spacing explicitly
                                 (e.g. original_spacing=1.0 if you know
                                 the segmentation was exported at 1 mm),
                                 overriding/bypassing the header.

        Ignored if target_spacing is not given.
    target_shape : tuple of int, optional
        DEPRECATED legacy path, kept only for backward compatibility.
        Forces the output to an exact voxel count regardless of physical
        spacing (naive count-to-count zoom) -- this is the behaviour that
        silently rescales the heart if the assumed spacing is wrong.
        Prefer target_spacing (+ original_spacing) instead. Ignored if
        target_spacing is given.
    pad_width : int or tuple, optional
        Number of zero-valued slices to pad on EACH face, applied AFTER
        resampling (so it's in units of the output/target spacing).
        - int: same padding on all faces
        - tuple of 3 ints: (px, py, pz)
    plot : bool, optional
        If True, visualize the binary volume using PyVista.
    threshold : float or int, optional
        Values > threshold are considered myocardium (default: 0).

    Returns
    -------
    S : np.ndarray (uint8)
        Binary myocardium mask with values {0, 1}.
    header : dict
        NRRD header (returned in case spacing/origin metadata is needed).
    spacing_mm : (3,) float64 ndarray
        The spacing (mm/voxel) that S is actually sampled at after this
        call -- i.e. target_spacing when that path was used, or the
        best-effort legacy estimate when only target_shape was given, or
        the native NRRD spacing if neither was given. Downstream stages
        (Purkinje growth, monodomain solve, ECG forward solve) should use
        THIS value as their voxel_size rather than a separately hand-typed
        constant, so the two can't drift apart.
    """

    # -----------------------------
    # Load segmentation
    # -----------------------------
    data, header = nrrd.read(seg_path)

    # Binary mask
    S = (data > threshold).astype(np.uint8)

    # -----------------------------
    # Optional resampling
    # -----------------------------
    if target_spacing is not None:
        # Physically-correct path: resample by (original / target) spacing
        # ratio, so the output grid has a known, requested mm/voxel size.
        target_spacing_arr = np.broadcast_to(
            np.asarray(target_spacing, dtype=np.float64), (3,)
        ).copy()
        orig_spacing_arr = resolve_original_spacing(
            original_spacing, target_spacing_arr, header
        )

        zoom_factors = orig_spacing_arr / target_spacing_arr
        S = zoom(S, zoom_factors, order=0).astype(np.uint8)
        spacing_mm = target_spacing_arr

    elif target_shape is not None:
        # Legacy path: force an exact voxel count, independent of physical
        # spacing. Kept only for backward compatibility -- this is the
        # behaviour that silently rescales the heart if the true spacing
        # doesn't match what downstream code assumes.
        if len(target_shape) != 3:
            raise ValueError("target_shape must be a 3-tuple (nx, ny, nz)")

        native_spacing_arr = resolve_original_spacing(
            "nrrd" if original_spacing is None else original_spacing,
            np.ones(3, dtype=np.float64),
            header,
        )
        zoom_factors = np.array(target_shape, dtype=np.float64) / np.array(S.shape, dtype=np.float64)
        S = zoom(S, zoom_factors, order=0).astype(np.uint8)
        # Best-effort spacing estimate implied by this legacy resize --
        # not guaranteed meaningful, since target_shape was chosen without
        # regard to physical size.
        spacing_mm = native_spacing_arr / zoom_factors

    else:
        # No resampling requested at all -- return the native NRRD spacing.
        spacing_mm = resolve_original_spacing("nrrd", np.ones(3, dtype=np.float64), header)

    # -----------------------------
    # Optional padding (applied after resampling, in output-spacing units)
    # -----------------------------
    if pad_width:
        if isinstance(pad_width, int):
            pad = ((pad_width, pad_width),
                   (pad_width, pad_width),
                   (pad_width, pad_width))
        elif len(pad_width) == 3:
            pad = tuple((p, p) for p in pad_width)
        else:
            raise ValueError(
                "pad_width must be an int or a 3-tuple (px, py, pz)"
            )

        S = np.pad(S, pad, mode="constant", constant_values=0)

    # -----------------------------
    # Optional visualization
    # -----------------------------
    if plot:
        # Taubin-smoothed; replaces the old per-voxel-cell
        # ImageData()+threshold()+Laplacian .smooth() (see
        # mesh_viz_utils.mask_to_pv_surface docstring). plot_voxel_size is
        # an approximate value for sizing this debug plot's mesh density
        # only -- it isn't traced through target_spacing/original_spacing's
        # resampling logic above, so it won't exactly match S's true
        # physical voxel size if you resampled; that's fine here since
        # only the mesh's visual density depends on it, not correctness.
        surface = mask_to_pv_surface(S, voxel_size=plot_voxel_size, target_mm=plot_target_mm,
                                      smooth_iter=30, pass_band=0.1)

        plotter = pv.Plotter()
        plotter.add_mesh(
            surface,
            color="salmon",
            smooth_shading=True
        )
        plotter.add_axes()
        plotter.show()

    return S, header, spacing_mm

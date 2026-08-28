import numpy as np
from pathlib import Path
from typing import Tuple

def save_purkinje(
    path: str,
    nodes: np.ndarray,
    elements: np.ndarray,
    activation_times: np.ndarray
) -> None:
    """
    Save a Purkinje network to a compressed NumPy archive.

    Parameters
    ----------
    path : str
        Output file path. The '.npz' extension will be added if not present.
    nodes : np.ndarray
        Node positions in voxel coordinates, shape (n_nodes, 3).
    elements : np.ndarray
        Element connectivity, shape (n_elements, 2).
    activation_times : np.ndarray
        Activation time at each node in milliseconds, shape (n_nodes,).
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    np.savez_compressed(
        path,
        nodes=nodes,
        elements=elements,
        activation_times=activation_times
    )


def load_purkinje(
    path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a Purkinje network from a compressed NumPy archive.

    Parameters
    ----------
    path : str
        Path to the '.npz' file produced by save_purkinje.

    Returns
    -------
    nodes : np.ndarray
        Node positions in voxel coordinates, shape (n_nodes, 3).
    elements : np.ndarray
        Element connectivity, shape (n_elements, 2).
    activation_times : np.ndarray
        Activation time at each node in milliseconds, shape (n_nodes,).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    KeyError
        If the file is missing one of the expected arrays.
    """
    path = Path(path)
    if not path.exists():
        # Also try adding .npz if the user omitted the extension
        path_npz = path.with_suffix(".npz")
        if path_npz.exists():
            path = path_npz
        else:
            raise FileNotFoundError(f"No Purkinje file found at '{path}'")

    with np.load(path) as data:
        try:
            nodes = data["nodes"]
            elements = data["elements"]
            activation_times = data["activation_times"]
        except KeyError as e:
            raise KeyError(
                f"File '{path}' is missing expected array {e}. "
                "Was it saved with save_purkinje()?"
            ) from e

    return nodes, elements, activation_times

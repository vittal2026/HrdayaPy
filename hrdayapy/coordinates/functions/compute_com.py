import numpy as np

def compute_com(binary_array):
    """
    Calculate the center of mass (COM) coordinates of a binary segmentation.
    
    Parameters:
    -----------
    binary_array : np.ndarray
        A binary array (2D or 3D) where non-zero values represent the object
        
    Returns:
    --------
    tuple
        Coordinates of the center of mass (y, x) for 2D or (z, y, x) for 3D
        Returns None if the array is empty (all zeros)
    
    Examples:
    ---------
    >>> arr = np.array([[0, 0, 0],
    ...                 [0, 1, 1],
    ...                 [0, 1, 1]])
    >>> compute_com(arr)
    (1.5, 1.5)
    """
    # Check if array is empty
    if not np.any(binary_array):
        return None
    
    # Get indices of all non-zero elements
    coords = np.where(binary_array > 0)
    
    # Calculate mean position along each axis
    com = tuple(np.mean(coord) for coord in coords)
    
    return com


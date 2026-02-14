import numpy as np
import scipy.ndimage as ndimage
try:
    from skimage import feature, morphology
except ImportError:
    # Feature detection might be optional depending on environment
    pass

def find_conv_lines(u: np.ndarray, v: np.ndarray, threshold: float = -1e-5) -> np.ndarray:
    """Detect horizontal convergence lines from wind components.

    This implementation uses spatial derivatives to compute convergence and then
    identifies linear features.

    Args:
        u (np.ndarray): Zonal wind component.
        v (np.ndarray): Meridional wind component.
        threshold (float): Threshold for convergence intensity. Defaults to -1e-5.

    Returns:
        np.ndarray: A binary mask of detected convergence lines.
    
    Note:
        Adapted from the original legacy implementation in `mogreps_process.py`.
    """
    # Compute divergence (negative is convergence)
    du_dx = np.gradient(u, axis=1)
    dv_dy = np.gradient(v, axis=0)
    convergence = -(du_dx + dv_dy)
    
    # Thresholding
    mask = convergence > -threshold
    
    # Skeletonize to find lines (if skimage available)
    try:
        mask = morphology.skeletonize(mask)
    except NameError:
        pass
        
    return mask

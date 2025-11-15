# src/methods/illumination.py

import numpy as np
from typing import Tuple
from skimage.restoration import denoise_tv_chambolle

def estimate_illumination_max_rgb(img: np.ndarray) -> np.ndarray:
    """
    Initial illumination estimate using Max-RGB.

    Args:
        img: float32 RGB image in [0,1], shape (H,W,3).

    Returns:
        Illumination map T0, shape (H,W), float32 in [0,1].
    """
    assert img.ndim == 3 and img.shape[2] == 3
    T0 = img.max(axis=2)
    return T0.astype(np.float32)

def refine_illumination_tv(T0: np.ndarray, weight: float = 0.1, n_iter: int = 100) -> np.ndarray:
    """
    Refine illumination map using total-variation denoising for smoothness.

    Args:
        T0: Initial illumination (H,W).
        weight: TV weight parameter.
        n_iter: Number of iterations.

    Returns:
        Refined illumination T (H,W).
    """
    T = denoise_tv_chambolle(T0, weight=weight, max_num_iter=n_iter)
    return T.astype(np.float32)

def normalize_illumination(T: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize illumination to [0,1].

    Args:
        T: Illumination map (H,W).

    Returns:
        Normalized illumination T_norm in [0,1].
    """
    t_min = float(T.min())
    t_max = float(T.max())
    T_norm = (T - t_min) / (t_max - t_min + eps)
    return T_norm.astype(np.float32)

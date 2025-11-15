# src/methods/baselines.py

import numpy as np
import cv2

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization on the luminance channel.

    Args:
        img: float32 RGB [0,1].

    Returns:
        Equalized RGB image.
    """
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    L_eq = cv2.equalizeHist(L)
    lab_eq = cv2.merge([L_eq, a, b])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return rgb_eq.astype(np.float32) / 255.0

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    CLAHE on luminance channel.

    Args:
        img: float32 RGB [0,1].

    Returns:
        CLAHE-enhanced RGB image.
    """
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_clahe = clahe.apply(L)
    lab_clahe = cv2.merge([L_clahe, a, b])
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return rgb_clahe.astype(np.float32) / 255.0

def apply_gamma(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """
    Global gamma correction.

    Args:
        img: float32 RGB [0,1].
        gamma: gamma exponent (<1 brightens, >1 darkens).

    Returns:
        Gamma-corrected image.
    """
    img = np.clip(img, 0.0, 1.0)
    return img ** gamma

def apply_simple_retinex(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Simple single-scale Retinex-style enhancement.

    Args:
        img: float32 RGB [0,1].

    Returns:
        Enhanced RGB image.
    """
    img = np.clip(img, 0.0, 1.0)
    # Avoid log(0)
    eps = 1e-6
    log_I = np.log(img + eps)
    # Gaussian blur per channel
    blurred = np.stack(
        [cv2.GaussianBlur(img[..., c], (0, 0), sigma) for c in range(3)],
        axis=2
    )
    log_blur = np.log(blurred + eps)
    R = log_I - log_blur
    # Normalize to [0,1]
    R_min = R.min()
    R_max = R.max()
    R_norm = (R - R_min) / (R_max - R_min + eps)
    return R_norm.astype(np.float32)

def lime_enhance_simplified(img: np.ndarray) -> np.ndarray:
    """
    Simplified LIME-style enhancement:
        - Illumination = max RGB
        - Refine with smoothing (could call your refine_illumination_tv)
        - Divide image by illumination^gamma

    You can later hook this into illumination.py functions.
    """
    from .illumination import estimate_illumination_max_rgb, refine_illumination_tv

    I = np.clip(img, 0.0, 1.0)
    T0 = estimate_illumination_max_rgb(I)
    T = refine_illumination_tv(T0, weight=0.1, n_iter=50)
    eps = 1e-6
    gamma = 0.8
    enhanced = I / (np.maximum(T, eps) ** gamma)[..., None]
    return np.clip(enhanced, 0.0, 1.0)

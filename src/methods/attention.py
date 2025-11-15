# src/methods/attention.py

import numpy as np
from src.config import ILLUM_ALPHA, GAMMA_MIN, GAMMA_MAX

def compute_darkness_attention(T_norm: np.ndarray, alpha: float = ILLUM_ALPHA) -> np.ndarray:
    """
    Compute darkness-based attention from normalized illumination.

    Args:
        T_norm: Illumination normalized to [0,1].
        alpha: Exponent to emphasize darker areas (alpha > 1).

    Returns:
        Attention map A_d' in [0,1], high where dark.
    """
    A_d = 1.0 - T_norm
    A_d = np.clip(A_d, 0.0, 1.0)
    A_d_prime = A_d ** alpha
    return A_d_prime.astype(np.float32)

def map_attention_to_gamma(
    attention: np.ndarray,
    gamma_min: float = GAMMA_MIN,
    gamma_max: float = GAMMA_MAX
) -> np.ndarray:
    """
    Map attention values to per-pixel gamma values.

    Args:
        attention: Attention map in [0,1] (higher = darker).
        gamma_min: Gamma at max attention (strong brightening).
        gamma_max: Gamma at zero attention (little/no brightening).

    Returns:
        Gamma map same shape as attention.
    """
    att = np.clip(attention, 0.0, 1.0)
    gamma_map = gamma_min + (gamma_max - gamma_min) * (1.0 - att)
    return gamma_map.astype(np.float32)

def attention_mask_for_clahe(attention: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    """
    Binary mask indicating where to apply local contrast enhancement.

    Args:
        attention: Attention map [0,1].
        thresh: Threshold; values > thresh considered dark enough for CLAHE.

    Returns:
        Boolean mask (H,W).
    """
    return (attention > thresh)

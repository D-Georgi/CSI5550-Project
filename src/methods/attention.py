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
    gamma_min: float = 0.7,
    gamma_max: float = 1.0,
) -> np.ndarray:
    """
    Map attention to gamma in a gentler, piecewise way:
      - Very dark (A > 0.7): gamma in [0.75, 0.85]
      - Medium (0.3 < A <= 0.7): gamma in [0.85, 1.0]
      - Bright (A <= 0.3): gamma ~ 1.0 (no change)
    """
    A = np.clip(attention, 0.0, 1.0)
    gamma = np.ones_like(A, dtype=np.float32)

    # medium dark
    mask_med = (A > 0.3) & (A <= 0.7)
    gamma[mask_med] = 0.85 + (1.0 - 0.85) * (0.7 - A[mask_med]) / (0.7 - 0.3 + 1e-6)

    # very dark
    mask_dark = A > 0.7
    gamma[mask_dark] = 0.75 + (0.85 - 0.75) * (1.0 - A[mask_dark]) / (1.0 - 0.7 + 1e-6)

    return gamma

def attention_mask_for_clahe(attention: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """
    Binary mask indicating where to apply local contrast enhancement.

    Args:
        attention: Attention map [0,1].
        thresh: Threshold; values > thresh considered dark enough for CLAHE.

    Returns:
        Boolean mask (H,W).
    """
    return (attention > thresh)

def compute_scaled_attention(T_norm: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    SIAM-style scaled illumination attention.
    Assume T_norm is illumination in [0,1] (higher = brighter).
    We first build naive Att = 1 - T_norm, then scale:
      S_Att = -Att * (Att - 2)
    Optionally apply alpha exponent to tweak shape.
    """
    I = np.clip(T_norm, 0.0, 1.0)
    Att = 1.0 - I
    S_Att = -Att * (Att - 2.0)  # Eq. (3) in paper
    if alpha != 1.0:
        S_Att = np.clip(S_Att, 0.0, 1.0) ** alpha
    return np.clip(S_Att, 0.0, 1.0).astype(np.float32)

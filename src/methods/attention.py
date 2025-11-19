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
    Map attention to gamma in a gentler, piecewise way, respecting the input ranges.

    Logic:
      - Very dark (A > 0.7): gamma scales from gamma_mid -> gamma_min
      - Medium (0.3 < A <= 0.7): gamma scales from gamma_max -> gamma_mid
      - Bright (A <= 0.3): gamma = gamma_max

    This preserves the 'knee' shape but allows gamma_min to actually drive the brightness.
    """
    A = np.clip(attention, 0.0, 1.0)
    gamma = np.full_like(A, gamma_max, dtype=np.float32)

    # Calculate an intermediate gamma value to maintain the piecewise curve
    # Originally: min=0.75, max=1.0 -> mid=0.85. (0.85 is ~40% up from min)
    gamma_range = gamma_max - gamma_min
    gamma_mid = gamma_min + (0.4 * gamma_range)

    # 1. Medium Dark (0.3 < A <= 0.7)
    # Interpolate between gamma_max (at A=0.3) and gamma_mid (at A=0.7)
    mask_med = (A > 0.3) & (A <= 0.7)
    if np.any(mask_med):
        # Normalized position in this band (0.0 at A=0.3, 1.0 at A=0.7)
        t = (A[mask_med] - 0.3) / (0.7 - 0.3 + 1e-6)
        # Linearly interpolate
        gamma[mask_med] = gamma_max - t * (gamma_max - gamma_mid)

    # 2. Very Dark (A > 0.7)
    # Interpolate between gamma_mid (at A=0.7) and gamma_min (at A=1.0)
    mask_dark = A > 0.7
    if np.any(mask_dark):
        # Normalized position in this band (0.0 at A=0.7, 1.0 at A=1.0)
        t = (A[mask_dark] - 0.7) / (1.0 - 0.7 + 1e-6)
        # Linearly interpolate
        gamma[mask_dark] = gamma_mid - t * (gamma_mid - gamma_min)

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
    first build naive Att = 1 - T_norm, then scale:
      S_Att = -Att * (Att - 2)
    apply alpha exponent to tweak shape.
    """
    I = np.clip(T_norm, 0.0, 1.0)
    Att = 1.0 - I
    S_Att = -Att * (Att - 2.0)  # Eq. (3) in paper
    if alpha != 1.0:
        S_Att = np.clip(S_Att, 0.0, 1.0) ** alpha
    return np.clip(S_Att, 0.0, 1.0).astype(np.float32)

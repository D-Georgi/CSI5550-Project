# src/methods/denoising.py

import numpy as np
import cv2

def bilateral_denoise(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Apply bilateral filter to RGB image (channel-wise).

    Args:
        img: float32 RGB [0,1].

    Returns:
        Denoised image.
    """
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = np.zeros_like(u8)
    for c in range(3):
        out[..., c] = cv2.bilateralFilter(u8[..., c], d, sigma_color, sigma_space)
    return (out.astype(np.float32) / 255.0)

def attention_weighted_blend(
    original: np.ndarray,
    denoised: np.ndarray,
    attention: np.ndarray
) -> np.ndarray:
    """
    Blend original and denoised images based on attention (more denoising when attention is high).

    Args:
        original: Original image [0,1].
        denoised: Denoised version [0,1].
        attention: Attention map [0,1] (H,W).

    Returns:
        Blended image [0,1].
    """
    att = attention[..., None]  # broadcast to (H,W,1)
    return (1 - att) * original + att * denoised

# src/methods/enhancement.py

import numpy as np
from .illumination import estimate_illumination_max_rgb, refine_illumination_tv, normalize_illumination
from .attention import compute_darkness_attention, map_attention_to_gamma, attention_mask_for_clahe
from .denoising import bilateral_denoise, attention_weighted_blend
from .baselines import apply_clahe

def enhance_with_illumination_attention(
    img: np.ndarray,
    tv_weight: float = 0.1,
    tv_iter: int = 50,
    apply_local_contrast: bool = True,
    apply_denoising_step: bool = True
) -> dict:
    """
    Full proposed pipeline: illumination-based attention for non-deep enhancement.

    Args:
        img: float32 RGB [0,1] low-light image.
        tv_weight: TV refinement weight for illumination.
        tv_iter: Number of iterations for TV refinement.
        apply_local_contrast: Whether to use attention-guided CLAHE.
        apply_denoising_step: Whether to use attention-guided denoising.

    Returns:
        dict with intermediate outputs:
            {
              "illumination": T,
              "attention": A,
              "gamma_map": gamma_map,
              "enhanced_gamma": img_gamma,
              "enhanced_final": img_final
            }
    """
    I = np.clip(img, 0.0, 1.0)

    # 1) Illumination estimation & refinement
    T0 = estimate_illumination_max_rgb(I)
    T = refine_illumination_tv(T0, weight=tv_weight, n_iter=tv_iter)
    T_norm = normalize_illumination(T)

    # 2) Attention from illumination
    A = compute_darkness_attention(T_norm)

    ultra_dark_mask = (T_norm < 0.05)

    gamma_map = map_attention_to_gamma(A)

    # 3) Per-pixel gamma correction
    img_gamma = _apply_per_pixel_gamma(I, gamma_map)

    # 4) Optional attention-guided local contrast (CLAHE in dark regions)
    if apply_local_contrast:
        mask = attention_mask_for_clahe(A)
        # apply CLAHE only where mask = True
        img_local = _apply_clahe_masked(img_gamma, mask)
    else:
        img_local = img_gamma

    # 5) Optional attention-guided denoising
    if apply_denoising_step:
        den = bilateral_denoise(img_local)
        # Boost denoising weight in ultra-dark areas
        A_boost = A.copy()
        A_boost[ultra_dark_mask] = 1.0
        img_final = attention_weighted_blend(img_local, den, A_boost)
    else:
        img_final = img_local

    return {
        "illumination": T,
        "attention": A,
        "gamma_map": gamma_map,
        "enhanced_gamma": img_gamma,
        "enhanced_final": np.clip(img_final, 0.0, 1.0),
    }

def _apply_per_pixel_gamma(img: np.ndarray, gamma_map: np.ndarray) -> np.ndarray:
    """Apply per-pixel gamma correction given a gamma map."""
    eps = 1e-6
    return np.power(np.clip(img, 0.0, 1.0) + eps, gamma_map[..., None])

def _apply_clahe_masked(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE only in masked regions. Simple way:
      - Apply CLAHE globally
      - Blend: mask * clahe + (1-mask) * original
    """
    clahe_img = apply_clahe(img)
    mask3 = mask[..., None].astype(np.float32)
    return mask3 * clahe_img + (1.0 - mask3) * img

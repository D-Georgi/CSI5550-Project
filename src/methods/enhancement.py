# src/methods/enhancement.py

import numpy as np
from .illumination import estimate_illumination_max_rgb, refine_illumination_tv, normalize_illumination
from .attention import compute_scaled_attention, compute_darkness_attention, map_attention_to_gamma, attention_mask_for_clahe
from .denoising import bilateral_denoise, attention_weighted_blend
from .baselines import apply_clahe
from dataclasses import dataclass


@dataclass
class IllumAttentionParams:
    # attention
    alpha: float = 1
    use_scaled_attention: bool = True   # vs naive darkness attention

    # gamma mapping
    gamma_min: float = 0.7
    gamma_max: float = 1.0

    # CLAHE
    use_clahe: bool = True
    clahe_thresh: float = 0.99
    clahe_clip_limit: float = 1.0

    # denoising
    use_denoise: bool = True
    denoise_strength: float = 0.9  # multiplier on attention during blend

    # illumination refinement
    tv_weight: float = 0.1
    tv_iter: int = 50

def enhance_with_illumination_attention(
        img: np.ndarray,
        params: IllumAttentionParams | None = None,
) -> dict:
    if params is None:
        #print("Using default illumination params")
        # Update defaults for NIGHT MODE
        params = IllumAttentionParams()

    I = np.clip(img, 0.0, 1.0)

    # 1) Illumination
    T0 = estimate_illumination_max_rgb(I)
    T = refine_illumination_tv(T0, weight=params.tv_weight, n_iter=params.tv_iter)
    T_norm = normalize_illumination(T)

    # 2) Attention (SIAM)
    if params.use_scaled_attention:
        A = compute_scaled_attention(T_norm, alpha=params.alpha)
    else:
        A = compute_darkness_attention(T_norm, alpha=params.alpha)

    # 3) Gamma Mapping
    gamma_map = map_attention_to_gamma(A, gamma_min=params.gamma_min, gamma_max=params.gamma_max)
    img_gamma = _apply_per_pixel_gamma(I, gamma_map)

    # 4) CLAHE & Denoising Swap
    if params.use_denoise:
        # Denoise base enhancement first
        img_clean = bilateral_denoise(img_gamma)
    else:
        img_clean = img_gamma

    # 5) CLAHE
    if params.use_clahe:
        mask = attention_mask_for_clahe(A, thresh=params.clahe_thresh)
        img_final = _apply_clahe_masked(img_clean, mask, clip_limit=params.clahe_clip_limit)
    else:
        img_final = img_clean

    # 6) Final Blend
    blend_factor = params.denoise_strength # Increase blend for night to see more result
    img_residual = (1 - blend_factor * A[..., None]) * I + (blend_factor * A[..., None]) * img_final

    return {
        "illumination": T,
        "attention": A,
        "gamma_map": gamma_map,
        "enhanced_gamma": img_gamma,
        "enhanced_final": np.clip(img_residual, 0.0, 1.0),
    }

def _apply_per_pixel_gamma(img: np.ndarray, gamma_map: np.ndarray) -> np.ndarray:
    """Apply per-pixel gamma correction given a gamma map."""
    eps = 1e-6
    return np.power(np.clip(img, 0.0, 1.0) + eps, gamma_map[..., None])

def _apply_clahe_masked(img: np.ndarray, mask: np.ndarray, clip_limit: float) -> np.ndarray:
    """
    Apply CLAHE only in masked regions. Simple way:
      - Apply CLAHE globally
      - Blend: mask * clahe + (1-mask) * original
    """
    clahe_img = apply_clahe(img, clip_limit=clip_limit)
    mask3 = mask[..., None].astype(np.float32)
    return mask3 * clahe_img + (1.0 - mask3) * img

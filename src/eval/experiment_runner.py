# src/eval/experiment_runner.py

from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from src.data.dataset_loader import load_lol_pairs
from src.data.image_io import load_image, save_image
from src.methods.baselines import (
    apply_histogram_equalization,
    apply_clahe,
    apply_gamma,
    apply_simple_retinex,
    lime_enhance_simplified,
)
from src.methods.enhancement import IllumAttentionParams, enhance_with_illumination_attention
from src.eval.metrics import compute_psnr, compute_ssim
from src.config import RESULTS_DIR

def evaluate_params_on_lol(
    params: IllumAttentionParams,
    max_images: int | None = 50,
) -> dict:
    """
    Run proposed method with given params on a subset of LOL and return avg PSNR/SSIM.
    """
    pairs = load_lol_pairs()
    if max_images is not None:
        pairs = pairs[:max_images]

    psnr_sum, ssim_sum = 0.0, 0.0
    count = 0

    for low_path, gt_path in tqdm(pairs, desc="Ablation subset"):
        low = load_image(low_path)
        gt = load_image(gt_path)

        out = enhance_with_illumination_attention(low, params=params)
        enhanced = out["enhanced_final"]

        psnr_sum += compute_psnr(gt, enhanced)
        ssim_sum += compute_ssim(gt, enhanced)
        count += 1

    return {
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
    }

def run_lol_experiment(
    subset_name: str = "lol_subset",
    save_images: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run baselines and proposed method on LOL dataset subset.

    Args:
        subset_name: Name for result folder.
        save_images: Whether to save enhanced outputs.

    Returns:
        metrics: {method_name: {"psnr": avg_psnr, "ssim": avg_ssim}}
    """
    pairs = load_lol_pairs()
    methods = {
        "he": apply_histogram_equalization,
        "clahe": apply_clahe,
        "gamma06": lambda x: apply_gamma(x, gamma=0.6),
        "retinex": apply_simple_retinex,
        "lime": lime_enhance_simplified,
        "illum_attention": lambda x: enhance_with_illumination_attention(x)["enhanced_final"],
    }

    metrics_sum = {m: {"psnr": 0.0, "ssim": 0.0} for m in methods}
    count = 0

    out_dir = RESULTS_DIR / "datasets" / subset_name
    if save_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    for low_path, gt_path in tqdm(pairs, desc="Running LOL experiment"):
        low = load_image(low_path)
        gt = load_image(gt_path)

        for name, func in methods.items():
            enhanced = func(low)
            psnr = compute_psnr(gt, enhanced)
            ssim = compute_ssim(gt, enhanced)
            metrics_sum[name]["psnr"] += psnr
            metrics_sum[name]["ssim"] += ssim

            if save_images:
                rel_name = f"{low_path.stem}_{name}.png"
                save_image(out_dir / rel_name, enhanced)

        count += 1

    metrics_avg = {
        name: {
            "psnr": m["psnr"] / count,
            "ssim": m["ssim"] / count
        }
        for name, m in metrics_sum.items()
    }
    return metrics_avg

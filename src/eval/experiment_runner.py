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
from src.methods.enhancement import enhance_with_illumination_attention
from metrics import compute_psnr, compute_ssim
from src.config import RESULTS_DIR

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

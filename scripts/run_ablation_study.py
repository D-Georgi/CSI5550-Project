# scripts/run_ablation_study.py

from itertools import product
from pathlib import Path
import csv

from src.methods.enhancement import IllumAttentionParams
from src.eval.experiment_runner import evaluate_params_on_lol

def main():
    # --- define search space (coarse) ---
    alphas          = [1.0, 1.2, 1.5]
    gamma_mins      = [0.7, 0.8, 0.9]
    clahe_thresholds = [0.4, 0.6]
    clahe_clip_limits = [1.5, 2.0]
    denoise_strengths = [0.5, 1.0]
    use_scaled_opts = [False, True]   # naive vs SIAM

    # limit combos to something manageable
    combos = list(product(
        use_scaled_opts,
        alphas,
        gamma_mins,
        clahe_thresholds,
        clahe_clip_limits,
        denoise_strengths,
    ))

    out_path = Path("experiments/logs/ablation_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "use_scaled", "alpha", "gamma_min",
            "clahe_thresh", "clahe_clip_limit",
            "denoise_strength", "psnr", "ssim",
        ])

        for (use_scaled, alpha, gmin, ct, cl, ds) in combos:
            params = IllumAttentionParams(
                use_scaled_attention=use_scaled,
                alpha=alpha,
                gamma_min=gmin,
                gamma_max=1.0,
                clahe_thresh=ct,
                clahe_clip_limit=cl,
                denoise_strength=ds,
            )
            metrics = evaluate_params_on_lol(params, max_images=40)  # subset
            print(
                f"scaled={use_scaled}, alpha={alpha}, gmin={gmin}, "
                f"ct={ct}, cl={cl}, ds={ds} -> PSNR={metrics['psnr']:.2f}, "
                f"SSIM={metrics['ssim']:.3f}"
            )
            writer.writerow([
                use_scaled, alpha, gmin, ct, cl, ds,
                metrics["psnr"], metrics["ssim"],
            ])

if __name__ == "__main__":
    main()

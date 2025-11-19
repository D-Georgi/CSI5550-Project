# scripts/run_comparison_grid.py

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.image_io import load_image, save_image
from src.methods.baselines import apply_histogram_equalization, apply_clahe, apply_gamma, lime_enhance_simplified
from src.methods.enhancement import enhance_with_illumination_attention, IllumAttentionParams


def make_comparison_grid(image_paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    my_params = IllumAttentionParams(
        gamma_min=0.5,  # Night mode aggressive gamma
        gamma_max=1.0,
        clahe_thresh=0.5,
        denoise_strength=0.85,  # Tunable blending
        tv_iter=50  # High quality iterations
    )

    for img_path in image_paths:
        print(f"Processing {img_path.name}...")
        img = load_image(img_path)

        # 1. Baselines
        # HE (Histogram Equalization) - often noisy/washed out
        he = apply_histogram_equalization(img)

        # CLAHE - better, but can amplify noise in sky
        clahe = apply_clahe(img, clip_limit=2.0)

        # Gamma - simple brightening
        gamma_img = apply_gamma(img, gamma=0.5)

        # LIME - Retinex based baseline
        lime = lime_enhance_simplified(img)

        # 2. SIAM
        res = enhance_with_illumination_attention(img, params=my_params)
        my_result = res["enhanced_final"]
        attention_map = res["attention"]

        # 3. Visualization
        # Row 1: Inputs & Baselines
        # Row 2: Components & Final Result

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Helper to show image
        def show(ax, i, title, cmap=None):
            if i.ndim == 2:
                ax.imshow(i, cmap=cmap if cmap else 'gray')
            else:
                ax.imshow(np.clip(i, 0, 1))
            ax.set_title(title, fontsize=14)
            ax.axis("off")

        show(axes[0, 0], img, "Original Input")
        show(axes[0, 1], he, "Global Hist. Eq. (Baseline)")
        show(axes[0, 2], lime, "LIME (Baseline)")

        show(axes[1, 0], attention_map, "Scaled Attention (SIAM)", cmap='inferno')
        show(axes[1, 1], clahe, "CLAHE Only")
        show(axes[1, 2], my_result, "Method")

        plt.tight_layout()

        # Save the figure
        save_path = out_dir / f"comparison_{img_path.stem}.jpg"
        plt.savefig(str(save_path), dpi=150)
        print(f"Saved comparison to {save_path}")
        plt.close()


if __name__ == "__main__":
    # input folder
    test_folder = Path("data/test_unpaired")
    output_folder = Path("data/results/comparisons")

    images = list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.png"))
    make_comparison_grid(images, output_folder)
# scripts/run_ablation_study.py

import sys
import itertools
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.image_io import load_image, save_image
from src.methods.enhancement import enhance_with_illumination_attention, IllumAttentionParams


def run_grid_search():
    INPUT_DIR = Path("data/test_night_bdd")
    BASE_OUTPUT_DIR = Path("data/results/ablation_study")

    grid = {
        "gamma_min": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 0.4=Very Bright/Noisy, 0.7=Darker/Cleaner, 0.8/0.9=Halo Effect
        "denoise_strength": [0.85, 0.9, 0.95, 1.0],  # 1.0=No original grain, 0.85=Texture preserved
        "clahe_thresh": [0.3, 0.5]  # 0.3=More local contrast, 0.5=Less noise in sky
    }

    # Get all images
    images = sorted(list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.png")))
    print(f"Found {len(images)} input images.")

    # Generate all combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Running {len(combinations)} parameter combinations per image...")
    print(f"Total processing steps: {len(combinations) * len(images)}")

    for params_dict in tqdm(combinations, desc="Experiments"):
        exp_name = (f"g{params_dict['gamma_min']}_"
                    f"d{params_dict['denoise_strength']}_"
                    f"c{params_dict['clahe_thresh']}")

        exp_dir = BASE_OUTPUT_DIR / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Setup Params object
        params = IllumAttentionParams(
            gamma_min=params_dict['gamma_min'],
            gamma_max=1.0,

            clahe_thresh=params_dict['clahe_thresh'],
            clahe_clip_limit=2.0,

            denoise_strength=params_dict['denoise_strength'],

            use_scaled_attention=True,
            tv_iter=30
        )

        # Process all images for this setting
        for img_file in images:
            # Skip if already exists (resuming capability)
            out_path = exp_dir / img_file.name
            if out_path.exists():
                continue

            try:
                img = load_image(img_file)
                res = enhance_with_illumination_attention(img, params=params)
                save_image(out_path, res["enhanced_final"])
            except Exception as e:
                print(f"Failed on {img_file.name}: {e}")

    print(f"\nAblation generation complete. Check {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    run_grid_search()
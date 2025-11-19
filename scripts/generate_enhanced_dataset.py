# scripts/generate_enhanced_dataset.py

import sys
import argparse
from pathlib import Path
from tqdm import tqdm  # Install with: pip install tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.image_io import load_image, save_image
from src.methods.enhancement import enhance_with_illumination_attention, IllumAttentionParams

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")))
    print(f"Found {len(images)} images in {input_dir}")

    for img_file in tqdm(images, desc="Enhancing"):
        try:
            # Load
            img = load_image(img_file)

            # Enhance
            res = enhance_with_illumination_attention(img)
            enhanced = res["enhanced_final"]

            # Save
            save_image(output_path / img_file.name, enhanced)

        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

if __name__ == "__main__":
    # Hardcoded paths for easy usage
    INPUT_DIR = "data/test_night_bdd"
    OUTPUT_DIR = "data/results/my_bdd_enhanced"

    print("Starting Batch Enhancement...")
    process_dataset(INPUT_DIR, OUTPUT_DIR)
    print(f"Done! Enhanced images saved to {OUTPUT_DIR}")
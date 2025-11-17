# scripts/run_batch_enhance_lol.py

from pathlib import Path
from src.data.image_io import load_image, save_image
from src.methods.baselines import apply_histogram_equalization, apply_clahe, apply_gamma
from src.methods.enhancement import enhance_with_illumination_attention

def main():
    project_root = Path(__file__).resolve().parents[1]

    input_dir = project_root / "data" / "raw" / "lol_low"
    output_dir = project_root / "data" / "results" / "lol_low_enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {input_dir}")
    print(f"Saving enhanced images to: {output_dir}")

    image_paths = list(input_dir.glob("*.png"))
    print(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        print(f"Processing {img_path.name}...")

        img = load_image(img_path)
        result = enhance_with_illumination_attention(img)
        illum_att = result["enhanced_final"]
        output_path = output_dir / img_path.name

        # Save the final enhanced image
        save_image(output_path, illum_att)

    print("Done.")

if __name__ == "__main__":
    main()

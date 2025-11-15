# scripts/run_single_image_demo.py

from pathlib import Path
from src.data.image_io import load_image, save_image
from src.methods.baselines import apply_histogram_equalization, apply_clahe, apply_gamma
from src.methods.enhancement import enhance_with_illumination_attention

def main():
    img_path = Path("data/test_unpaired/example.png")
    img = load_image(img_path)

    he = apply_histogram_equalization(img)
    clahe = apply_clahe(img)
    gamma = apply_gamma(img, gamma=0.6)

    result = enhance_with_illumination_attention(img)
    illum_att = result["enhanced_final"]

    out_dir = Path("data/results/single_image")
    save_image(out_dir / "input.png", img)
    save_image(out_dir / "he.png", he)
    save_image(out_dir / "clahe.png", clahe)
    save_image(out_dir / "gamma06.png", gamma)
    save_image(out_dir / "illum_attention.png", illum_att)

if __name__ == "__main__":
    main()

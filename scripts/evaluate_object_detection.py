# scripts/evaluate_object_detection.py

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO  # pip install ultralytics

# --- CONFIGURATION ---
# Path to trained YOLO11 weights
MODEL_PATH = "bdd100k_yolo11_best.pt"

# Folders
DIR_ORIGINAL = Path("data/test_night_bdd")
DIR_OURS = Path("data/results/ours_bdd_enhanced")
DIR_AWN = Path("data/results/allweathernet_bdd")

OUTPUT_DIR = Path("data/results/detection_comparisons")


# ---------------------

def run_inference(model, img_path):
    """
    Runs YOLO inference and returns:
    - plotted_img: Image with boxes drawn
    - count: Number of objects detected
    - avg_conf: Average confidence score of detections
    """
    if not img_path.exists():
        return None, 0, 0.0

    # Run inference
    # conf=0.25 is standard, for night 0.15 to see if recall improves
    results = model.predict(str(img_path), conf=0.25, verbose=False)
    result = results[0]

    # Get stats
    boxes = result.boxes
    count = len(boxes)
    avg_conf = float(boxes.conf.mean()) if count > 0 else 0.0

    # Get visualization (BGR numpy array)
    plotted_img = result.plot()

    # Convert BGR to RGB for Matplotlib
    plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)

    return plotted_img, count, avg_conf


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Model
    try:
        print(f"Loading YOLO model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'ultralytics' is installed and the path is correct.")
        return

    # Get list of images from the Original folder
    images = sorted(list(DIR_ORIGINAL.glob("*.jpg")) + list(DIR_ORIGINAL.glob("*.png")))

    if not images:
        print(f"No images found in {DIR_ORIGINAL}")
        return

    print(f"Processing {len(images)} images for detection comparison...")

    stats = []

    for img_file in images:
        img_name = img_file.name

        # Paths to counterparts
        path_ours = DIR_OURS / img_name
        path_awn = DIR_AWN / img_name

        # 1. Detect on Original
        viz_orig, n_orig, conf_orig = run_inference(model, img_file)

        # 2. Detect on Ours
        viz_ours, n_ours, conf_ours = run_inference(model, path_ours)

        # 3. Detect on AllWeatherNet (if available)
        viz_awn, n_awn, conf_awn = run_inference(model, path_awn)

        if viz_ours is None:
            print(f"Skipping {img_name}, enhanced version not found.")
            continue

        # Log stats
        stats.append({
            "file": img_name,
            "orig_count": n_orig,
            "ours_count": n_ours,
            "awn_count": n_awn if viz_awn is not None else 0
        })

        # --- Visualization Grid ---
        cols = 3 if viz_awn is not None else 2
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5))

        # Helper
        def show_ax(ax, img, title, count, conf):
            ax.imshow(img)
            ax.set_title(f"{title}\nObjects: {count} | Avg Conf: {conf:.2f}", fontsize=12, fontweight='bold')
            ax.axis('off')

        if cols == 2:
            axes = axes.flatten()
            show_ax(axes[0], viz_orig, "Original (Night)", n_orig, conf_orig)
            show_ax(axes[1], viz_ours, "Ours (Classical SIAM)", n_ours, conf_ours)
        else:
            axes = axes.flatten()
            show_ax(axes[0], viz_orig, "Original (Night)", n_orig, conf_orig)
            show_ax(axes[1], viz_awn, "AllWeatherNet", n_awn, conf_awn)
            show_ax(axes[2], viz_ours, "Ours (Classical SIAM)", n_ours, conf_ours)

        plt.tight_layout()
        save_path = OUTPUT_DIR / f"detect_{img_file.stem}.jpg"
        plt.savefig(str(save_path))
        plt.close()

        # Print quick diff
        diff = n_ours - n_orig
        sign = "+" if diff > 0 else ""
        print(f"[{img_name}] Orig: {n_orig} -> Ours: {n_ours} ({sign}{diff})")

    # Summary
    total_orig = sum(s['orig_count'] for s in stats)
    total_ours = sum(s['ours_count'] for s in stats)
    print("\n--- SUMMARY ---")
    print(f"Total Detections (Original): {total_orig}")
    print(f"Total Detections (Ours):     {total_ours}")
    print(f"Improvement: {((total_ours - total_orig) / total_orig) * 100:.1f}%")


if __name__ == "__main__":
    main()
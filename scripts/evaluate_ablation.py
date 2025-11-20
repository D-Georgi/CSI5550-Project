# scripts/evaluate_ablation.py

import sys
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# CONFIG
MODEL_PATH = "yolo/weights/best.pt"
ABLATION_DIR = Path("data/results/ablation_study")


def evaluate_experiments():
    # Load Model
    model = YOLO(MODEL_PATH)

    # Find all experiment folders
    exp_folders = sorted([f for f in ABLATION_DIR.iterdir() if f.is_dir()])

    results = []

    print(f"Evaluating {len(exp_folders)} experiments...")

    for folder in exp_folders:
        exp_name = folder.name
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        total_objects = 0
        total_conf_sum = 0.0

        # Run inference on batch
        # Use a slightly lower conf
        for img_path in tqdm(images, desc=exp_name, leave=False):
            res = model.predict(str(img_path), conf=0.25, verbose=False, device=0)[0]

            # Stats for this image
            boxes = res.boxes
            if len(boxes) > 0:
                total_objects += len(boxes)
                total_conf_sum += float(boxes.conf.sum())

        # Calculate Metrics
        avg_objects_per_img = total_objects / len(images) if images else 0
        avg_conf_per_object = (total_conf_sum / total_objects) if total_objects > 0 else 0

        # High Yield = Many objects detected with high confidence.
        yield_score = total_conf_sum / len(images)

        results.append({
            "Experiment": exp_name,
            "Total Detections": total_objects,
            "Avg Conf": round(avg_conf_per_object, 3),
            "Yield Score": round(yield_score, 3)
        })

    # Create DataFrame and Sort
    df = pd.DataFrame(results)
    df = df.sort_values(by="Yield Score", ascending=False)

    print("\n--- ABLATION RESULTS (Ranked by Yield) ---")
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv("ablation_results.csv", index=False)
    print("\nResults saved to ablation_results.csv")

    # Recommendation
    best = df.iloc[0]
    print(f"\n🏆 WINNER: {best['Experiment']}")
    print(f"   It detected {best['Total Detections']} objects (avg {best['Avg Conf']} conf).")


if __name__ == "__main__":
    evaluate_experiments()
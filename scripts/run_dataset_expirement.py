# scripts/run_dataset_experiment.py

from src.eval.experiment_runner import run_lol_experiment
from src.eval.visualization import plot_metric_bars

def main():
    metrics = run_lol_experiment(subset_name="lol_full", save_images=False)
    print("Average metrics:", metrics)
    plot_metric_bars(metrics, metric_name="psnr")
    plot_metric_bars(metrics, metric_name="ssim")

if __name__ == "__main__":
    main()

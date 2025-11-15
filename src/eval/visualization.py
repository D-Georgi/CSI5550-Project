# src/eval/visualization.py

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

def show_comparison_row(
    images: Dict[str, np.ndarray],
    figsize=(15, 3)
) -> None:
    """
    Show a row of images: {title: image} side-by-side.

    Args:
        images: Ordered dict-like {title: img}.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, images.items()):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_metric_bars(metric_dict: Dict[str, Dict[str, float]], metric_name: str = "psnr") -> None:
    """
    Bar plot of metric (e.g., PSNR, SSIM) across methods.

    Args:
        metric_dict: {method: {"psnr": val, "ssim": val}}.
        metric_name: Which metric to plot.
    """
    methods = list(metric_dict.keys())
    values = [metric_dict[m][metric_name] for m in methods]

    plt.figure()
    plt.bar(methods, values)
    plt.ylabel(metric_name.upper())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

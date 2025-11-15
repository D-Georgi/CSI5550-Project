# src/eval/metrics.py

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """PSNR between ground truth and prediction (RGB [0,1])."""
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))

def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """SSIM between ground truth and prediction (RGB [0,1])."""
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))

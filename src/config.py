# src/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_LOW_DIR = DATA_DIR / "raw" / "lol_low"
RAW_GT_DIR = DATA_DIR / "raw" / "lol_gt"
RESULTS_DIR = DATA_DIR / "results"

# Method hyperparameters
ILLUM_ALPHA = 1.5          # exponent for attention nonlinearity
GAMMA_MIN = 0.4
GAMMA_MAX = 1.1

# Denoising parameters
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 15

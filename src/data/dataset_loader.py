# src/data/dataset_loader.py

from pathlib import Path
from typing import List, Tuple
from .image_io import load_image
from src.config import RAW_LOW_DIR, RAW_GT_DIR

def list_image_files(folder: Path, exts=(".png", ".jpg", ".jpeg")) -> List[Path]:
    """List all image files in a folder with given extensions."""
    files = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)

def load_lol_pairs(
    low_dir: Path = RAW_LOW_DIR,
    gt_dir: Path = RAW_GT_DIR
) -> List[Tuple[Path, Path]]:
    """
    Create list of (low_light_path, gt_path) pairs for LOL dataset.

    Assumes filenames match between low_dir and gt_dir.
    """
    low_files = list_image_files(low_dir)
    pairs = []
    for lf in low_files:
        gt = gt_dir / lf.name
        if gt.exists():
            pairs.append((lf, gt))
    return pairs

def load_unpaired_folder(folder: Path) -> List[Path]:
    """Return list of image paths for arbitrary test images."""
    return list_image_files(folder)

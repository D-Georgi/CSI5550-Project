# src/data/image_io.py

from pathlib import Path
import numpy as np
import cv2

def load_image(path: str | Path, as_float: bool = True) -> np.ndarray:
    """
    Load an RGB image from disk.

    Args:
        path: Path to image file.
        as_float: If True, return float32 array in [0, 1]. Else uint8 in [0, 255].

    Returns:
        np.ndarray of shape (H, W, 3), dtype float32 or uint8.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        img = img.astype(np.float32) / 255.0
    return img

def save_image(path: str | Path, img: np.ndarray) -> None:
    """
    Save an RGB image to disk. Accepts [0,1] float or [0,255] uint8.

    Args:
        path: Output path.
        img: Image array (H,W,3).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def to_float01(img: np.ndarray) -> np.ndarray:
    """Ensure image is float32 in [0,1]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float [0,1] to uint8 [0,255] if needed."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0.0, 1.0)
        return (img * 255.0).round().astype(np.uint8)
    return img

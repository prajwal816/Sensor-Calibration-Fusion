"""
I/O helpers — load/save images, point clouds, configs, NumPy arrays.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from .logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  YAML / Config
# --------------------------------------------------------------------------- #

def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return cfg


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dict to YAML."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved YAML to %s", path)


# --------------------------------------------------------------------------- #
#  Images
# --------------------------------------------------------------------------- #

def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Load an image via OpenCV."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    """Save an image via OpenCV."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    cv2.imwrite(str(path), img)
    logger.info("Saved image to %s", path)


# --------------------------------------------------------------------------- #
#  NumPy arrays
# --------------------------------------------------------------------------- #

def save_numpy(arr: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array (.npy)."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    np.save(str(path), arr)
    logger.info("Saved numpy array to %s", path)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array (.npy)."""
    return np.load(str(path))


# --------------------------------------------------------------------------- #
#  Point clouds (Open3D)
# --------------------------------------------------------------------------- #

def save_pointcloud(pcd, path: str | Path) -> None:
    """Save an Open3D point cloud (PLY/PCD)."""
    import open3d as o3d  # lazy import
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    logger.info("Saved point cloud to %s  (%d points)", path, len(pcd.points))


def load_pointcloud(path: str | Path):
    """Load an Open3D point cloud."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    logger.info("Loaded point cloud from %s  (%d points)", path, len(pcd.points))
    return pcd

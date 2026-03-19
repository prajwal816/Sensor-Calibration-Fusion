"""
Visualization helpers — Matplotlib and Open3D utilities for
reprojection-error plots, point-cloud rendering, and image overlays.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Reprojection error plot
# --------------------------------------------------------------------------- #

def plot_reprojection_errors(
    errors: np.ndarray,
    title: str = "Reprojection Error Distribution",
    save_path: Optional[str] = None,
) -> None:
    """Histogram of reprojection errors (px).

    Parameters
    ----------
    errors : np.ndarray
        1-D array of per-point reprojection error magnitudes.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.75, color="#4A90D9")
    ax.axvline(errors.mean(), color="red", linestyle="--", label=f"Mean = {errors.mean():.4f} px")
    ax.set_xlabel("Reprojection Error (px)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved reprojection plot to %s", save_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Draw detected corners on an image
# --------------------------------------------------------------------------- #

def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    pattern_size: tuple[int, int],
    found: bool = True,
) -> np.ndarray:
    """Return image with drawn checkerboard corners."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    return vis


# --------------------------------------------------------------------------- #
#  Side-by-side RGB + depth overlay
# --------------------------------------------------------------------------- #

def overlay_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Show RGB and colourised depth side-by-side."""
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(depth_color)
    axes[1].set_title("Depth (colourised)")
    axes[1].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved overlay image to %s", save_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Open3D point cloud viewer (optional, non-blocking)
# --------------------------------------------------------------------------- #

def show_pointclouds(
    pcds: Sequence,
    window_name: str = "Point Clouds",
    point_size: float = 2.0,
    bg_color: tuple = (0.1, 0.1, 0.1),
) -> None:
    """Render one or more Open3D point clouds in a window.

    Falls back to a log message when running headless.
    """
    try:
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=720)
        for pcd in pcds:
            vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.asarray(bg_color)
        vis.run()
        vis.destroy_window()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not open visualizer: %s", exc)

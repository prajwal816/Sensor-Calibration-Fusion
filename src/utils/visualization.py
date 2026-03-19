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


# --------------------------------------------------------------------------- #
#  ICP convergence curve
# --------------------------------------------------------------------------- #

def plot_convergence(
    convergence_history: list[dict],
    title: str = "ICP Convergence",
    save_path: Optional[str] = None,
) -> None:
    """Plot fitness and inlier RMSE over ICP iterations.

    Parameters
    ----------
    convergence_history : list of dicts with keys ``iteration``,
        ``fitness``, ``inlier_rmse``, and optionally ``scale``.
    """
    iters = [h["iteration"] for h in convergence_history]
    fitness = [h["fitness"] for h in convergence_history]
    rmse = [h["inlier_rmse"] for h in convergence_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Fitness
    ax1.plot(iters, fitness, "o-", color="#2ECC71", markersize=3, linewidth=1.5, label="Fitness")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fitness Convergence")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # RMSE
    ax2.plot(iters, rmse, "s-", color="#E74C3C", markersize=3, linewidth=1.5, label="Inlier RMSE")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Inlier RMSE")
    ax2.set_title("RMSE Convergence")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Shade multi-scale regions if present
    if "scale" in convergence_history[0]:
        scale_colors = ["#3498DB", "#9B59B6", "#F39C12", "#1ABC9C"]
        scales = sorted(set(h.get("scale", 0) for h in convergence_history))
        for s in scales:
            s_entries = [h for h in convergence_history if h.get("scale", 0) == s]
            s_iters = [h["iteration"] for h in s_entries]
            color = scale_colors[s % len(scale_colors)]
            if s_iters:
                for ax in (ax1, ax2):
                    ax.axvspan(
                        min(s_iters) - 0.5,
                        max(s_iters) + 0.5,
                        alpha=0.1,
                        color=color,
                        label=f"Scale {s}" if ax is ax1 else None,
                    )
        ax1.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved convergence plot to %s", save_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Before / After registration comparison
# --------------------------------------------------------------------------- #

def plot_registration_comparison(
    before_metrics: dict,
    after_metrics: dict,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing before and after registration metrics.

    Parameters
    ----------
    before_metrics, after_metrics : dicts with keys ``fitness``,
        ``inlier_rmse``, and optionally ``chamfer``.
    """
    labels = []
    before_vals = []
    after_vals = []

    for key in ["fitness", "inlier_rmse", "chamfer"]:
        if key in before_metrics and key in after_metrics:
            labels.append(key.replace("_", " ").title())
            before_vals.append(before_metrics[key])
            after_vals.append(after_metrics[key])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, before_vals, width, label="Before", color="#E74C3C", alpha=0.85)
    bars2 = ax.bar(x + width / 2, after_vals, width, label="After", color="#2ECC71", alpha=0.85)

    ax.set_ylabel("Value")
    ax.set_title("Registration: Before vs After", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Value annotations
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved registration comparison to %s", save_path)
    plt.close(fig)


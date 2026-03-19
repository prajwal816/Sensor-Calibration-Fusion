"""
Registration quality metrics.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_alignment_error(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    max_distance: float = 0.05,
) -> dict:
    """Evaluate alignment quality after registration.

    Parameters
    ----------
    source, target : Open3D point clouds.
    transformation : (4, 4) matrix applied to source.
    max_distance : max correspondence distance for evaluation.

    Returns
    -------
    dict with: fitness, inlier_rmse, num_correspondences.
    """
    result = o3d.pipelines.registration.evaluate_registration(
        source, target, max_distance, transformation
    )
    metrics = {
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "num_correspondences": len(result.correspondence_set),
    }
    logger.info(
        "Alignment — fitness=%.4f  RMSE=%.6f  correspondences=%d",
        metrics["fitness"], metrics["inlier_rmse"], metrics["num_correspondences"],
    )
    return metrics


def compute_rmse(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """Point-wise RMSE between two aligned (N, 3) point arrays.

    Both arrays must have the same number of rows (matched correspondences).
    """
    diff = np.asarray(points_a) - np.asarray(points_b)
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    logger.info("Point-wise RMSE: %.6f", rmse)
    return rmse


def compute_chamfer_distance(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> float:
    """Bidirectional Chamfer distance between two point clouds."""
    dist_s2t = np.asarray(source.compute_point_cloud_distance(target))
    dist_t2s = np.asarray(target.compute_point_cloud_distance(source))
    chamfer = float(np.mean(dist_s2t)) + float(np.mean(dist_t2s))
    logger.info("Chamfer distance: %.6f", chamfer)
    return chamfer

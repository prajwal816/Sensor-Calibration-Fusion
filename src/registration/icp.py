"""
ICP (Iterative Closest Point) registration using Open3D.

Supports:
- Point-to-point ICP
- Point-to-plane ICP
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import open3d as o3d

from ..transforms.rigid_transform import RigidTransform
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _prepare_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.005,
    estimate_normals: bool = True,
) -> o3d.geometry.PointCloud:
    """Down-sample and optionally estimate normals."""
    down = pcd.voxel_down_sample(voxel_size)
    if estimate_normals and not down.has_normals():
        down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2, max_nn=30
            )
        )
    return down


def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    method: str = "point_to_plane",
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    max_correspondence_distance: float = 0.05,
    voxel_size: float = 0.005,
    init_transform: Optional[np.ndarray] = None,
) -> dict:
    """Run ICP registration.

    Parameters
    ----------
    source, target : Open3D point clouds.
    method : ``"point_to_point"`` or ``"point_to_plane"``.
    max_iterations : ICP iteration cap.
    tolerance : convergence threshold.
    max_correspondence_distance : max point-pair distance.
    voxel_size : voxel down-sampling size.
    init_transform : (4, 4) initial alignment; identity by default.

    Returns
    -------
    dict with: transform (RigidTransform), fitness, inlier_rmse,
               correspondence_set_size, elapsed_seconds, transformation_matrix.
    """
    source_down = _prepare_pcd(source, voxel_size, estimate_normals=(method == "point_to_plane"))
    target_down = _prepare_pcd(target, voxel_size, estimate_normals=(method == "point_to_plane"))

    if init_transform is None:
        init_transform = np.eye(4)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=tolerance,
        relative_rmse=tolerance,
    )

    if method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    logger.info(
        "Running %s ICP  (max_iter=%d, max_dist=%.4f, voxel=%.4f) …",
        method, max_iterations, max_correspondence_distance, voxel_size,
    )

    t0 = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance,
        init_transform,
        estimation,
        criteria,
    )
    elapsed = time.perf_counter() - t0

    transform = RigidTransform(matrix=result.transformation)
    logger.info(
        "ICP converged — fitness=%.4f  inlier_rmse=%.6f  time=%.3fs",
        result.fitness, result.inlier_rmse, elapsed,
    )

    return {
        "transform": transform,
        "transformation_matrix": result.transformation,
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "correspondence_set_size": len(result.correspondence_set),
        "elapsed_seconds": elapsed,
    }

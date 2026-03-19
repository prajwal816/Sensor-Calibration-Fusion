"""
Global registration using FPFH features + RANSAC.

Used as an initialisation step before ICP for large-displacement scenarios.
"""

from __future__ import annotations

import time

import numpy as np
import open3d as o3d

from ..transforms.rigid_transform import RigidTransform
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _compute_fpfh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    radius_multiplier: float = 5.0,
    max_nn: int = 100,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Down-sample, estimate normals, and compute FPFH features."""
    down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * radius_multiplier
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn),
    )
    return down, fpfh


def run_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
    fpfh_radius_multiplier: float = 5.0,
    fpfh_max_nn: int = 100,
    distance_threshold_multiplier: float = 1.5,
    max_iterations: int = 4_000_000,
    confidence: float = 0.999,
) -> dict:
    """FPFH-based RANSAC global registration.

    Parameters
    ----------
    source, target : Open3D point clouds.
    voxel_size : down-sampling voxel size.
    distance_threshold_multiplier : RANSAC inlier distance = voxel_size × this.
    max_iterations, confidence : RANSAC parameters.

    Returns
    -------
    dict with: transform, fitness, inlier_rmse, elapsed_seconds.
    """
    logger.info("Computing FPFH features (voxel_size=%.4f) …", voxel_size)
    src_down, src_fpfh = _compute_fpfh(source, voxel_size, fpfh_radius_multiplier, fpfh_max_nn)
    tgt_down, tgt_fpfh = _compute_fpfh(target, voxel_size, fpfh_radius_multiplier, fpfh_max_nn)

    distance_threshold = voxel_size * distance_threshold_multiplier

    logger.info(
        "Running RANSAC global registration (dist_thresh=%.4f, max_iter=%d) …",
        distance_threshold, max_iterations,
    )

    t0 = time.perf_counter()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=max_iterations,
            confidence=confidence,
        ),
    )
    elapsed = time.perf_counter() - t0

    transform = RigidTransform(matrix=result.transformation)
    logger.info(
        "Global registration — fitness=%.4f  inlier_rmse=%.6f  time=%.3fs",
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

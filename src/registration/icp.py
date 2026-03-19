"""
ICP (Iterative Closest Point) registration using Open3D.

Supports:
- Point-to-point ICP
- Point-to-plane ICP
- Multi-scale ICP with convergence tracking
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


def run_icp_with_convergence(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    method: str = "point_to_plane",
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    max_correspondence_distance: float = 0.05,
    voxel_size: float = 0.005,
    init_transform: Optional[np.ndarray] = None,
    convergence_step: int = 1,
) -> dict:
    """Run ICP with per-iteration convergence tracking.

    Runs ICP in small iteration batches to record fitness and RMSE at
    each step, producing a convergence history useful for plotting.

    Parameters
    ----------
    convergence_step : number of ICP iterations per batch (1 = per iteration).

    Returns
    -------
    dict — same as :func:`run_icp`, plus:
        convergence_history : list[dict] with keys iteration, fitness, inlier_rmse.
    """
    source_down = _prepare_pcd(source, voxel_size, estimate_normals=(method == "point_to_plane"))
    target_down = _prepare_pcd(target, voxel_size, estimate_normals=(method == "point_to_plane"))

    if init_transform is None:
        init_transform = np.eye(4)

    if method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    logger.info(
        "Running %s ICP with convergence tracking (max_iter=%d, step=%d) …",
        method, max_iterations, convergence_step,
    )

    convergence_history: list[dict] = []
    current_transform = init_transform.copy()
    total_iterations = 0
    t0 = time.perf_counter()

    for batch_start in range(0, max_iterations, convergence_step):
        batch_size = min(convergence_step, max_iterations - batch_start)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=batch_size,
            relative_fitness=tolerance,
            relative_rmse=tolerance,
        )

        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            max_correspondence_distance,
            current_transform,
            estimation,
            criteria,
        )

        total_iterations += batch_size
        current_transform = result.transformation

        convergence_history.append({
            "iteration": total_iterations,
            "fitness": result.fitness,
            "inlier_rmse": result.inlier_rmse,
            "correspondences": len(result.correspondence_set),
        })

        # Early stopping: check if converged
        if len(convergence_history) >= 2:
            prev = convergence_history[-2]
            curr = convergence_history[-1]
            delta_fitness = abs(curr["fitness"] - prev["fitness"])
            delta_rmse = abs(curr["inlier_rmse"] - prev["inlier_rmse"])
            if delta_fitness < tolerance and delta_rmse < tolerance:
                logger.info("ICP converged early at iteration %d", total_iterations)
                break

    elapsed = time.perf_counter() - t0
    transform = RigidTransform(matrix=current_transform)

    logger.info(
        "ICP (tracked) — fitness=%.4f  inlier_rmse=%.6f  iterations=%d  time=%.3fs",
        result.fitness, result.inlier_rmse, total_iterations, elapsed,
    )

    return {
        "transform": transform,
        "transformation_matrix": current_transform,
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "correspondence_set_size": len(result.correspondence_set),
        "elapsed_seconds": elapsed,
        "total_iterations": total_iterations,
        "convergence_history": convergence_history,
    }


def run_multiscale_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_scales: list[float] | None = None,
    max_iterations_per_scale: list[int] | None = None,
    method: str = "point_to_plane",
    init_transform: Optional[np.ndarray] = None,
) -> dict:
    """Coarse-to-fine multi-scale ICP.

    Runs ICP at progressively finer voxel resolutions for robust convergence.

    Parameters
    ----------
    voxel_scales : list of voxel sizes from coarse to fine.
    max_iterations_per_scale : iteration cap at each scale.

    Returns
    -------
    dict — same as :func:`run_icp`, plus convergence_history across all scales.
    """
    if voxel_scales is None:
        voxel_scales = [0.04, 0.02, 0.005]
    if max_iterations_per_scale is None:
        max_iterations_per_scale = [50, 30, 20]

    if init_transform is None:
        current_transform = np.eye(4)
    else:
        current_transform = init_transform.copy()

    convergence_history: list[dict] = []
    cumulative_iter = 0
    t0 = time.perf_counter()

    for scale_idx, (vs, max_it) in enumerate(zip(voxel_scales, max_iterations_per_scale)):
        max_dist = vs * 5
        logger.info("Multi-scale ICP — scale %d  voxel=%.4f  max_dist=%.4f", scale_idx, vs, max_dist)

        result = run_icp_with_convergence(
            source, target,
            method=method,
            max_iterations=max_it,
            max_correspondence_distance=max_dist,
            voxel_size=vs,
            init_transform=current_transform,
            convergence_step=1,
        )
        current_transform = result["transformation_matrix"]

        for entry in result["convergence_history"]:
            convergence_history.append({
                "iteration": cumulative_iter + entry["iteration"],
                "fitness": entry["fitness"],
                "inlier_rmse": entry["inlier_rmse"],
                "scale": scale_idx,
                "voxel_size": vs,
            })
        cumulative_iter += result["total_iterations"]

    elapsed = time.perf_counter() - t0
    transform = RigidTransform(matrix=current_transform)

    logger.info(
        "Multi-scale ICP done — fitness=%.4f  rmse=%.6f  total_iter=%d  time=%.3fs",
        result["fitness"], result["inlier_rmse"], cumulative_iter, elapsed,
    )

    return {
        "transform": transform,
        "transformation_matrix": current_transform,
        "fitness": result["fitness"],
        "inlier_rmse": result["inlier_rmse"],
        "elapsed_seconds": elapsed,
        "total_iterations": cumulative_iter,
        "convergence_history": convergence_history,
    }

"""
Merge multiple sensor point clouds into one unified representation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d

from ..transforms.rigid_transform import RigidTransform
from ..utils.logger import get_logger

logger = get_logger(__name__)


def fuse_multi_sensor(
    point_clouds: Sequence[o3d.geometry.PointCloud],
    transforms: Sequence[RigidTransform],
    voxel_size: float = 0.005,
    statistical_outlier_nb: int = 20,
    statistical_outlier_std: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Merge N point clouds using their calibrated transforms.

    Each ``transforms[i]`` maps ``point_clouds[i]`` into a common
    reference frame.

    Parameters
    ----------
    point_clouds : list of Open3D point clouds.
    transforms : list of RigidTransform (sensor → reference).
    voxel_size : post-merge voxel down-sample size (0 = skip).
    statistical_outlier_nb, statistical_outlier_std : outlier filter params.

    Returns
    -------
    o3d.geometry.PointCloud — unified, cleaned cloud.
    """
    if len(point_clouds) != len(transforms):
        raise ValueError("Must have one transform per point cloud")

    merged = o3d.geometry.PointCloud()
    for i, (pcd, tf) in enumerate(zip(point_clouds, transforms)):
        transformed = o3d.geometry.PointCloud(pcd)
        transformed.transform(tf.matrix)
        merged += transformed
        logger.info("Merged sensor %d: %d points", i, len(pcd.points))

    logger.info("Total merged points: %d", len(merged.points))

    # Voxel down-sample
    if voxel_size > 0:
        merged = merged.voxel_down_sample(voxel_size)
        logger.info("After voxel down-sample (%.4f): %d points", voxel_size, len(merged.points))

    # Statistical outlier removal
    if statistical_outlier_nb > 0:
        merged, ind = merged.remove_statistical_outlier(
            nb_neighbors=statistical_outlier_nb,
            std_ratio=statistical_outlier_std,
        )
        logger.info("After outlier removal: %d points", len(merged.points))

    return merged

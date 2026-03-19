"""
Back-project a depth map into a 3-D point cloud using camera intrinsics.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from ..utils.logger import get_logger

logger = get_logger(__name__)


def depth_to_pointcloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1000.0,
    depth_trunc: float = 3.0,
    step: int = 1,
) -> o3d.geometry.PointCloud:
    """Convert a depth image to an Open3D point cloud.

    Parameters
    ----------
    depth : (H, W) uint16 or float depth image.
    fx, fy, cx, cy : intrinsic parameters.
    depth_scale : divisor to convert raw depth to metres.
    depth_trunc : truncation in metres.
    step : pixel stride (1 = full resolution).

    Returns
    -------
    o3d.geometry.PointCloud
    """
    h, w = depth.shape[:2]
    u = np.arange(0, w, step, dtype=np.float64)
    v = np.arange(0, h, step, dtype=np.float64)
    u, v = np.meshgrid(u, v)

    z = depth[::step, ::step].astype(np.float64) / depth_scale
    mask = (z > 0) & (z < depth_trunc)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    logger.info("Depth → point cloud: %d valid points (of %d)", len(points), mask.size)
    return pcd


def depth_image_to_o3d(
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    depth_scale: float = 1000.0,
    depth_trunc: float = 3.0,
) -> o3d.geometry.PointCloud:
    """Convenience wrapper using Open3D's own RGBD pipeline (depth-only)."""
    h, w = depth.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,
        fx=intrinsic_matrix[0, 0], fy=intrinsic_matrix[1, 1],
        cx=intrinsic_matrix[0, 2], cy=intrinsic_matrix[1, 2],
    )
    depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        intrinsic,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
    )
    logger.info("Open3D depth→pcd: %d points", len(pcd.points))
    return pcd

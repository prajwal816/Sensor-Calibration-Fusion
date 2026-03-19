"""
Fuse aligned RGB and depth into a coloured 3-D point cloud.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from ..utils.logger import get_logger

logger = get_logger(__name__)


def fuse_rgb_depth(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    depth_scale: float = 1000.0,
    depth_trunc: float = 3.0,
) -> o3d.geometry.PointCloud:
    """Create a coloured point cloud from an aligned RGB + depth pair.

    Parameters
    ----------
    rgb : (H, W, 3) uint8 BGR image (OpenCV convention).
    depth : (H, W) uint16 depth image.
    intrinsic_matrix : (3, 3) camera intrinsic K.
    depth_scale : raw → metres divisor.
    depth_trunc : max depth in metres.

    Returns
    -------
    o3d.geometry.PointCloud with colours.
    """
    h, w = depth.shape[:2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    u, v = np.meshgrid(u, v)

    z = depth.astype(np.float64) / depth_scale
    mask = (z > 0) & (z < depth_trunc)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x[mask], y[mask], z[mask]], axis=-1)

    # Convert BGR → RGB, normalise to [0, 1]
    rgb_float = rgb[:, :, ::-1].astype(np.float64) / 255.0
    colors = rgb_float[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    logger.info("Fused RGB-D → %d coloured points", len(points))
    return pcd


def fuse_with_o3d(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic_matrix: np.ndarray,
    depth_scale: float = 1000.0,
    depth_trunc: float = 3.0,
) -> o3d.geometry.PointCloud:
    """Create coloured point cloud via Open3D RGBD image."""
    import cv2

    h, w = depth.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,
        fx=intrinsic_matrix[0, 0], fy=intrinsic_matrix[1, 1],
        cx=intrinsic_matrix[0, 2], cy=intrinsic_matrix[1, 2],
    )
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    logger.info("Open3D RGBD fusion → %d points", len(pcd.points))
    return pcd

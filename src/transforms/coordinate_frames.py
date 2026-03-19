"""
Coordinate-frame conversion utilities.

Common conversions between camera, world, and depth sensor frames.
"""

from __future__ import annotations

import numpy as np

from .rigid_transform import RigidTransform
from .homogeneous import make_homogeneous


def camera_to_world(
    points: np.ndarray,
    extrinsic: RigidTransform,
) -> np.ndarray:
    """Transform points from camera frame to world frame.

    Parameters
    ----------
    points : (N, 3) array in camera coordinates.
    extrinsic : RigidTransform  — camera-to-world transform (T_wc).

    Returns
    -------
    (N, 3) array in world coordinates.
    """
    return extrinsic.apply(points)


def world_to_camera(
    points: np.ndarray,
    extrinsic: RigidTransform,
) -> np.ndarray:
    """Transform points from world frame to camera frame.

    T_cw = T_wc⁻¹

    Parameters
    ----------
    points : (N, 3)
    extrinsic : RigidTransform — camera-to-world transform (T_wc).
    """
    return extrinsic.inverse().apply(points)


def depth_to_camera(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Back-project pixel + depth to 3-D camera coordinates.

    Parameters
    ----------
    u, v : (N,) pixel columns, rows.
    depth : (N,) depth values (metres).
    fx, fy, cx, cy : intrinsic parameters.

    Returns
    -------
    (N, 3) points in the camera coordinate system.
    """
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack([x, y, z])


def align_depth_to_rgb(
    depth_points: np.ndarray,
    T_rgb_depth: RigidTransform,
) -> np.ndarray:
    """Re-project depth-sensor points into the RGB camera's frame.

    Parameters
    ----------
    depth_points : (N, 3) in depth-sensor frame.
    T_rgb_depth : RigidTransform — depth→RGB extrinsic.

    Returns
    -------
    (N, 3) points in RGB camera frame.
    """
    return T_rgb_depth.apply(depth_points)

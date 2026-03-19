"""
Extrinsic (stereo) calibration between RGB and depth sensors.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..transforms.rigid_transform import RigidTransform
from ..transforms.homogeneous import make_homogeneous
from ..utils.logger import get_logger
from .intrinsic import detect_checkerboard

logger = get_logger(__name__)


def calibrate_extrinsics(
    rgb_image_dir: str | Path,
    depth_image_dir: str | Path,
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    image_format: str = "png",
    rgb_camera_matrix: Optional[np.ndarray] = None,
    rgb_dist_coeffs: Optional[np.ndarray] = None,
    depth_camera_matrix: Optional[np.ndarray] = None,
    depth_dist_coeffs: Optional[np.ndarray] = None,
) -> dict:
    """Compute the rigid transform between an RGB camera and a depth sensor.

    If intrinsic matrices are not provided, they are estimated internally
    via ``cv2.stereoCalibrate`` (flag ``CALIB_FIX_INTRINSIC`` is omitted).

    Parameters
    ----------
    rgb_image_dir, depth_image_dir : directories of paired images.
    board_size : inner corners (cols, rows).
    square_size : physical square size (metres).
    rgb_camera_matrix, rgb_dist_coeffs : pre-calibrated RGB intrinsics.
    depth_camera_matrix, depth_dist_coeffs : pre-calibrated depth intrinsics.

    Returns
    -------
    dict with keys:
        transform (RigidTransform), R, T,
        essential_matrix, fundamental_matrix, reprojection_error
    """
    rgb_dir = Path(rgb_image_dir)
    depth_dir = Path(depth_image_dir)
    rgb_images = sorted(glob.glob(str(rgb_dir / f"*.{image_format}")))
    depth_images = sorted(glob.glob(str(depth_dir / f"*.{image_format}")))

    if len(rgb_images) != len(depth_images):
        raise ValueError(
            f"Mismatched image counts: {len(rgb_images)} RGB vs {len(depth_images)} depth"
        )

    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    rgb_points = []
    depth_points = []
    image_size = None

    for rgb_path, depth_path in zip(rgb_images, depth_images):
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if image_size is None:
            image_size = (rgb_img.shape[1], rgb_img.shape[0])

        found_rgb, corners_rgb = detect_checkerboard(rgb_img, board_size)
        found_depth, corners_depth = detect_checkerboard(depth_img, board_size)

        if found_rgb and found_depth:
            obj_points.append(objp)
            rgb_points.append(corners_rgb)
            depth_points.append(corners_depth)
        else:
            logger.warning("Skipping pair: %s / %s", rgb_path, depth_path)

    if len(obj_points) < 3:
        raise RuntimeError(
            f"Only {len(obj_points)} usable stereo pairs — need ≥ 3."
        )

    logger.info("Stereo-calibrating with %d pairs …", len(obj_points))

    flags = 0
    if rgb_camera_matrix is not None and depth_camera_matrix is not None:
        flags = cv2.CALIB_FIX_INTRINSIC

    if rgb_camera_matrix is None:
        rgb_camera_matrix = np.eye(3, dtype=np.float64)
    if rgb_dist_coeffs is None:
        rgb_dist_coeffs = np.zeros(5, dtype=np.float64)
    if depth_camera_matrix is None:
        depth_camera_matrix = np.eye(3, dtype=np.float64)
    if depth_dist_coeffs is None:
        depth_dist_coeffs = np.zeros(5, dtype=np.float64)

    ret, cm1, dc1, cm2, dc2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        rgb_points,
        depth_points,
        rgb_camera_matrix,
        rgb_dist_coeffs,
        depth_camera_matrix,
        depth_dist_coeffs,
        image_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )

    transform = RigidTransform(R=R, t=T)
    logger.info("Stereo reprojection error: %.4f px", ret)

    return {
        "transform": transform,
        "R": R,
        "T": T,
        "essential_matrix": E,
        "fundamental_matrix": F,
        "reprojection_error": ret,
        "rgb_camera_matrix": cm1,
        "rgb_dist_coeffs": dc1,
        "depth_camera_matrix": cm2,
        "depth_dist_coeffs": dc2,
    }

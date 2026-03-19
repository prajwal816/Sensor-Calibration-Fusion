"""
Stereo rectification — compute rectification maps for aligned RGB ↔ depth.

After stereo calibration, rectification warps both images so that
corresponding epipolar lines become horizontal scan-lines, enabling
efficient dense stereo matching or pixel-aligned RGB-D fusion.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..transforms.rigid_transform import RigidTransform
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_rectification_maps(
    camera_matrix_1: np.ndarray,
    dist_coeffs_1: np.ndarray,
    camera_matrix_2: np.ndarray,
    dist_coeffs_2: np.ndarray,
    image_size: tuple[int, int],
    R: np.ndarray,
    T: np.ndarray,
    alpha: float = 0.0,
) -> dict:
    """Compute stereo rectification transforms and undistort-rectify maps.

    Parameters
    ----------
    camera_matrix_1, dist_coeffs_1 : Intrinsics of camera 1 (e.g. RGB).
    camera_matrix_2, dist_coeffs_2 : Intrinsics of camera 2 (e.g. depth).
    image_size : (width, height).
    R, T : Rotation (3×3) and translation (3×1) from stereo calibration
           (camera 1 → camera 2).
    alpha : Free scaling parameter.
            0 = crop all invalid pixels,
            1 = keep all pixels (some black borders).

    Returns
    -------
    dict with keys:
        R1, R2          — 3×3 rectification rotations.
        P1, P2          — 3×4 projection matrices in rectified coords.
        Q               — 4×4 disparity-to-depth mapping matrix.
        map1_x, map1_y  — undistort+rectify maps for camera 1.
        map2_x, map2_y  — undistort+rectify maps for camera 2.
        valid_roi_1, valid_roi_2 — valid pixel regions after rectification.
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=camera_matrix_1,
        distCoeffs1=dist_coeffs_1,
        cameraMatrix2=camera_matrix_2,
        distCoeffs2=dist_coeffs_2,
        imageSize=image_size,
        R=R,
        T=T,
        alpha=alpha,
        flags=cv2.CALIB_ZERO_DISPARITY,
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(
        camera_matrix_1, dist_coeffs_1, R1, P1, image_size, cv2.CV_32FC1
    )
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        camera_matrix_2, dist_coeffs_2, R2, P2, image_size, cv2.CV_32FC1
    )

    logger.info(
        "Stereo rectification computed  |  ROI1=%s  ROI2=%s  image_size=%s",
        roi1, roi2, image_size,
    )

    return {
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "map1_x": map1_x,
        "map1_y": map1_y,
        "map2_x": map2_x,
        "map2_y": map2_y,
        "valid_roi_1": roi1,
        "valid_roi_2": roi2,
    }


def rectify_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    rectification_maps: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply precomputed rectification maps to a pair of images.

    Parameters
    ----------
    img1, img2 : Input images (can be RGB or grayscale).
    rectification_maps : dict from :func:`compute_rectification_maps`.

    Returns
    -------
    (rectified_1, rectified_2)
    """
    rect1 = cv2.remap(img1, rectification_maps["map1_x"], rectification_maps["map1_y"], cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, rectification_maps["map2_x"], rectification_maps["map2_y"], cv2.INTER_LINEAR)
    return rect1, rect2


def draw_rectified_pair(
    rect1: np.ndarray,
    rect2: np.ndarray,
    num_lines: int = 20,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Horizontally concatenate rectified images with epipolar guide-lines.

    Useful for visual validation: corresponding features should lie on the
    same horizontal line after successful rectification.

    Parameters
    ----------
    rect1, rect2 : Rectified images (same height).
    num_lines : number of horizontal lines to draw.
    save_path : optional path to save the visualisation.

    Returns
    -------
    np.ndarray — concatenated visualisation image.
    """
    if len(rect1.shape) == 2:
        rect1 = cv2.cvtColor(rect1, cv2.COLOR_GRAY2BGR)
    if len(rect2.shape) == 2:
        rect2 = cv2.cvtColor(rect2, cv2.COLOR_GRAY2BGR)

    h = rect1.shape[0]
    canvas = np.concatenate([rect1, rect2], axis=1)
    step = max(1, h // num_lines)
    for y in range(0, h, step):
        color = tuple(int(c) for c in np.random.randint(60, 255, 3))
        cv2.line(canvas, (0, y), (canvas.shape[1], y), color, 1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, canvas)
        logger.info("Saved rectified pair visualisation → %s", save_path)

    return canvas


def compute_disparity_map(
    rect1_gray: np.ndarray,
    rect2_gray: np.ndarray,
    num_disparities: int = 64,
    block_size: int = 9,
) -> np.ndarray:
    """Compute a disparity map from a rectified stereo pair.

    Parameters
    ----------
    rect1_gray, rect2_gray : Grayscale rectified images.
    num_disparities : must be divisible by 16.
    block_size : matching window size (odd).

    Returns
    -------
    np.ndarray — disparity map (float32, divide by 16 for true disparities).
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(rect1_gray, rect2_gray).astype(np.float32) / 16.0
    logger.info(
        "Disparity computed  |  range=[%.1f, %.1f]",
        disparity[disparity > 0].min() if (disparity > 0).any() else 0,
        disparity.max(),
    )
    return disparity


def disparity_to_depth(
    disparity: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Convert disparity map to a 3-D point cloud using the Q matrix.

    Parameters
    ----------
    disparity : float32 disparity map.
    Q : 4×4 disparity-to-depth matrix from ``cv2.stereoRectify``.

    Returns
    -------
    np.ndarray (H, W, 3) — 3-D coordinates per pixel.
    """
    points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    return points_3d

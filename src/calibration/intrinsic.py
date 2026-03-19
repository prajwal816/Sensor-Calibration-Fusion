"""
Intrinsic camera calibration using OpenCV's checkerboard detector.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..utils.logger import get_logger
from ..utils.visualization import draw_corners, plot_reprojection_errors

logger = get_logger(__name__)


def detect_checkerboard(
    image: np.ndarray,
    board_size: tuple[int, int],
    refine: bool = True,
    criteria: tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
) -> tuple[bool, Optional[np.ndarray]]:
    """Detect checkerboard corners in a single image.

    Parameters
    ----------
    image : grayscale image.
    board_size : (cols, rows) inner corners.

    Returns
    -------
    found : bool
    corners : (N, 1, 2) float32 or None.
    """
    found, corners = cv2.findChessboardCorners(image, board_size, None)
    if found and refine:
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def calibrate_intrinsics(
    image_dir: str | Path,
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    image_format: str = "png",
    save_dir: Optional[str | Path] = None,
) -> dict:
    """Run full intrinsic calibration on a folder of checkerboard images.

    Parameters
    ----------
    image_dir : path to directory containing calibration images.
    board_size : inner-corner count (cols, rows).
    square_size : physical size of a square (metres).
    image_format : file extension to glob.
    save_dir : if given, save visualisations here.

    Returns
    -------
    dict with keys:
        camera_matrix, dist_coeffs, rvecs, tvecs,
        reprojection_error, image_size, per_image_errors
    """
    image_dir = Path(image_dir)
    images = sorted(glob.glob(str(image_dir / f"*.{image_format}")))
    if not images:
        raise FileNotFoundError(f"No .{image_format} images in {image_dir}")

    logger.info("Found %d calibration images in %s", len(images), image_dir)

    # 3-D object points for one board
    objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    for img_path in images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        found, corners = detect_checkerboard(gray, board_size)
        if found:
            obj_points.append(objp)
            img_points.append(corners)
            logger.debug("✔ Corners found: %s", img_path)
            if save_dir:
                vis = draw_corners(img, corners, board_size, found)
                out_path = Path(save_dir) / f"corners_{Path(img_path).stem}.png"
                os.makedirs(Path(save_dir), exist_ok=True)
                cv2.imwrite(str(out_path), vis)
        else:
            logger.warning("✘ No corners: %s", img_path)

    if len(obj_points) < 3:
        raise RuntimeError(
            f"Only {len(obj_points)} usable images — need at least 3 for calibration."
        )

    logger.info("Calibrating with %d / %d images …", len(obj_points), len(images))
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Per-image reprojection errors
    per_image_errors = []
    all_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(err)
        diff = (img_points[i].reshape(-1, 2) - projected.reshape(-1, 2))
        all_errors.append(np.linalg.norm(diff, axis=1))

    all_errors_flat = np.concatenate(all_errors)
    logger.info("Mean reprojection error: %.4f px", ret)

    if save_dir:
        plot_reprojection_errors(
            all_errors_flat,
            save_path=str(Path(save_dir) / "reprojection_errors.png"),
        )

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "reprojection_error": ret,
        "image_size": image_size,
        "per_image_errors": per_image_errors,
    }

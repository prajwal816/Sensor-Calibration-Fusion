"""
Generate synthetic sample data for testing the calibration & fusion pipeline.

Creates:
- Checkerboard calibration images (rendered programmatically).
- Synthetic depth maps.
- Synthetic RGB images.
- Paired point clouds with a known rigid-body transform.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import open3d as o3d

from src.transforms.rigid_transform import RigidTransform
from src.transforms.homogeneous import rotation_from_euler, make_homogeneous
from src.utils.logger import get_logger
from src.utils.io_utils import save_pointcloud

logger = get_logger("generate_data")

SEED = 42
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
#  Checkerboard image generation
# --------------------------------------------------------------------------- #

def _render_checkerboard(
    board_size: tuple[int, int] = (9, 6),
    square_px: int = 40,
    margin: int = 60,
) -> np.ndarray:
    """Render a clean checkerboard image."""
    cols, rows = board_size
    w = (cols + 1) * square_px + 2 * margin
    h = (rows + 1) * square_px + 2 * margin
    img = np.ones((h, w), dtype=np.uint8) * 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = margin + r * square_px
                x0 = margin + c * square_px
                img[y0: y0 + square_px, x0: x0 + square_px] = 0
    return img


def generate_calibration_images(
    out_dir: str,
    board_size: tuple[int, int] = (9, 6),
    num_images: int = 15,
    img_size: tuple[int, int] = (640, 480),
) -> None:
    """Generate synthetic checkerboard images with random perspective warps."""
    os.makedirs(out_dir, exist_ok=True)
    board = _render_checkerboard(board_size)
    bh, bw = board.shape[:2]

    for i in range(num_images):
        # Random perspective perturbation
        src_pts = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        noise = np.random.uniform(-30, 30, (4, 2)).astype(np.float32)
        dst_pts = np.float32([
            [50, 50], [img_size[0] - 50, 60],
            [img_size[0] - 60, img_size[1] - 50], [60, img_size[1] - 60],
        ]) + noise

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(board, M, img_size, borderValue=200)

        # Add mild noise
        noise_img = np.random.normal(0, 5, warped.shape).astype(np.int16)
        warped = np.clip(warped.astype(np.int16) + noise_img, 0, 255).astype(np.uint8)

        path = os.path.join(out_dir, f"calib_{i:03d}.png")
        cv2.imwrite(path, warped)

    logger.info("Generated %d calibration images in %s", num_images, out_dir)


# --------------------------------------------------------------------------- #
#  Synthetic depth + RGB
# --------------------------------------------------------------------------- #

def generate_depth_rgb_pair(
    out_dir: str,
    img_size: tuple[int, int] = (640, 480),
    fx: float = 525.0,
    fy: float = 525.0,
    cx: float = 320.0,
    cy: float = 240.0,
    num_pairs: int = 5,
) -> np.ndarray:
    """Generate synthetic RGB images and corresponding 16-bit depth maps.

    Returns the intrinsic matrix used.
    """
    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    for idx in range(num_pairs):
        # Random colourful RGB image with geometric shapes
        rgb = np.random.randint(40, 220, (img_size[1], img_size[0], 3), dtype=np.uint8)
        for _ in range(10):
            pt = (np.random.randint(50, img_size[0] - 50), np.random.randint(50, img_size[1] - 50))
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))
            cv2.circle(rgb, pt, np.random.randint(20, 80), color, -1)

        # Synthetic depth: gradient + random bumps (mm scale)
        base_depth = np.linspace(800, 2500, img_size[1])[:, None] * np.ones((1, img_size[0]))
        bumps = np.random.uniform(-200, 200, (img_size[1], img_size[0]))
        depth = (base_depth + bumps).clip(500, 3000).astype(np.uint16)

        cv2.imwrite(os.path.join(rgb_dir, f"rgb_{idx:03d}.png"), rgb)
        cv2.imwrite(os.path.join(depth_dir, f"depth_{idx:03d}.png"), depth)

    logger.info("Generated %d RGB-D pairs in %s", num_pairs, out_dir)
    return K


# --------------------------------------------------------------------------- #
#  Synthetic point clouds with known transform
# --------------------------------------------------------------------------- #

def generate_point_cloud_pair(
    out_dir: str,
    num_points: int = 10000,
    noise_std: float = 0.002,
) -> dict:
    """Generate two point clouds related by a known rigid transform.

    Returns
    -------
    dict  with keys: source_path, target_path, ground_truth_transform.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create a "bunny-like" surface: hemisphere
    theta = np.random.uniform(0, np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    r = 0.5 + np.random.uniform(-0.05, 0.05, num_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pts = np.column_stack([x, y, z])

    # Ground-truth transform
    R_gt = rotation_from_euler((15.0, -10.0, 5.0), degrees=True)
    t_gt = np.array([0.1, -0.05, 0.2])
    T_gt = make_homogeneous(R_gt, t_gt)
    gt_transform = RigidTransform(matrix=T_gt)

    pts_transformed = gt_transform.apply(pts)
    pts_transformed += np.random.normal(0, noise_std, pts_transformed.shape)

    # Add partial overlap noise
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(pts)
    target_pcd.paint_uniform_color([0.1, 0.7, 0.3])

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(pts_transformed)
    source_pcd.paint_uniform_color([0.7, 0.1, 0.3])

    src_path = os.path.join(out_dir, "source.ply")
    tgt_path = os.path.join(out_dir, "target.ply")
    save_pointcloud(source_pcd, src_path)
    save_pointcloud(target_pcd, tgt_path)

    logger.info("Generated point cloud pair (%d points each) with known transform", num_points)
    return {
        "source_path": src_path,
        "target_path": tgt_path,
        "ground_truth_transform": gt_transform,
    }


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    base = "data"
    logger.info("=== Generating sample data ===")

    generate_calibration_images(
        os.path.join(base, "calibration_images"),
        board_size=(9, 6),
        num_images=15,
    )
    generate_calibration_images(
        os.path.join(base, "calibration_images_depth"),
        board_size=(9, 6),
        num_images=15,
    )

    K = generate_depth_rgb_pair(os.path.join(base, "rgbd"), num_pairs=5)
    # Save intrinsics for convenience
    from src.calibration.calibration_data import CalibrationResult
    cal = CalibrationResult(
        camera_matrix=K,
        dist_coeffs=np.zeros(5),
        image_size=(640, 480),
        reprojection_error=0.0,
        metadata={"source": "synthetic"},
    )
    os.makedirs(os.path.join(base, "output"), exist_ok=True)
    cal.save(os.path.join(base, "output", "synthetic_intrinsics.json"))

    generate_point_cloud_pair(os.path.join(base, "point_clouds"), num_points=10000)

    logger.info("=== Sample data generation complete ===")


if __name__ == "__main__":
    main()

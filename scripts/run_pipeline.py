"""
End-to-end automation pipeline.

    Raw sensor data → Calibrate → Register → Fuse → Metrics report

Usage:
    python scripts/run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import cv2
import numpy as np
import open3d as o3d

from src.calibration.intrinsic import calibrate_intrinsics
from src.calibration.calibration_data import CalibrationResult
from src.registration.icp import run_icp
from src.registration.global_registration import run_global_registration
from src.registration.metrics import compute_alignment_error, compute_rmse, compute_chamfer_distance
from src.fusion.rgb_depth_fusion import fuse_rgb_depth
from src.fusion.multi_sensor_fusion import fuse_multi_sensor
from src.transforms.rigid_transform import RigidTransform
from src.utils.io_utils import load_config, load_pointcloud, save_pointcloud
from src.utils.logger import get_logger


def _divider(msg: str, logger):
    logger.info("=" * 60)
    logger.info("  %s", msg)
    logger.info("=" * 60)


@click.command()
@click.option("--config", default="configs/default.yaml", help="YAML config path.")
@click.option("--skip-calibration", is_flag=True, help="Skip intrinsic calibration step.")
def main(config, skip_calibration):
    """Run the full calibration + registration + fusion pipeline."""
    cfg = load_config(config)
    log_cfg = cfg.get("logging", {})
    logger = get_logger(
        "pipeline",
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file"),
    )

    paths = cfg["paths"]
    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    report: dict = {"stages": {}}
    t_pipeline = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Intrinsic Calibration
    # ------------------------------------------------------------------ #
    _divider("Stage 1: Intrinsic Calibration", logger)

    intrinsics_path = os.path.join(output_dir, "intrinsics.json")
    if skip_calibration and os.path.exists(intrinsics_path):
        logger.info("Skipping calibration — loading %s", intrinsics_path)
        cal = CalibrationResult.load(intrinsics_path)
    else:
        cal_cfg = cfg["calibration"]
        calib_img_dir = paths.get("calibration_images", "data/calibration_images")
        if not os.path.isdir(calib_img_dir):
            logger.warning("No calibration images at %s — using synthetic intrinsics", calib_img_dir)
            synth_path = os.path.join(output_dir, "synthetic_intrinsics.json")
            if os.path.exists(synth_path):
                cal = CalibrationResult.load(synth_path)
            else:
                # Fallback default
                cal = CalibrationResult(
                    camera_matrix=np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float64),
                    dist_coeffs=np.zeros(5),
                    image_size=(640, 480),
                    reprojection_error=0.0,
                )
        else:
            result = calibrate_intrinsics(
                image_dir=calib_img_dir,
                board_size=tuple(cal_cfg["board_size"]),
                square_size=cal_cfg["square_size"],
                image_format=cal_cfg.get("image_format", "png"),
                save_dir=output_dir,
            )
            cal = CalibrationResult(
                camera_matrix=result["camera_matrix"],
                dist_coeffs=result["dist_coeffs"],
                image_size=result["image_size"],
                reprojection_error=result["reprojection_error"],
            )
        cal.save(intrinsics_path)

    report["stages"]["calibration"] = {
        "reprojection_error": cal.reprojection_error,
        "image_size": list(cal.image_size),
    }
    logger.info("Intrinsics RMSE: %.4f px", cal.reprojection_error)

    # ------------------------------------------------------------------ #
    # 2. Point Cloud Registration
    # ------------------------------------------------------------------ #
    _divider("Stage 2: Point Cloud Registration", logger)

    pc_dir = paths.get("point_clouds", "data/point_clouds")
    source_path = os.path.join(pc_dir, "source.ply")
    target_path = os.path.join(pc_dir, "target.ply")

    if os.path.exists(source_path) and os.path.exists(target_path):
        source_pcd = load_pointcloud(source_path)
        target_pcd = load_pointcloud(target_path)

        # Before-registration metrics
        identity_metrics = compute_alignment_error(
            source_pcd, target_pcd, np.eye(4), max_distance=0.1
        )
        before_chamfer = compute_chamfer_distance(source_pcd, target_pcd)
        logger.info("BEFORE registration — fitness=%.4f  RMSE=%.6f  Chamfer=%.6f",
                     identity_metrics["fitness"], identity_metrics["inlier_rmse"], before_chamfer)

        # Global registration (coarse)
        reg_cfg = cfg.get("registration", {})
        global_cfg = reg_cfg.get("global", {})
        global_result = run_global_registration(
            source_pcd, target_pcd,
            voxel_size=global_cfg.get("voxel_size", 0.05),
        )

        # ICP refinement
        icp_cfg = reg_cfg.get("icp", {})
        icp_result = run_icp(
            source_pcd, target_pcd,
            method=icp_cfg.get("method", "point_to_plane"),
            max_iterations=icp_cfg.get("max_iterations", 200),
            tolerance=icp_cfg.get("tolerance", 1e-6),
            max_correspondence_distance=icp_cfg.get("max_correspondence_distance", 0.05),
            voxel_size=icp_cfg.get("voxel_size", 0.005),
            init_transform=global_result["transformation_matrix"],
        )

        # After-registration metrics
        after_metrics = compute_alignment_error(
            source_pcd, target_pcd, icp_result["transformation_matrix"], max_distance=0.1
        )

        aligned = o3d.geometry.PointCloud(source_pcd)
        aligned.transform(icp_result["transformation_matrix"])
        after_chamfer = compute_chamfer_distance(aligned, target_pcd)

        logger.info("AFTER registration  — fitness=%.4f  RMSE=%.6f  Chamfer=%.6f",
                     after_metrics["fitness"], after_metrics["inlier_rmse"], after_chamfer)

        save_pointcloud(aligned, os.path.join(output_dir, "registered_source.ply"))

        report["stages"]["registration"] = {
            "before": {
                "fitness": identity_metrics["fitness"],
                "inlier_rmse": identity_metrics["inlier_rmse"],
                "chamfer": before_chamfer,
            },
            "global": {
                "fitness": global_result["fitness"],
                "inlier_rmse": global_result["inlier_rmse"],
                "elapsed_seconds": global_result["elapsed_seconds"],
            },
            "icp": {
                "fitness": icp_result["fitness"],
                "inlier_rmse": icp_result["inlier_rmse"],
                "elapsed_seconds": icp_result["elapsed_seconds"],
            },
            "after": {
                "fitness": after_metrics["fitness"],
                "inlier_rmse": after_metrics["inlier_rmse"],
                "chamfer": after_chamfer,
            },
        }
    else:
        logger.warning("No source/target point clouds found — skipping registration")

    # ------------------------------------------------------------------ #
    # 3. Sensor Fusion (RGB-D)
    # ------------------------------------------------------------------ #
    _divider("Stage 3: Sensor Fusion", logger)

    rgbd_dir = os.path.join(paths.get("data_dir", "data"), "rgbd")
    rgb_dir = os.path.join(rgbd_dir, "rgb")
    depth_dir_rgbd = os.path.join(rgbd_dir, "depth")

    if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir_rgbd):
        import glob
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        depth_files = sorted(glob.glob(os.path.join(depth_dir_rgbd, "*.png")))

        fused_clouds = []
        fus_cfg = cfg.get("fusion", {})

        for rgb_f, depth_f in zip(rgb_files[:3], depth_files[:3]):
            rgb_img = cv2.imread(rgb_f)
            depth_img = cv2.imread(depth_f, cv2.IMREAD_UNCHANGED)
            pcd = fuse_rgb_depth(
                rgb_img, depth_img,
                intrinsic_matrix=cal.camera_matrix,
                depth_scale=fus_cfg.get("depth_scale", 1000.0),
                depth_trunc=fus_cfg.get("depth_trunc", 3.0),
            )
            fused_clouds.append(pcd)

        if fused_clouds:
            # Multi-sensor merge with identity transforms (same viewpoint for synthetic data)
            identity_tfs = [RigidTransform() for _ in fused_clouds]
            merged = fuse_multi_sensor(fused_clouds, identity_tfs, voxel_size=fus_cfg.get("voxel_size", 0.005))
            save_pointcloud(merged, os.path.join(output_dir, "fused_scene.ply"))
            report["stages"]["fusion"] = {
                "num_views": len(fused_clouds),
                "total_points": len(merged.points),
            }
            logger.info("Fused %d views → %d points", len(fused_clouds), len(merged.points))
    else:
        logger.warning("No RGBD data found — skipping fusion")

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    elapsed = time.perf_counter() - t_pipeline
    report["total_time_seconds"] = elapsed
    report_path = os.path.join(output_dir, "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Pipeline complete in %.2f s — report saved to %s", elapsed, report_path)

    # Print summary table
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    if "calibration" in report["stages"]:
        print(f"  Calibration RMSE:        {report['stages']['calibration']['reprojection_error']:.4f} px")
    if "registration" in report["stages"]:
        reg = report["stages"]["registration"]
        print(f"  Registration (before):   fitness={reg['before']['fitness']:.4f}  RMSE={reg['before']['inlier_rmse']:.6f}")
        print(f"  Registration (after):    fitness={reg['after']['fitness']:.4f}  RMSE={reg['after']['inlier_rmse']:.6f}")
        print(f"  Chamfer before→after:    {reg['before']['chamfer']:.6f} → {reg['after']['chamfer']:.6f}")
        print(f"  ICP convergence time:    {reg['icp']['elapsed_seconds']:.3f}s")
    if "fusion" in report["stages"]:
        fus = report["stages"]["fusion"]
        print(f"  Fusion views / points:   {fus['num_views']} / {fus['total_points']}")
    print(f"  Total pipeline time:     {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

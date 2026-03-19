"""
CLI: Sensor fusion commands.

Usage:
    python scripts/fuse.py rgbd --rgb image.png --depth depth.png --intrinsics intrinsics.json
    python scripts/fuse.py multi --clouds c1.ply c2.ply --transforms t1.json t2.json
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import cv2
import numpy as np
import open3d as o3d

from src.fusion.rgb_depth_fusion import fuse_rgb_depth
from src.fusion.multi_sensor_fusion import fuse_multi_sensor
from src.transforms.rigid_transform import RigidTransform
from src.calibration.calibration_data import CalibrationResult
from src.utils.io_utils import load_config, save_pointcloud, load_pointcloud
from src.utils.logger import get_logger


@click.group()
@click.option("--config", default="configs/default.yaml", help="Path to YAML config.")
@click.pass_context
def cli(ctx, config):
    """Sensor fusion CLI."""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config)


@cli.command()
@click.option("--rgb", required=True, help="RGB image path.")
@click.option("--depth", required=True, help="Depth image path (16-bit PNG).")
@click.option("--intrinsics", required=True, help="Intrinsics JSON path.")
@click.option("--output", default="data/output/fused_rgbd.ply", help="Output point cloud.")
@click.pass_context
def rgbd(ctx, rgb, depth, intrinsics, output):
    """Fuse an RGB + depth pair into a coloured point cloud."""
    cfg = ctx.obj["cfg"]
    fus_cfg = cfg.get("fusion", {})
    logger = get_logger("fuse", level=cfg.get("logging", {}).get("level", "INFO"))

    cal = CalibrationResult.load(intrinsics)
    rgb_img = cv2.imread(rgb)
    depth_img = cv2.imread(depth, cv2.IMREAD_UNCHANGED)

    pcd = fuse_rgb_depth(
        rgb_img, depth_img,
        intrinsic_matrix=cal.camera_matrix,
        depth_scale=fus_cfg.get("depth_scale", 1000.0),
        depth_trunc=fus_cfg.get("depth_trunc", 3.0),
    )

    # Statistical outlier removal
    so = fus_cfg.get("statistical_outlier", {})
    if so:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=so.get("nb_neighbors", 20),
            std_ratio=so.get("std_ratio", 2.0),
        )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_pointcloud(pcd, output)
    logger.info("Fused RGBD cloud saved → %s  (%d points)", output, len(pcd.points))


@cli.command()
@click.option("--clouds", required=True, multiple=True, help="Point cloud files.")
@click.option("--transforms", required=True, multiple=True, help="Transform JSON files (one per cloud).")
@click.option("--output", default="data/output/fused_multi.ply", help="Output point cloud.")
@click.pass_context
def multi(ctx, clouds, transforms, output):
    """Merge multiple point clouds via calibrated transforms."""
    cfg = ctx.obj["cfg"]
    fus_cfg = cfg.get("fusion", {})
    logger = get_logger("fuse", level=cfg.get("logging", {}).get("level", "INFO"))

    pcds = [load_pointcloud(p) for p in clouds]
    tfs = []
    for tp in transforms:
        with open(tp, "r") as f:
            d = json.load(f)
        tfs.append(RigidTransform.from_dict(d.get("transform", d)))

    merged = fuse_multi_sensor(
        pcds, tfs,
        voxel_size=fus_cfg.get("voxel_size", 0.005),
        statistical_outlier_nb=fus_cfg.get("statistical_outlier", {}).get("nb_neighbors", 20),
        statistical_outlier_std=fus_cfg.get("statistical_outlier", {}).get("std_ratio", 2.0),
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_pointcloud(merged, output)
    logger.info("Multi-sensor fusion saved → %s  (%d points)", output, len(merged.points))


if __name__ == "__main__":
    cli()

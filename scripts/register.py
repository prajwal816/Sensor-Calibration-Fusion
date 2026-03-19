"""
CLI: Point cloud registration commands.

Usage:
    python scripts/register.py icp --source cloud_a.ply --target cloud_b.ply
    python scripts/register.py global --source cloud_a.ply --target cloud_b.ply
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import open3d as o3d

from src.registration.icp import run_icp
from src.registration.global_registration import run_global_registration
from src.registration.metrics import compute_alignment_error
from src.utils.io_utils import load_config, save_pointcloud, load_pointcloud
from src.utils.logger import get_logger


@click.group()
@click.option("--config", default="configs/default.yaml", help="Path to YAML config.")
@click.pass_context
def cli(ctx, config):
    """Point cloud registration CLI."""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config)


@cli.command()
@click.option("--source", required=True, help="Source point cloud file.")
@click.option("--target", required=True, help="Target point cloud file.")
@click.option("--output", default="data/output/registered.ply", help="Output aligned cloud.")
@click.pass_context
def icp(ctx, source, target, output):
    """Run ICP registration."""
    cfg = ctx.obj["cfg"]
    reg_cfg = cfg.get("registration", {}).get("icp", {})
    logger = get_logger("register", level=cfg.get("logging", {}).get("level", "INFO"))

    src_pcd = load_pointcloud(source)
    tgt_pcd = load_pointcloud(target)

    result = run_icp(
        src_pcd, tgt_pcd,
        method=reg_cfg.get("method", "point_to_plane"),
        max_iterations=reg_cfg.get("max_iterations", 200),
        tolerance=reg_cfg.get("tolerance", 1e-6),
        max_correspondence_distance=reg_cfg.get("max_correspondence_distance", 0.05),
        voxel_size=reg_cfg.get("voxel_size", 0.005),
    )

    aligned = o3d.geometry.PointCloud(src_pcd)
    aligned.transform(result["transformation_matrix"])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_pointcloud(aligned, output)

    metrics = compute_alignment_error(
        src_pcd, tgt_pcd, result["transformation_matrix"],
        max_distance=reg_cfg.get("max_correspondence_distance", 0.05),
    )
    logger.info("ICP result — fitness=%.4f  RMSE=%.6f  time=%.3fs",
                result["fitness"], result["inlier_rmse"], result["elapsed_seconds"])


@cli.command(name="global")
@click.option("--source", required=True, help="Source point cloud file.")
@click.option("--target", required=True, help="Target point cloud file.")
@click.option("--output", default="data/output/global_registered.ply", help="Output aligned cloud.")
@click.pass_context
def global_reg(ctx, source, target, output):
    """Run global (FPFH + RANSAC) registration."""
    cfg = ctx.obj["cfg"]
    reg_cfg = cfg.get("registration", {}).get("global", {})
    logger = get_logger("register", level=cfg.get("logging", {}).get("level", "INFO"))

    src_pcd = load_pointcloud(source)
    tgt_pcd = load_pointcloud(target)

    result = run_global_registration(
        src_pcd, tgt_pcd,
        voxel_size=reg_cfg.get("voxel_size", 0.05),
        fpfh_radius_multiplier=reg_cfg.get("fpfh_radius_multiplier", 5.0),
        fpfh_max_nn=reg_cfg.get("fpfh_max_nn", 100),
        distance_threshold_multiplier=reg_cfg.get("ransac_distance_threshold_multiplier", 1.5),
        max_iterations=reg_cfg.get("ransac_max_iterations", 4_000_000),
        confidence=reg_cfg.get("ransac_confidence", 0.999),
    )

    aligned = o3d.geometry.PointCloud(src_pcd)
    aligned.transform(result["transformation_matrix"])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_pointcloud(aligned, output)

    logger.info("Global registration — fitness=%.4f  RMSE=%.6f  time=%.3fs",
                result["fitness"], result["inlier_rmse"], result["elapsed_seconds"])


if __name__ == "__main__":
    cli()

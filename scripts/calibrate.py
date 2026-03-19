"""
CLI: Camera calibration commands.

Usage:
    python scripts/calibrate.py intrinsic --image-dir data/calibration_images
    python scripts/calibrate.py extrinsic --rgb-dir data/rgb_images --depth-dir data/depth_images
"""

from __future__ import annotations

import json
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import numpy as np

from src.calibration.intrinsic import calibrate_intrinsics
from src.calibration.extrinsic import calibrate_extrinsics
from src.calibration.calibration_data import CalibrationResult
from src.utils.io_utils import load_config
from src.utils.logger import get_logger


@click.group()
@click.option("--config", default="configs/default.yaml", help="Path to YAML config.")
@click.pass_context
def cli(ctx, config):
    """Camera calibration CLI."""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config)


@cli.command()
@click.option("--image-dir", required=True, help="Directory of checkerboard images.")
@click.option("--output", default="data/output/intrinsics.json", help="Output JSON.")
@click.pass_context
def intrinsic(ctx, image_dir, output):
    """Run intrinsic calibration."""
    cfg = ctx.obj["cfg"]
    cal_cfg = cfg.get("calibration", {})
    logger = get_logger("calibrate", level=cfg.get("logging", {}).get("level", "INFO"))

    result = calibrate_intrinsics(
        image_dir=image_dir,
        board_size=tuple(cal_cfg.get("board_size", [9, 6])),
        square_size=cal_cfg.get("square_size", 0.025),
        image_format=cal_cfg.get("image_format", "png"),
        save_dir=os.path.dirname(output),
    )

    cal = CalibrationResult(
        camera_matrix=result["camera_matrix"],
        dist_coeffs=result["dist_coeffs"],
        image_size=result["image_size"],
        reprojection_error=result["reprojection_error"],
    )
    cal.save(output)
    logger.info("Intrinsic calibration complete → %s  (RMSE %.4f px)", output, result["reprojection_error"])


@cli.command()
@click.option("--rgb-dir", required=True, help="RGB calibration images.")
@click.option("--depth-dir", required=True, help="Depth calibration images.")
@click.option("--rgb-intrinsics", default=None, help="Pre-calibrated RGB intrinsics JSON.")
@click.option("--depth-intrinsics", default=None, help="Pre-calibrated depth intrinsics JSON.")
@click.option("--output", default="data/output/extrinsics.json", help="Output JSON.")
@click.pass_context
def extrinsic(ctx, rgb_dir, depth_dir, rgb_intrinsics, depth_intrinsics, output):
    """Run extrinsic (stereo) calibration."""
    cfg = ctx.obj["cfg"]
    cal_cfg = cfg.get("calibration", {})
    logger = get_logger("calibrate", level=cfg.get("logging", {}).get("level", "INFO"))

    rgb_cm, rgb_dc, depth_cm, depth_dc = None, None, None, None
    if rgb_intrinsics:
        rgb_cal = CalibrationResult.load(rgb_intrinsics)
        rgb_cm, rgb_dc = rgb_cal.camera_matrix, rgb_cal.dist_coeffs
    if depth_intrinsics:
        depth_cal = CalibrationResult.load(depth_intrinsics)
        depth_cm, depth_dc = depth_cal.camera_matrix, depth_cal.dist_coeffs

    result = calibrate_extrinsics(
        rgb_image_dir=rgb_dir,
        depth_image_dir=depth_dir,
        board_size=tuple(cal_cfg.get("board_size", [9, 6])),
        square_size=cal_cfg.get("square_size", 0.025),
        image_format=cal_cfg.get("image_format", "png"),
        rgb_camera_matrix=rgb_cm,
        rgb_dist_coeffs=rgb_dc,
        depth_camera_matrix=depth_cm,
        depth_dist_coeffs=depth_dc,
    )

    out_data = {
        "R": result["R"].tolist(),
        "T": result["T"].tolist(),
        "reprojection_error": result["reprojection_error"],
        "transform": result["transform"].to_dict(),
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(out_data, f, indent=2)
    logger.info("Extrinsic calibration complete → %s", output)


if __name__ == "__main__":
    cli()

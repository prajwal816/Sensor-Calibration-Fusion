"""
Simulated ROS-style fusion node.

Subscribes to calibration + point cloud topics and publishes a fused
point cloud. Uses a lightweight shim when real ROS is unavailable.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import open3d as o3d

from src.fusion.multi_sensor_fusion import fuse_multi_sensor
from src.transforms.rigid_transform import RigidTransform
from src.utils.io_utils import load_pointcloud, save_pointcloud
from src.utils.logger import get_logger

logger = get_logger("fusion_node")


# --------------------------------------------------------------------------- #
#  Lightweight ROS shim
# --------------------------------------------------------------------------- #

class _FakeSubscriber:
    def __init__(self, topic: str, msg_type: str, callback):
        self.topic = topic
        logger.info("[ROS-shim] Subscriber created: %s (%s)", topic, msg_type)
        self._callback = callback

    def simulate_message(self, data):
        self._callback(data)


class _FakePublisher:
    def __init__(self, topic: str, msg_type: str):
        self.topic = topic
        logger.info("[ROS-shim] Publisher created: %s (%s)", topic, msg_type)

    def publish(self, data):
        logger.info("[ROS-shim] Published on %s: (PointCloud with %d points)", self.topic, data)


def _init_node(name: str):
    logger.info("[ROS-shim] Node initialised: %s", name)


# --------------------------------------------------------------------------- #
#  Fusion Node
# --------------------------------------------------------------------------- #

class FusionNode:
    """Subscribes to point cloud sources and publishes a fused cloud.

    Parameters
    ----------
    cloud_paths : list of PLY/PCD file paths to simulate subscribed messages.
    transform_dicts : list of transform dicts (one per cloud).
    output_path : where to save the fused result.
    """

    def __init__(
        self,
        cloud_paths: list[str],
        transform_dicts: list[dict],
        output_path: str = "data/output/ros_fused.ply",
    ):
        _init_node("fusion_node")

        self._clouds = [load_pointcloud(p) for p in cloud_paths]
        self._transforms = [
            RigidTransform.from_dict(d.get("transform", d)) for d in transform_dicts
        ]
        self._output_path = output_path
        self._pub = _FakePublisher("/sensor_fusion/fused_cloud", "PointCloud2")

    def spin_once(self) -> o3d.geometry.PointCloud:
        """Perform one fusion cycle and publish."""
        merged = fuse_multi_sensor(self._clouds, self._transforms)
        self._pub.publish(len(merged.points))
        save_pointcloud(merged, self._output_path)
        logger.info("FusionNode published fused cloud → %s", self._output_path)
        return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fusion subscriber/publisher node (simulated).")
    parser.add_argument("--clouds", nargs="+", required=True, help="Point cloud file paths.")
    parser.add_argument("--output", default="data/output/ros_fused.ply")
    args = parser.parse_args()

    # Use identity transforms by default
    identity = {"matrix": np.eye(4).tolist()}
    node = FusionNode(args.clouds, [identity] * len(args.clouds), output_path=args.output)
    node.spin_once()

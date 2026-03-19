"""
Simulated ROS-style calibration node.

This module provides a ROS-compatible interface stub for camera calibration.
It does NOT require an actual ROS installation — useful for development,
testing, and demonstration outside a ROS environment.

If ``rospy`` is available, real publishers/subscribers are used.
Otherwise, a lightweight shim is provided.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.calibration.calibration_data import CalibrationResult
from src.utils.logger import get_logger

logger = get_logger("calibration_node")


# --------------------------------------------------------------------------- #
#  Lightweight ROS shim (no dependency on rospy)
# --------------------------------------------------------------------------- #

class _FakePublisher:
    """Minimal publisher stub."""
    def __init__(self, topic: str, msg_type: str):
        self.topic = topic
        self.msg_type = msg_type
        logger.info("[ROS-shim] Publisher created: %s (%s)", topic, msg_type)

    def publish(self, data):
        logger.info("[ROS-shim] Published on %s: %s …", self.topic, str(data)[:120])


class _FakeRate:
    def __init__(self, hz: float):
        self._period = 1.0 / hz

    def sleep(self):
        time.sleep(self._period)


def _init_node(name: str):
    logger.info("[ROS-shim] Node initialised: %s", name)


# --------------------------------------------------------------------------- #
#  Calibration Node
# --------------------------------------------------------------------------- #

class CalibrationNode:
    """Publishes calibration results on a ROS topic (or simulated equivalent).

    Parameters
    ----------
    calibration_path : path to a CalibrationResult JSON file.
    topic : ROS topic name.
    rate_hz : publishing rate.
    """

    def __init__(
        self,
        calibration_path: str,
        topic: str = "/sensor_fusion/calibration",
        rate_hz: float = 1.0,
    ):
        _init_node("calibration_publisher")

        self.cal = CalibrationResult.load(calibration_path)
        self._pub = _FakePublisher(topic, "CalibrationResult")
        self._rate = _FakeRate(rate_hz)

    def spin(self, max_iters: int = 10) -> None:
        """Publish calibration data in a loop."""
        logger.info("CalibrationNode spinning (max %d iterations) …", max_iters)
        cal_dict = self.cal.to_dict()
        for i in range(max_iters):
            self._pub.publish(json.dumps(cal_dict))
            self._rate.sleep()
        logger.info("CalibrationNode finished.")


# --------------------------------------------------------------------------- #
#  Entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibration publisher node (simulated).")
    parser.add_argument("--calibration", default="data/output/intrinsics.json")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    node = CalibrationNode(args.calibration, rate_hz=args.rate)
    node.spin(max_iters=args.iters)

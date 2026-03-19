"""Tests for src/fusion — depth_to_pointcloud, rgb_depth_fusion, multi_sensor_fusion."""

import numpy as np
import open3d as o3d
import pytest

from src.fusion.depth_to_pointcloud import depth_to_pointcloud
from src.fusion.rgb_depth_fusion import fuse_rgb_depth
from src.fusion.multi_sensor_fusion import fuse_multi_sensor
from src.transforms.rigid_transform import RigidTransform


class TestDepthToPointcloud:
    """Back-projection from depth images to point clouds."""

    def _make_depth(self, w=640, h=480):
        """Synthetic depth image — gradient 800..2500 mm."""
        return np.linspace(800, 2500, h)[:, None].repeat(w, axis=1).astype(np.uint16)

    def test_output_has_points(self):
        depth = self._make_depth()
        pcd = depth_to_pointcloud(depth, fx=525, fy=525, cx=320, cy=240)
        assert len(pcd.points) > 0

    def test_no_points_for_zero_depth(self):
        depth = np.zeros((480, 640), dtype=np.uint16)
        pcd = depth_to_pointcloud(depth, fx=525, fy=525, cx=320, cy=240)
        assert len(pcd.points) == 0

    def test_depth_truncation(self):
        depth = self._make_depth()
        pcd = depth_to_pointcloud(depth, fx=525, fy=525, cx=320, cy=240,
                                   depth_scale=1000, depth_trunc=1.5)
        pts = np.asarray(pcd.points)
        assert pts[:, 2].max() < 1.5

    def test_step_reduces_points(self):
        depth = self._make_depth()
        pcd_full = depth_to_pointcloud(depth, fx=525, fy=525, cx=320, cy=240, step=1)
        pcd_step = depth_to_pointcloud(depth, fx=525, fy=525, cx=320, cy=240, step=2)
        assert len(pcd_step.points) < len(pcd_full.points)


class TestRGBDepthFusion:
    """RGB + depth fusion into coloured point clouds."""

    def _make_pair(self, w=640, h=480):
        rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        depth = np.linspace(800, 2500, h)[:, None].repeat(w, axis=1).astype(np.uint16)
        K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float64)
        return rgb, depth, K

    def test_fused_cloud_has_colors(self):
        rgb, depth, K = self._make_pair()
        pcd = fuse_rgb_depth(rgb, depth, K)
        assert len(pcd.points) > 0
        assert len(pcd.colors) == len(pcd.points)

    def test_colors_in_range(self):
        rgb, depth, K = self._make_pair()
        pcd = fuse_rgb_depth(rgb, depth, K)
        colors = np.asarray(pcd.colors)
        assert colors.min() >= 0.0
        assert colors.max() <= 1.0


class TestMultiSensorFusion:
    """Merging multiple point clouds."""

    def _make_random_cloud(self, n=500):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.randn(n, 3))
        return pcd

    def test_merge_identity_transforms(self):
        clouds = [self._make_random_cloud() for _ in range(3)]
        transforms = [RigidTransform() for _ in range(3)]
        merged = fuse_multi_sensor(clouds, transforms, voxel_size=0)
        total_input = sum(len(c.points) for c in clouds)
        assert len(merged.points) == total_input

    def test_voxel_downsample_reduces(self):
        clouds = [self._make_random_cloud(n=1000) for _ in range(2)]
        transforms = [RigidTransform() for _ in range(2)]
        merged = fuse_multi_sensor(clouds, transforms, voxel_size=0.5)
        assert len(merged.points) < 2000

    def test_mismatched_lengths_raises(self):
        clouds = [self._make_random_cloud() for _ in range(2)]
        transforms = [RigidTransform()]  # only one
        with pytest.raises(ValueError):
            fuse_multi_sensor(clouds, transforms)

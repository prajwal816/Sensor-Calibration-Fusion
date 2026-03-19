"""Tests for src/registration — ICP, global registration, metrics."""

import numpy as np
import open3d as o3d
import pytest

from src.registration.icp import run_icp, run_icp_with_convergence, run_multiscale_icp
from src.registration.metrics import compute_alignment_error, compute_rmse, compute_chamfer_distance
from src.transforms.rigid_transform import RigidTransform
from src.transforms.homogeneous import rotation_from_euler, make_homogeneous


def _make_test_clouds(num_points=5000, noise=0.002):
    """Create a source & target cloud with a known rigid transform."""
    np.random.seed(42)
    theta = np.random.uniform(0, np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    r = 0.5
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pts = np.column_stack([x, y, z])

    R_gt = rotation_from_euler((10, -5, 3), degrees=True)
    t_gt = np.array([0.05, -0.03, 0.1])
    T_gt = make_homogeneous(R_gt, t_gt)

    tf = RigidTransform(matrix=T_gt)
    pts_transformed = tf.apply(pts) + np.random.normal(0, noise, pts.shape)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pts_transformed)

    return source, target, T_gt


class TestICP:
    """ICP registration tests."""

    def test_icp_improves_alignment(self):
        source, target, T_gt = _make_test_clouds()
        before = compute_alignment_error(source, target, np.eye(4), max_distance=0.5)
        result = run_icp(source, target, method="point_to_point",
                         max_correspondence_distance=0.5, voxel_size=0.02)
        after = compute_alignment_error(source, target,
                                        result["transformation_matrix"], max_distance=0.5)
        assert after["inlier_rmse"] < before["inlier_rmse"]
        assert after["fitness"] >= before["fitness"]

    def test_icp_result_keys(self):
        source, target, _ = _make_test_clouds(num_points=500)
        result = run_icp(source, target, voxel_size=0.02,
                         max_correspondence_distance=0.5)
        expected = {"transform", "transformation_matrix", "fitness",
                    "inlier_rmse", "correspondence_set_size", "elapsed_seconds"}
        assert expected.issubset(result.keys())
        assert isinstance(result["transform"], RigidTransform)

    def test_icp_point_to_plane(self):
        source, target, _ = _make_test_clouds(num_points=2000)
        result = run_icp(source, target, method="point_to_plane",
                         max_correspondence_distance=0.5, voxel_size=0.02)
        assert result["fitness"] > 0

    def test_icp_with_init_transform(self):
        source, target, T_gt = _make_test_clouds()
        # Invert the ground truth as initialization (rough alignment)
        init = np.linalg.inv(T_gt)
        result = run_icp(source, target, init_transform=init,
                         max_correspondence_distance=0.1, voxel_size=0.02)
        assert result["fitness"] > 0.8


class TestICPConvergence:
    """ICP with convergence tracking."""

    def test_convergence_history_populated(self):
        source, target, _ = _make_test_clouds(num_points=1000)
        result = run_icp_with_convergence(
            source, target, max_iterations=20,
            max_correspondence_distance=0.5, voxel_size=0.02,
            convergence_step=5,
        )
        assert "convergence_history" in result
        assert len(result["convergence_history"]) > 0
        assert "iteration" in result["convergence_history"][0]
        assert "fitness" in result["convergence_history"][0]
        assert "inlier_rmse" in result["convergence_history"][0]

    def test_convergence_rmse_decreases(self):
        source, target, _ = _make_test_clouds(num_points=2000)
        result = run_icp_with_convergence(
            source, target, max_iterations=50,
            max_correspondence_distance=0.5, voxel_size=0.02,
            convergence_step=5,
        )
        history = result["convergence_history"]
        if len(history) >= 2:
            # RMSE should generally decrease (first vs last)
            assert history[-1]["inlier_rmse"] <= history[0]["inlier_rmse"] + 0.01

    def test_multiscale_icp(self):
        source, target, _ = _make_test_clouds(num_points=2000)
        result = run_multiscale_icp(
            source, target,
            voxel_scales=[0.04, 0.02],
            max_iterations_per_scale=[20, 10],
        )
        assert "convergence_history" in result
        assert result["fitness"] > 0
        # Check that multiple scales are recorded
        scales = set(h.get("scale") for h in result["convergence_history"])
        assert len(scales) >= 2


class TestMetrics:
    """Registration metrics tests."""

    def test_rmse_identical_points(self):
        pts = np.random.randn(100, 3)
        assert compute_rmse(pts, pts) == pytest.approx(0.0, abs=1e-10)

    def test_rmse_known_offset(self):
        pts = np.zeros((100, 3))
        offset = np.ones((100, 3)) * 0.1
        rmse = compute_rmse(pts, offset)
        expected = np.sqrt(0.03)  # sqrt(3 * 0.01)
        assert rmse == pytest.approx(expected, rel=1e-5)

    def test_chamfer_identical(self):
        pts = np.random.randn(200, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        cd = compute_chamfer_distance(pcd, pcd)
        assert cd == pytest.approx(0.0, abs=1e-10)

    def test_alignment_error_identity(self):
        pts = np.random.randn(200, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        metrics = compute_alignment_error(pcd, pcd, np.eye(4), max_distance=1.0)
        assert metrics["fitness"] == pytest.approx(1.0)
        assert metrics["inlier_rmse"] == pytest.approx(0.0, abs=1e-10)

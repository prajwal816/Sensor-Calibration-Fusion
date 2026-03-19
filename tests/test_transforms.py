"""Tests for src/transforms — RigidTransform, homogeneous matrices, frames."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.transforms.rigid_transform import RigidTransform
from src.transforms.homogeneous import (
    make_homogeneous,
    rotation_from_euler,
    rotation_from_axis_angle,
    rotation_from_quaternion,
    decompose_homogeneous,
)
from src.transforms.coordinate_frames import (
    camera_to_world,
    world_to_camera,
    depth_to_camera,
)


class TestHomogeneous:
    """Homogeneous matrix construction and decomposition."""

    def test_make_homogeneous_identity(self):
        T = make_homogeneous(np.eye(3), np.zeros(3))
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_make_homogeneous_with_translation(self):
        t = np.array([1.0, 2.0, 3.0])
        T = make_homogeneous(np.eye(3), t)
        np.testing.assert_array_almost_equal(T[:3, 3], t)
        np.testing.assert_array_almost_equal(T[3, :], [0, 0, 0, 1])

    def test_decompose_roundtrip(self):
        R = rotation_from_euler((30, 45, 60), degrees=True)
        t = np.array([0.5, -1.0, 2.0])
        T = make_homogeneous(R, t)
        R_out, t_out = decompose_homogeneous(T)
        np.testing.assert_array_almost_equal(R_out, R)
        np.testing.assert_array_almost_equal(t_out, t)

    def test_rotation_from_euler_identity(self):
        R = rotation_from_euler((0, 0, 0), degrees=True)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rotation_from_axis_angle(self):
        R = rotation_from_axis_angle([0, 0, 1], 90, degrees=True)
        point = np.array([1, 0, 0])
        rotated = R @ point
        np.testing.assert_array_almost_equal(rotated, [0, 1, 0], decimal=5)

    def test_rotation_from_quaternion(self):
        # 90° about z  → quat [0, 0, sin(45°), cos(45°)] scipy ordering
        q = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        R = rotation_from_quaternion(q)
        point = np.array([1, 0, 0])
        rotated = R @ point
        np.testing.assert_array_almost_equal(rotated, [0, 1, 0], decimal=5)

    def test_rotation_is_orthogonal(self):
        R = rotation_from_euler((10, 20, 30), degrees=True)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestRigidTransform:
    """SE(3) rigid transform operations."""

    def test_identity(self):
        T = RigidTransform()
        np.testing.assert_array_almost_equal(T.matrix, np.eye(4))

    def test_apply_identity(self):
        T = RigidTransform()
        pts = np.random.randn(100, 3)
        transformed = T.apply(pts)
        np.testing.assert_array_almost_equal(transformed, pts)

    def test_apply_translation_only(self):
        T = RigidTransform(t=np.array([1, 2, 3]))
        pts = np.array([[0, 0, 0], [1, 1, 1]])
        result = T.apply(pts)
        np.testing.assert_array_almost_equal(result, [[1, 2, 3], [2, 3, 4]])

    def test_inverse(self):
        R = rotation_from_euler((15, -10, 5), degrees=True)
        t = np.array([0.1, -0.05, 0.2])
        T = RigidTransform(R=R, t=t)
        T_inv = T.inverse()
        identity = T.compose(T_inv)
        np.testing.assert_array_almost_equal(identity.matrix, np.eye(4), decimal=10)

    def test_compose(self):
        T1 = RigidTransform(t=np.array([1, 0, 0]))
        T2 = RigidTransform(t=np.array([0, 1, 0]))
        T12 = T1.compose(T2)
        np.testing.assert_array_almost_equal(T12.translation, [1, 1, 0])

    def test_matmul_operator(self):
        T1 = RigidTransform(t=np.array([1, 0, 0]))
        T2 = RigidTransform(t=np.array([0, 1, 0]))
        T12 = T1 @ T2
        np.testing.assert_array_almost_equal(T12.matrix, T1.compose(T2).matrix)

    def test_serialisation_roundtrip(self):
        R = rotation_from_euler((30, 45, 60), degrees=True)
        T = RigidTransform(R=R, t=np.array([1, 2, 3]))
        d = T.to_dict()
        T2 = RigidTransform.from_dict(d)
        np.testing.assert_array_almost_equal(T.matrix, T2.matrix)

    def test_apply_inverse_roundtrip(self):
        R = rotation_from_euler((30, 45, 60), degrees=True)
        T = RigidTransform(R=R, t=np.array([1, 2, 3]))
        pts = np.random.randn(50, 3)
        transformed = T.apply(pts)
        recovered = T.inverse().apply(transformed)
        np.testing.assert_array_almost_equal(recovered, pts, decimal=10)


class TestCoordinateFrames:
    """Coordinate frame conversions."""

    def test_camera_world_roundtrip(self):
        R = rotation_from_euler((10, 20, 30), degrees=True)
        ext = RigidTransform(R=R, t=np.array([1, 0, 0]))
        pts = np.random.randn(20, 3)
        world = camera_to_world(pts, ext)
        back = world_to_camera(world, ext)
        np.testing.assert_array_almost_equal(back, pts, decimal=10)

    def test_depth_to_camera_center(self):
        # Principal point should map to (0,0,d)
        pts = depth_to_camera(
            u=np.array([320.0]),
            v=np.array([240.0]),
            depth=np.array([1.5]),
            fx=525, fy=525, cx=320, cy=240,
        )
        np.testing.assert_array_almost_equal(pts[0], [0, 0, 1.5])

    def test_depth_to_camera_batch(self):
        N = 100
        u = np.random.uniform(0, 640, N)
        v = np.random.uniform(0, 480, N)
        d = np.random.uniform(0.5, 3.0, N)
        pts = depth_to_camera(u, v, d, fx=525, fy=525, cx=320, cy=240)
        assert pts.shape == (N, 3)
        np.testing.assert_array_almost_equal(pts[:, 2], d)

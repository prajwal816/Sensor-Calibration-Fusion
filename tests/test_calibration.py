"""Tests for src/calibration — CalibrationResult, stereo_rectify."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from src.calibration.calibration_data import CalibrationResult
from src.calibration.intrinsic import detect_checkerboard
from src.calibration.stereo_rectify import (
    compute_rectification_maps,
    rectify_image_pair,
    draw_rectified_pair,
    compute_disparity_map,
)
from src.transforms.rigid_transform import RigidTransform


class TestCalibrationResult:
    """Serialisation roundtrip and validation."""

    def _make_sample(self) -> CalibrationResult:
        K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float64)
        return CalibrationResult(
            camera_matrix=K,
            dist_coeffs=np.zeros(5),
            image_size=(640, 480),
            reprojection_error=0.35,
            extrinsic=RigidTransform(t=np.array([0.1, 0, 0])),
            metadata={"sensor": "test"},
        )

    def test_to_dict_from_dict_roundtrip(self):
        cal = self._make_sample()
        d = cal.to_dict()
        cal2 = CalibrationResult.from_dict(d)
        np.testing.assert_array_almost_equal(cal.camera_matrix, cal2.camera_matrix)
        np.testing.assert_array_almost_equal(cal.dist_coeffs, cal2.dist_coeffs)
        assert cal.image_size == cal2.image_size
        assert cal.reprojection_error == cal2.reprojection_error
        np.testing.assert_array_almost_equal(
            cal.extrinsic.matrix, cal2.extrinsic.matrix
        )

    def test_save_load_roundtrip(self):
        cal = self._make_sample()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cal.json")
            cal.save(path)
            cal2 = CalibrationResult.load(path)
        np.testing.assert_array_almost_equal(cal.camera_matrix, cal2.camera_matrix)
        assert cal.reprojection_error == cal2.reprojection_error

    def test_without_extrinsic(self):
        K = np.eye(3)
        cal = CalibrationResult(
            camera_matrix=K,
            dist_coeffs=np.zeros(5),
            image_size=(640, 480),
            reprojection_error=0.5,
        )
        d = cal.to_dict()
        assert "extrinsic" not in d
        cal2 = CalibrationResult.from_dict(d)
        assert cal2.extrinsic is None


class TestCheckerboardDetection:
    """Checkerboard corner detection on synthetic images."""

    def _render_board(self, board_size=(9, 6), square_px=30, margin=50):
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

    def test_detect_clean_board(self):
        img = self._render_board()
        found, corners = detect_checkerboard(img, (9, 6))
        assert found
        assert corners is not None
        assert corners.shape[0] == 54

    def test_no_board_in_blank_image(self):
        blank = np.ones((480, 640), dtype=np.uint8) * 128
        found, corners = detect_checkerboard(blank, (9, 6))
        assert not found
        assert corners is None


class TestStereoRectification:
    """Stereo rectification map computation and image warping."""

    def _make_stereo_params(self):
        K1 = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float64)
        K2 = K1.copy()
        D1 = np.zeros(5)
        D2 = np.zeros(5)
        R = np.eye(3, dtype=np.float64)
        T = np.array([[0.05], [0.0], [0.0]], dtype=np.float64)  # 5cm baseline
        return K1, D1, K2, D2, (640, 480), R, T

    def test_compute_maps_keys(self):
        K1, D1, K2, D2, size, R, T = self._make_stereo_params()
        maps = compute_rectification_maps(K1, D1, K2, D2, size, R, T)
        expected_keys = {"R1", "R2", "P1", "P2", "Q", "map1_x", "map1_y",
                         "map2_x", "map2_y", "valid_roi_1", "valid_roi_2"}
        assert expected_keys.issubset(maps.keys())

    def test_rectify_preserves_shape(self):
        K1, D1, K2, D2, size, R, T = self._make_stereo_params()
        maps = compute_rectification_maps(K1, D1, K2, D2, size, R, T)
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        r1, r2 = rectify_image_pair(img1, img2, maps)
        assert r1.shape == img1.shape
        assert r2.shape == img2.shape

    def test_draw_rectified_pair_output_shape(self):
        K1, D1, K2, D2, size, R, T = self._make_stereo_params()
        maps = compute_rectification_maps(K1, D1, K2, D2, size, R, T)
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        r1, r2 = rectify_image_pair(img1, img2, maps)
        canvas = draw_rectified_pair(r1, r2)
        assert canvas.shape[0] == 480
        assert canvas.shape[1] == 1280  # side-by-side

    def test_disparity_map_shape(self):
        g1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        g2 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        disp = compute_disparity_map(g1, g2, num_disparities=64, block_size=9)
        assert disp.shape == (480, 640)
        assert disp.dtype == np.float32

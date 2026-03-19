"""Tests for src/utils — logger, io_utils, visualization."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.utils.logger import get_logger
from src.utils.io_utils import load_config, save_yaml, load_image, save_image
from src.utils.visualization import (
    plot_reprojection_errors,
    plot_convergence,
    plot_registration_comparison,
    draw_corners,
)


class TestLogger:
    """Logger factory tests."""

    def test_returns_logger(self):
        log = get_logger("test_logger_1")
        assert log.name == "test_logger_1"

    def test_singleton(self):
        log1 = get_logger("test_singleton")
        log2 = get_logger("test_singleton")
        assert log1 is log2

    def test_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            log = get_logger("test_file_handler", log_file=log_file)
            log.info("hello")
            assert os.path.exists(log_file)
            # Close handlers so Windows can delete the file
            for handler in log.handlers[:]:
                handler.close()
                log.removeHandler(handler)
            # Remove from logger singleton cache
            import src.utils.logger as _lmod
            _lmod._INITIALISED_LOGGERS.pop("test_file_handler", None)


class TestIOUtils:
    """Config, image, and array I/O tests."""

    def test_save_load_yaml_roundtrip(self):
        data = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.yaml")
            save_yaml(data, path)
            loaded = load_config(path)
        assert loaded == data

    def test_load_config_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_save_load_image_roundtrip(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            save_image(img, path)
            loaded = load_image(path)
        assert loaded.shape == img.shape

    def test_load_image_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.png")


class TestVisualization:
    """Visualization helpers (save-to-file only, no display)."""

    def test_reprojection_error_plot(self):
        errors = np.random.uniform(0, 2, 200)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reproj.png")
            plot_reprojection_errors(errors, save_path=path)
            assert os.path.exists(path)

    def test_convergence_plot(self):
        history = [
            {"iteration": i, "fitness": min(1.0, 0.1 * i), "inlier_rmse": max(0.001, 0.05 - 0.005 * i)}
            for i in range(1, 11)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "convergence.png")
            plot_convergence(history, save_path=path)
            assert os.path.exists(path)

    def test_convergence_plot_multiscale(self):
        history = [
            {"iteration": i, "fitness": 0.5 + 0.05 * i, "inlier_rmse": 0.05 - 0.004 * i, "scale": 0}
            for i in range(1, 6)
        ] + [
            {"iteration": i + 5, "fitness": 0.8 + 0.02 * i, "inlier_rmse": 0.02 - 0.002 * i, "scale": 1}
            for i in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "multiscale.png")
            plot_convergence(history, title="Multi-scale ICP", save_path=path)
            assert os.path.exists(path)

    def test_registration_comparison_plot(self):
        before = {"fitness": 0.3, "inlier_rmse": 0.05, "chamfer": 0.2}
        after = {"fitness": 1.0, "inlier_rmse": 0.003, "chamfer": 0.006}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "comparison.png")
            plot_registration_comparison(before, after, save_path=path)
            assert os.path.exists(path)

    def test_draw_corners(self):
        img = np.ones((300, 400), dtype=np.uint8) * 200
        fake_corners = np.random.rand(54, 1, 2).astype(np.float32) * 200 + 50
        result = draw_corners(img, fake_corners, (9, 6), found=True)
        assert result.shape == (300, 400, 3)

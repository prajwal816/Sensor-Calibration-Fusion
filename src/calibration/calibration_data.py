"""
CalibrationResult — serialisable container for calibration data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..transforms.rigid_transform import RigidTransform
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Holds intrinsic and extrinsic calibration results.

    Attributes
    ----------
    camera_matrix : (3, 3) ndarray — intrinsic matrix K.
    dist_coeffs : ndarray — distortion coefficients.
    image_size : (width, height).
    reprojection_error : float — mean reprojection RMSE (px).
    extrinsic : RigidTransform or None — camera-to-reference transform.
    metadata : arbitrary extra info.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float
    extrinsic: Optional[RigidTransform] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----- serialisation --------------------------------------------------- #

    def to_dict(self) -> dict:
        d: dict = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "image_size": list(self.image_size),
            "reprojection_error": float(self.reprojection_error),
            "metadata": self.metadata,
        }
        if self.extrinsic is not None:
            d["extrinsic"] = self.extrinsic.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationResult":
        ext = None
        if "extrinsic" in d:
            ext = RigidTransform.from_dict(d["extrinsic"])
        return cls(
            camera_matrix=np.array(d["camera_matrix"]),
            dist_coeffs=np.array(d["dist_coeffs"]),
            image_size=tuple(d["image_size"]),
            reprojection_error=d["reprojection_error"],
            extrinsic=ext,
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        """Write to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved calibration to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        """Read from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        logger.info("Loaded calibration from %s", path)
        return cls.from_dict(d)

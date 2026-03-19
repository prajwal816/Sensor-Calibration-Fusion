"""
RigidTransform — a composable rigid-body (SE(3)) transformation.

Stores a 4×4 homogeneous matrix internally and exposes compose, invert,
and apply operations.
"""

from __future__ import annotations

import numpy as np

from .homogeneous import make_homogeneous, decompose_homogeneous


class RigidTransform:
    """Rigid-body transform in SE(3).

    Parameters
    ----------
    R : (3, 3) ndarray — rotation matrix.  Identity by default.
    t : (3,) ndarray  — translation vector.  Zero by default.
    matrix : (4, 4) ndarray — full homogeneous matrix (overrides R, t).
    """

    def __init__(
        self,
        R: np.ndarray | None = None,
        t: np.ndarray | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        if matrix is not None:
            self._T = np.array(matrix, dtype=np.float64)
        else:
            R = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
            t = np.zeros(3, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
            self._T = make_homogeneous(R, t)

    # ---- properties ---- #

    @property
    def matrix(self) -> np.ndarray:
        """4×4 homogeneous matrix."""
        return self._T.copy()

    @property
    def rotation(self) -> np.ndarray:
        """3×3 rotation component."""
        return self._T[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        """3-element translation component."""
        return self._T[:3, 3].copy()

    # ---- operations ---- #

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        """Return ``self ∘ other``  (i.e. self @ other)."""
        return RigidTransform(matrix=self._T @ other._T)

    def inverse(self) -> "RigidTransform":
        """Return T⁻¹."""
        R, t = decompose_homogeneous(self._T)
        R_inv = R.T
        t_inv = -R_inv @ t
        return RigidTransform(R=R_inv, t=t_inv)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the transform to an (N, 3) point array.

        Parameters
        ----------
        points : (N, 3) array

        Returns
        -------
        transformed : (N, 3) array
        """
        pts = np.asarray(points, dtype=np.float64)
        N = pts.shape[0]
        ones = np.ones((N, 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])             # (N, 4)
        transformed_h = (self._T @ pts_h.T).T       # (N, 4)
        return transformed_h[:, :3]

    # ---- serialisation ---- #

    def to_dict(self) -> dict:
        """Serialise to plain dict (JSON-safe lists)."""
        return {"matrix": self._T.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "RigidTransform":
        """Deserialise from dict."""
        return cls(matrix=np.array(d["matrix"]))

    # ---- dunder ---- #

    def __repr__(self) -> str:
        return f"RigidTransform(\n{self._T}\n)"

    def __matmul__(self, other: "RigidTransform") -> "RigidTransform":
        return self.compose(other)

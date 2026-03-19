"""
Homogeneous-matrix construction from various rotation representations.

All functions return 4×4 ``numpy.ndarray`` transformation matrices.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous matrix from a 3×3 rotation and 3×1 translation.

    Parameters
    ----------
    R : (3, 3) array
    t : (3,) or (3, 1) array

    Returns
    -------
    T : (4, 4) array
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).flatten()
    return T


def rotation_from_euler(
    angles: tuple[float, float, float],
    seq: str = "xyz",
    degrees: bool = False,
) -> np.ndarray:
    """3×3 rotation matrix from Euler angles.

    Parameters
    ----------
    angles : (roll, pitch, yaw) or matching *seq*.
    seq : str  — axis sequence, e.g. ``"xyz"``, ``"ZYX"``.
    degrees : bool
    """
    return Rotation.from_euler(seq, angles, degrees=degrees).as_matrix()


def rotation_from_axis_angle(axis: np.ndarray, angle: float, degrees: bool = False) -> np.ndarray:
    """3×3 rotation from an axis + angle.

    Parameters
    ----------
    axis : (3,) unit vector.
    angle : scalar.
    degrees : bool
    """
    if degrees:
        angle = np.radians(angle)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    rotvec = axis * angle
    return Rotation.from_rotvec(rotvec).as_matrix()


def rotation_from_quaternion(q: np.ndarray) -> np.ndarray:
    """3×3 rotation from a quaternion ``[x, y, z, w]`` (scipy convention).

    Parameters
    ----------
    q : (4,) array  — ``[x, y, z, w]``.
    """
    return Rotation.from_quat(q).as_matrix()


def decompose_homogeneous(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract R (3×3) and t (3,) from a 4×4 homogeneous matrix."""
    return T[:3, :3].copy(), T[:3, 3].copy()

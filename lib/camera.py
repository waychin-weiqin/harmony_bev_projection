"""Camera math utilities.

All functions accept / produce plain numpy arrays.  No OpenCV dependency except
for project_points (which uses cv2.projectPoints for distortion correctness).
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


Intrinsics = Dict  # {fx, fy, cx, cy, k1, k2, p1, p2, w, h}


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (COLMAP / Hamilton convention) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qz*qw,  2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx*qx - 2*qz*qz,  2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,  1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def R_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3×3 rotation matrix → quaternion (qw, qx, qy, qz), normalised."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    return qw/norm, qx/norm, qy/norm, qz/norm


# ---------------------------------------------------------------------------
# Intrinsics helpers
# ---------------------------------------------------------------------------

def make_K(intr: Intrinsics) -> np.ndarray:
    """Build 3×3 camera matrix from an intrinsics dict."""
    return np.array([
        [intr["fx"], 0,          intr["cx"]],
        [0,          intr["fy"], intr["cy"]],
        [0,          0,          1         ],
    ], dtype=np.float64)


def make_dist(intr: Intrinsics) -> np.ndarray:
    """Build (k1, k2, p1, p2) distortion vector from an intrinsics dict."""
    return np.array(
        [intr.get("k1", 0), intr.get("k2", 0),
         intr.get("p1", 0), intr.get("p2", 0)],
        dtype=np.float64,
    )


def make_K_scaled(intr: Intrinsics, scale: float) -> np.ndarray:
    """Return K with focal lengths and principal point scaled by *scale*."""
    return np.array([
        [intr["fx"] * scale, 0,                  intr["cx"] * scale],
        [0,                  intr["fy"] * scale,  intr["cy"] * scale],
        [0,                  0,                   1                  ],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_points(
    pts_3d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project N×3 world points to N×2 image pixels (with distortion)."""
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    pts_2d, _ = cv2.projectPoints(pts_3d.astype(np.float64), rvec, tvec, K, dist)
    return pts_2d.reshape(-1, 2)


def reprojection_errors(
    pts_3d: np.ndarray,
    pts_2d_obs: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (per-point errors, projected_2d) for a set of correspondences."""
    projected = project_points(pts_3d, K, dist, R, t)
    errors = np.linalg.norm(projected - pts_2d_obs.astype(np.float64), axis=1)
    return errors, projected


# ---------------------------------------------------------------------------
# Pose utility
# ---------------------------------------------------------------------------

def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space camera centre C = -R^T t."""
    return -R.T @ t


def pose_from_dict(pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (R, t) from a pose dict with qw/qx/qy/qz/tx/ty/tz keys."""
    R = quat_to_R(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
    t = np.array([pose["tx"], pose["ty"], pose["tz"]], dtype=np.float64)
    return R, t


def is_nearly_coplanar(pts_3d: np.ndarray, threshold: float = 0.05) -> bool:
    """Return True when the points lie approximately in a single plane.

    Uses the ratio of the smallest to largest singular value of the centered
    point matrix.  Threshold ≈ 0.05 means the smallest axis is < 5% of the
    largest — effectively planar.
    """
    if len(pts_3d) < 4:
        return True
    centered = pts_3d - pts_3d.mean(axis=0)
    _, s, _ = np.linalg.svd(centered)
    return float(s[-1]) / (float(s[0]) + 1e-9) < threshold

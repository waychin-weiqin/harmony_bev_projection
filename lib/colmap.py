"""COLMAP and custom pose file readers.

Supported camera models (COLMAP cameras.txt):
  PINHOLE      fx fy cx cy
  RADIAL       fx fy cx cy k1 k2
  OPENCV       fx fy cx cy k1 k2 p1 p2
  FULL_OPENCV  fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6  (k3..k6 ignored)

All models are normalized to the same dict layout:
  {fx, fy, cx, cy, k1, k2, p1, p2, w, h}

For unknown models a warning is emitted and distortion is set to zero.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Intrinsics = Dict  # {fx, fy, cx, cy, k1, k2, p1, p2, w, h}
Pose = Dict        # {name, qw, qx, qy, qz, tx, ty, tz}


# ---------------------------------------------------------------------------
# cameras.txt
# ---------------------------------------------------------------------------

def read_cameras_txt(path: str | Path) -> Dict[int, Intrinsics]:
    """Parse a COLMAP cameras.txt and return a dict of camera_id → intrinsics."""
    path = Path(path)
    cameras: Dict[int, Intrinsics] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1].upper()
            w, h = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = _parse_camera_model(cam_id, model, w, h, params)

    return cameras


def _parse_camera_model(cam_id: int, model: str, w: int, h: int,
                         params: list) -> Intrinsics:
    base = {"w": w, "h": h, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}

    if model in ("SIMPLE_PINHOLE",):
        f = params[0]
        base.update(fx=f, fy=f, cx=params[1], cy=params[2])

    elif model in ("PINHOLE",):
        base.update(fx=params[0], fy=params[1], cx=params[2], cy=params[3])

    elif model in ("SIMPLE_RADIAL",):
        f = params[0]
        base.update(fx=f, fy=f, cx=params[1], cy=params[2], k1=params[3])

    elif model in ("RADIAL",):
        f = params[0]
        base.update(fx=f, fy=f, cx=params[1], cy=params[2],
                    k1=params[3], k2=params[4])

    elif model in ("OPENCV",):
        base.update(fx=params[0], fy=params[1], cx=params[2], cy=params[3],
                    k1=params[4], k2=params[5], p1=params[6], p2=params[7])

    elif model in ("FULL_OPENCV",):
        base.update(fx=params[0], fy=params[1], cx=params[2], cy=params[3],
                    k1=params[4], k2=params[5], p1=params[6], p2=params[7])
        # k3..k6 are ignored — not supported by cv2.solvePnP default

    elif model in ("OPENCV_FISHEYE",):
        # Use only the first two radial distortion params; no tangential
        base.update(fx=params[0], fy=params[1], cx=params[2], cy=params[3],
                    k1=params[4], k2=params[5])
        print(f"  Warning: camera {cam_id} uses OPENCV_FISHEYE — "
              "fisheye coefficients mapped to k1/k2, p1/p2 set to 0.",
              file=sys.stderr)

    else:
        print(f"  Warning: camera {cam_id} has unsupported model '{model}' — "
              "intrinsics set to identity with zero distortion.", file=sys.stderr)
        cx, cy = w / 2.0, h / 2.0
        f = max(w, h)
        base.update(fx=f, fy=f, cx=cx, cy=cy)

    return base


# ---------------------------------------------------------------------------
# Custom poses file (camID_to_imagename.txt)
# ---------------------------------------------------------------------------

def read_poses_txt(path: str | Path) -> Dict[int, Pose]:
    """Parse a custom poses file.

    Format (one camera per line, comment lines with '#' are skipped):
        CAMERA_ID  image_name  qw qx qy qz  tx ty tz
    """
    path = Path(path)
    poses: Dict[int, Pose] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.upper().startswith("CAMERA_ID"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            cam_id = int(parts[0])
            name = parts[1]
            qw, qx, qy, qz = map(float, parts[2:6])
            tx, ty, tz = map(float, parts[6:9])
            poses[cam_id] = dict(
                name=name,
                qw=qw, qx=qx, qy=qy, qz=qz,
                tx=tx, ty=ty, tz=tz,
            )

    return poses


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------

def build_intrinsics(
    colmap_cameras_path: Optional[Path] = None,
    manual_table: Optional[Dict] = None,
    poses: Optional[Dict[int, Pose]] = None,
) -> Dict[int, Intrinsics]:
    """Return a cam_id → Intrinsics dict.

    Priority:
      1. colmap_cameras_path (if provided)
      2. manual_table from config['camera_intrinsics'] (if provided)

    When using COLMAP cameras, only cameras whose IDs appear in *poses* are
    returned (if poses is given); this avoids loading intrinsics for cameras
    that are not being evaluated.
    """
    if colmap_cameras_path is not None:
        all_intr = read_cameras_txt(colmap_cameras_path)
        if poses is not None:
            return {k: v for k, v in all_intr.items() if k in poses}
        return all_intr

    if manual_table is not None:
        result = {}
        for cam_id, raw in manual_table.items():
            result[int(cam_id)] = {
                "fx": float(raw.get("fx", raw.get("f", 1000))),
                "fy": float(raw.get("fy", raw.get("f", 1000))),
                "cx": float(raw["cx"]),
                "cy": float(raw["cy"]),
                "k1": float(raw.get("k1", 0)),
                "k2": float(raw.get("k2", 0)),
                "p1": float(raw.get("p1", 0)),
                "p2": float(raw.get("p2", 0)),
                "w": int(raw.get("w", raw.get("width", 0))),
                "h": int(raw.get("h", raw.get("height", 0))),
            }
        return result

    raise ValueError("Neither colmap_cameras_path nor manual_table provided.")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_poses_txt(path: str | Path, poses: Dict[int, Pose]) -> None:
    """Write poses in the custom CAMERA_ID name qw qx qy qz tx ty tz format.

    Parameters
    ----------
    path:
        Output file path (created or overwritten).
    poses:
        Dict mapping camera_id → pose dict as returned by :func:`read_poses_txt`.
        Each dict must have keys: name, qw, qx, qy, qz, tx, ty, tz.
    """
    path = Path(path)
    with open(path, "w") as f:
        f.write("# CAMERA_ID  name  qw qx qy qz  tx ty tz\n")
        for cam_id in sorted(poses):
            p = poses[cam_id]
            f.write(
                f"{cam_id}  {p['name']}  "
                f"{p['qw']:.10f} {p['qx']:.10f} {p['qy']:.10f} {p['qz']:.10f}  "
                f"{p['tx']:.10f} {p['ty']:.10f} {p['tz']:.10f}\n"
            )

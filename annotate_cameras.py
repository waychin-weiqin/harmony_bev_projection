"""Annotate camera positions on the aligned BEV image.

Reads refined_poses.txt, computes each camera's world-space centre
(C = -R^T @ t), projects it through BevTransform, and draws the
camera ID as a labelled marker.

Cameras that fall outside the drone BEV footprint (common — they sit at
the perimeter of the scene) are drawn on an expanded canvas so every
camera is visible.

Usage
-----
    python annotate_cameras.py \\
        --poses   /path/to/refined_poses.txt \\
        --bev-transform /path/to/bev_transform.json \\
        --bev     data/drone_bev_aligned.png \\
        --output  data/bev_cameras.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from lib.bev import BevTransform
from lib.camera import quat_to_R
from lib.colmap import read_poses_txt

# BGR colours — one per camera in sorted ID order
_COLOURS = [
    (57,  183, 255),   # amber
    (60,  220, 60),    # green
    (255, 100, 60),    # blue
    (80,  80,  255),   # red
    (255, 60,  200),   # magenta
    (60,  220, 220),   # yellow
    (200, 130, 60),    # teal
    (160, 60,  255),   # orange-red
]

_PADDING = 160          # pixels of dark border added around the BEV
_MARKER_R = 18          # outer ring radius
_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_THICKNESS = 2


def camera_centre(pose: dict) -> np.ndarray:
    """Return the 3-D world position of the camera optical centre (C = -R^T t)."""
    qw, qx, qy, qz = pose["qw"], pose["qx"], pose["qy"], pose["qz"]
    R = quat_to_R(qw, qx, qy, qz)
    t = np.array([pose["tx"], pose["ty"], pose["tz"]])
    return -R.T @ t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate camera positions on the aligned BEV image.",
    )
    parser.add_argument(
        "--poses", metavar="FILE",
        default=str(_HERE.parent / "outputs" / "round2" / "refined_poses.txt"),
    )
    parser.add_argument(
        "--bev-transform", metavar="FILE",
        default=str(_HERE.parent / "outputs" / "round2" / "bev_transform.json"),
    )
    parser.add_argument(
        "--bev", metavar="FILE",
        default=str(_HERE / "data" / "drone_bev_aligned.png"),
    )
    parser.add_argument(
        "--output", metavar="FILE",
        default=str(_HERE / "data" / "bev_cameras.png"),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    poses = read_poses_txt(args.poses)
    with open(args.bev_transform) as f:
        bev_transform = BevTransform.from_dict(json.load(f))

    bev_img = cv2.imread(args.bev)
    if bev_img is None:
        print(f"ERROR: cannot load BEV image: {args.bev}")
        sys.exit(1)

    bev_h, bev_w = bev_img.shape[:2]

    # world_to_aligned_bev() returns coordinates in the pre-rotation BEV space
    # (rotate_ccw_90=True in bev_transform).  The BEV image stored in data/ has
    # been rotated 90° CW to correct the orientation, so we apply the same
    # 90° CW mapping: (x, y) → (H_pre - 1 - y, x), where H_pre is the height
    # of the image before the CW rotation = current BEV width (bev_w).
    H_pre = bev_w   # pre-rotation height = post-rotation width (1136 → 913 swap)

    # ------------------------------------------------------------------
    # Compute every camera's BEV pixel position
    # ------------------------------------------------------------------
    cam_pixels: dict[int, tuple[int, int]] = {}
    for cam_id, pose in sorted(poses.items()):
        C = camera_centre(pose)
        bx_pre, by_pre = bev_transform.world_to_aligned_bev(C[0], C[1])
        # Apply 90° CW correction
        bx = H_pre - 1 - by_pre
        by = bx_pre
        cam_pixels[cam_id] = (int(round(bx)), int(round(by)))
        print(f"  Camera {cam_id}: world ({C[0]:.2f}, {C[1]:.2f}) "
              f"→ BEV ({cam_pixels[cam_id][0]}, {cam_pixels[cam_id][1]})")

    # ------------------------------------------------------------------
    # Compute how much border to add so all cameras fit
    # ------------------------------------------------------------------
    all_x = [px for px, py in cam_pixels.values()]
    all_y = [py for px, py in cam_pixels.values()]

    pad_left  = max(_PADDING, _PADDING - min(all_x))
    pad_top   = max(_PADDING, _PADDING - min(all_y))
    pad_right = max(_PADDING, max(all_x) - bev_w  + _PADDING)
    pad_bot   = max(_PADDING, max(all_y) - bev_h + _PADDING)

    canvas_w = bev_w + pad_left + pad_right
    canvas_h = bev_h + pad_top  + pad_bot

    # Dark background with slight texture
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

    # Paste BEV into canvas
    canvas[pad_top:pad_top + bev_h, pad_left:pad_left + bev_w] = bev_img

    # Light border around BEV footprint
    cv2.rectangle(canvas,
                  (pad_left - 1, pad_top - 1),
                  (pad_left + bev_w, pad_top + bev_h),
                  (80, 80, 80), 1)

    # ------------------------------------------------------------------
    # Draw each camera
    # ------------------------------------------------------------------
    for i, (cam_id, (bx, by)) in enumerate(sorted(cam_pixels.items())):
        cx = bx + pad_left
        cy = by + pad_top
        colour = _COLOURS[i % len(_COLOURS)]
        label  = str(cam_id)

        # If camera is outside the BEV region, draw a dashed leader line
        # from the nearest BEV border to the marker
        bx_clamped = max(0, min(bev_w - 1, bx))
        by_clamped = max(0, min(bev_h - 1, by))
        outside = (bx != bx_clamped) or (by != by_clamped)

        if outside:
            # Leader line from clamped border point to marker
            border_cx = bx_clamped + pad_left
            border_cy = by_clamped + pad_top
            _draw_dashed_line(canvas, (border_cx, border_cy), (cx, cy), colour, 1, 10)

        # Marker: dark filled disc + coloured ring + inner dot
        cv2.circle(canvas, (cx, cy), _MARKER_R + 2, (20, 20, 20), -1)
        cv2.circle(canvas, (cx, cy), _MARKER_R,     colour,       2)
        cv2.circle(canvas, (cx, cy), 5,              colour,       -1)

        # Label centred above marker
        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        tx = cx - tw // 2
        ty = cy - _MARKER_R - 8
        # Shadow
        cv2.putText(canvas, label, (tx + 1, ty + 1),
                    _FONT, _FONT_SCALE, (10, 10, 10), _FONT_THICKNESS + 1, cv2.LINE_AA)
        # Text
        cv2.putText(canvas, label, (tx, ty),
                    _FONT, _FONT_SCALE, colour, _FONT_THICKNESS, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"\nSaved → {out_path}  ({canvas_w} × {canvas_h} px)")


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    colour: tuple,
    thickness: int = 1,
    dash_len: int = 10,
) -> None:
    """Draw a dashed line between pt1 and pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    t = 0.0
    draw = True
    while t < dist:
        t_end = min(t + dash_len, dist)
        if draw:
            p1 = (int(x1 + dx * t), int(y1 + dy * t))
            p2 = (int(x1 + dx * t_end), int(y1 + dy * t_end))
            cv2.line(img, p1, p2, colour, thickness)
        t = t_end
        draw = not draw


if __name__ == "__main__":
    main()

"""Bird's Eye View coordinate utilities.

The BEV is an orthographic projection of the XY ground plane viewed from
above.  Y increases upward in the world but downward in image coordinates,
so images are flipped on the Y axis.

Coordinate chain from world → final aligned BEV image pixel:

  1. world (x, y)  →  ortho BEV pixel  (OrthoParams)
  2. ortho BEV px  →  rectified px      (perspective_M from corner selection)
  3. rectified px  →  final px          (optional 90° CCW rotation)

All three steps are encoded in BevTransform, which is serialisable to/from
the JSON file produced by step 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# OrthoParams
# ---------------------------------------------------------------------------

@dataclass
class OrthoParams:
    """Parameters for the orthographic top-down (BEV) projection."""
    scale: float       # pixels per metre
    x_min: float       # world-X at left edge of image
    y_max: float       # world-Y at top edge of image  (note: larger Y = higher up)
    width: int         # image width  in pixels
    height: int        # image height in pixels

    def world_to_bev(self, x: float, y: float) -> Tuple[float, float]:
        """World (x, y) → ortho BEV pixel (px, py).  Y is flipped."""
        px = (x - self.x_min) * self.scale
        py = (self.y_max - y) * self.scale
        return px, py

    def bev_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Ortho BEV pixel (px, py) → world (x, y)."""
        x = px / self.scale + self.x_min
        y = self.y_max - py / self.scale
        return x, y

    @classmethod
    def from_cones(cls, cones_xy: np.ndarray, scale: int = 200,
                   padding: float = 0.5) -> "OrthoParams":
        """Compute OrthoParams that tightly bounds the given XY cone positions."""
        x_min = float(cones_xy[:, 0].min()) - padding
        x_max = float(cones_xy[:, 0].max()) + padding
        y_min = float(cones_xy[:, 1].min()) - padding
        y_max = float(cones_xy[:, 1].max()) + padding
        w = int((x_max - x_min) * scale)
        h = int((y_max - y_min) * scale)
        return cls(scale=scale, x_min=x_min, y_max=y_max, width=w, height=h)

    def to_dict(self) -> dict:
        return dict(scale=self.scale, x_min=self.x_min, y_max=self.y_max,
                    width=self.width, height=self.height)

    @classmethod
    def from_dict(cls, d: dict) -> "OrthoParams":
        return cls(scale=d["scale"], x_min=d["x_min"], y_max=d["y_max"],
                   width=d["width"], height=d["height"])


# ---------------------------------------------------------------------------
# BevTransform — full world-to-aligned-BEV chain
# ---------------------------------------------------------------------------

@dataclass
class BevTransform:
    """Serialisable transform chain from world coordinates to aligned BEV pixels.

    Chain:
      world (x,y)
        → ortho_params.world_to_bev()
        → perspective_M @ [ox, oy, 1]  (from manual corner selection)
        → optional 90° CCW: (ax, ay) → (ay, pre_rotation_w - 1 - ax)
        → final pixel (px, py)
    """
    ortho_params: OrthoParams
    perspective_M: np.ndarray      # 3×3, ortho BEV → rectified image
    pre_rotation_size: Tuple[int, int]  # (w, h) before the 90° rotation
    rotate_ccw_90: bool
    aligned_size: Tuple[int, int]  # (w, h) of the final output image

    def world_to_aligned_bev(self, x: float, y: float) -> Tuple[float, float]:
        """Project world (x, y) to aligned BEV pixel (px, py)."""
        ox, oy = self.ortho_params.world_to_bev(x, y)
        pt = self.perspective_M @ np.array([ox, oy, 1.0])
        ax, ay = pt[0] / pt[2], pt[1] / pt[2]
        if self.rotate_ccw_90:
            pre_w = self.pre_rotation_size[0]
            return ay, pre_w - 1 - ax
        return ax, ay

    def to_dict(self) -> dict:
        return {
            "ortho_params": self.ortho_params.to_dict(),
            "perspective_M": self.perspective_M.tolist(),
            "pre_rotation_size": list(self.pre_rotation_size),
            "rotate_ccw_90": self.rotate_ccw_90,
            "aligned_size": list(self.aligned_size),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BevTransform":
        return cls(
            ortho_params=OrthoParams.from_dict(d["ortho_params"]),
            perspective_M=np.array(d["perspective_M"], dtype=np.float64),
            pre_rotation_size=tuple(d["pre_rotation_size"]),
            rotate_ccw_90=bool(d["rotate_ccw_90"]),
            aligned_size=tuple(d["aligned_size"]),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BevTransform":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

CAMERA_COLORS = [
    (255,   0,   0),  # Blue
    (  0, 255,   0),  # Green
    (  0,   0, 255),  # Red
    (255, 255,   0),  # Cyan
    (255,   0, 255),  # Magenta
    (  0, 255, 255),  # Yellow
    (128,   0, 255),  # Purple
    (  0, 128, 255),  # Orange
]


def color_for_camera(cam_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR colour for a given camera id."""
    return CAMERA_COLORS[cam_id % len(CAMERA_COLORS)]

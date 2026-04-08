"""Project detected image-space points onto the aligned drone BEV.

Pre-computed homographies for cameras 175–182 are included in ``data/``.
Simply provide your detections and run — no calibration step needed.

Input detection format (one JSON file per camera)
--------------------------------------------------
Minimal::

    {"detections": [[u1, v1], [u2, v2], ...]}

With optional labels::

    {
        "detections": [[u1, v1], [u2, v2], ...],
        "labels":     ["cone",   "car",   ...]
    }

Coordinates must be in undistorted image space (i.e. from images processed
with step 0.1 undistortion).

Usage — single camera
---------------------
::

    python project_to_bev.py \\
        --input detections/cam175.json \\
        --camera 175

Usage — batch (all cameras in a directory)
------------------------------------------
::

    python project_to_bev.py \\
        --input-dir detections/

Camera IDs are inferred from filenames that contain a number, e.g.:
  ``cam175.json``  →  camera 175
  ``175.json``     →  camera 175
  ``det_180.json`` →  camera 180

Outputs (all in ``--output-dir``, default: ``outputs/``)
---------------------------------------------------------
  ``bev_projection_cam{id}.json``  — per-camera projected coordinates
  ``bev_projection_cam{id}.png``   — detections overlaid on drone BEV
  ``bev_projection_all.json``      — all cameras combined
  ``bev_projection_all.png``       — all cameras combined on one drone BEV
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from lib.bev import color_for_camera


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DEFAULT_HOMOGRAPHY = _HERE / "data" / "image_bev_homographies_analytical.json"
_DEFAULT_BEV        = _HERE / "data" / "drone_bev_aligned.png"
_DEFAULT_OUTPUT_DIR = _HERE / "outputs"


# ---------------------------------------------------------------------------
# Homography loading
# ---------------------------------------------------------------------------

def load_homographies(path: Path) -> Dict[int, np.ndarray]:
    """Return {cam_id: H_3x3} from either a step-4b or analytical homography file.

    Accepts two JSON schemas:

    * **Step-4b** (``clicked_cone_tips.json``): entries nested under a
      ``"homographies"`` key.
    * **Analytical** (``image_bev_homographies_analytical*.json``): flat dict
      with camera IDs at the top level.
    """
    with open(path) as f:
        data = json.load(f)

    entries = data["homographies"] if "homographies" in data else data

    homos: Dict[int, np.ndarray] = {}
    for cam_id_str, info in entries.items():
        if not isinstance(info, dict) or "H" not in info:
            continue
        homos[int(cam_id_str)] = np.array(info["H"], dtype=np.float64)
    return homos


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_detections(path: Path) -> Tuple[List[List[float]], List[Optional[str]]]:
    """Load detections from a JSON file.

    Returns
    -------
    detections : list of [u, v] pixel coordinates
    labels     : list of string labels (or None per entry if not provided)
    """
    with open(path) as f:
        data = json.load(f)

    raw = data.get("detections", [])
    detections = [[float(pt[0]), float(pt[1])] for pt in raw]

    raw_labels = data.get("labels", [])
    labels: List[Optional[str]] = []
    for i in range(len(detections)):
        labels.append(str(raw_labels[i]) if i < len(raw_labels) else None)

    return detections, labels


def _cam_id_from_filename(path: Path) -> Optional[int]:
    """Extract the first run of digits from a stem as the camera ID."""
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_detections(
    detections: List[List[float]],
    H: np.ndarray,
) -> np.ndarray:
    """Apply homography H to a list of [u, v] image pixels.

    Returns an N×2 float32 array of BEV pixel positions.
    """
    if not detections:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array(detections, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return projected


# ---------------------------------------------------------------------------
# Per-camera output
# ---------------------------------------------------------------------------

def _draw_detections(
    canvas: np.ndarray,
    bev_pts: np.ndarray,
    labels: List[Optional[str]],
    cam_id: int,
    draw_cam_label: bool = True,
) -> None:
    color = color_for_camera(cam_id)
    h, w = canvas.shape[:2]

    for i, (bx, by) in enumerate(bev_pts):
        px, py = int(round(float(bx))), int(round(float(by)))
        if not (0 <= px < w and 0 <= py < h):
            continue

        cv2.circle(canvas, (px, py), 8, color, -1)
        lbl = f"{cam_id}:{i}" if draw_cam_label else str(i)
        if labels[i] is not None:
            lbl = f"{lbl} {labels[i]}"
        cv2.putText(canvas, lbl, (px + 10, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def process_camera(
    cam_id: int,
    detections: List[List[float]],
    labels: List[Optional[str]],
    H: np.ndarray,
    bev_base: np.ndarray,
    homography_source: str,
    output_dir: Path,
) -> List[dict]:
    """Project detections for one camera, save JSON + PNG, return projection dicts."""
    bev_pts = project_detections(detections, H)

    projections = []
    for i, (img_pt, label) in enumerate(zip(detections, labels)):
        bx, by = float(bev_pts[i, 0]), float(bev_pts[i, 1])
        entry: dict = {
            "detection_id": i,
            "img_pixel": [round(img_pt[0], 3), round(img_pt[1], 3)],
            "bev_pixel": [round(bx, 3), round(by, 3)],
        }
        if label is not None:
            entry["label"] = label
        projections.append(entry)

    out_json = output_dir / f"bev_projection_cam{cam_id}.json"
    with open(out_json, "w") as f:
        json.dump({
            "cam_id": cam_id,
            "homography_source": homography_source,
            "n_detections": len(detections),
            "projections": projections,
        }, f, indent=2)
    print(f"  [{cam_id}] Saved {out_json.name}")

    canvas = bev_base.copy()
    _draw_detections(canvas, bev_pts, labels, cam_id, draw_cam_label=False)

    h_c, w_c = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w_c, 32), (30, 30, 30), -1)
    cv2.putText(canvas,
                f"Cam {cam_id}  —  {len(detections)} detection(s)  "
                f"[{homography_source}]",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_png = output_dir / f"bev_projection_cam{cam_id}.png"
    cv2.imwrite(str(out_png), canvas)
    print(f"  [{cam_id}] Saved {out_png.name}")

    return projections


# ---------------------------------------------------------------------------
# Combined output
# ---------------------------------------------------------------------------

def save_combined(
    all_projections: List[dict],
    cameras: List[int],
    bev_base: np.ndarray,
    homography_source: str,
    homos: Dict[int, np.ndarray],
    cam_detections: Dict[int, Tuple[List[List[float]], List[Optional[str]]]],
    output_dir: Path,
) -> None:
    out_json = output_dir / "bev_projection_all.json"
    with open(out_json, "w") as f:
        json.dump({
            "homography_source": homography_source,
            "cameras": cameras,
            "n_total_detections": len(all_projections),
            "projections": all_projections,
        }, f, indent=2)
    print(f"  Saved {out_json.name}  ({len(all_projections)} total projections)")

    canvas = bev_base.copy()
    for cam_id in cameras:
        if cam_id not in homos or cam_id not in cam_detections:
            continue
        H = homos[cam_id]
        dets, lbls = cam_detections[cam_id]
        bev_pts = project_detections(dets, H)
        _draw_detections(canvas, bev_pts, lbls, cam_id, draw_cam_label=True)

    lx, ly_start = 8, canvas.shape[0] - 14 - 18 * len(cameras)
    bg_y = ly_start - 6
    cv2.rectangle(canvas,
                  (lx - 4, bg_y),
                  (lx + 180, bg_y + 18 * len(cameras) + 8),
                  (30, 30, 30), -1)
    for i, cam_id in enumerate(sorted(cameras)):
        color = color_for_camera(cam_id)
        ly = ly_start + i * 18
        cv2.circle(canvas, (lx + 6, ly + 4), 6, color, -1)
        cv2.putText(canvas, f"Camera {cam_id}", (lx + 16, ly + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out_png = output_dir / "bev_projection_all.png"
    cv2.imwrite(str(out_png), canvas)
    print(f"  Saved {out_png.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project detected image-space points onto the aligned drone BEV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", metavar="FILE",
        help="Path to a single detection JSON file.",
    )
    input_group.add_argument(
        "--input-dir", metavar="DIR",
        help="Directory of detection JSON files; camera IDs inferred from filenames.",
    )

    parser.add_argument(
        "--camera", type=int, metavar="ID",
        help="Camera ID (required when --input is used).",
    )
    parser.add_argument(
        "--homography", default=str(_DEFAULT_HOMOGRAPHY), metavar="FILE",
        help=f"Homography JSON file (default: data/image_bev_homographies_analytical.json).",
    )
    parser.add_argument(
        "--bev", default=str(_DEFAULT_BEV), metavar="FILE",
        help="Path to drone_bev_aligned.png (default: data/drone_bev_aligned.png).",
    )
    parser.add_argument(
        "--output-dir", default=str(_DEFAULT_OUTPUT_DIR), metavar="DIR",
        help="Directory for output files (default: outputs/).",
    )

    args = parser.parse_args()

    homo_path = Path(args.homography)
    bev_path  = Path(args.bev)
    out_dir   = Path(args.output_dir)

    if not homo_path.exists():
        print(f"ERROR: Homography file not found: {homo_path}")
        sys.exit(1)
    if not bev_path.exists():
        print(f"ERROR: BEV image not found: {bev_path}")
        sys.exit(1)

    # Collect input files → {cam_id: Path}
    input_files: Dict[int, Path] = {}

    if args.input:
        if args.camera is None:
            parser.error("--camera is required when --input is used.")
        input_files[args.camera] = Path(args.input)
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Input directory not found: {input_dir}")
            sys.exit(1)
        for json_file in sorted(input_dir.glob("*.json")):
            cam_id = _cam_id_from_filename(json_file)
            if cam_id is None:
                print(f"  WARNING: Could not infer camera ID from '{json_file.name}' — skipped.")
                continue
            input_files[cam_id] = json_file

    if not input_files:
        print("ERROR: No detection files found.")
        sys.exit(1)

    print(f"\nLoading homographies from: {homo_path}")
    homos = load_homographies(homo_path)
    if not homos:
        print("ERROR: No valid homographies found in the file.")
        sys.exit(1)
    print(f"  Available cameras: {sorted(homos)}")

    bev_base = cv2.imread(str(bev_path))
    if bev_base is None:
        print(f"ERROR: Cannot load BEV image: {bev_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    homo_source = homo_path.name

    print(f"\nProjecting to BEV  →  output dir: {out_dir}\n")

    all_projections: List[dict] = []
    processed_cameras: List[int] = []
    cam_detections: Dict[int, Tuple[List[List[float]], List[Optional[str]]]] = {}

    for cam_id in sorted(input_files):
        det_path = input_files[cam_id]

        if cam_id not in homos:
            print(f"  [{cam_id}] WARNING: no homography available for this camera — skipped.")
            continue

        print(f"  [{cam_id}] Reading {det_path.name} ...")
        try:
            detections, labels = load_detections(det_path)
        except Exception as e:
            print(f"  [{cam_id}] ERROR reading file: {e}. Skipped.")
            continue

        if not detections:
            print(f"  [{cam_id}] No detections in file — skipped.")
            continue

        print(f"  [{cam_id}] {len(detections)} detection(s)")

        cam_detections[cam_id] = (detections, labels)
        projections = process_camera(
            cam_id=cam_id,
            detections=detections,
            labels=labels,
            H=homos[cam_id],
            bev_base=bev_base,
            homography_source=homo_source,
            output_dir=out_dir,
        )

        for proj in projections:
            all_projections.append({"cam_id": cam_id, **proj})
        processed_cameras.append(cam_id)

    if not processed_cameras:
        print("\nNo cameras were processed (check warnings above).")
        sys.exit(1)

    print(f"\nSaving combined outputs ({len(all_projections)} total projections) ...")
    save_combined(
        all_projections=all_projections,
        cameras=processed_cameras,
        bev_base=bev_base,
        homography_source=homo_source,
        homos=homos,
        cam_detections=cam_detections,
        output_dir=out_dir,
    )

    print(f"\nDone.  Results in: {out_dir}/")


if __name__ == "__main__":
    main()

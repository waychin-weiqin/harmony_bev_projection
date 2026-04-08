"""Compute analytical per-camera image → aligned BEV homographies.

For a point on the plane Z = z_g the projection from world (X, Y) to
camera image pixel is a planar projective transform (homography):

    λ [u, v, 1]ᵀ = A · [X, Y, 1]ᵀ
    A = [ P[:,0]  P[:,1]  P[:,2]·z_g + P[:,3] ]    (3×3)
    P = K · [R | t]                                  (3×4)

Inverting A gives image pixel → world XY.  Composing with the BevTransform
chain (OrthoParams → perspective_M → optional 90° CCW rotation) yields a
single 3×3 matrix H mapping camera image pixels to aligned BEV pixels.

Pre-computed homographies for cameras 175–182 are already provided in
``data/image_bev_homographies_analytical.json``.  Run this script only if
you need to recompute them (e.g. after updating poses or bev_transform.json).

Usage
-----
    python compute_homography.py \\
        --correspondences /path/to/correspondences.json \\
        --bev-transform   /path/to/bev_transform.json \\
        --poses           /path/to/refined_poses.txt \\
        --colmap-cameras  /path/to/cameras.txt \\
        --bev             /path/to/drone_bev_aligned.png \\
        --output-dir      data/

    # Custom plane height (for detections above cone tips):
    python compute_homography.py ... --z-plane 1.2

Outputs (in --output-dir)
-------------------------
    image_bev_homographies_analytical.json   — H matrices + validation stats
    homo_viz_analytical_cam{id}.png          — per-camera residual on BEV
    homo_viz_analytical_all.png              — mosaic of all cameras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from lib.bev import BevTransform, OrthoParams
from lib.camera import make_K, pose_from_dict
from lib.colmap import read_poses_txt, build_intrinsics


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _ortho_matrix(ortho: OrthoParams) -> np.ndarray:
    """3×3 homogeneous matrix: world (X, Y, 1) → ortho BEV pixel."""
    s = ortho.scale
    return np.array(
        [[s,  0, -s * ortho.x_min],
         [0, -s,  s * ortho.y_max],
         [0,  0,  1.0            ]],
        dtype=np.float64,
    )


def _rotation_90ccw_matrix(pre_rot_w: int) -> np.ndarray:
    """3×3 homogeneous matrix for 90° CCW rotation."""
    return np.array(
        [[ 0,  1,              0],
         [-1,  0, pre_rot_w - 1],
         [ 0,  0,              1]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Core analytical H
# ---------------------------------------------------------------------------

def compute_analytical_H(
    intrinsics: dict,
    pose: dict,
    z_ground: float,
    bev_transform: BevTransform,
) -> np.ndarray:
    """Return the 3×3 homography mapping image pixel → aligned BEV pixel.

    Parameters
    ----------
    intrinsics  : {fx, fy, cx, cy, ...}  (distortion ignored — see note)
    pose        : {qw, qx, qy, qz, tx, ty, tz}
    z_ground    : world Z of the projection plane (metres)
    bev_transform : loaded from bev_transform.json

    Note on distortion
    ------------------
    A homography cannot model lens distortion.  Pass intrinsics from
    undistorted images (K_cropped, zero distortion) for best accuracy.
    """
    K = make_K(intrinsics)
    R, t = pose_from_dict(pose)

    P = K @ np.hstack([R, t.reshape(3, 1)])
    A = np.column_stack([P[:, 0], P[:, 1], P[:, 2] * z_ground + P[:, 3]])
    A_inv = np.linalg.inv(A)

    T_ortho = _ortho_matrix(bev_transform.ortho_params)
    M_persp = bev_transform.perspective_M

    if bev_transform.rotate_ccw_90:
        T_rot = _rotation_90ccw_matrix(bev_transform.pre_rotation_size[0])
        H = T_rot @ M_persp @ T_ortho @ A_inv
    else:
        H = M_persp @ T_ortho @ A_inv

    return H


# ---------------------------------------------------------------------------
# Computation + evaluation
# ---------------------------------------------------------------------------

def compute_and_save(
    correspondences_json: Path,
    bev_transform_json: Path,
    poses_path: Path,
    colmap_cameras_path: Optional[Path],
    manual_intrinsics: Optional[dict],
    undistorted_cameras_json: Path,
    output_path: Path,
    z_plane_override: Optional[float] = None,
) -> Dict[str, dict]:
    """Compute analytical H for every camera and evaluate against noise-free GT."""
    with open(correspondences_json) as f:
        corr_data = json.load(f)

    image_space = corr_data.get("image_space", "raw")

    with open(bev_transform_json) as f:
        bev_transform = BevTransform.from_dict(json.load(f))

    # Determine projection plane Z
    z_vals = [c["coords_3d"][2] for c in corr_data["cones"]]
    z_median = float(np.median(z_vals))

    if z_plane_override is not None:
        z_ground = z_plane_override
        print(f"  Projection plane Z = {z_ground:.4f} m  "
              f"[user-specified; cone-tip median = {z_median:.4f} m, "
              f"delta = {z_ground - z_median:+.4f} m]")
    else:
        z_ground = z_median
        print(f"  Ground plane Z = {z_ground:.4f} m  "
              f"(median over {len(z_vals)} cone tips, "
              f"range {min(z_vals):.3f}–{max(z_vals):.3f} m)")

    poses = read_poses_txt(poses_path)
    intrinsics = build_intrinsics(
        colmap_cameras_path=colmap_cameras_path,
        manual_table=manual_intrinsics,
        poses=poses,
    )

    if image_space == "undistorted":
        if undistorted_cameras_json.exists():
            with open(undistorted_cameras_json) as f:
                undist_data = json.load(f)
            for cam_id_str, cam_info in undist_data.items():
                cid = int(cam_id_str)
                if cid in intrinsics:
                    K_cr = np.array(cam_info["K_cropped"])
                    intr = dict(intrinsics[cid])
                    intr.update(
                        fx=float(K_cr[0, 0]), fy=float(K_cr[1, 1]),
                        cx=float(K_cr[0, 2]), cy=float(K_cr[1, 2]),
                        k1=0.0, k2=0.0, p1=0.0, p2=0.0,
                        w=cam_info["width"], h=cam_info["height"],
                    )
                    intrinsics[cid] = intr
            print(f"  Using K_cropped + zero distortion (image_space={image_space})")
        else:
            print(f"  WARNING: image_space={image_space} but undistorted_cameras.json "
                  "not found — falling back to original K with distortion ignored")

    cam_obs: Dict[int, List[Tuple]] = {}
    for cone in corr_data["cones"]:
        world_xy = cone["coords_3d"][:2]
        for cam_id_str, img_pt in cone.get("observations", {}).items():
            cam_id = int(cam_id_str)
            cam_obs.setdefault(cam_id, []).append(
                (img_pt, world_xy, cone["cone_id"])
            )

    result: Dict[str, dict] = {}

    for cam_id in sorted(cam_obs):
        if cam_id not in poses:
            print(f"  Camera {cam_id}: no pose — skipping")
            continue
        if cam_id not in intrinsics:
            print(f"  Camera {cam_id}: no intrinsics — skipping")
            continue

        H = compute_analytical_H(
            intrinsics=intrinsics[cam_id],
            pose=poses[cam_id],
            z_ground=z_ground,
            bev_transform=bev_transform,
        )

        obs_list = cam_obs[cam_id]
        img_pts = np.array([o[0] for o in obs_list], dtype=np.float32)
        gt_pts = np.array(
            [bev_transform.world_to_aligned_bev(o[1][0], o[1][1]) for o in obs_list],
            dtype=np.float32,
        )

        proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        errs = np.linalg.norm(proj - gt_pts, axis=1)

        print(f"  Camera {cam_id}: {len(obs_list)} pts  "
              f"mean {errs.mean():.2f} px  max {errs.max():.2f} px")

        result[str(cam_id)] = {
            "H": H.tolist(),
            "n_correspondences": len(obs_list),
            "n_inliers": len(obs_list),
            "mean_error_px": round(float(errs.mean()), 3),
            "max_error_px":  round(float(errs.max()), 3),
            "z_ground": round(z_ground, 6),
        }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved {len(result)} analytical homographies → {output_path}")
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(
    correspondences_json: Path,
    bev_transform_json: Path,
    analytical_json: Path,
    aligned_bev_path: Path,
    output_dir: Path,
    viz_suffix: str = "_analytical",
) -> None:
    """Draw GT vs projected (H_analytical * image_pt) on drone_bev_aligned.png."""
    with open(correspondences_json) as f:
        corr_data = json.load(f)
    with open(bev_transform_json) as f:
        bev_transform = BevTransform.from_dict(json.load(f))
    with open(analytical_json) as f:
        homo_H = json.load(f)

    aligned_bev_base = cv2.imread(str(aligned_bev_path))
    if aligned_bev_base is None:
        raise FileNotFoundError(f"Cannot load aligned BEV: {aligned_bev_path}")

    cam_obs: Dict[int, List[Tuple]] = {}
    for cone in corr_data["cones"]:
        world_xy = cone["coords_3d"][:2]
        for cam_id_str, img_pt in cone.get("observations", {}).items():
            cam_id = int(cam_id_str)
            cam_obs.setdefault(cam_id, []).append(
                (img_pt, world_xy, cone["cone_id"])
            )

    tile_imgs = []
    bev_h, bev_w = aligned_bev_base.shape[:2]

    for cam_id in sorted(cam_obs):
        if str(cam_id) not in homo_H:
            continue

        H = np.array(homo_H[str(cam_id)]["H"], dtype=np.float64)
        canvas = aligned_bev_base.copy()

        for (img_pt, world_xy, cid) in cam_obs[cam_id]:
            gx_f, gy_f = bev_transform.world_to_aligned_bev(world_xy[0], world_xy[1])
            gx, gy = int(round(gx_f)), int(round(gy_f))

            src = np.array([[[float(img_pt[0]), float(img_pt[1])]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(src, H).reshape(2)
            px, py = int(round(proj[0])), int(round(proj[1]))

            in_bev = 0 <= px < bev_w and 0 <= py < bev_h
            gt_in_bev = 0 <= gx < bev_w and 0 <= gy < bev_h

            if in_bev and gt_in_bev:
                cv2.line(canvas, (gx, gy), (px, py), (0, 140, 255), 1)
                err = float(np.hypot(px - gx, py - gy))
                mid = ((gx + px) // 2 + 4, (gy + py) // 2)
                cv2.putText(canvas, f"{err:.0f}", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

            if gt_in_bev:
                cv2.circle(canvas, (gx, gy), 9, (0, 200, 0), 2)

            if in_bev:
                cv2.drawMarker(canvas, (px, py), (0, 0, 220),
                               cv2.MARKER_CROSS, 16, 2)
                cv2.putText(canvas, str(cid), (px + 10, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)

        stats = homo_H[str(cam_id)]
        label = (f"Cam {cam_id}  n={stats['n_correspondences']}"
                 f"  mean {stats['mean_error_px']:.1f} px"
                 f"  max {stats['max_error_px']:.1f} px"
                 f"  [analytical]")
        cv2.rectangle(canvas, (0, 0), (bev_w, 34), (30, 30, 30), -1)
        cv2.putText(canvas, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        lx, ly = 10, bev_h - 55
        cv2.rectangle(canvas, (lx - 4, ly - 20), (lx + 300, ly + 40), (30, 30, 30), -1)
        cv2.circle(canvas, (lx + 9, ly), 8, (0, 200, 0), 2)
        cv2.putText(canvas, "GT (world_to_aligned_bev, no click noise)",
                    (lx + 22, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.drawMarker(canvas, (lx + 9, ly + 22), (0, 0, 220),
                       cv2.MARKER_CROSS, 14, 2)
        cv2.putText(canvas, "H_analytical * image pixel",
                    (lx + 22, ly + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out_path = output_dir / f"homo_viz{viz_suffix}_cam{cam_id}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"  Saved {out_path}")
        tile_imgs.append(canvas)

    if tile_imgs:
        max_h = max(i.shape[0] for i in tile_imgs)
        max_w = max(i.shape[1] for i in tile_imgs)
        padded = []
        for img in tile_imgs:
            pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            pad[:img.shape[0], :img.shape[1]] = img
            padded.append(pad)
        cols = 2
        rows = (len(padded) + 1) // cols
        scale = min(1.0, 3200 / (max_w * cols))
        tw, th = int(max_w * scale), int(max_h * scale)
        tiles = [cv2.resize(p, (tw, th)) for p in padded]
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))
        rows_imgs = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
        mosaic = np.vstack(rows_imgs)
        mosaic_path = output_dir / f"homo_viz{viz_suffix}_all.png"
        cv2.imwrite(str(mosaic_path), mosaic)
        print(f"  Saved mosaic → {mosaic_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute analytical image → aligned BEV homographies from poses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--z-plane", type=float, default=None, metavar="Z",
        help=(
            "World Z (metres) of the projection plane. Overrides the median "
            "cone-tip Z. Use when detecting features at a known height above "
            "the cones (e.g. --z-plane 1.2)."
        ),
    )
    parser.add_argument(
        "--output-dir", default=str(_DATA), metavar="DIR",
        help="Directory for output JSON and visualisation images (default: data/).",
    )

    # Override arguments — all default to the bundled data/ files
    parser.add_argument(
        "--correspondences", default=str(_DATA / "correspondences.json"), metavar="FILE",
        help="Path to correspondences.json (default: data/correspondences.json).",
    )
    parser.add_argument(
        "--bev-transform", default=str(_DATA / "bev_transform.json"), metavar="FILE",
        help="Path to bev_transform.json (default: data/bev_transform.json).",
    )
    parser.add_argument(
        "--poses", default=str(_DATA / "refined_poses.txt"), metavar="FILE",
        help="Path to poses file (default: data/refined_poses.txt).",
    )
    parser.add_argument(
        "--bev", default=str(_DATA / "drone_bev_aligned.png"), metavar="FILE",
        help="Path to drone_bev_aligned.png (default: data/drone_bev_aligned.png).",
    )
    parser.add_argument(
        "--colmap-cameras", default=str(_DATA / "cameras.txt"), metavar="FILE",
        help="Path to COLMAP cameras.txt (default: data/cameras.txt).",
    )
    parser.add_argument(
        "--undistorted-cameras", default=str(_DATA / "undistorted_cameras.json"),
        metavar="FILE",
        help="Path to undistorted_cameras.json (default: data/undistorted_cameras.json).",
    )
    parser.add_argument(
        "--no-visualise", action="store_true",
        help="Skip generating visualisation images.",
    )

    args = parser.parse_args()

    z_plane_override: Optional[float] = args.z_plane

    print("\n" + "=" * 60)
    print("Analytical image → aligned BEV homographies")
    print("=" * 60)

    for label, p in [
        ("--correspondences",    args.correspondences),
        ("--bev-transform",      args.bev_transform),
        ("--poses",              args.poses),
        ("--bev",                args.bev),
        ("--colmap-cameras",     args.colmap_cameras),
    ]:
        if not Path(p).exists():
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if z_plane_override is not None:
        z_tag = f"z_{z_plane_override:.4f}".replace("-", "m")
        out_json = out_dir / f"image_bev_homographies_analytical_{z_tag}.json"
    else:
        z_tag = None
        out_json = out_dir / "image_bev_homographies_analytical.json"

    undist_path = Path(args.undistorted_cameras)

    compute_and_save(
        correspondences_json=Path(args.correspondences),
        bev_transform_json=Path(args.bev_transform),
        poses_path=Path(args.poses),
        colmap_cameras_path=Path(args.colmap_cameras),
        manual_intrinsics=None,
        undistorted_cameras_json=undist_path,
        output_path=out_json,
        z_plane_override=z_plane_override,
    )

    if not args.no_visualise:
        print("\n--- Visualisation ---")
        viz_suffix = f"_analytical_{z_tag}" if z_tag is not None else "_analytical"
        visualise(
            correspondences_json=Path(args.correspondences),
            bev_transform_json=Path(args.bev_transform),
            analytical_json=out_json,
            aligned_bev_path=Path(args.bev),
            output_dir=out_dir,
            viz_suffix=viz_suffix,
        )


if __name__ == "__main__":
    main()

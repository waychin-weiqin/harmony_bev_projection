# bev-projection

Project detected image-space points from any of the eight calibrated cameras
onto the aligned Bird's Eye View (BEV) image using pre-computed analytical
homographies.

## Installation

```bash
git clone https://github.com/waychin-weiqin/harmony_bev_projection.git
cd bev_projection
pip install .
```

This installs `opencv-python` and `numpy`. Python 3.9+ is required.

## Bundled data

Everything needed to run out of the box is included in `data/`:

| File | Description |
|---|---|
| `image_bev_homographies_analytical.json` | Pre-computed H matrices for cameras 175–182 |
| `drone_bev_aligned.png` | Reference BEV image |
| `bev_cameras.png` | BEV annotated with camera positions |
| `correspondences.json` | Cone 3D positions and image observations |
| `bev_transform.json` | World → BEV transform chain |
| `refined_poses.txt` | Camera extrinsics for all 8 cameras |
| `cameras.txt` | COLMAP intrinsics for cameras 175–182 |
| `undistorted_cameras.json` | Cropped K matrices for undistorted image space |

---

## Project detections onto the BEV

Prepare a detection JSON for each camera:

```json
{
    "detections": [[u1, v1], [u2, v2]],
    "labels": ["cone", "car"]
}
```

Pixel coordinates must be in **undistorted image space** (i.e. from images
processed with step 0.1 of the main pipeline).

### Single camera

```bash
python project_to_bev.py --input detections/cam175.json --camera 175
```

### All cameras at once

```bash
python project_to_bev.py --input-dir detections/
```

Camera IDs are inferred from filenames (`cam175.json` → camera 175).
Results are written to `outputs/` by default.

### Output files

| File | Description |
|---|---|
| `bev_projection_cam{id}.json` | Per-camera projected BEV pixel coordinates |
| `bev_projection_cam{id}.png` | Detections overlaid on the BEV image |
| `bev_projection_all.json` | All cameras combined |
| `bev_projection_all.png` | All cameras combined on one BEV image |

### Options

```
--homography FILE    Homography JSON (default: data/image_bev_homographies_analytical.json)
--bev FILE          BEV image (default: data/drone_bev_aligned.png)
--output-dir DIR    Output directory (default: outputs/)
```

---

## Recompute homographies for a different plane height

The pre-computed homographies in `data/` assume detections lie on the
**cone-tip plane** (~0.03 m above the world origin). If your detections
correspond to features at a different height (e.g. the mid-body of a person,
a mounted marker), recompute with `--z-plane`:

```bash
# Cone-tip plane (default)
python compute_homography.py

# Custom plane at 1.2 m
python compute_homography.py --z-plane 1.2
```

All required files are bundled in `data/` and picked up automatically.
Output is written back to `data/`:

- Default: `data/image_bev_homographies_analytical.json`
- Custom:  `data/image_bev_homographies_analytical_z_1.2000.json`

Pass the custom file to `project_to_bev.py` via `--homography`:

```bash
python project_to_bev.py \
    --input detections/cam175.json --camera 175 \
    --homography data/image_bev_homographies_analytical_z_1.2000.json
```

> **Note:** The validation errors printed during recomputation compare
> projected cone observations against the cone-tip GT. Errors will appear
> large when using a non-default plane height — this is expected, since the
> ground truth is still at cone-tip level.

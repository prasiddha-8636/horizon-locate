#!/usr/bin/env python
"""Measure GSV mask boundary error against DB-predicted skyline rows.

For samples with reliable exp_corr (the 'OK' class), the DB horizon predicts
where the skyline should be in the crop.  Compare the actual U-Net mask
boundary to that prediction:

  systematic offset  ->  fixable by post-processing (e.g. cloud-base bias)
  random jitter      ->  needs better segmentation
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.horizon_format import decode_horizon_uint8

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
MASKS_DIR = ROOT / "data" / "street_view" / "masks"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"

FOV_Y_DEG = 65.0


def fetch_horizon(vp):
    pf = pq.ParquetFile(DB_PATH)
    rg_sizes = np.array(
        [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    )
    cum = np.concatenate([[0], np.cumsum(rg_sizes)])
    rg = int(np.searchsorted(cum[1:], vp, side="right"))
    pos = vp - cum[rg]
    return decode_horizon_uint8(
        pf.read_row_group(rg, columns=["raw_horizon_deg"])
        .to_pandas()["raw_horizon_deg"]
        .iloc[pos]
    )


def expected_skyline_rows(horizon, heading, pitch_deg, H, W):
    """Predict skyline pixel row per column from DB horizon + camera geometry."""
    aspect = W / H
    hfov_deg = np.degrees(2 * np.arctan(np.tan(np.radians(FOV_Y_DEG) / 2) * aspect))
    fx = W / (2 * np.tan(np.radians(hfov_deg) / 2))
    fy = H / (2 * np.tan(np.radians(FOV_Y_DEG) / 2))
    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    # world azimuth for each column (camera frame azim + heading)
    az_cam = np.degrees(np.arctan2((cols - x_c) / fx, 1.0))
    world_az = (heading + az_cam) % 360
    # elevation from horizon (interpolate)
    hor_deg = np.interp(world_az, np.arange(360), horizon)
    # camera-frame elevation = world_elev - pitch
    cam_elev = hor_deg - pitch_deg
    row = y_c - fy * np.tan(np.radians(cam_elev))
    return row, world_az


first = next(
    pq.ParquetFile(DB_PATH).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    from src.query_profile import extract_elevation_profile

    all_offsets = []
    all_abs = []
    n_samples = 0
    n_ok = 0
    for sid, g in gt.items():
        mask_path = MASKS_DIR / f"{sid}.png"
        if not mask_path.exists():
            continue
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        H, W = mask.shape
        binary = (mask >= 128).astype(np.uint8)
        skyline_px = np.full(W, H - 1, dtype=np.float32)
        for c in range(W):
            rows = np.where(binary[:, c] == 1)[0]
            if len(rows) > 0:
                skyline_px[c] = rows[0]

        # Only include samples where the profile matches the horizon at the
        # expected offset (rotation-free).  This isolates mask quality.
        pr = extract_elevation_profile(
            str(mask_path),
            fov_y_deg=g["fov_y_deg"],
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=BIN_DEG,
        )
        if not pr["ok"]:
            continue
        prof = pr["profile"]
        sa = pr["start_az"]
        n = len(prof)
        exp = int(round((g["true_heading_deg"] + sa) % 360))
        hor = fetch_horizon(vp)
        w_exp = hor[np.arange(exp, exp + n) % 360]
        exp_corr = float(np.corrcoef(prof, w_exp)[0, 1])
        if exp_corr < 0.9:
            continue
        n_ok += 1

        hor = fetch_horizon(vp)
        pitch_deg = g["crop_pitch_deg"]
        pred_row, world_az = expected_skyline_rows(
            hor, g["true_heading_deg"], pitch_deg, H, W
        )

        err = skyline_px - pred_row
        # drop columns where prediction is out of image (unreliable)
        valid = (pred_row > 0) & (pred_row < H - 1)
        all_offsets.append(err[valid])
        all_abs.append(np.abs(err[valid]))
        n_samples += 1

    e = np.concatenate(all_offsets)
    a = np.concatenate(all_abs)
    print(f"Samples (exp_corr>=0.9): {n_samples}, columns: {len(e)}")
    print(
        f"Boundary error (mask - db_pred), pixels (H={720}, 65deg VFOV => ~{720 / 65:.1f} px/deg):"
    )
    print(f"  mean={np.mean(e):+.1f} median={np.median(e):+.1f} std={np.std(e):.1f}")
    print(
        f"  p10={np.percentile(e, 10):+.1f} p25={np.percentile(e, 25):+.1f} p75={np.percentile(e, 75):+.1f} p90={np.percentile(e, 90):+.1f}"
    )
    print(
        f"  |err| median={np.median(a):.1f} mean={np.mean(a):.1f} p90={np.percentile(a, 90):.1f}"
    )
    print(
        f"  |err|<6px (1 deg): {100 * np.mean(a < 6):.0f}%   <18px (3 deg): {100 * np.mean(a < 18):.0f}%"
    )
    print(
        f"  |err|<36px (6 deg): {100 * np.mean(a < 36):.0f}%   <72px (12 deg): {100 * np.mean(a < 72):.0f}%"
    )


if __name__ == "__main__":
    main()

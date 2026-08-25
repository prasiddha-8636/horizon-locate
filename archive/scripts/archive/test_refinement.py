#!/usr/bin/env python
"""Test improved skyline boundary post-processing.

Measures mask-boundary error vs DB-predicted rows for the 'OK' samples using
an improved boundary: heavy smoothing + wide gradient snapping + bias fix.
Compares against the raw U-Net boundary.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
MASKS_DIR = ROOT / "data" / "street_view" / "masks"
IMAGES_DIR = ROOT / "data" / "street_view" / "images"
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
    aspect = W / H
    hfov_deg = np.degrees(2 * np.arctan(np.tan(np.radians(FOV_Y_DEG) / 2) * aspect))
    fx = W / (2 * np.tan(np.radians(hfov_deg) / 2))
    fy = H / (2 * np.tan(np.radians(FOV_Y_DEG) / 2))
    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    az_cam = np.degrees(np.arctan2((cols - x_c) / fx, 1.0))
    world_az = (heading + az_cam) % 360
    hor_deg = np.interp(world_az, np.arange(360), horizon)
    cam_elev = hor_deg - pitch_deg
    row = y_c - fy * np.tan(np.radians(cam_elev))
    return row


def raw_boundary(mask):
    H, W = mask.shape
    binary = (mask >= 128).astype(np.uint8)
    b = np.full(W, H - 1, dtype=np.float32)
    for c in range(W):
        rows = np.where(binary[:, c] == 1)[0]
        if len(rows) > 0:
            b[c] = rows[0]
    return b


def improved_boundary(mask, img, window_px=60, snap_cols=45):
    """Smooth + gradient-snap boundary."""
    H, W = mask.shape
    b = raw_boundary(mask)
    # 1. heavy median filtering along columns (ridge is smooth at ~1deg = ~11px)
    b_smooth = ndimage.median_filter(b, size=45)
    # 2. low-pass to kill remaining jitter
    b_smooth = ndimage.gaussian_filter1d(b_smooth, sigma=6.0)
    # 3. gradient snap: sharpest downward brightness drop within window_px
    gray = np.asarray(Image.open(img).convert("L"), dtype=np.float32)
    gray = ndimage.gaussian_filter(gray, 2.0)
    vgrad = np.diff(gray, axis=0)  # + = brighter below
    out = b_smooth.copy()
    for c in range(0, W, 2):
        lo = max(1, int(b_smooth[c]) - window_px)
        hi = min(H - 2, int(b_smooth[c]) + window_px)
        seg = vgrad[lo:hi, c]
        if len(seg) == 0:
            continue
        # sharpest downward brightness transition = min gradient
        # weight by distance from smoothed boundary (ridge close to it)
        rows = np.arange(lo, hi)
        dist = np.abs(rows - b_smooth[c])
        score = seg - 0.15 * dist / window_px  # encourage nearby, strong drop
        best = int(np.argmin(score))
        out[c] = rows[best]
    # snap remaining odd columns by interpolation
    for c in range(1, W, 2):
        out[c] = 0.5 * (out[c - 1] + out[min(c + 1, W - 1)])
    # 4. final smoothing
    out = ndimage.median_filter(out, size=15)
    return out


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    raw_err = []
    imp_err = []
    n_samples = 0
    for sid, g in gt.items():
        mask_path = MASKS_DIR / f"{sid}.png"
        img_path = IMAGES_DIR / f"{sid}.png"
        if not mask_path.exists() or not img_path.exists():
            continue
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        mask = np.array(Image.open(mask_path).convert("L"))

        pr = extract_elevation_profile(
            str(mask_path),
            fov_y_deg=g["fov_y_deg"],
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=1.0,
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
        n_samples += 1

        H, W = mask.shape
        pred_row = expected_skyline_rows(
            hor, g["true_heading_deg"], g["crop_pitch_deg"], H, W
        )
        valid = (pred_row > 0) & (pred_row < H - 1)

        rb = raw_boundary(mask)
        ib = improved_boundary(mask, str(img_path))
        raw_err.append((rb - pred_row)[valid])
        imp_err.append((ib - pred_row)[valid])
        if n_samples <= 3:
            print(
                f"{sid[:24]} raw_med={np.median((rb - pred_row)[valid]):+.0f} imp_med={np.median((ib - pred_row)[valid]):+.0f}"
            )

    re = np.concatenate(raw_err)
    ie = np.concatenate(imp_err)
    print(f"\nSamples: {n_samples}")
    print(
        f"RAW : median={np.median(re):+.1f} std={np.std(re):.1f} |err|med={np.median(np.abs(re)):.1f} <72px:{100 * np.mean(np.abs(re) < 72):.0f}%"
    )
    print(
        f"IMPR: median={np.median(ie):+.1f} std={np.std(ie):.1f} |err|med={np.median(np.abs(ie)):.1f} <72px:{100 * np.mean(np.abs(ie) < 72):.0f}%"
    )


if __name__ == "__main__":
    main()

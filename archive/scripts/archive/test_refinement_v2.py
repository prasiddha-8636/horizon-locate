#!/usr/bin/env python
"""Smart boundary refinement v2: reliability-weighted gradient snap + interpolation.

Measures FB correlation at the true VP before/after to quantify improvement.
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
from src.matching import feature_bundle_matrix, ncc_scores
from src.evaluation import load_db_metadata
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


def raw_boundary(mask):
    H, W = mask.shape
    binary = (mask >= 128).astype(np.uint8)
    b = np.full(W, H - 1, dtype=np.float32)
    for c in range(W):
        rows = np.where(binary[:, c] == 1)[0]
        if len(rows) > 0:
            b[c] = rows[0]
    return b


def refine_boundary_v2(mask, img, window_px=60):
    """Snap boundary to sharpest brightness drop; interpolate unreliable cols."""
    H, W = mask.shape
    b = raw_boundary(mask)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    gray = ndimage.gaussian_filter(gray, 2.0)
    vgrad = np.diff(gray, axis=0)  # + = brighter below

    out = np.full(W, np.nan, dtype=np.float32)
    edge_strength = np.zeros(W)
    for c in range(W):
        lo = max(1, int(b[c]) - window_px)
        hi = min(H - 2, int(b[c]) + window_px)
        seg = vgrad[lo:hi, c]
        if len(seg) == 0:
            continue
        # sharpest downward transition (most negative gradient)
        rows = np.arange(lo, hi)
        best = int(np.argmin(seg))
        out[c] = rows[best]
        edge_strength[c] = -seg[best]

    # reliability: edge_strength above the 40th percentile within the row's
    # local context; weak edges = diffuse cloud base -> interpolate
    thresh = (
        np.percentile(edge_strength[edge_strength > 0], 40)
        if np.any(edge_strength > 0)
        else 0
    )
    reliable = (edge_strength >= max(thresh, 1e-6)) & np.isfinite(out)
    # require spatial coherence: a reliable col must have reliable neighbours
    rel = reliable.astype(int)
    rel = ndimage.binary_dilation(rel, structure=np.ones(5)) & reliable

    # interpolate unreliable columns from the smooth ridge
    cols = np.arange(W)
    smooth = ndimage.gaussian_filter1d(np.where(np.isfinite(out), out, b), sigma=8.0)
    final = np.where(rel, out, smooth)
    final = ndimage.median_filter(final, size=11)
    return final, reliable


def profile_from_boundary(b, H, W, fov_y, r_tilt, bin_deg=1.0):
    """Replicate query_profile's pixel->elevation mapping for a boundary array."""
    aspect = W / H
    hfov_deg = np.degrees(2 * np.arctan(np.tan(np.radians(fov_y) / 2) * aspect))
    fx = W / (2 * np.tan(np.radians(hfov_deg) / 2))
    fy = H / (2 * np.tan(np.radians(fov_y) / 2))
    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    rays = np.vstack([(cols - x_c) / fx, (y_c - b) / fy, -np.ones(W)])
    rays /= np.linalg.norm(rays, axis=0)
    azim_cam = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))
    if r_tilt is not None:
        rays = np.asarray(r_tilt) @ rays
    elev = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))
    order = np.argsort(azim_cam)
    azim_cam, elev = azim_cam[order], elev[order]
    sa = np.ceil(azim_cam[0] / bin_deg) * bin_deg
    ea = np.floor(azim_cam[-1] / bin_deg) * bin_deg
    grid = np.arange(sa, ea + 1e-6, bin_deg)
    return np.interp(grid, azim_cam, elev), sa


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    # subsample for speed: every 20th sample
    sids = list(gt.keys())[::20][:40]
    rows = []
    for sid in sids:
        mask_path = MASKS_DIR / f"{sid}.png"
        img_path = IMAGES_DIR / f"{sid}.png"
        if not mask_path.exists() or not img_path.exists():
            continue
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        H, W = mask.shape
        hor = fetch_horizon(vp)
        r_tilt = np.array(g["cam_R_tilt"])
        fov = g["fov_y_deg"]

        # original profile
        pr = extract_elevation_profile(
            str(mask_path), fov_y_deg=fov, r_tilt=r_tilt, bin_deg=1.0
        )
        if not pr["ok"]:
            continue
        prof_orig = pr["profile"]
        sa = pr["start_az"]
        n = len(prof_orig)
        exp = int(round((g["true_heading_deg"] + sa) % 360))
        w = hor[np.arange(exp, exp + n) % 360]

        # refined profile
        img = Image.open(img_path)
        rb = raw_boundary(mask)
        fb2, rel = refine_boundary_v2(mask, img)
        prof_new, sa_new = profile_from_boundary(fb2, H, W, fov, r_tilt, 1.0)
        # align sa (should match)
        n2 = len(prof_new)
        exp2 = int(round((g["true_heading_deg"] + sa_new) % 360))
        w2 = hor[np.arange(exp2, exp2 + n2) % 360]

        # FB correlation at true VP: per-feature corr, averaged
        from src.matching import _feature_bundle

        def fb_score(p, h):
            pv, pd = _feature_bundle(p)
            hv, hd = _feature_bundle(h)
            c0 = (
                float(np.corrcoef(pv, hv)[0, 1])
                if np.std(pv) > 0 and np.std(hv) > 0
                else 0.0
            )
            c1 = (
                float(np.corrcoef(pd, hd)[0, 1])
                if np.std(pd) > 0 and np.std(hd) > 0
                else 0.0
            )
            return 0.5 * (c0 + c1)

        fb_orig_score = fb_score(prof_orig, w)
        fb_new_score = fb_score(prof_new, w2)

        # also raw corr at expected offset
        rc_orig = float(np.corrcoef(prof_orig, w)[0, 1])
        rc_new = float(np.corrcoef(prof_new, w2)[0, 1])

        rows.append(
            (sid, rc_orig, rc_new, fb_orig_score, fb_new_score, float(rel.mean()))
        )
        if len(rows) <= 5:
            print(
                f"{sid[:24]} rc {rc_orig:+.3f}->{rc_new:+.3f}  fb {fb_orig_score:+.3f}->{fb_new_score:+.3f}  reliable={rel.mean():.0%}"
            )

    v = np.array(rows)
    if len(v):
        print(f"\nn={len(v)}")
        print(
            f"raw corr  : {np.median(v[:, 1]):+.3f} -> {np.median(v[:, 2]):+.3f}  (mean {np.mean(v[:, 2] - v[:, 1]):+.3f})"
        )
        print(
            f"fb score  : {np.median(v[:, 3]):+.3f} -> {np.median(v[:, 4]):+.3f}  (mean {np.mean(v[:, 4] - v[:, 3]):+.3f})"
        )
        print(f"reliable cols: median {np.median(v[:, 5]):.0%}")


if __name__ == "__main__":
    main()

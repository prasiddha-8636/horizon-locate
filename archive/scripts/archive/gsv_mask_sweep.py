#!/usr/bin/env python
"""Offline threshold sweep on GSV using saved-in-memory prob maps.

Runs the U-Net ONCE per image (returns full-res prob map), then sweeps the P(sky)
threshold, rebuilding masks + profiles + true-VP FB per threshold — no disk artifacts.

Mask built as: sky where prob >= t, top-connected component kept, convention
sky=0/black, terrain=255/white (matches src/segmentation.py). No Canny refine here.

Usage:
  python scripts/gsv_mask_sweep.py --limit 150
  python scripts/gsv_mask_sweep.py                # all 1808
"""

import argparse, json, os, sys, time
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.segmentation import load_segmentation_model, segment_image
from src.query_profile import extract_elevation_profile
from scripts.gsv_eval import fetch_horizon, fb_at_best

GT_PATH = ROOT / "data/street_view/ground_truth.json"
IMAGES_DIR = ROOT / "data/street_view/images"
MODEL_PATH = ROOT / "data/sky_segmentation_unet_model.pth"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def mask_from_prob(prob, t):
    sky = (prob >= t).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = np.zeros_like(sky, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_TOP] == 0 and stats[i, cv2.CC_STAT_AREA] > 100:
            keep[labels == i] = True
    return np.where(keep, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    args = ap.parse_args()

    with open(GT_PATH) as f:
        gt = json.load(f)
    sids = list(gt.keys())
    if args.limit:
        sids = sids[: args.limit]
    print(f"Samples: {len(sids)}  thresholds: {THRESHOLDS}", flush=True)

    print("Loading model (CPU).", flush=True)
    model = load_segmentation_model(str(MODEL_PATH), "cpu")

    # fb[sid_idx][t_idx]
    fb = np.full((len(sids), len(THRESHOLDS)), np.nan)
    prof_fail = 0
    t0 = time.time()
    for i, sid in enumerate(sids):
        g = gt[sid]
        img_path = IMAGES_DIR / f"{sid}.png"
        if not img_path.exists():
            prof_fail += 1
            continue
        seg = segment_image(
            model, str(img_path), None, "cpu", tta=True, return_prob=True
        )
        prob = seg["diagnostics"].get("prob_map")
        if prob is None:
            prof_fail += 1
            continue
        fov = g.get("fov_y_deg", 65.0)
        r_tilt = np.array(g["cam_R_tilt"]) if g.get("cam_R_tilt") else None
        vp = int(g.get("closest_viewpoint_id", -1))
        if vp < 0:
            prof_fail += 1
            continue
        horizon = fetch_horizon(vp)
        for j, t in enumerate(THRESHOLDS):
            mask = mask_from_prob(prob, t)
            pr = extract_elevation_profile(
                mask, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=1.0
            )
            if not pr["ok"]:
                continue
            fb[i, j] = fb_at_best(pr["profile"], horizon)[0]
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(sids)} done, {el / (i + 1):.1f}s/sample", flush=True)

    print(f"\n=== THRESHOLD SWEEP (true-VP FB_best) ===", flush=True)
    medians = []
    for j, t in enumerate(THRESHOLDS):
        col = fb[:, j]
        col = col[~np.isnan(col)]
        medians.append(np.median(col))
        print(
            f"  t={t:.2f}: median FB={np.median(col):.3f}  "
            f"p25={np.percentile(col, 25):.3f}  p75={np.percentile(col, 75):.3f}  "
            f"n={len(col)}",
            flush=True,
        )
    best = int(np.argmax(medians))
    print(
        f"\nBest threshold: {THRESHOLDS[best]:.2f} (median FB {medians[best]:.3f})",
        flush=True,
    )
    print(
        f"Baseline threshold 0.50 median FB: {medians[THRESHOLDS.index(0.50)]:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

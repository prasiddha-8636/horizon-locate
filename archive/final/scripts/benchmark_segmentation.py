"""Benchmark sky-boundary refinement methods against hand-annotated skylines.

Methods compared on the 18 annotated GSV samples:
  baseline_unet  — raw U-Net mask boundary
  refine_unet    — U-Net + refine_sky_mask_with_guidance (LAB b* snap, top-connected, interp)
  canny_direct   — pure Canny edge skyline
  lab_b          — LAB b* thresholding

Metric: mean/median absolute boundary error (px) vs annotated skyline,
        counted only over columns where both prediction and GT exist.
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/admin/SkylineGeolocation/src")

from segmentation import (
    refine_sky_mask_with_guidance,
    segment_image,
)

DATA_DIR = "/home/admin/SkylineGeolocation/data/street_view"
MODEL_PATH = "/home/admin/SkylineGeolocation/data/sky_segmentation_unet_model.pth"
N_SAMPLES = 18


def load_model():
    from segmentation import load_segmentation_model

    return load_segmentation_model(MODEL_PATH, "cpu")


def run_unet_mask(model, img):
    """Run U-Net via segment_image (writes to temp file), return raw mask.

    Returns sky=0, terrain=255 (segment_image convention).
    Even when result['ok']=False (LOW_CONFIDENCE), the saved mask is still usable.
    """
    tmp_in = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(img).save(tmp_in.name)
    tmp_in.close()
    tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_out.close()
    try:
        segment_image(
            model,
            tmp_in.name,
            tmp_out.name,
            "cpu",
            min_sky_ratio=0.0,
            max_sky_ratio=1.0,
            min_boundary_coverage=0.0,
        )
        mask = np.array(Image.open(tmp_out.name))
        return mask
    finally:
        os.unlink(tmp_in.name)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)


def run_canny_direct(img):
    """Pure Canny edge skyline. Returns sky=0, terrain=255."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    H, W = edges.shape
    sky_mask = np.full((H, W), 255, dtype=np.uint8)  # default = terrain
    for col in range(W):
        col_edges = np.where(edges[:, col] > 0)[0]
        if len(col_edges) > 0:
            b = int(col_edges[0])
            sky_mask[:b, col] = 0  # sky=0
    return sky_mask


def run_lab_b(img):
    """LAB b* thresholding. Returns sky=0, terrain=255."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    sky_255 = (b_channel < 120).astype(np.uint8) * 255  # sky=255 intermediate
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        sky_255, connectivity=8
    )
    top_sky_255 = np.zeros_like(sky_255)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_TOP] == 0 and stats[i, cv2.CC_STAT_AREA] > 100:
            top_sky_255[labels == i] = 255
    return np.where(top_sky_255 == 255, 0, 255).astype(np.uint8)


def compute_skyline_from_mask(mask):
    """Sky convention: zero pixels are sky. Boundary row per column.

    Boundary = FIRST sky->terrain transition (first nonzero below top sky),
    not last sky row (which fails when a second terrain band sits at bottom).
    """
    H, W = mask.shape
    boundaries = np.full(W, -1, dtype=np.int32)
    col_nonzero = mask != 0
    for col in range(W):
        nz_rows = np.where(col_nonzero[:, col])[0]
        if len(nz_rows) == 0:
            continue  # all sky -> no terrain boundary found
        boundaries[col] = int(nz_rows[0]) - 1 if nz_rows[0] > 0 else -1
    # For columns where the very first row is terrain, boundary stays -1;
    # fall back to last-sky-row only when no nonzero exists above.
    return boundaries


def compute_boundary_error(pred_rows, gt_rows):
    valid = (pred_rows >= 0) & (gt_rows >= 0)
    n = int(valid.sum())
    if n == 0:
        return float("inf"), 0
    err = float(
        np.mean(np.abs(pred_rows[valid].astype(float) - gt_rows[valid].astype(float)))
    )
    return err, n


def main():
    with open(os.path.join(DATA_DIR, "annotations.json")) as f:
        annotations = json.load(f)["annotations"]

    model = load_model()
    sample_ids = sorted(annotations.keys())[:N_SAMPLES]

    methods = ["baseline_unet", "refine_unet", "canny_direct", "lab_b"]
    results = {m: [] for m in methods}

    for sid in sample_ids:
        img_path = os.path.join(DATA_DIR, "images", f"{sid}.jpg")
        if not os.path.exists(img_path):
            alt = [p for p in (f"{sid}.png", f"{sid}.jpeg")]
            found = False
            for a in alt:
                p = os.path.join(DATA_DIR, "images", a)
                if os.path.exists(p):
                    img_path = p
                    found = True
                    break
            if not found:
                print(f"SKIP {sid}: no image")
                continue
        img = np.array(Image.open(img_path).convert("RGB"))

        # GT boundary array at image width
        W = img.shape[1]
        gt_boundary = np.full(W, -1, dtype=np.int32)
        for c, r in annotations[sid]:
            ci, ri = int(c), int(r)
            if 0 <= ci < W and 0 <= ri < img.shape[0]:
                gt_boundary[ci] = ri

        raw_mask = run_unet_mask(model, img)

        masks = {}
        masks["baseline_unet"] = (
            raw_mask
            if raw_mask is not None
            else np.zeros(img.shape[:2], dtype=np.uint8)
        )
        masks["refine_unet"] = (
            refine_sky_mask_with_guidance(img, masks["baseline_unet"])
            if raw_mask is not None
            else np.zeros(img.shape[:2], dtype=np.uint8)
        )
        masks["canny_direct"] = run_canny_direct(img)
        masks["lab_b"] = run_lab_b(img)

        row = {"sample_id": sid}
        for name, m in masks.items():
            pred = compute_skyline_from_mask(m)
            err, nv = compute_boundary_error(pred, gt_boundary)
            results[name].append(
                {"sample_id": sid, "mean_error_px": err, "valid_cols": nv}
            )
            row[name] = f"{err:.1f}" if err != float("inf") else "inf"
        print(f"{sid[:12]}: " + "  ".join(f"{n}={row[n]}" for n in methods))

    print(f"\nBenchmark on {len(sample_ids)} annotated samples:\n")
    print(
        f"{'Method':<16} {'Mean Err (px)':<14} {'Median (px)':<13} {'Mean Valid Cols':<15}"
    )
    print("-" * 60)
    summary = {}
    for name in methods:
        errs = [
            r["mean_error_px"]
            for r in results[name]
            if r["mean_error_px"] != float("inf")
        ]
        cols = [r["valid_cols"] for r in results[name]]
        mean_err = float(np.mean(errs)) if errs else float("inf")
        med_err = float(np.median(errs)) if errs else float("inf")
        mean_cols = float(np.mean(cols)) if cols else 0.0
        summary[name] = {
            "mean_error_px": mean_err,
            "median_error_px": med_err,
            "mean_valid_cols": mean_cols,
            "finite_samples": len(errs),
        }
        print(f"{name:<16} {mean_err:<14.2f} {med_err:<13.2f} {mean_cols:<15.1f}")

    out_path = "/home/admin/SkylineGeolocation/results/segmentation_benchmark.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

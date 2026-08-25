"""Comprehensive benchmark for sky-boundary refinement methods and inference configurations.

Evaluated Post-Processing Methods:
  1. baseline_raw_unet: Raw MobileNetV3 U-Net mask without refinement
  2. canny_direct: Direct Canny edge boundary
  3. lab_b_threshold: CIE LAB b* threshold (<120)
  4. grayscale_fixed_window: Grayscale vertical gradient with fixed +/-25 row search
  5. lab_b_subpixel_snap: LAB b* channel parabolic sub-pixel fitting + morphological ridge coupling
  6. multichannel_gradient_fusion: Weighted fusion (LAB b*, HSV Saturation, Grayscale gradients)
  7. dynamic_programming_skyline: Viterbi shortest-path cost-grid line extraction

Evaluated Inference Configurations:
  - Input Resolutions: 256x256, 384x384
  - Test-Time Augmentation (TTA): False, True
  - Cutoff Thresholds: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
"""

import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/admin/SkylineGeolocation/src")

from segmentation import (
    load_segmentation_model,
    refine_dynamic_programming_skyline,
    refine_grayscale_fixed_window,
    refine_multichannel_gradient_fusion,
    refine_sky_mask_with_guidance,
    segment_image,
)

DATA_DIR = "/home/admin/SkylineGeolocation/data/street_view"
MODEL_PATH = "/home/admin/SkylineGeolocation/data/sky_segmentation_unet_model.pth"
OUTPUT_JSON = "/home/admin/SkylineGeolocation/results/segmentation_benchmark.json"
N_SAMPLES = 18


def run_unet_mask(model, img, input_size=256, tta=True, threshold=0.5, refinement_method="none"):
    """Run segment_image via temporary files and return boundary mask (sky=0, terrain=255)."""
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
            tta=tta,
            threshold=threshold,
            input_size=input_size,
            refinement_method=refinement_method,
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
    sky_mask = np.full((H, W), 255, dtype=np.uint8)
    for col in range(W):
        col_edges = np.where(edges[:, col] > 0)[0]
        if len(col_edges) > 0:
            b = int(col_edges[0])
            sky_mask[:b, col] = 0
    return sky_mask


def run_lab_b_threshold(img):
    """Pure CIE LAB b* channel thresholding. Returns sky=0, terrain=255."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    sky_255 = (b_channel < 120).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        sky_255, connectivity=8
    )
    top_sky_255 = np.zeros_like(sky_255)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_TOP] == 0 and stats[i, cv2.CC_STAT_AREA] > 100:
            top_sky_255[labels == i] = 255
    return np.where(top_sky_255 == 255, 0, 255).astype(np.uint8)


def compute_skyline_from_mask(mask):
    """Extract boundary row index per column. Sky convention: 0 is sky, 255 is terrain."""
    H, W = mask.shape
    boundaries = np.full(W, -1, dtype=np.int32)
    col_nonzero = mask != 0
    for col in range(W):
        nz_rows = np.where(col_nonzero[:, col])[0]
        if len(nz_rows) == 0:
            continue
        boundaries[col] = int(nz_rows[0]) - 1 if nz_rows[0] > 0 else -1
    return boundaries


def compute_boundary_error(pred_rows, gt_rows):
    """Compute mean absolute row error over valid overlapping columns."""
    valid = (pred_rows >= 0) & (gt_rows >= 0)
    n = int(valid.sum())
    if n == 0:
        return float("inf"), 0
    err = float(
        np.mean(np.abs(pred_rows[valid].astype(float) - gt_rows[valid].astype(float)))
    )
    return err, n


def benchmark_post_processing_algorithms(model, sample_ids, annotations):
    """Benchmark all post-processing algorithms on standard inference settings."""
    print("=" * 80)
    print("PART 1: Post-Processing Boundary Extraction Algorithms")
    print("=" * 80)

    algorithms = [
        "baseline_raw_unet",
        "canny_direct",
        "lab_b_threshold",
        "grayscale_fixed_window",
        "grayscale_fixed_clahe",
        "lab_b_subpixel_snap",
        "lab_b_subpixel_clahe",
        "multichannel_gradient_fusion",
        "multichannel_clahe",
        "dynamic_programming_skyline",
    ]

    results = {alg: [] for alg in algorithms}

    for sid in sample_ids:
        img_path = os.path.join(DATA_DIR, "images", f"{sid}.jpg")
        if not os.path.exists(img_path):
            for ext in [".png", ".jpeg"]:
                p = os.path.join(DATA_DIR, "images", f"{sid}{ext}")
                if os.path.exists(p):
                    img_path = p
                    break
        if not os.path.exists(img_path):
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        W = img.shape[1]

        gt_boundary = np.full(W, -1, dtype=np.int32)
        for c, r in annotations[sid]:
            ci, ri = int(c), int(r)
            if 0 <= ci < W and 0 <= ri < img.shape[0]:
                gt_boundary[ci] = ri

        raw_unet_mask = run_unet_mask(
            model, img, input_size=256, tta=True, threshold=0.5, refinement_method="none"
        )

        masks = {}
        masks["baseline_raw_unet"] = raw_unet_mask
        masks["canny_direct"] = run_canny_direct(img)
        masks["lab_b_threshold"] = run_lab_b_threshold(img)
        masks["grayscale_fixed_window"] = refine_grayscale_fixed_window(img, raw_unet_mask, use_clahe=False)
        masks["grayscale_fixed_clahe"] = refine_grayscale_fixed_window(img, raw_unet_mask, use_clahe=True)
        masks["lab_b_subpixel_snap"] = refine_sky_mask_with_guidance(img, raw_unet_mask, use_clahe=False)
        masks["lab_b_subpixel_clahe"] = refine_sky_mask_with_guidance(img, raw_unet_mask, use_clahe=True)
        masks["multichannel_gradient_fusion"] = refine_multichannel_gradient_fusion(img, raw_unet_mask, use_clahe=False)
        masks["multichannel_clahe"] = refine_multichannel_gradient_fusion(img, raw_unet_mask, use_clahe=True)
        masks["dynamic_programming_skyline"] = refine_dynamic_programming_skyline(img, raw_unet_mask)

        for alg, m in masks.items():
            pred = compute_skyline_from_mask(m)
            err, nv = compute_boundary_error(pred, gt_boundary)
            results[alg].append({"sample_id": sid, "mean_error_px": err, "valid_cols": nv})

    summary = {}
    print(f"\n{'Algorithm':<32} {'Mean Err (px)':<14} {'Median (px)':<13} {'Mean Valid Cols':<15}")
    print("-" * 75)
    for alg in algorithms:
        errs = [r["mean_error_px"] for r in results[alg] if r["mean_error_px"] != float("inf")]
        cols = [r["valid_cols"] for r in results[alg]]
        mean_err = float(np.mean(errs)) if errs else float("inf")
        med_err = float(np.median(errs)) if errs else float("inf")
        mean_cols = float(np.mean(cols)) if cols else 0.0
        summary[alg] = {
            "mean_error_px": round(mean_err, 2),
            "median_error_px": round(med_err, 2),
            "mean_valid_cols": round(mean_cols, 1),
            "finite_samples": len(errs),
        }
        print(f"{alg:<32} {mean_err:<14.2f} {med_err:<13.2f} {mean_cols:<15.1f}")

    return summary


def benchmark_inference_configurations(model, sample_ids, annotations):
    """Benchmark combinations of Input Resolution, TTA, and Cutoff Thresholds."""
    print("\n" + "=" * 80)
    print("PART 2: Inference Configuration Parameters Sweep")
    print("=" * 80)

    input_sizes = [256, 384]
    tta_options = [False, True]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    config_results = []

    for size in input_sizes:
        for tta in tta_options:
            for thresh in thresholds:
                config_key = f"input_size={size}_tta={tta}_threshold={thresh}"
                errors = []
                valid_counts = []

                t0 = time.time()
                for sid in sample_ids:
                    img_path = os.path.join(DATA_DIR, "images", f"{sid}.jpg")
                    if not os.path.exists(img_path):
                        for ext in [".png", ".jpeg"]:
                            p = os.path.join(DATA_DIR, "images", f"{sid}{ext}")
                            if os.path.exists(p):
                                img_path = p
                                break
                    if not os.path.exists(img_path):
                        continue

                    img = np.array(Image.open(img_path).convert("RGB"))
                    W = img.shape[1]

                    gt_boundary = np.full(W, -1, dtype=np.int32)
                    for c, r in annotations[sid]:
                        ci, ri = int(c), int(r)
                        if 0 <= ci < W and 0 <= ri < img.shape[0]:
                            gt_boundary[ci] = ri

                    mask = run_unet_mask(
                        model,
                        img,
                        input_size=size,
                        tta=tta,
                        threshold=thresh,
                        refinement_method="lab_b_subpixel",
                    )
                    pred_boundary = compute_skyline_from_mask(mask)
                    err, nv = compute_boundary_error(pred_boundary, gt_boundary)

                    if err != float("inf"):
                        errors.append(err)
                        valid_counts.append(nv)

                elapsed = time.time() - t0
                mean_err = float(np.mean(errors)) if errors else float("inf")
                med_err = float(np.median(errors)) if errors else float("inf")
                mean_cols = float(np.mean(valid_counts)) if valid_counts else 0.0

                res = {
                    "config_key": config_key,
                    "input_size": size,
                    "tta": tta,
                    "threshold": thresh,
                    "mean_error_px": round(mean_err, 2),
                    "median_error_px": round(med_err, 2),
                    "mean_valid_cols": round(mean_cols, 1),
                    "elapsed_sec": round(elapsed, 2),
                }
                config_results.append(res)
                print(
                    f"Config: {config_key:<45} | "
                    f"Mean Err: {mean_err:6.2f} px | "
                    f"Median: {med_err:6.2f} px | "
                    f"Time: {elapsed:5.2f}s"
                )

    return config_results


def main():
    with open(os.path.join(DATA_DIR, "annotations.json")) as f:
        annotations = json.load(f)["annotations"]

    model = load_segmentation_model(MODEL_PATH, "cpu")
    sample_ids = sorted(annotations.keys())[:N_SAMPLES]

    alg_summary = benchmark_post_processing_algorithms(model, sample_ids, annotations)
    config_summary = benchmark_inference_configurations(model, sample_ids, annotations)

    full_report = {
        "post_processing_algorithms": alg_summary,
        "inference_configurations": config_summary,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved full benchmark report to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

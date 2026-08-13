#!/usr/bin/env python3
"""Generate many additional example figures for the report."""

import os, json, glob, numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

SRC = Path("/home/admin/SkylineGeolocation")
OUT = Path("/home/admin/MinorFinalReport/src/images")
OUT.mkdir(parents=True, exist_ok=True)

SYNTH_IMAGES = SRC / "archive/synthetic_dataset/images"
SYNTH_MASKS_GT = SRC / "archive/synthetic_dataset/masks"
SYNTH_MASKS_PRED = SRC / "archive/synthetic_dataset/predicted_masks"
GSV_IMAGES = SRC / "data/street_view/images"
GSV_MASKS = SRC / "data/street_view/masks"
GSV_CROPS = SRC / "data/street_view/gsv_crops"

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ═══════════════════════════════════════════════════════════════════════
# A. MANY GSV SAMPLES (real photos + masks)
# ═══════════════════════════════════════════════════════════════════════
def gsv_many_samples():
    """Generate multiple grids of GSV samples showing image + mask + overlay."""
    img_files = sorted(glob.glob(str(GSV_IMAGES / "*.png")))
    mask_files = sorted(glob.glob(str(GSV_MASKS / "*.png")))

    # Generate 3 grids of 6 samples each (18 total)
    batch_size = 6
    for batch_idx in range(3):
        start = batch_idx * 30  # spread across dataset
        indices = [start + i * 5 for i in range(batch_size)]
        indices = [i for i in indices if i < len(img_files) and i < len(mask_files)]

        fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3 * len(indices)))
        if len(indices) == 1:
            axes = axes.reshape(1, -1)

        for row, idx in enumerate(indices):
            img = cv2.cvtColor(cv2.imread(img_files[idx]), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            axes[row, 0].imshow(img)
            axes[row, 0].set_ylabel(f"#{idx}", fontsize=9, fontweight="bold")
            if row == 0: axes[row, 0].set_title("GSV Image", fontsize=10)

            axes[row, 1].imshow(mask_r, cmap="gray", vmin=0, vmax=255)
            if row == 0: axes[row, 1].set_title("Predicted Mask", fontsize=10)

            overlay = img.copy()
            sky = (mask_r < 128)
            overlay[sky] = (overlay[sky] * 0.4 + np.array([100, 150, 255]) * 0.6).astype(np.uint8)
            # Draw boundary
            for col in range(w):
                sky_rows = np.where(mask_r[:, col] < 128)[0]
                if len(sky_rows) > 0:
                    r = int(sky_rows[-1])
                    cv2.circle(overlay, (col, r), 1, (255, 50, 50), -1)
            axes[row, 2].imshow(overlay)
            if row == 0: axes[row, 2].set_title("Overlay + Boundary", fontsize=10)

            for ax in axes[row]:
                ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f"GSV Samples (Batch {batch_idx + 1})", fontsize=12, fontweight="bold")
        fig.tight_layout()
        save(fig, f"fig_gsv_batch_{batch_idx + 1}.png")


# ═══════════════════════════════════════════════════════════════════════
# B. SYNTHETIC SAMPLES WITH DIFFERENT WEATHER
# ═══════════════════════════════════════════════════════════════════════
def synth_weather_grid():
    """Show synthetic samples with different visual conditions."""
    img_files = sorted(glob.glob(str(SYNTH_IMAGES / "*.png")))
    mask_files = sorted(glob.glob(str(SYNTH_MASKS_GT / "*.png")))

    # Pick samples that look different
    indices = [0, 15, 30, 60, 100, 150, 200, 250]
    indices = [i for i in indices if i < len(img_files)]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.1)

    for pos, idx in enumerate(indices):
        row, col = pos // 4, pos % 4
        ax = fig.add_subplot(gs[row, col])
        img = cv2.cvtColor(cv2.imread(img_files[idx]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"Sample {idx}", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Synthetic Training Samples (Varied Weather/Lighting)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_synth_weather_grid.png")


# ═══════════════════════════════════════════════════════════════════════
# C. GT vs PREDICTED MASK COMPARISON (many samples)
# ═══════════════════════════════════════════════════════════════════════
def gt_vs_pred_many():
    """Side-by-side GT vs predicted for many samples."""
    gt_files = sorted(glob.glob(str(SYNTH_MASKS_GT / "*.png")))
    pred_files = sorted(glob.glob(str(SYNTH_MASKS_PRED / "*.png")))

    indices = list(range(0, min(20, len(gt_files)), 2))  # 10 samples

    fig, axes = plt.subplots(len(indices), 2, figsize=(6, 2.5 * len(indices)))

    for row, idx in enumerate(indices):
        gt = cv2.imread(gt_files[idx], cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_files[idx], cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape[:2]
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        axes[row, 0].imshow(gt, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_ylabel(f"#{idx}", fontsize=9)
        if row == 0: axes[row, 0].set_title("Ground Truth", fontsize=10)

        axes[row, 1].imshow(pred, cmap="gray", vmin=0, vmax=255)
        if row == 0: axes[row, 1].set_title("Predicted", fontsize=10)

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Ground Truth vs Predicted Masks (10 Synthetic Samples)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_gt_vs_pred_many.png")


# ═══════════════════════════════════════════════════════════════════════
# D. PROFILE EXTRACTION: MANY SAMPLES
# ═══════════════════════════════════════════════════════════════════════
def profile_many():
    """Show image → mask → profile for 4 different samples."""
    img_files = sorted(glob.glob(str(SYNTH_IMAGES / "*.png")))
    mask_files = sorted(glob.glob(str(SYNTH_MASKS_GT / "*.png")))

    indices = [20, 80, 160, 240]
    indices = [i for i in indices if i < len(img_files)]

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 3.2 * len(indices)))

    for row, idx in enumerate(indices):
        img = cv2.cvtColor(cv2.imread(img_files[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"#{idx}", fontsize=10, fontweight="bold")
        if row == 0: axes[row, 0].set_title("Image", fontsize=10)

        axes[row, 1].imshow(mask_r, cmap="gray", vmin=0, vmax=255)
        if row == 0: axes[row, 1].set_title("Mask", fontsize=10)

        # Boundary overlay
        overlay = img.copy()
        for col in range(w):
            sky_rows = np.where(mask_r[:, col] < 128)[0]
            if len(sky_rows) > 0:
                r = int(sky_rows[-1])
                cv2.circle(overlay, (col, r), 1, (255, 0, 0), -1)
        axes[row, 2].imshow(overlay)
        if row == 0: axes[row, 2].set_title("Skyline Boundary", fontsize=10)

        # Profile
        boundary = np.full(w, -1, dtype=np.float64)
        for col in range(w):
            sky_rows = np.where(mask_r[:, col] < 128)[0]
            if len(sky_rows) > 0:
                boundary[col] = sky_rows[-1]
        valid = boundary >= 0
        if np.any(valid):
            all_cols = np.arange(w, dtype=np.float64)
            b_interp = np.interp(all_cols, all_cols[valid], boundary[valid])
        else:
            b_interp = np.full(w, h // 2)

        fov_y = np.radians(65.0)
        focal_y = h / (2 * np.tan(fov_y / 2))
        elev = np.degrees(np.arctan((h / 2 - b_interp) / focal_y))

        axes[row, 3].plot(range(w), elev, "b-", linewidth=1.2)
        axes[row, 3].fill_between(range(w), elev, alpha=0.15)
        axes[row, 3].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        if row == 0: axes[row, 3].set_title("Elevation Profile", fontsize=10)
        axes[row, 3].set_xlabel("Column")
        axes[row, 3].set_ylabel("Elev (°)")
        axes[row, 3].tick_params(labelsize=8)

        for ax in axes[row, :3]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Profile Extraction (4 Samples)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_profile_many.png")


# ═══════════════════════════════════════════════════════════════════════
# E. GSV CROP EXAMPLES (individual crops from panoramas)
# ═══════════════════════════════════════════════════════════════════════
def gsv_crops_grid():
    """Show individual GSV crop images from different panoramas."""
    crop_files = sorted(glob.glob(str(GSV_CROPS / "*.png")))

    # Pick 8 diverse crops
    indices = [0, 50, 100, 200, 400, 600, 800, 1000]
    indices = [i for i in indices if i < len(crop_files)]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for pos, idx in enumerate(indices):
        row, col = pos // 4, pos % 4
        img = cv2.cvtColor(cv2.imread(crop_files[idx]), cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(img)
        name = os.path.basename(crop_files[idx]).split("_h")[0][:12]
        axes[row, col].set_title(f"{name}...", fontsize=8)
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    fig.suptitle("Individual GSV Crops (from Different Panoramas)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_gsv_crops_grid.png")


# ═══════════════════════════════════════════════════════════════════════
# F. ERROR DISTRIBUTION HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════
def error_histogram():
    """Histogram of errors from GSV evaluation."""
    with open(SRC / "archive/final/results/gsv_improve_eval_results.json") as f:
        data = json.load(f)
    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: All errors (log scale)
    errs = [r["err_rrf"] for r in results]
    errs_km = [e / 1000 for e in errs]
    ax1.hist(errs_km, bins=30, color="#3498db", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax1.set_xlabel("Error (km)")
    ax1.set_ylabel("Count")
    ax1.set_title("Error Distribution (All 68 Panos)", fontsize=11, fontweight="bold")
    ax1.axvline(x=np.median(errs_km), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(errs_km):.1f} km")
    ax1.legend(fontsize=9)

    # Right: Zoomed in (< 1km)
    close = [e for e in errs if e < 1000]
    ax2.hist(close, bins=15, color="#2ecc71", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.set_xlabel("Error (m)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Zoomed: {len(close)} Panos Within 1 km", fontsize=11, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig_error_histogram.png")


# ═══════════════════════════════════════════════════════════════════════
# G. FOV COVERAGE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
def fov_distribution():
    """Histogram of FOV coverage across GSV panos."""
    with open(SRC / "archive/final/results/gsv_improve_eval_results.json") as f:
        data = json.load(f)
    results = data["results"]

    fovs = [r.get("coverage_deg", 0) for r in results]
    errs = [r["err_rrf"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: FOV histogram
    ax1.hist(fovs, bins=15, color="#e67e22", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax1.axvline(x=200, color="red", linestyle="--", linewidth=1.5, label="200° threshold")
    ax1.set_xlabel("Fused FOV (degrees)")
    ax1.set_ylabel("Count")
    ax1.set_title("FOV Coverage Distribution", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)

    # Right: FOV vs Error scatter
    colors = ["#2ecc71" if e < 1000 else "#e74c3c" for e in errs]
    ax2.scatter(fovs, [e / 1000 for e in errs], c=colors, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax2.set_xlabel("Fused FOV (degrees)")
    ax2.set_ylabel("Error (km)")
    ax2.set_title("FOV vs Error", fontsize=11, fontweight="bold")
    ax2.axvline(x=200, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(y=1, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    fig.tight_layout()
    save(fig, "fig_fov_distribution.png")


# ═══════════════════════════════════════════════════════════════════════
# H. CLAHE BEFORE/AFTER
# ═══════════════════════════════════════════════════════════════════════
def clahe_comparison():
    """Show CLAHE effect on a foggy/hazy sample."""
    img_files = sorted(glob.glob(str(SYNTH_IMAGES / "*.png")))

    # Pick a hazy-looking sample
    idx = 120
    img = cv2.cvtColor(cv2.imread(img_files[idx]), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale", fontsize=11, fontweight="bold")
    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("CLAHE Enhanced", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("CLAHE Enhancement for Fog/Haze Removal", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_clahe_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# I. CANNY EDGE DETECTION STAGES
# ═══════════════════════════════════════════════════════════════════════
def canny_stages():
    """Show Canny edge detection at different scales."""
    img_files = sorted(glob.glob(str(SYNTH_IMAGES / "*.png")))
    idx = 80
    img = cv2.cvtColor(cv2.imread(img_files[idx]), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    fine_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    coarse_blur = cv2.GaussianBlur(enhanced, (7, 7), 0)
    edges_fine = cv2.Canny(fine_blur, 30, 150)
    edges_coarse = cv2.Canny(coarse_blur, 20, 100)
    fused = ((edges_fine > 0) | (edges_coarse > 0)).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(img)
    axes[0].set_title("Input", fontsize=10, fontweight="bold")
    axes[1].imshow(edges_fine, cmap="gray")
    axes[1].set_title("Fine Canny (30/150)", fontsize=10, fontweight="bold")
    axes[2].imshow(edges_coarse, cmap="gray")
    axes[2].set_title("Coarse Canny (20/100)", fontsize=10, fontweight="bold")
    axes[3].imshow(fused, cmap="gray")
    axes[3].set_title("Fused Multi-Scale", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Multi-Scale Canny Edge Detection", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_canny_stages.png")


# ═══════════════════════════════════════════════════════════════════════
# J. SUCCESS CASES DETAILED
# ═══════════════════════════════════════════════════════════════════════
def success_detailed():
    """Show the 3 best GSV results with their images."""
    with open(SRC / "archive/final/results/gsv_improve_eval_results.json") as f:
        data = json.load(f)
    results = sorted(data["results"], key=lambda r: r["err_rrf"])

    best3 = results[:3]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, r in enumerate(best3):
        pano_id = r["pano_id"]
        # Find the crop image
        crop_pattern = str(GSV_CROPS / f"{pano_id}*.png")
        crops = sorted(glob.glob(crop_pattern))
        if crops:
            img = cv2.cvtColor(cv2.imread(crops[0]), cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
        else:
            axes[i].text(0.5, 0.5, f"Image not found\n{pano_id[:15]}...",
                        ha="center", va="center", fontsize=9)
        axes[i].set_title(f"Error: {r['err_rrf']:.0f}m\nFOV: {r.get('coverage_deg', 0):.0f}°",
                         fontsize=10, fontweight="bold", color="green")
        axes[i].set_xticks([]); axes[i].set_yticks([])

    fig.suptitle("Top 3 Best Localizations (GSV)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_success_detailed.png")


# ═══════════════════════════════════════════════════════════════════════
# K. FAILURE CASES DETAILED
# ═══════════════════════════════════════════════════════════════════════
def failure_detailed():
    """Show 3 worst GSV results."""
    with open(SRC / "archive/final/results/gsv_improve_eval_results.json") as f:
        data = json.load(f)
    results = sorted(data["results"], key=lambda r: r["err_rrf"], reverse=True)

    worst3 = results[:3]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, r in enumerate(worst3):
        pano_id = r["pano_id"]
        crop_pattern = str(GSV_CROPS / f"{pano_id}*.png")
        crops = sorted(glob.glob(crop_pattern))
        if crops:
            img = cv2.cvtColor(cv2.imread(crops[0]), cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
        else:
            axes[i].text(0.5, 0.5, f"Image not found\n{pano_id[:15]}...",
                        ha="center", va="center", fontsize=9)
        fov = r.get("coverage_deg", 0)
        axes[i].set_title(f"Error: {r['err_rrf']/1000:.1f}km\nFOV: {fov:.0f}°",
                         fontsize=10, fontweight="bold", color="red")
        axes[i].set_xticks([]); axes[i].set_yticks([])

    fig.suptitle("3 Worst Failures (GSV)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_failure_detailed.png")


# ═══════════════════════════════════════════════════════════════════════
# L. TRAINING DATA: IMAGE + MASK PAIRS
# ═══════════════════════════════════════════════════════════════════════
def training_pairs():
    """Show training image-mask pairs."""
    img_files = sorted(glob.glob(str(SYNTH_IMAGES / "*.png")))
    mask_files = sorted(glob.glob(str(SYNTH_MASKS_GT / "*.png")))

    indices = [0, 10, 25, 50, 75, 100, 150, 200]
    indices = [i for i in indices if i < len(img_files)]

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))

    for pos, idx in enumerate(indices[:8]):
        row = pos // 2
        col_base = (pos % 2) * 2
        img = cv2.cvtColor(cv2.imread(img_files[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_files[idx], cv2.IMREAD_GRAYSCALE)

        axes[row, col_base].imshow(img)
        axes[row, col_base].set_title(f"Image #{idx}", fontsize=9)
        axes[row, col_base + 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row, col_base + 1].set_title(f"Mask #{idx}", fontsize=9)

        for c in [col_base, col_base + 1]:
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])

    fig.suptitle("Training Data: Image-Mask Pairs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_training_pairs.png")


# ═══════════════════════════════════════════════════════════════════════
# M. BANDPASS FILTER EFFECT
# ═══════════════════════════════════════════════════════════════════════
def bandpass_effect():
    """Show how bandpass filtering affects a profile."""
    np.random.seed(42)
    # Generate a synthetic profile
    x = np.linspace(0, 2 * np.pi, 720)
    profile = 10 * np.sin(x) + 5 * np.sin(3 * x) + 2 * np.sin(10 * x) + np.random.randn(720) * 0.5

    from scipy.ndimage import gaussian_filter1d
    bp_narrow = gaussian_filter1d(profile, 2, mode="wrap") - gaussian_filter1d(profile, 8, mode="wrap")
    bp_wide = gaussian_filter1d(profile, 3, mode="wrap") - gaussian_filter1d(profile, 16, mode="wrap")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(profile, "b-", linewidth=0.8)
    axes[0].set_title("Original Profile", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Elevation (°)")

    axes[1].plot(bp_narrow, "g-", linewidth=0.8)
    axes[1].set_title("Bandpass σ(2→8)", fontsize=11, fontweight="bold")

    axes[2].plot(bp_wide, "r-", linewidth=0.8)
    axes[2].set_title("Bandpass σ(3→16)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Azimuth bin")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle("Bandpass Filtering: Isolating Different Terrain Scales", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_bandpass_effect.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating additional figures...")
    gsv_many_samples()
    synth_weather_grid()
    gt_vs_pred_many()
    profile_many()
    gsv_crops_grid()
    error_histogram()
    fov_distribution()
    clahe_comparison()
    canny_stages()
    success_detailed()
    failure_detailed()
    training_pairs()
    bandpass_effect()
    print(f"\nAll figures saved to {OUT}")

"""Data-driven scientific sweep of sky_ratio and boundary_coverage on GT annotations."""

import json, os, cv2, numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "street_view"

with open(DATA_DIR / "annotations.json") as f:
    annots = json.load(f)["annotations"]

sky_ratios = []
coverages = []

for sid, pts in annots.items():
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = DATA_DIR / "images" / f"{sid}{ext}"
        if p.exists(): img_path = p; break
    if not img_path: continue

    img = Image.open(img_path)
    W, H = img.size

    gt_bnd = np.full(W, -1, dtype=np.int32)
    for c, r in pts:
        ci, ri = int(c), int(r)
        if 0 <= ci < W and 0 <= ri < H:
            gt_bnd[ci] = ri

    valid_cols = gt_bnd >= 0
    coverage = float(valid_cols.sum()) / W
    coverages.append(coverage)

    # Calculate sky area above GT boundary
    sky_pixels = 0
    for c in range(W):
        if gt_bnd[c] >= 0:
            sky_pixels += gt_bnd[c]
        else:
            sky_pixels += H // 3  # Estimate for unannotated edge columns

    sky_ratio = float(sky_pixels) / (W * H)
    sky_ratios.append(sky_ratio)

print("=" * 60)
print("SCIENTIFIC DATA-DRIVEN DISTRIBUTION ON HUMAN GROUND TRUTH")
print("=" * 60)
print(f"Sample Count: {len(sky_ratios)}")
print(f"Sky Ratio Range : Min={np.min(sky_ratios):.3f} | Max={np.max(sky_ratios):.3f} | Median={np.median(sky_ratios):.3f}")
print(f"Sky Ratio 5th-95th Percentile: [{np.percentile(sky_ratios, 5):.3f}, {np.percentile(sky_ratios, 95):.3f}]")
print(f"Boundary Coverage Range      : Min={np.min(coverages):.3f} | Max={np.max(coverages):.3f} | Median={np.median(coverages):.3f}")
print(f"Boundary Coverage 5th Pct    : {np.percentile(coverages, 5):.3f}")

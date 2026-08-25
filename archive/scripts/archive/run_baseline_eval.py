#!/usr/bin/env python
"""Honest whole-region baseline evaluation on a stratified GSV subset + full synthetic.

Produces:
  data/eval/baseline/subset_ids.json
  data/eval/baseline/gsv_subset_gt.json
  data/eval/baseline/gsv_baseline.csv
  data/eval/baseline/gsv_baseline_summary.json
  data/eval/baseline/synthetic_baseline.csv
  data/eval/baseline/synthetic_baseline_summary.json
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation import load_ground_truth, run_evaluation

SV_GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
SV_MASKS_DIR = ROOT / "data" / "street_view" / "masks"
SYN_GT_PATH = ROOT / "data" / "synthetic_dataset" / "ground_truth.json"
SYN_MASKS_DIR = ROOT / "data" / "synthetic_dataset" / "predicted_masks"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
OUT_DIR = ROOT / "data" / "eval" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_stratified_subset(gt_data, n=250, seed=42):
    """Pick n samples spread uniformly across the region's lat/lon grid."""
    rng = np.random.default_rng(seed)
    sids = list(gt_data.keys())
    lats = np.array([gt_data[s]["true_lat"] for s in sids])
    lons = np.array([gt_data[s]["true_lon"] for s in sids])

    lat_bins = np.linspace(lats.min(), lats.max(), 11)
    lon_bins = np.linspace(lons.min(), lons.max(), 11)
    selected = []
    per_cell = max(1, n // 100)
    for i in range(10):
        for j in range(10):
            mask = (
                (lats >= lat_bins[i])
                & (lats < lat_bins[i + 1])
                & (lons >= lon_bins[j])
                & (lons < lon_bins[j + 1])
            )
            cells = np.where(mask)[0]
            if len(cells) > 0:
                take = rng.choice(cells, size=min(per_cell, len(cells)), replace=False)
                selected.extend(sids[idx] for idx in take)
    remaining = [s for s in sids if s not in set(selected)]
    need = n - len(selected)
    if need > 0 and remaining:
        extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        selected.extend(extra)
    return selected[:n]


def run_and_save(gt_path, masks_dir, tag):
    """Run run_evaluation on a GT file and save results + summary."""
    gt_data, _ = load_ground_truth(str(gt_path))
    n_gt = len(gt_data)

    t0 = time.time()
    df, summary = run_evaluation(
        ground_truth_path=str(gt_path),
        db_path=str(DB_PATH),
        masks_dir=str(masks_dir),
        use_altimeter=True,
        use_compass=True,
        weights=(0.5, 0.5),
        min_std_deg=1.5,
        min_max_elev_deg=1.0,
        spatial_stride=12,
        sample_batch_size=3,
        chunk_rows=4000,
    )
    elapsed = time.time() - t0

    summary["eval_time_s"] = elapsed
    summary["n_gt"] = n_gt
    summary["seconds_per_sample"] = elapsed / max(n_gt, 1)

    df.to_csv(OUT_DIR / f"{tag}_baseline.csv", index=False)
    with open(OUT_DIR / f"{tag}_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {tag} summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return df, summary


if __name__ == "__main__":
    # --- GSV stratified subset ---
    with open(SV_GT_PATH) as f:
        sv_gt = json.load(f)
    subset_ids = make_stratified_subset(sv_gt, n=250, seed=42)
    subset_gt = {sid: sv_gt[sid] for sid in subset_ids}

    subset_gt_path = OUT_DIR / "gsv_subset_gt.json"
    with open(subset_gt_path, "w") as f:
        json.dump(subset_gt, f, indent=2)
    with open(OUT_DIR / "subset_ids.json", "w") as f:
        json.dump(subset_ids, f, indent=2)
    print(f"GSV subset: {len(subset_ids)} samples -> {subset_gt_path}")

    df_sv, sum_sv = run_and_save(subset_gt_path, str(SV_MASKS_DIR), "gsv")

    # --- Full synthetic (sanity) ---
    df_syn, sum_syn = run_and_save(str(SYN_GT_PATH), str(SYN_MASKS_DIR), "synthetic")

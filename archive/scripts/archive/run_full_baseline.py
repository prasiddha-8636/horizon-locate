#!/usr/bin/env python
"""Run baseline evaluation on synthetic dataset."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluation import load_ground_truth, run_evaluation

# Load synthetic ground truth
with open("data/synthetic_dataset/ground_truth.json") as f:
    sy_gt_data = json.load(f)

# Select first 50 samples for quick evaluation
sample_ids = list(sy_gt_data.keys())[:50]
subset_gt = {sid: sy_gt_data[sid] for sid in sample_ids}

# Save subset for evaluation
subset_path = "tmp_synth_subset.json"
with open(subset_path, "w") as f:
    json.dump(subset_gt, f, indent=2)

# Prepare mask directory (假设已准备好)
mask_dir = "data/synthetic_dataset/predicted_masks"
if not os.path.exists(mask_dir):
    print(f"Warning: mask directory {mask_dir} not found, skipping segmentation phase")
    exit(0)

# Run evaluation
t0 = time.time()
df, summary = run_evaluation(
    ground_truth_path=str(subset_path),
    expected_offset_deg=None,
    use_altimeter=True,
    use_compass=True,
    sample_list=sample_ids,
    db_path="notebooks/02_SkylineDatabase/output/skyline_db.parquet",
    masks_dir=mask_dir,
    weights=(0.5, 0.5),
    min_std_deg=1.5,
    min_max_elev_deg=1.0,
    spatial_stride=12,
)

elapsed = time.time() - t0
print(f"Evaluation completed in {elapsed:.1f}s")
print(f"Samples processed: {len(sample_ids)}")
print("Summary keys:", list(summary.keys()))
print("Acquired status:", df["status"].value_counts().to_dict())

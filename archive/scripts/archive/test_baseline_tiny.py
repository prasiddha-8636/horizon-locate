#!/usr/bin/env python
"""Smoke test: run baseline evaluator on 3 GSV samples."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import load_ground_truth, run_evaluation

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
MASKS_DIR = ROOT / "data" / "street_view" / "masks"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"

# Load GT and pick 3 samples
with open(GT_PATH) as f:
    gt_data = json.load(f)
sample_ids = list(gt_data.keys())[:3]
subset_gt = {sid: gt_data[sid] for sid in sample_ids}

subset_path = ROOT / "tmp_test_gt.json"
subset_path.parent.mkdir(exist_ok=True)
with open(subset_path, "w") as f:
    json.dump(subset_gt, f, indent=2)

print("Testing on:", sample_ids)

t0 = time.time()
df, summary = run_evaluation(
    ground_truth_path=str(subset_path),
    db_path=str(DB_PATH),
    masks_dir=str(MASKS_DIR),
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
print(f"Elapsed: {elapsed:.1f}s")
print("Summary:")
print(json.dumps(summary, indent=2, default=str))
print("DF head:")
print(df.head())

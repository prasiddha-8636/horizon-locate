#!/usr/bin/env python
"""Quick test: match synthetic samples against subset of skyline DB."""
import json, os, gc, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.matching import match_query
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8, decode_horizon_column

db_path = "notebooks/02_SkylineDatabase/output/skyline_db.parquet"
gt_path = "data/synthetic_dataset/ground_truth.json"
masks_dir = "data/synthetic_dataset/predicted_masks"

print("Loading ground truth...", flush=True)
with open(gt_path) as f:
    gt_data = json.load(f)
sample_ids = list(gt_data.keys())[:5]
print(f"Samples: {sample_ids}", flush=True)

print("DB metadata...", flush=True)
meta = pd.read_parquet(db_path, columns=["lon", "lat", "elevation_m"])
n_vp = len(meta)
lon_arr, lat_arr = meta["lon"].to_numpy(), meta["lat"].to_numpy()
del meta; gc.collect()
print(f"{n_vp} viewpoints", flush=True)

print("Bin size...", flush=True)
pf = pq.ParquetFile(db_path)
first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
print(f"bin_deg={bin_deg}", flush=True)

print("Loading subset DB (first 5000 rows)...", flush=True)
batch = next(pf.iter_batches(batch_size=5000, columns=["raw_horizon_deg"]))
chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
db_matrix = chunk
print(f"DB matrix: {db_matrix.shape}", flush=True)

for sid in sample_ids:
    gt = gt_data[sid]
    mask_path = os.path.join(masks_dir, f"sample_{int(sid):04d}.png")
    if not os.path.exists(mask_path):
        print(f"  {sid}: no mask", flush=True)
        continue

    fov = gt.get("fov_y_deg", 65.0)
    r_tilt = np.array(gt["cam_R_tilt"]) if gt.get("cam_R_tilt") else None

    pr = extract_elevation_profile(mask_path, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=bin_deg)
    if not pr["ok"]:
        print(f"  {sid}: profile: {pr['reason']}", flush=True)
        continue

    mr = match_query(db_matrix, bin_deg, pr["profile"], spatial_stride=2, min_corr=0.1, min_score_gap=0.02)

    tl, tn = gt["true_lat"], gt["true_lon"]
    if mr["ok"] and mr["matches"]:
        b = mr["matches"][0]
        err = geodesic((tl, tn), (lat_arr[b["row_index"]], lon_arr[b["row_index"]])).meters
        print(f"  {sid}: err={err:.0f}m, corr={b['fft_corr']:.3f}, dtw={b['dtw_distance']:.1f}, status={mr['status']}", flush=True)
    else:
        print(f"  {sid}: {mr['status']}: {mr['reason']}", flush=True)

print("DONE", flush=True)

#!/usr/bin/env python
"""Run matching eval on 300 synthetic samples using streaming DB chunks."""

import json, os, gc, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.matching import fft_prefilter, finalize_matches
from src.query_profile import extract_elevation_profile, is_profile_applicable
from src.horizon_format import decode_horizon_column

DB_PATH = "notebooks/02_SkylineDatabase/output/skyline_db.parquet"
GT_PATH = "data/synthetic_dataset/ground_truth.json"
MASKS_DIR = "data/synthetic_dataset/predicted_masks"

print("Loading ground truth...", flush=True)
with open(GT_PATH) as f:
    gt_data = json.load(f)
sample_ids = sorted(gt_data.keys(), key=int)
print(f"Total samples: {len(sample_ids)}", flush=True)

print("Loading DB metadata...", flush=True)
meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
lon_arr, lat_arr = meta["lon"].to_numpy(), meta["lat"].to_numpy()
del meta
gc.collect()

print("Getting bin size...", flush=True)
pf = pq.ParquetFile(DB_PATH)
first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])

results = []
n_ok = 0
n_skip = 0
n_fail = 0
t0 = time.time()

for sid in sample_ids:
    gt = gt_data[sid]
    mask_path = os.path.join(MASKS_DIR, f"sample_{int(sid):04d}.png")
    if not os.path.exists(mask_path):
        n_skip += 1
        continue

    fov = gt.get("fov_y_deg", 65.0)
    r_tilt = np.array(gt["cam_R_tilt"]) if gt.get("cam_R_tilt") else None

    pr = extract_elevation_profile(
        mask_path, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=bin_deg
    )
    if not pr["ok"]:
        n_skip += 1
        continue

    profile = pr["profile"]
    tl, tn = gt["true_lat"], gt["true_lon"]

    # Stream DB chunks (memory-safe)
    best_overall = None
    best_dist = float("inf")
    best_overall = None
    pf2 = pq.ParquetFile(DB_PATH)
    chunk_start = 0
    for batch in pf2.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        corr, offsets = fft_prefilter(chunk, profile, bin_deg)
        top5 = np.argsort(-corr)[:5]
        for idx in top5:
            if corr[idx] == -np.inf:
                continue
            global_idx = chunk_start + idx
            err = geodesic((tl, tn), (lat_arr[global_idx], lon_arr[global_idx])).meters
            if err < best_dist:
                best_dist = err
                best_overall = {
                    "row_index": int(global_idx),
                    "error_m": err,
                    "fft_corr": float(corr[idx]),
                }
        chunk_start += len(chunk)
        del chunk, corr, offsets
        gc.collect()

    if best_overall:
        n_ok += 1
        results.append(best_overall)
    else:
        n_fail += 1

    if len(results) % 50 == 0 and len(results) > 0:
        t = time.time() - t0
        print(
            f"  {len(results)}/{len(sample_ids)} matched, {t / len(results):.1f}s/sample",
            flush=True,
        )

t = time.time() - t0
print(f"\n=== RESULTS ===", flush=True)
print(f"Matched: {n_ok}, Skipped (bad profile): {n_skip}, Failed: {n_fail}", flush=True)
print(f"Time: {t:.0f}s ({t / max(1, n_ok):.1f}s/sample)", flush=True)
if results:
    errors = np.array([r["error_m"] for r in results])
    print(f"Median error: {np.median(errors):.0f}m", flush=True)
    print(f"Mean error:   {np.mean(errors):.0f}m", flush=True)
    print(f"Top-1@500m:   {np.mean(errors <= 500) * 100:.1f}%", flush=True)
    print(f"Top-1@1000m:  {np.mean(errors <= 1000) * 100:.1f}%", flush=True)
    print(f"Top-1@5000m:  {np.mean(errors <= 5000) * 100:.1f}%", flush=True)

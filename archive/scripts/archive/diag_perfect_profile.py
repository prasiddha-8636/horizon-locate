#!/usr/bin/env python
"""Diagnostic: isolate error sources on synthetic data.

Test A: Perfect profile (exact DB horizon at GT viewpoint, exact offset)
  -> if this fails, the matching/offset convention is broken.
Test B: Perfect profile but jittered viewpoint (real synthetic setup)
  -> error here = spatial/azimuth quantization limit.
Test C: Real extracted profile from predicted mask
  -> error here adds segmentation noise.
"""

import json, sys, time, os
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.matching import feature_bundle_matrix, ncc_scores
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8, decode_horizon_column
from geopy.distance import geodesic

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")

with open(os.path.join(ROOT, "data/synthetic_dataset/ground_truth.json")) as f:
    gt = json.load(f)

pf = pq.ParquetFile(DB)
meta = pq.read_table(DB, columns=["lon", "lat", "elevation_m"])
lon_arr = meta.column("lon").to_pandas().values
lat_arr = meta.column("lat").to_pandas().values
elev_arr = meta.column("elevation_m").to_pandas().values
first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
print(f"bin_deg = {bin_deg}")


# Load all horizons for fast per-query full scan (300 queries, 1.34M rows)
# Instead, stream per query.
def fetch_horizon(vp_idx):
    pf = pq.ParquetFile(DB)
    rg = vp_idx // 4096
    pos = vp_idx % 4096
    batch = pf.read_row_group(rg, columns=["raw_horizon_deg"])
    return decode_horizon_uint8(batch.to_pandas()["raw_horizon_deg"].iloc[pos])


def full_db_ncc(profile, expected_offset_deg=None, elevation_m=None):
    """Stream full DB, return (top_lat, top_lon, top_score, top_err)."""
    pf = pq.ParquetFile(DB)
    best_c = -np.inf
    best_i = -1
    cs = 0
    for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        cm = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        n = len(cm)
        db_val, db_d1 = feature_bundle_matrix(cm)
        corr, offs = ncc_scores(
            db_val,
            db_d1,
            profile,
            bin_deg,
            weights=(0.5, 0.5),
            expected_offset_deg=expected_offset_deg,
            tolerance_deg=20.0 if expected_offset_deg is not None else None,
        )
        if elevation_m is not None:
            ev = np.abs(elev_arr[cs : cs + n] - elevation_m) <= 200.0
            corr = np.where(ev, corr, -np.inf)
        m = np.argmax(corr)
        if corr[m] > best_c:
            best_c = corr[m]
            best_i = cs + m
        cs += n
        del cm, db_val, db_d1, corr
    return best_c, best_i


def make_query_from_horizon(horizon, fov_deg, heading_deg):
    """Simulate the query profile extraction: slice horizon over [heading-fov/2, heading+fov/2]."""
    # horizon[k] = elevation at azimuth k*bin_deg (North=0, clockwise?)
    half = int(fov_deg / 2 / bin_deg)
    start_bin = int(round((heading_deg - fov_deg / 2) / bin_deg)) % 360
    # slice circular
    idxs = (np.arange(start_bin, start_bin + 2 * half + 1)) % 360
    return horizon[idxs]


# ---------- Test A: perfect profile at exact GT viewpoint ----------
print("\n=== Test A: perfect profile at GT viewpoint ===")
errsA, errsB, errsC = [], [], []
for sid in list(gt.keys())[:60]:
    g = gt[sid]
    vp = int(g["closest_viewpoint_id"])
    # Some entries have closest_viewpoint_dist_m > 0 (jittered), some 0.0
    dist_m = g["closest_viewpoint_dist_m"]
    horizon = fetch_horizon(vp)
    fov = g["fov_y_deg"]
    heading = g["true_heading_deg"]
    # aspect ratio of synthetic render (let's assume square? check GT)
    fov_x = fov  # placeholder
    profile = make_query_from_horizon(horizon, fov_x, heading)
    expected_off = (heading - fov_x / 2) % 360.0
    score, idx = full_db_ncc(
        profile, expected_offset_deg=expected_off, elevation_m=g["eye_z_m"]
    )
    err = geodesic((g["true_lat"], g["true_lon"]), (lat_arr[idx], lon_arr[idx])).meters
    errsA.append((err, dist_m, score, vp, idx))
    if len(errsA) % 10 == 0:
        print(
            f"  {len(errsA)} samples, median err so far: {np.median([e[0] for e in errsA]):.0f}m"
        )

errA = [e[0] for e in errsA]
print(f"Test A median: {np.median(errA):.0f}m, p90: {np.percentile(errA, 90):.0f}m")
print(
    f"  @100m: {100 * np.mean(np.array(errA) <= 100):.0f}%, @500m: {100 * np.mean(np.array(errA) <= 500):.0f}%"
)
print(f"  dist-to-VP: median {np.median([e[1] for e in errsA]):.1f}m")
print(f"  scores: median {np.median([e[2] for e in errsA]):.3f}")
same_vp = [e for e in errsA if e[3] == e[4]]
print(f"  found exact GT VP: {len(same_vp)}/{len(errsA)}")

np.save("/tmp/diag_testA.npy", errsA, allow_pickle=True)

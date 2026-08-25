#!/usr/bin/env python
"""Fast diagnostic: measure per-VP NCC quality + ambiguity.

For each sample, compute:
  1. NCC score at the GT viewpoint (peak over all offsets) — if low, profile is bad.
  2. Peak NCC over the full DB (sampled) — if far from GT, match is ambiguous.
"""

import json, sys, os
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.matching import feature_bundle_matrix, ncc_scores
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8, decode_horizon_column
from geopy.distance import geodesic

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_PATH = os.path.join(ROOT, "data/synthetic_dataset/ground_truth.json")
MASK_DIR = os.path.join(ROOT, "data/synthetic_dataset/masks")
PRED_DIR = os.path.join(ROOT, "data/synthetic_dataset/predicted_masks")

with open(GT_PATH) as f:
    gt = json.load(f)

meta = pq.read_table(DB, columns=["lon", "lat", "elevation_m"])
lon_arr = meta.column("lon").to_pandas().values
lat_arr = meta.column("lat").to_pandas().values
elev_arr = meta.column("elevation_m").to_pandas().values

pf0 = pq.ParquetFile(DB)
first = next(pf0.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
del pf0


def fetch_horizon(vp_idx):
    pf = pq.ParquetFile(DB)
    rg = int(vp_idx) // 4096
    pos = int(vp_idx) % 4096
    batch = pf.read_row_group(rg, columns=["raw_horizon_deg"])
    return decode_horizon_uint8(batch.to_pandas()["raw_horizon_deg"].iloc[pos])


def ncc_at_horizon(horizon, profile, expected_off):
    db_val, db_d1 = feature_bundle_matrix(horizon[None, :])
    corr, offs = ncc_scores(
        db_val,
        db_d1,
        profile,
        bin_deg,
        weights=(0.5, 0.5),
        expected_offset_deg=expected_off,
        tolerance_deg=20.0 if expected_off is not None else None,
    )
    return float(corr[0]), float(offs[0])


def global_peak(profile, exp_off, elev, sample_every=500):
    pf = pq.ParquetFile(DB)
    best = -np.inf
    best_lat = best_lon = None
    best_err = np.nan
    cs = 0
    for batch in pf.iter_batches(batch_size=4096, columns=["raw_horizon_deg"]):
        cm = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        n = len(cm)
        sel = np.arange(0, n, sample_every)
        db_val, db_d1 = feature_bundle_matrix(cm[sel])
        corr, offs = ncc_scores(
            db_val,
            db_d1,
            profile,
            bin_deg,
            weights=(0.5, 0.5),
            expected_offset_deg=exp_off,
            tolerance_deg=20.0 if exp_off is not None else None,
        )
        ev = np.abs(elev_arr[cs + sel] - elev) <= 200.0
        corr = np.where(ev, corr, -np.inf)
        m = np.argmax(corr)
        if corr[m] > best:
            best = float(corr[m])
            best_lat = lat_arr[cs + sel[m]]
            best_lon = lon_arr[cs + sel[m]]
        cs += n
        del cm, db_val, db_d1, corr
    del pf
    return best, best_lat, best_lon


def run_tag(tag, mask_dir, sids):
    rows = []
    for sid in sids:
        g = gt[sid]
        pr = extract_elevation_profile(
            os.path.join(mask_dir, f"sample_{int(sid):04d}.png"),
            fov_y_deg=g["fov_y_deg"],
            bin_deg=bin_deg,
        )
        if not pr["ok"]:
            rows.append((sid, np.nan, np.nan, np.nan, np.nan))
            continue
        exp_off = (g["true_heading_deg"] + pr["start_az"]) % 360.0
        horizon = fetch_horizon(int(g["closest_viewpoint_id"]))
        gtscore, gtoff = ncc_at_horizon(horizon, pr["profile"], exp_off)
        gbest, gblat, gblon = global_peak(pr["profile"], exp_off, g["eye_z_m"])
        err = (
            geodesic((g["true_lat"], g["true_lon"]), (gblat, gblon)).meters
            if gblat is not None
            else np.nan
        )
        rows.append((sid, gtscore, gbest, err, gtoff))

    print(f"=== {tag} ===")
    arr = np.array(rows, dtype=float)
    gts = arr[:, 1]
    gb = arr[:, 2]
    errs = arr[:, 3]
    ok = ~np.isnan(errs)
    print(
        f"  GT-VP peak score: median={np.nanmedian(gts):.3f} p25={np.nanpercentile(gts, 25):.3f} p10={np.nanpercentile(gts, 10):.3f}"
    )
    print(f"  Global peak score: median={np.nanmedian(gb):.3f}")
    print(
        f"  Global peak error: median={np.nanmedian(errs[ok]):.0f}m "
        f"@50m={100 * np.mean(errs[ok] <= 50):.0f}% @100m={100 * np.mean(errs[ok] <= 100):.0f}% "
        f"@500m={100 * np.mean(errs[ok] <= 500):.0f}%"
    )
    gap = gb - gts
    print(
        f"  Gap (global peak - GT peak): median={np.nanmedian(gap):+.3f} "
        f"frac_gap_pos={100 * np.nanmean(gap > 0.05):.0f}%"
    )
    print()


sids = list(gt.keys())[:40]
print(f"Testing {len(sids)} synthetic samples (bin={bin_deg})\n")
run_tag("B: GT render mask", MASK_DIR, sids)
run_tag("C: predicted mask (real pipeline)", PRED_DIR, sids)

#!/usr/bin/env python
"""Measure coarse-scan localization quality on GSV.

For each sample: run the coarse NCC scan (stride configurable), then check
how far the top-1/top-5 coarse VPs are from the GT VP. This decides whether
refinement can work.
"""

import sys, json, os, time
import numpy as np
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.matching import feature_bundle_matrix, ncc_scores
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8, decode_horizon_column

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
N_VP = 1338650


_META = pq.read_table(DB, columns=["lon", "lat", "elevation_m"])


def coarse_topk(profile, exp_off, elev, stride, tol, k=10):
    """Return (dist of top1 to GT not available here) -> returns (topk_idx, topk_corr)."""
    pf = pq.ParquetFile(DB)
    meta = _META
    lat_arr = meta.column("lat").to_pandas().values
    lon_arr = meta.column("lon").to_pandas().values
    elev_arr = meta.column("elevation_m").to_pandas().values

    best_corr = np.full(N_VP, -np.inf)
    best_off = np.zeros(N_VP, dtype=np.int32)
    cs = 0
    for batch in pf.iter_batches(batch_size=20000, columns=["raw_horizon_deg"]):
        cm = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        n = len(cm)
        stride_idx = np.arange(0, n, stride)
        db_val, db_d1 = feature_bundle_matrix(cm[stride_idx])
        corr, offs = ncc_scores(
            db_val,
            db_d1,
            profile,
            1.0,
            weights=(0.5, 0.5),
            expected_offset_deg=exp_off,
            tolerance_deg=tol,
        )
        ev = np.abs(elev_arr[cs + stride_idx] - elev) <= 200.0
        corr = np.where(ev, corr, -np.inf)
        idxs = cs + stride_idx
        for j, ii in enumerate(idxs):
            if corr[j] > best_corr[ii]:
                best_corr[ii] = corr[j]
                best_off[ii] = offs[j]
        cs += n
        del cm, db_val, db_d1, corr
    del pf
    valid = np.where(best_corr > -np.inf)[0]
    order = valid[np.argsort(-best_corr[valid])][:k]
    return order, best_corr[order], lat_arr, lon_arr


first = next(
    pq.ParquetFile(DB_PATH).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
def main():
    with open(os.path.join(ROOT, "data/street_view/ground_truth.json")) as f:
        gt = json.load(f)
    sids = list(gt.keys())[:5]

    print(
        f"{'sid':<22} {'stride':>6} {'tol':>4} {'top1_dist':>9} {'top5_dist':>9} {'gt_in5':>7}"
    )
    for sid in sids:
        g = gt[sid]
        pr = extract_elevation_profile(
            os.path.join(ROOT, f"data/street_view/masks/{sid}.png"),
            fov_y_deg=g.get("fov_y_deg", 65.0),
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=BIN_DEG,
        )
        if not pr["ok"]:
            print(f"{sid:<22} PROFILE_FAIL")
            continue
        prof = pr["profile"]
        exp_off = (g["true_heading_deg"] + pr["start_az"]) % 360.0
        elev = g["eye_z_m"]

        for stride, tol in [(5, 20.0), (12, 20.0)]:
            t0 = time.time()
            order, corrs, lat_arr, lon_arr = coarse_topk(
                prof, exp_off, elev, stride, tol, k=10
            )
            dt = time.time() - t0
            dists = [
                geodesic(
                    (g["true_lat"], g["true_lon"]), (lat_arr[i], lon_arr[i])
                ).meters
                for i in order
            ]
            gt_in5 = min(dists) <= 500.0
            print(
                f"{sid:<22} {stride:>6} {tol:>4} {dists[0]:>9.0f} {min(dists[:5]):>9.0f} {str(gt_in5):>7} [{dt:.0f}s]"
            )


if __name__ == "__main__":
    main()

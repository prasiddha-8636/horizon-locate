#!/usr/bin/env python
"""Exhaustive stride=1 scan: true-VP rank in the full 1.3M VP search."""

import json, sys, time, os
import numpy as np
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.matching import _feature_bundle, _pearson_ncc_batch, feature_bundle_matrix
from src.horizon_format import decode_horizon_uint8, decode_horizon_column

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_PATH = os.path.join(ROOT, "data/synthetic_dataset/ground_truth.json")

meta = pq.read_table(DB, columns=["lon", "lat", "elevation_m"])
lon_arr = meta.column("lon").to_pandas().values
lat_arr = meta.column("lat").to_pandas().values
nv = len(lon_arr)
first = next(pq.ParquetFile(DB).iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
n_bins = int(360 / bin_deg)


def fetch_horizon(vp_idx):
    pf = pq.ParquetFile(DB)
    rg_starts = np.concatenate(
        [
            [0],
            np.cumsum(
                [pf.metadata.row_group(r).num_rows for r in range(pf.num_row_groups)]
            )[:-1],
        ]
    )
    rg = int(np.searchsorted(rg_starts, vp_idx, side="right") - 1)
    pos = vp_idx - rg_starts[rg]
    batch = pf.read_row_group(rg, columns=["raw_horizon_deg"])
    return decode_horizon_uint8(batch.to_pandas()["raw_horizon_deg"].iloc[pos])


def make_profile(horizon, fov_deg, heading_deg):
    half = int(fov_deg / 2 / bin_deg)
    start_bin = int(round((heading_deg - fov_deg / 2) / bin_deg)) % n_bins
    idxs = (np.arange(start_bin, start_bin + 2 * half + 1)) % n_bins
    return horizon[idxs]


def full_scan_stride1(profile):
    """Return per-VP best-offset correlation array of length nv."""
    pf = pq.ParquetFile(DB)
    all_corr = np.full(nv, -np.inf, dtype=np.float64)
    global_offset = 0
    for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        cm = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        n = len(cm)
        db_val, db_d1 = feature_bundle_matrix(cm)

        q_val, q_d1 = _feature_bundle(profile)
        q_val_zm = q_val - q_val.mean()
        q_d1_zm = q_d1 - q_d1.mean()
        db_ext_val = np.concatenate([db_val, db_val[:, : len(profile) - 1]], axis=1)
        db_ext_d1 = np.concatenate([db_d1, db_d1[:, : len(profile) - 1]], axis=1)
        ncc_val = _pearson_ncc_batch(db_ext_val, q_val_zm, np.linalg.norm(q_val_zm))
        ncc_d1 = _pearson_ncc_batch(db_ext_d1, q_d1_zm, np.linalg.norm(q_d1_zm))
        combined = 0.5 * ncc_val + 0.5 * ncc_d1
        best_off = np.argmax(combined, axis=1)
        best_corr = combined[np.arange(n), best_off]
        all_corr[global_offset : global_offset + n] = best_corr
        global_offset += n
        del cm, db_val, db_d1, combined
    return all_corr


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    print(f"bin_deg={bin_deg} n_bins={n_bins} n_vps={nv} stride=1 (exhaustive)")
    print()
    print(
        f"{'SID':>6} {'TrueVP':>8} {'Rank':>8}/1.3M {'%ile':>6} {'TrueFB':>8} "
        f"{'BestFB':>8} {'BestErr':>9}  time"
    )
    for sid in list(gt.keys())[:5]:
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        gt_lat, gt_lon = g["true_lat"], g["true_lon"]
        horizon = fetch_horizon(vp)
        profile = make_profile(horizon, g["fov_y_deg"], g["true_heading_deg"])

        t0 = time.time()
        corr = full_scan_stride1(profile)
        total = nv
        true_corr = corr[vp]
        rank = int(np.sum(corr > true_corr))
        best_vp = int(np.argmax(corr))
        best_err = geodesic(
            (gt_lat, gt_lon), (lat_arr[best_vp], lon_arr[best_vp])
        ).meters
        pct = rank / total * 100
        elapsed = time.time() - t0
        print(
            f"{sid:>6} {vp:>8} {rank:>8}/{total} {pct:>5.1f}% {true_corr:>8.4f} "
            f"{corr.max():>8.4f} {best_err:>9.0f}  ({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    main()

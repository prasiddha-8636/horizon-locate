#!/usr/bin/env python
"""Adaptive ensemble: baseline NCC + bandpass bp(2,8) + bp(3,16) + parallax-ratio.

Streams the DB once per sample (stride-12, calibrated pitch) and computes all
four full-DB score fields. Keeps top-K candidate (vp, score) lists per method
plus the true-VP score, then selects the method with the strongest relative
top-1 surge (score gap top1 vs top2 / top2) and reports its top-1 error.

Usage: python scripts/adaptive_ensemble_eval.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft, irfft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, _feature_bundle_matrix, _pearson_ncc_batch
from src.horizon_format import decode_horizon_column
from scripts.fixes_eval import (
    Rx,
    mask_from_ann,
    DB_PATH,
    GT_FILE,
    ANNOT_FILE,
    CALIB_FILE,
)

CHUNK = 4000
BIN_DEG = 0.5
STRIDE = 12
TOPK = 200


def dog_bandpass(x, s1, s2):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return gaussian_filter1d(x, s1, mode="wrap") - gaussian_filter1d(
            x, s2, mode="wrap"
        )
    return gaussian_filter1d(x, s1, mode="wrap", axis=1) - gaussian_filter1d(
        x, s2, mode="wrap", axis=1
    )


def split_layers_mat(mat, freq_cutoff_ratio=0.25):
    freq = rfft(mat, axis=1)
    n_keep = max(4, int(freq.shape[1] * freq_cutoff_ratio))
    far = irfft(freq[:, :n_keep], n=mat.shape[1], axis=1)
    return far, mat - far


def parallax_ratio_mat(mat, freq_cutoff_ratio=0.25):
    far, near = split_layers_mat(mat, freq_cutoff_ratio)
    d_far = np.gradient(far, axis=1)
    d_near = np.gradient(near, axis=1)
    ratio = np.zeros_like(mat)
    mask = np.abs(d_far) > 1e-6
    ratio[mask] = d_near[mask] / d_far[mask]
    return ratio


def ncc_fb_mat(query, db_mat):
    """Feature-bundle NCC (value + grad), best over circular shifts."""
    M = len(query)
    qv, qd = _feature_bundle(query)
    qv = qv - qv.mean()
    qd = qd - qd.mean()
    qvn = np.linalg.norm(qv)
    qdn = np.linalg.norm(qd)
    if qvn < 1e-12 or qdn < 1e-12:
        return np.full(len(db_mat), -np.inf)
    v, d1 = _feature_bundle_matrix(db_mat)
    ext_v = np.concatenate([v, v[:, : M - 1]], axis=1)
    ext_d = np.concatenate([d1, d1[:, : M - 1]], axis=1)
    return 0.5 * _pearson_ncc_batch(ext_v, qv, qvn).max(
        axis=1
    ) + 0.5 * _pearson_ncc_batch(ext_d, qd, qdn).max(axis=1)


def ncc_plain_mat(query, db_mat):
    M = len(query)
    qz = query - query.mean()
    qn = np.linalg.norm(qz)
    if qn < 1e-12:
        return np.full(len(db_mat), -np.inf)
    dz = db_mat - db_mat.mean(axis=1, keepdims=True)
    ext = np.concatenate([dz, dz[:, : M - 1]], axis=1)
    return _pearson_ncc_batch(ext, qz, qn).max(axis=1)


def err_km(vp_idx, tl, tn, vp_lat, vp_lon):
    return geodesic((tl, tn), (vp_lat[vp_idx], vp_lon[vp_idx])).meters / 1000


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    vp_lon = meta["lon"].to_numpy()
    vp_lat = meta["lat"].to_numpy()
    del meta

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))
    sids = [s for s in ann if s in gt]

    queries = {}
    tlat, tlon = {}, {}
    true_vp = {}
    for sid in sids:
        g = gt[sid]
        tilt = np.array(g["cam_R_tilt"])
        dp = float(calib.get(sid, {}).get("delta_pitch_deg", 0.0))
        mask, _ = mask_from_ann(ann[sid])
        pr = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=Rx(dp) @ tilt, bin_deg=BIN_DEG
        )
        if pr["ok"] and len(pr["profile"]) >= 4:
            queries[sid] = pr["profile"]
            tlat[sid], tlon[sid] = g["true_lat"], g["true_lon"]
            true_vp[sid] = g["closest_viewpoint_id"]
    print(f"Queries: {len(queries)}")

    methods = ["ncc", "bp28", "bp316", "para"]
    # Per snippet: keep running top-K per method per sample
    topk = {m: {s: [] for s in queries} for m in methods}  # list of (score, vp)
    best = {
        m: {s: -np.inf for s in queries} for m in methods
    }  # binned: we stream, so store all
    # For simplicity store full per-method arrays (float32) per sample in RAM.
    # 1.34M * 4 * 8 bytes/sample ~ 43MB sample, fine but we process per sample once.

    # Simplify: collect full score arrays per method per sample, one sample at a time
    pf = pq.ParquetFile(DB_PATH)
    t0 = time.time()

    # Gather DB into chunked arrays once to reuse across samples? DB is 485MB;
    # better: iterate once, computing all 17 samples' 4 methods simultaneously.
    scores = {
        m: {s: np.zeros(pf.metadata.num_rows, dtype=np.float32) for s in queries}
        for m in methods
    }

    n_chunks = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        chunk_s = chunk[::STRIDE]
        N = len(chunk_s)
        chunk_base = n_chunks * CHUNK
        idx_arr = chunk_base + np.arange(N) * STRIDE

        bp28 = dog_bandpass(chunk_s.astype(np.float64), 2, 8)
        bp316 = dog_bandpass(chunk_s.astype(np.float64), 3, 16)
        para_db = parallax_ratio_mat(chunk_s.astype(np.float64))

        for s in queries:
            q = queries[s]
            f = {
                "ncc": ncc_fb_mat(q, chunk_s),
                "bp28": ncc_plain_mat(dog_bandpass(q, 2, 8), bp28),
                "bp316": ncc_plain_mat(dog_bandpass(q, 3, 16), bp316),
                "para": ncc_fb_mat(parallax_ratio_mat(q[np.newaxis])[0], para_db),
            }
            for m in methods:
                scores[m][s][idx_arr] = f[m].astype(np.float32)
        n_chunks += 1
        if n_chunks % 50 == 0:
            print(f"  chunk {n_chunks} ({time.time() - t0:.0f}s)", flush=True)

    # ---- Per-method metrics ---
    print("\n=== Per-method full-DB top-1 (17 GSV, calibrated pitch, stride 12) ===")
    for m in methods:
        errs = []
        for s in queries:
            a = scores[m][s]
            bv = int(np.argmax(a))
            errs.append(err_km(bv, tlat[s], tlon[s], vp_lat, vp_lon))
        e = np.array(errs)
        print(
            f"{m:>6}: med={np.median(e):5.1f}km  <1km={int(np.sum(e < 1))}/{len(queries)}  "
            f"<2km={int(np.sum(e < 2))}/{len(queries)}  <5km={int(np.sum(e < 5))}/{len(queries)}  "
            f"<10km={int(np.sum(e < 10))}/{len(queries)}"
        )

    # ---- Adaptive ensemble: strongest relative surge among top-1 -> pick method
    print("\n=== Adaptive ensemble (surge-selected best-of-4) ===")
    ens_errs = []
    ens_true_rank = []
    for s in queries:
        # For each method, surge = (top1 - top2) / (top2 + eps) using 2nd best VP
        # We need top-2 per method (top1 + best runner-up).
        surges = {}
        for m in methods:
            a = scores[m][s]
            i1 = int(np.argmax(a))
            s1 = a[i1]
            a2 = np.ones(len(a)) * -np.inf
            a2[i1] = -np.inf
            i2 = int(np.argmax(a2))
            s2 = a[i2]
            surges[m] = (s1 - s2) / (abs(s2) + 1e-6)
        chosen = max(surges, key=surges.get)
        bv = int(np.argmax(scores[chosen][s]))
        ens_errs.append(err_km(bv, tlat[s], tlon[s], vp_lat, vp_lon))
        tv = true_vp[s]
        # rank of true VP under chosen method
        a = scores[chosen][s]
        rank = 1
        if tv < len(a):
            rank = int(np.sum(a > a[tv])) + 1
        ens_true_rank.append((chosen, rank))
        print(
            f"{s[:20]:20s} chosen={chosen:>5}  ens_err={ens_errs[-1]:6.1f}km  true_rank={rank}"
        )
    ee = np.array(ens_errs)
    print(
        f"\nENSEMBLE: med={np.median(ee):5.1f}km  <1km={int(np.sum(ee < 1))}/{len(queries)}  "
        f"<2km={int(np.sum(ee < 2))}/{len(queries)}  <5km={int(np.sum(ee < 5))}/{len(queries)}  "
        f"<10km={int(np.sum(ee < 10))}/{len(queries)}"
    )
    print(f"\nTotal: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

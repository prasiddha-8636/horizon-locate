#!/usr/bin/env python
"""Honest full-DB evaluation of bandpass/POC matching (no candidate
restriction, true VP NOT guaranteed in the search).

Streams the DB once per filter config and scores all annotated queries
simultaneously per chunk. Reports per-config full-DB top-1 error and median.

Usage: python scripts/full_db_bandpass_eval.py
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, _feature_bundle_matrix, _pearson_ncc_batch
from src.horizon_format import decode_horizon_column
from scripts.fixes_eval import Rx, mask_from_ann, DB_PATH, GT_FILE, ANNOT_FILE

CHUNK = 4000
BIN_DEG = 0.5
STRIDE = 12  # canonical honest-matcher spatial stride (annotated_gsv_eval)


def load_geometry():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    return (
        meta["lon"].to_numpy(),
        meta["lat"].to_numpy(),
        meta["elevation_m"].to_numpy(),
    )


def dog_bandpass(x, s1, s2):
    """Difference-of-Gaussians bandpass (circular) on 1-D or 2-D (N, L)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return gaussian_filter1d(x, s1, mode="wrap") - gaussian_filter1d(
            x, s2, mode="wrap"
        )
    return gaussian_filter1d(x, s1, mode="wrap", axis=1) - gaussian_filter1d(
        x, s2, mode="wrap", axis=1
    )


def best_corr_per_vp(query, db_bp):
    """Plain Pearson NCC on bandpassed values, best over circular shifts."""
    M = len(query)
    q_bp = query  # already bandpassed
    qz = q_bp - q_bp.mean()
    qn = np.linalg.norm(qz)
    if qn < 1e-12:
        return np.full(len(db_bp), -np.inf)
    dz = db_bp - db_bp.mean(axis=1, keepdims=True)
    ext = np.concatenate([dz, dz[:, : M - 1]], axis=1)
    return _pearson_ncc_batch(ext, qz, qn).max(axis=1)


def best_corr_fb(query, db_bp):
    """Feature-bundle (value + grad) NCC on bandpassed values."""
    M = len(query)
    qv, qd = _feature_bundle(query)
    qv = qv - qv.mean()
    qd = qd - qd.mean()
    qvn = np.linalg.norm(qv)
    qdn = np.linalg.norm(qd)
    if qvn < 1e-12 or qdn < 1e-12:
        return np.full(len(db_bp), -np.inf)
    v, d1 = _feature_bundle_matrix(db_bp)
    ext_v = np.concatenate([v, v[:, : M - 1]], axis=1)
    ext_d = np.concatenate([d1, d1[:, : M - 1]], axis=1)
    return 0.5 * _pearson_ncc_batch(ext_v, qv, qvn).max(
        axis=1
    ) + 0.5 * _pearson_ncc_batch(ext_d, qd, qdn).max(axis=1)


def main():
    vp_lon, vp_lat, _ = load_geometry()
    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [s for s in ann if s in gt]

    # Query profiles (shared across all DB passes)
    queries = {}
    tlat = {}
    tlon = {}
    for sid in sids:
        g = gt[sid]
        tilt = np.array(g["cam_R_tilt"])
        mask, _ = mask_from_ann(ann[sid])
        pr = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=tilt, bin_deg=BIN_DEG
        )
        if pr["ok"] and len(pr["profile"]) >= 4:
            queries[sid] = pr["profile"]
            tlat[sid], tlon[sid] = g["true_lat"], g["true_lon"]
    print(f"Queries: {len(queries)}")

    pf = pq.ParquetFile(DB_PATH)
    configs = {
        "base": None,
        "bp(1,4)": (1, 4),
        "bp(2,8)": (2, 8),
        "bp(3,16)": (3, 16),
        "bp_fb(1,4)": None,  # feature-bundle on bandpass (1,4)
    }

    results = {c: {s: -np.inf for s in queries} for c in configs}
    best_vp = {c: {s: -1 for s in queries} for c in configs}

    t0 = time.time()
    n_chunks = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        chunk = chunk[::STRIDE]  # spatial stride (canonical honest matcher)
        N = len(chunk)
        chunk_base = n_chunks * CHUNK  # global VP idx of chunk row 0

        # baseline: raw values, plain NCC
        for s in queries:
            q = queries[s]
            c = best_corr_per_vp(q, chunk)
            k = int(np.argmax(c))
            if c[k] > results["base"][s]:
                results["base"][s] = c[k]
                best_vp["base"][s] = chunk_base + k * STRIDE

        for name, (s1, s2) in [
            ("bp(1,4)", (1, 4)),
            ("bp(2,8)", (2, 8)),
            ("bp(3,16)", (3, 16)),
        ]:
            bp = dog_bandpass(chunk.astype(np.float64), s1, s2)
            for s in queries:
                qbp = dog_bandpass(queries[s], s1, s2)
                c = best_corr_per_vp(qbp, bp)
                k = int(np.argmax(c))
                if c[k] > results[name][s]:
                    results[name][s] = c[k]
                    best_vp[name][s] = chunk_base + k * STRIDE

        # feature-bundle on bandpass (1,4)
        bp14 = dog_bandpass(chunk.astype(np.float64), 1, 4)
        for s in queries:
            qbp = dog_bandpass(queries[s], 1, 4)
            c = best_corr_fb(qbp, bp14)
            k = int(np.argmax(c))
            if c[k] > results["bp_fb(1,4)"][s]:
                results["bp_fb(1,4)"][s] = c[k]
                best_vp["bp_fb(1,4)"][s] = chunk_base + k * STRIDE

        n_chunks += 1
        if n_chunks % 25 == 0:
            print(f"  chunk {n_chunks} ({time.time() - t0:.0f}s)", flush=True)

    # Report
    def err_of(vp_idx, tl, tn):
        return geodesic((tl, tn), (vp_lat[vp_idx], vp_lon[vp_idx])).meters / 1000

    print("\n=== Full-DB honest top-1 (no candidate restriction) ===")
    for name in configs:
        errs = {s: err_of(best_vp[name][s], tlat[s], tlon[s]) for s in queries}
        e = np.array(list(errs.values()))
        print(
            f"{name:>10}: med={np.median(e):5.1f}km  <1km={int(np.sum(e < 1))}/{len(queries)}  "
            f"<5km={int(np.sum(e < 5))}/{len(queries)}  <10km={int(np.sum(e < 10))}/{len(queries)}"
        )
    print(f"\nTotal: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

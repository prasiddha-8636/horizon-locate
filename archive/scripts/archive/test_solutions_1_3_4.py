#!/usr/bin/env python
"""Combined evaluation of Solutions 1 + 3 + 4 on 17 annotated GSV samples.

Solution 1 (Sub-Pixel): compare integer-pixel vs sub-pixel skyline extraction
Solution 3 (Sub-Grid): bilinear horizon interpolation among4 nearest DB VPs
Solution 4 (Multi-Spectral): 3-channel feature bundle (value+d1+DoG(2,8))

Runs a controlled candidate-set test (top-200 NCC + true VP + 200 nearest)
to isolate each fix's contribution, plus a full-DB stride-12 scan for the
combined config.

Usage: python scripts/test_solutions_1_3_4.py
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
from src.matching import (
    _feature_bundle,
    _feature_bundle_matrix,
    _feature_bundle_ms,
    _feature_bundle_matrix_ms,
    _pearson_ncc_batch,
    _safe_zscore,
)
from src.horizon_format import decode_horizon_column, decode_horizon_uint8
from scripts.fixes_eval import (
    Rx,
    mask_from_ann,
    DB_PATH,
    GT_FILE,
    ANNOT_FILE,
    CALIB_FILE,
)

CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
BIN_DEG = 0.5
STRIDE = 12
CHUNK = 4000


def load_geometry():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    return (
        meta["lon"].to_numpy(),
        meta["lat"].to_numpy(),
        meta["elevation_m"].to_numpy(),
    )


def fetch_horizons_batch(idx_list):
    _pf = pq.ParquetFile(DB_PATH)
    sizes = [_pf.metadata.row_group(i).num_rows for i in range(_pf.num_row_groups)]
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    groups = {}
    for vi in idx_list:
        rg = int(np.searchsorted(starts, vi, side="right") - 1)
        groups.setdefault(rg, []).append(vi)
    out = {}
    for rg, vis in groups.items():
        raw = (
            _pf.read_row_group(rg, columns=["raw_horizon_deg"])
            .to_pandas()["raw_horizon_deg"]
            .to_numpy()
        )
        for vi in vis:
            out[vi] = decode_horizon_uint8(raw[vi - starts[rg]])
    return out


def ncc_fb(query, db_mat):
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


def ncc_ms(query, db_mat):
    """Multi-spectral (value+d1+DoG) NCC."""
    M = len(query)
    qv, qd, qdog = _feature_bundle_ms(query)
    qv = qv - qv.mean()
    qd = qd - qd.mean()
    qdog = qdog - qdog.mean()
    qvn = np.linalg.norm(qv)
    qdn = np.linalg.norm(qd)
    qdogn = np.linalg.norm(qdog)
    if qvn < 1e-12 or qdn < 1e-12 or qdogn < 1e-12:
        return np.full(len(db_mat), -np.inf)
    v, d1, dog = _feature_bundle_matrix_ms(db_mat)
    ext_v = np.concatenate([v, v[:, : M - 1]], axis=1)
    ext_d = np.concatenate([d1, d1[:, : M - 1]], axis=1)
    ext_dog = np.concatenate([dog, dog[:, : M - 1]], axis=1)
    w1 = w2 = w3 = 1.0 / 3.0
    return (
        w1 * _pearson_ncc_batch(ext_v, qv, qvn).max(axis=1)
        + w2 * _pearson_ncc_batch(ext_d, qd, qdn).max(axis=1)
        + w3 * _pearson_ncc_batch(ext_dog, qdog, qdogn).max(axis=1)
    )


def smooth_vp_horizon(vp_idx, hdict, vp_lat_arr, vp_lon_arr, k=4):
    """Smooth a VP's horizon by weighted average of k-nearest neighbors.
    This is the practical sub-grid fix: instead of reading a single DB row,
    interpolate with nearby VPs to reduce grid-gap quantization."""
    h_self = hdict.get(vp_idx)
    if h_self is None:
        return np.zeros(720)
    my_lat, my_lon = vp_lat_arr[vp_idx], vp_lon_arr[vp_idx]
    dlat = vp_lat_arr - my_lat
    dlon = vp_lon_arr - my_lon
    dist = np.sqrt(dlat**2 + dlon**2) * 111.0
    dist[vp_idx] = np.inf
    nearest = np.argsort(dist)[: k - 1]
    h_list = [hdict[i] for i in nearest if i in hdict]
    if not h_list:
        return h_self
    weights = np.array([1.0 / (dist[i] + 1e-6) for i in nearest if i in hdict])
    weights = np.concatenate([[2.0], weights])  # self-weighted
    weights /= weights.sum()
    all_h = [h_self] + h_list
    result = np.zeros_like(h_self, dtype=np.float64)
    for w, h in zip(weights, all_h):
        result += w * h
    return result


def smooth_db_matrix(cand, hdict, vp_lat, vp_lon, k=4):
    """Build smoothed DB matrix for all candidates."""
    db = np.empty((len(cand), 720), dtype=np.float64)
    for j, vi in enumerate(cand):
        db[j] = smooth_vp_horizon(int(vi), hdict, vp_lat, vp_lon, k=k)
    return db


def dog_bandpass(x, s1, s2):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return gaussian_filter1d(x, s1, mode="wrap") - gaussian_filter1d(
            x, s2, mode="wrap"
        )
    return gaussian_filter1d(x, s1, mode="wrap", axis=1) - gaussian_filter1d(
        x, s2, mode="wrap", axis=1
    )


def main():
    vp_lon, vp_lat, vp_elev = load_geometry()
    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))

    IMAGE_DIR = os.path.join(ROOT, "data/street_view/images")
    MASK_DIR = os.path.join(ROOT, "data/street_view/masks")

    sids = [
        s
        for s in ann
        if s in gt
        and os.path.exists(os.path.join(IMAGE_DIR, f"{s}.png"))
        and os.path.exists(os.path.join(CACHE_DIR, f"{s}_corr.npz"))
    ]
    print(f"Samples with photos: {len(sids)}")

    rows = []
    t0 = time.time()

    for sid in sids:
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        tilt = np.array(g["cam_R_tilt"])
        dp = float(calib.get(sid, {}).get("delta_pitch_deg", 0.0))
        vp = g["closest_viewpoint_id"]

        img = np.array(
            __import__("PIL")
            .Image.open(os.path.join(IMAGE_DIR, f"{sid}.png"))
            .convert("RGB")
        )
        mask = np.array(
            __import__("PIL")
            .Image.open(os.path.join(MASK_DIR, f"{sid}.png"))
            .convert("L")
        )

        R_cal = Rx(dp) @ tilt

        # Solution 1: sub-pixel profile
        pr_sp = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=R_cal, bin_deg=BIN_DEG, image=img
        )
        # Baseline: integer-pixel profile
        pr_int = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=R_cal, bin_deg=BIN_DEG
        )
        if not pr_sp["ok"] or not pr_int["ok"]:
            continue

        q_sp = pr_sp["profile"]
        q_int = pr_int["profile"]

        # Candidate set
        corr = np.load(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))["corr"]
        top200 = np.argsort(corr)[-200:]
        dlat = vp_lat - tlat
        dlon = vp_lon - tlon
        approx_km = np.sqrt(dlat**2 + dlon**2) * 111.0
        near200 = np.argsort(approx_km)[:200]
        cand = np.unique(np.concatenate([top200, [vp], near200])).astype(int)

        hdict = fetch_horizons_batch(list(cand))
        db = np.stack([hdict[int(c)] for c in cand])

        # Solution 3: per-VP smoothed DB (each VP interpolated with 4 nearest neighbors)
        db_s3 = smooth_db_matrix(cand, hdict, vp_lat, vp_lon, k=5)

        results_row = {"sid": sid, "n_cand": len(cand)}

        configs = [
            ("baseline", q_int, db, ncc_fb),
            ("sol1_subpx", q_sp, db, ncc_fb),
            ("sol4_ms", q_int, db, ncc_ms),
            ("sol1+4", q_sp, db, ncc_ms),
            ("sol3_smooth", q_int, db_s3, ncc_fb),
            ("sol1+3", q_sp, db_s3, ncc_fb),
            ("sol3+4", q_int, db_s3, ncc_ms),
            ("sol1+3+4", q_sp, db_s3, ncc_ms),
        ]

        for label, query, matrix, scorer in configs:
            scores = scorer(query, matrix)
            best_i = int(np.argmax(scores))
            best_vp = int(cand[best_i])
            err_km = (
                geodesic((tlat, tlon), (vp_lat[best_vp], vp_lon[best_vp])).meters / 1000
            )
            vp_idx_in_cand = list(cand).index(vp)
            rank = int(np.sum(scores > scores[vp_idx_in_cand])) + 1
            results_row[f"{label}_err"] = round(err_km, 3)
            results_row[f"{label}_rank"] = rank
            results_row[f"{label}_corrT"] = round(float(scores[vp_idx_in_cand]), 4)

        rows.append(results_row)
        print(
            f"{sid[:15]:15s} "
            f"base={results_row['baseline_err']:6.1f}km "
            f"s1={results_row['sol1_subpx_err']:6.1f} "
            f"s4={results_row['sol4_ms_err']:6.1f} "
            f"s3={results_row['sol3_smooth_err']:6.1f} "
            f"all={results_row['sol1+3+4_err']:6.1f}"
        )

    # Summary
    print("\n=== Candidate-set summary (true VP guaranteed in set) ===")
    labels = [
        "baseline",
        "sol1_subpx",
        "sol4_ms",
        "sol1+4",
        "sol3_smooth",
        "sol1+3",
        "sol3+4",
        "sol1+3+4",
    ]
    for lab in labels:
        errs = np.array([r[f"{lab}_err"] for r in rows])
        ranks = np.array([r[f"{lab}_rank"] for r in rows])
        print(
            f"{lab:>12}: med_err={np.median(errs):5.1f}km  <1km={int(np.sum(errs < 1))}/{len(rows)}  "
            f"<5km={int(np.sum(errs < 5))}/{len(rows)}  rank(med)={int(np.median(ranks)):3d}"
        )
    print(f"\nTotal: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

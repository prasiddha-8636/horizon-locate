#!/usr/bin/env python
"""Test bandpass-filtered Pearson NCC (BPN-NCC, Difference-of-Gaussians) and
Phase-Only Correlation (POC) against the baseline NCC on the 17 annotated GSV
samples.

Hypothesis (from the imposter post-mortem): the far-away imposter's horizon is
a smooth low-frequency wave that Pearson NCC rewards; the true VP carries
mid-frequency ridge detail that 30m-DEM micro-noise penalizes. Bandpassing both
query and DB through DoG(s1,s2) should strip the smooth imposter's energy while
retaining the true VP's shared mid-band structure.

Fast candidate-set test: per sample take the baseline top-200 NCC VPs + the
true VP + 200 nearest VPs, decode horizons, and re-rank by bandpassed NCC /
POC. Reports true-VP rank and top-1 error inside the candidate set.

Usage: python scripts/test_bandpass_ncc.py
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
from src.horizon_format import decode_horizon_uint8
from scripts.fixes_eval import Rx, mask_from_ann, DB_PATH, GT_FILE, ANNOT_FILE

CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
TOPS = 200
NEIGHBORS = 200
BIN_DEG = 0.5


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


def dog_bandpass(x, s1, s2):
    """Difference-of-Gaussians bandpass (circular)."""
    return gaussian_filter1d(x, s1, mode="wrap") - gaussian_filter1d(x, s2, mode="wrap")


def ncc_over_shifts(query, db_matrix):
    """Pearson NCC feature-bundle, best over circular shifts."""
    M = len(query)
    qv, qd = _feature_bundle(query)
    qv = qv - qv.mean()
    qd = qd - qd.mean()
    qvn = np.linalg.norm(qv)
    qdn = np.linalg.norm(qd)
    if qvn < 1e-12 or qdn < 1e-12:
        return np.full(len(db_matrix), -np.inf)
    v, d1 = _feature_bundle_matrix(db_matrix)
    ext_v = np.concatenate([v, v[:, : M - 1]], axis=1)
    ext_d = np.concatenate([d1, d1[:, : M - 1]], axis=1)
    comb = 0.5 * _pearson_ncc_batch(ext_v, qv, qvn).max(
        axis=1
    ) + 0.5 * _pearson_ncc_batch(ext_d, qd, qdn).max(axis=1)
    return comb


def ncc_over_shifts_raw(query, db_matrix):
    """Plain Pearson on the raw (bandpassed) values, best over shifts."""
    M = len(query)
    qz = query - query.mean()
    qn = np.linalg.norm(qz)
    if qn < 1e-12:
        return np.full(len(db_matrix), -np.inf)
    dz = db_matrix - db_matrix.mean(axis=1, keepdims=True)
    ext = np.concatenate([dz, dz[:, : M - 1]], axis=1)
    return _pearson_ncc_batch(ext, qz, qn).max(axis=1)


def poc_over_shifts(query, db_matrix):
    """Phase-only correlation over circular shifts on a sub-window query.

    Whitens the magnitude spectrum of both the circularly-extended DB row and
    the zero-padded query, then cross-correlates (phase-only) -> POC over shifts.
    """
    M = len(query)
    D = db_matrix.shape[1]
    L = D + M - 1  # extended length for circular cross-correlation
    q_pad = np.zeros(L, dtype=np.float64)
    q_pad[:M] = query - query.mean()
    fq = np.fft.rfft(q_pad)
    fq_white = fq / (np.abs(fq) + 1e-6)
    poc = np.empty(db_matrix.shape[0])
    for i, row in enumerate(db_matrix):
        ext = np.concatenate([row, row[: M - 1]]) if M > 1 else row
        fdb = np.fft.rfft(ext)
        fdb_white = fdb / (np.abs(fdb) + 1e-6)
        # cross-power over the circular shift dimension (first L bins)
        cross = fdb_white * np.conj(fq_white)
        c = np.fft.irfft(cross, n=L)
        poc[i] = c.max()
    return poc


def eval_sample(sid, gt, ann_vps, corr, vp_lat, vp_lon, vp_elev):
    g = gt[sid]
    tlat, tlon = g["true_lat"], g["true_lon"]
    tilt = np.array(g["cam_R_tilt"])
    mask, _ = mask_from_ann(ann_vps[sid])
    pr = extract_elevation_profile(
        mask, fov_y_deg=g["fov_y_deg"], r_tilt=tilt, bin_deg=BIN_DEG
    )
    if not pr["ok"]:
        return None
    q = pr["profile"]
    vp = g["closest_viewpoint_id"]

    # Candidate set: baseline top-200 + true VP + 200 nearest by geodesic
    top = np.argsort(corr)[-TOPS:]
    dlat = vp_lat - g["true_lat"]
    dlon = vp_lon - g["true_lon"]
    approx_km = np.sqrt(dlat**2 + dlon**2) * 111.0
    near = np.argsort(approx_km)[:NEIGHBORS]
    cand = np.unique(np.concatenate([top, [vp], near])).astype(int)
    cand = [int(c) for c in cand]

    hdict = fetch_horizons_batch(cand)
    db = np.stack([hdict[c] for c in cand])

    res = {"sid": sid, "true_vp": int(vp), "n_cand": len(cand)}
    for label, fn in [
        ("baseline", None),  # cached corr
        (
            "bp(1,4)",
            lambda a, b: ncc_over_shifts_raw(
                dog_bandpass(a, 1, 4), dog_bandpass(b, 1, 4)
            ),
        ),
        (
            "bp(2,8)",
            lambda a, b: ncc_over_shifts_raw(
                dog_bandpass(a, 2, 8), dog_bandpass(b, 2, 8)
            ),
        ),
        (
            "bp(3,16)",
            lambda a, b: ncc_over_shifts_raw(
                dog_bandpass(a, 3, 16), dog_bandpass(b, 3, 16)
            ),
        ),
        (
            "bp_fb(2,8)",
            lambda a, b: ncc_over_shifts(dog_bandpass(a, 2, 8), dog_bandpass(b, 2, 8)),
        ),
        ("poc", lambda a, b: poc_over_shifts(a, b)),
    ]:
        if label == "baseline":
            scores = corr[cand]
        else:
            scores = fn(q, db)
        best_i = int(np.argmax(scores))
        best_vp = cand[best_i]
        err_km = (
            geodesic((tlat, tlon), (vp_lat[best_vp], vp_lon[best_vp])).meters / 1000
        )
        rank = int(np.sum(scores > scores[list(cand).index(vp)])) + 1  # rank of true VP
        res[f"{label}_err_km"] = round(err_km, 3)
        res[f"{label}_trank"] = rank
        res[f"{label}_corrt"] = round(float(scores[list(cand).index(vp)]), 4)
    return res


def main():
    vp_lon, vp_lat, vp_elev = load_geometry()
    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [
        s
        for s in ann
        if s in gt and os.path.exists(os.path.join(CACHE_DIR, f"{s}_corr.npz"))
    ]

    rows = []
    for sid in sids:
        corr = np.load(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))["corr"]
        r = eval_sample(sid, gt, ann, corr, vp_lat, vp_lon, vp_elev)
        if r is None:
            continue
        rows.append(r)
        print(
            f"{sid[:20]:20s} n_cand={r['n_cand']:4d}  "
            f"base err={r['baseline_err_km']:7.1f} km  "
            f"bp(1,4) err={r['bp(1,4)_err_km']:7.1f}  "
            f"bp(2,8) err={r['bp(2,8)_err_km']:7.1f}  "
            f"bp(3,16) err={r['bp(3,16)_err_km']:7.1f}  "
            f"bp_fb err={r['bp_fb(2,8)_err_km']:7.1f}  "
            f"poc err={r['poc_err_km']:7.1f}"
        )

    print("\n=== Summary: median error (km) + true-VP rank (+ correlations) ===")
    methods = ["baseline", "bp(1,4)", "bp(2,8)", "bp(3,16)", "bp_fb(2,8)", "poc"]
    for m in methods:
        errs = np.array([r[f"{m}_err_km"] for r in rows])
        ranks = np.array([r[f"{m}_trank"] for r in rows])
        corrts = np.array([r[f"{m}_corrt"] for r in rows])
        print(
            f"{m:>10}: med_err={np.median(errs):6.1f}km  "
            f"<1km={int(np.sum(errs < 1))}/{len(rows)}  <5km={int(np.sum(errs < 5))}/{len(rows)}  "
            f"trueVP_rank(med)={int(np.median(ranks)):4d}  corrt(med)={np.median(corrts):.3f}"
        )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time() - t0:.0f}s")

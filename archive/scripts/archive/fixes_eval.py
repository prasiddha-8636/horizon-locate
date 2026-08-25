#!/usr/bin/env python
"""
Three fixes for GSV skyline matching:

1. Two-Layer Horizon Extraction (Far vs Near Ground via frequency split)
2. Inter-Peak Angular Spacing Fingerprint (Δθ tuples)
3. Parallax Ratio (dθ_near / dθ_far)

All work on EXISTING DB + query profiles — no DB regeneration needed.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import find_peaks
from scipy.fft import rfft, irfft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, _pearson_ncc_batch
from src.horizon_format import decode_horizon_column

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
CALIB_FILE = os.path.join(ROOT, "data/street_view/calibrated_ground_truth.json")
W, H_IMG = 1080, 720
CORRECT_M = 500.0
NCC_RADIUS_KM = 5.0


def Rx(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def mask_from_ann(points):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.unique(xs).size < 2:
        return None, None
    mask = np.zeros((H_IMG, W), dtype=np.uint8)
    ycol = np.full(W, H_IMG, dtype=np.int64)
    ii = 0
    for x in range(W):
        while ii < len(xs) - 1 and xs[ii + 1] <= x:
            ii += 1
        if xs[ii] <= x <= xs[-1]:
            ycol[x] = int(np.interp(x, xs, ys))
    for x in range(W):
        mask[min(H_IMG - 1, max(0, int(ycol[x]))) :, x] = 255
    return mask, ycol


def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2_arr))
        * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


# ============================================================
# FIX 1: Two-Layer Horizon Extraction
# ============================================================
def split_horizon_layers(profile, freq_cutoff_ratio=0.25):
    """
    Split profile into far-ground (low-freq) and near-ground (high-freq) layers.

    Args:
        profile: 1D elevation angle array
        freq_cutoff_ratio: fraction of FFT coefficients to keep for far-ground

    Returns:
        far_layer: low-frequency component (distant peaks)
        near_layer: high-frequency component (local ridges)
    """
    freq = rfft(profile)
    n_keep = max(4, int(len(freq) * freq_cutoff_ratio))  # keep lowest 25%
    far = irfft(freq[:n_keep], n=len(profile))
    near = profile - far
    return far, near


def match_two_layer(db_far, db_near, q_far, q_near, w_far=0.6, w_near=0.4):
    """Match using weighted combination of far and near layer NCC."""
    M = len(q_far)
    qv_f, qd_f = _feature_bundle(q_far)
    qv_n, qd_n = _feature_bundle(q_near)
    qv_fz, qd_fz = qv_f - np.mean(qv_f), qd_f - np.mean(qd_f)
    qv_nz, qd_nz = qv_n - np.mean(qv_n), qd_n - np.mean(qd_n)
    qvn_f, qdn_f = np.linalg.norm(qv_fz), np.linalg.norm(qd_fz)
    qvn_n, qdn_n = np.linalg.norm(qv_nz), np.linalg.norm(qd_nz)

    v_f, d1_f = _feature_bundle(db_far)
    v_n, d1_n = _feature_bundle(db_near)
    # Extend circularly for NCC
    ext_f = np.concatenate([v_f, v_f[: M - 1]])
    ext_d1_f = np.concatenate([d1_f, d1_f[: M - 1]])
    ext_n = np.concatenate([v_n, v_n[: M - 1]])
    ext_d1_n = np.concatenate([d1_n, d1_n[: M - 1]])

    ncc_f = 0.5 * _pearson_ncc_batch(
        ext_f[np.newaxis], qv_fz, qvn_f
    ) + 0.5 * _pearson_ncc_batch(ext_d1_f[np.newaxis], qd_fz, qdn_f)
    ncc_n = 0.5 * _pearson_ncc_batch(
        ext_n[np.newaxis], qv_nz, qvn_n
    ) + 0.5 * _pearson_ncc_batch(ext_d1_n[np.newaxis], qd_nz, qdn_n)

    return (w_far * ncc_f + w_near * ncc_n).max()


# ============================================================
# FIX 2: Inter-Peak Angular Spacing Fingerprint
# ============================================================
def extract_peak_fingerprint(profile, prominence=0.5, min_dist=5):
    """Extract peak positions and angular spacings Δθ tuple."""
    peaks, props = find_peaks(profile, prominence=prominence, distance=min_dist)
    if len(peaks) < 2:
        return None
    # Keep top 6 peaks by prominence
    if len(peaks) > 6:
        idx = np.argsort(props["prominences"])[-6:]
        peaks = peaks[idx]
    peaks = np.sort(peaks)
    spacings = np.diff(peaks) * 0.5  # convert bins to degrees
    return {
        "peaks": peaks,
        "elevations": profile[peaks],
        "spacings": spacings,
        "n_peaks": len(peaks),
    }


def match_fingerprint(db_profile, q_profile, min_peaks=2):
    """Match using peak spacing fingerprint similarity.

    Compares sub-sequences of inter-peak spacings (Δθ tuple) of the query
    against the DB profile's peaks, allowing different total peak counts.
    """
    db_fp = extract_peak_fingerprint(db_profile)
    q_fp = extract_peak_fingerprint(q_profile)

    if db_fp is None or q_fp is None:
        return 0.0
    if db_fp["n_peaks"] < min_peaks or q_fp["n_peaks"] < min_peaks:
        return 0.0

    db_spacings = db_fp["spacings"]
    q_spacings = q_fp["spacings"]

    # Slide the query spacing sequence across the DB spacing sequence
    # and compute correlation at each alignment (when lengths allow).
    best_score = 0.0
    n_q = len(q_spacings)
    n_db = len(db_spacings)

    if n_q > n_db:
        # Query has more peaks than DB: slide DB across query
        for start in range(n_q - n_db + 1):
            window = q_spacings[start : start + n_db]
            corr = np.corrcoef(db_spacings, window)[0, 1]
            if not np.isnan(corr):
                best_score = max(best_score, corr)
    else:
        for start in range(n_db - n_q + 1):
            window = db_spacings[start : start + n_q]
            corr = np.corrcoef(window, q_spacings)[0, 1]
            if not np.isnan(corr):
                best_score = max(best_score, corr)

    return max(0.0, best_score)


# ============================================================
# FIX 3: Parallax Ratio
# ============================================================
def compute_parallax_ratio(profile, freq_cutoff_ratio=0.25):
    """Compute dθ_near / dθ_far as a function of azimuth."""
    far, near = split_horizon_layers(profile, freq_cutoff_ratio)
    # Gradients (slopes)
    d_far = np.gradient(far)
    d_near = np.gradient(near)
    # Avoid division by zero
    ratio = np.zeros_like(profile)
    mask = np.abs(d_far) > 1e-6
    ratio[mask] = d_near[mask] / d_far[mask]
    return ratio


def match_parallax_ratio(db_profile, q_profile):
    """Match using parallax ratio profile similarity."""
    db_ratio = compute_parallax_ratio(db_profile)
    q_ratio = compute_parallax_ratio(q_profile)
    # Use NCC on ratio profiles (less sensitive to absolute elevation)
    M = len(q_profile)
    qv, qd = _feature_bundle(q_ratio)
    qvz, qdz = qv - np.mean(qv), qd - np.mean(qd)
    qvn, qdn = np.linalg.norm(qvz), np.linalg.norm(qdz)

    v, d1 = _feature_bundle(db_ratio)
    ve = np.concatenate([v, v[: M - 1]])
    de = np.concatenate([d1, d1[: M - 1]])

    ncc = 0.5 * _pearson_ncc_batch(ve[np.newaxis], qvz, qvn) + 0.5 * _pearson_ncc_batch(
        de[np.newaxis], qdz, qdn
    )
    return ncc[0].max()


# ============================================================
# Main Evaluation
# ============================================================
def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    vp_lon = meta["lon"].to_numpy()
    vp_lat = meta["lat"].to_numpy()
    vp_elev = meta["elevation_m"].to_numpy()
    N_vps = len(vp_lon)

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))
    sids = [s for s in ann if s in gt and ann[s] is not None and s in calib]

    pf = pq.ParquetFile(DB_PATH)

    results = {
        "ncc": [],
        "layer": [],
        "parallax": [],
        "combined": [],
    }

    t0 = time.time()
    for si, sid in enumerate(sids):
        ts = time.time()
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        tilt = np.array(g["cam_R_tilt"])
        dp_calib = float(calib[sid].get("delta_pitch_deg", 0.0))
        fov_y = g["fov_y_deg"]

        mask, ycol = mask_from_ann(ann[sid])
        if mask is None:
            continue

        R_cal = Rx(dp_calib) @ tilt
        pr = extract_elevation_profile(mask, fov_y_deg=fov_y, r_tilt=R_cal, bin_deg=0.5)
        if not pr["ok"]:
            continue
        q_profile = pr["profile"]
        M = len(q_profile)

        # Load cached NCC
        cache_file = os.path.join(CACHE_DIR, f"{sid}_corr.npz")
        if not os.path.exists(cache_file):
            continue
        cached = np.load(cache_file)
        corr = cached["corr"]

        dist_km = haversine_km(tlat, tlon, vp_lat, vp_lon)
        valid = dist_km <= NCC_RADIUS_KM
        masked_corr = corr.copy()
        masked_corr[~valid] = -np.inf
        bv = int(np.argmax(masked_corr))
        ncc_err = dist_km[bv] * 1000
        results["ncc"].append(ncc_err)

        # Pre-split query profile
        q_far, q_near = split_horizon_layers(q_profile)
        q_fingerprint = extract_peak_fingerprint(q_profile)
        q_ratio = compute_parallax_ratio(q_profile)

        # Decode horizons for top-200 VPs (re-ranking)
        K = min(200, len(masked_corr))
        top_idxs = np.argsort(masked_corr)[-K:]
        needed = set(int(i) for i in top_idxs)
        vp_far = {}
        vp_near = {}
        vp_fp = {}
        vp_ratio = {}
        cs = 0
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            raw = batch.to_pandas()["raw_horizon_deg"].to_numpy()
            chunk = decode_horizon_column(raw)
            N = len(chunk)
            for vi in needed:
                if cs <= vi < cs + N:
                    h = chunk[vi - cs]
                    f, n = split_horizon_layers(h)
                    vp_far[vi] = f
                    vp_near[vi] = n
                    vp_fp[vi] = extract_peak_fingerprint(h)
                    vp_ratio[vi] = compute_parallax_ratio(h)
            cs += N
            if len(vp_far) == len(needed):
                break

        # Compute all fix scores on top-200 NCC candidates
        ncc_scores = np.array([masked_corr[vi] for vi in top_idxs])
        layer_scores = np.zeros(K)
        parallax_scores = np.zeros(K)

        for j, vi in enumerate(top_idxs):
            if vi not in vp_far:
                continue
            # Reconstruct full profile from far + near layers
            db_profile = vp_far[vi] + vp_near[vi]
            # Fix 1: two-layer
            layer_scores[j] = match_two_layer(vp_far[vi], vp_near[vi], q_far, q_near)
            # Fix 3: parallax ratio
            parallax_scores[j] = match_parallax_ratio(db_profile, q_profile)

        # ---- Adaptive ensemble within top-10 ----
        # Each method picks its top-1 from the NCC top-10.
        # Ensemble picks whichever method's top-1 is closest to true (oracle-free).
        # Practical heuristic: pick method with highest raw score, breaking ties
        # by NCC preference. We compare absolute scores to decide.

        ncc_best_idx = int(np.argmax(ncc_scores))
        layer_best_idx = int(np.argmax(layer_scores))
        parallax_best_idx = int(np.argmax(parallax_scores))

        layer_best_vp = top_idxs[layer_best_idx]
        parallax_best_vp = top_idxs[parallax_best_idx]

        # Method confidence: score of top-1 minus mean (how much it stands out).
        layer_surge = layer_scores[layer_best_idx] - np.mean(layer_scores)
        par_surge = parallax_scores[parallax_best_idx] - np.mean(parallax_scores)

        # Default to NCC top-1, override if layer/parallax is clearly better.
        best_ens = top_idxs[ncc_best_idx]

        # Parallax wins if it picks a different VP with strong surge AND is competitive.
        if (
            parallax_best_vp != best_ens
            and par_surge > 0.005
            and parallax_scores[parallax_best_idx] >= layer_scores[layer_best_idx]
        ):
            best_ens = parallax_best_vp

        # Layer wins if it picks a different VP with strong surge AND beats parallax.
        if (
            layer_best_vp != best_ens
            and layer_surge > 0.005
            and layer_scores[layer_best_idx] > parallax_scores[parallax_best_idx]
        ):
            best_ens = layer_best_vp

        # Errors
        layer_err = dist_km[layer_best_vp] * 1000
        parallax_err = dist_km[parallax_best_vp] * 1000
        ens_err = dist_km[best_ens] * 1000

        results["layer"].append(layer_err)
        results["parallax"].append(parallax_err)
        results["combined"].append(ens_err)

        print(
            f"[{si + 1:2d}] {sid[:20]:<20} NCC={ncc_err / 1000:.1f}km "
            f"Layer={layer_err / 1000:.1f}km Parallax={parallax_err / 1000:.1f}km "
            f"Ens={ens_err / 1000:.1f}km ({time.time() - ts:.0f}s)",
            flush=True,
        )

    # Summary
    for key in ["ncc", "layer", "parallax", "combined"]:
        errs = np.array(results[key])
        if len(errs) > 0:
            print(
                f"{key:>12}: top1@500m={sum(e < 500 for e in errs)}/{len(errs)}  "
                f"median={np.median(errs) / 1000:.1f}km  "
                f"<1km={sum(e < 1000 for e in errs)}/{len(errs)}  "
                f"<2km={sum(e < 2000 for e in errs)}/{len(errs)}"
            )
    print(f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

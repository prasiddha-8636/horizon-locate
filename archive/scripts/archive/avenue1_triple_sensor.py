#!/usr/bin/env python
"""Avenue 1: Triple-sensor gating — simulating the mobile app.

Applies all three priors simultaneously:
  1. Pitch: calibrated Δp from calibrated_ground_truth.json (simulates IMU pitch)
  2. Heading: ±15° window around calibrated heading (simulates compass)
  3. Position: ≤5km radius around true location (simulates coarse cell)

Also tests each subset to show marginal sensor value.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, _pearson_ncc_batch, feature_bundle_matrix
from src.horizon_format import decode_horizon_column

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
CALIB_FILE = os.path.join(ROOT, "data/street_view/calibrated_ground_truth.json")
W, H = 1080, 720
CORRECT_M = 500.0


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
        return None
    mask = np.zeros((H, W), dtype=np.uint8)
    ycol = np.full(W, H, dtype=np.int64)
    ii = 0
    for x in range(W):
        while ii < len(xs) - 1 and xs[ii + 1] <= x:
            ii += 1
        if xs[ii] <= x <= xs[-1]:
            ycol[x] = int(np.interp(x, xs, ys))
    for x in range(W):
        mask[min(H - 1, max(0, int(ycol[x]))) :, x] = 255
    return mask


def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    """Vectorized haversine distance in km. (lat1,lon1) scalar, arrays for VP coords."""
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


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    vp_lon, vp_lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()
    N_vps = len(vp_lon)
    pf = pq.ParquetFile(DB_PATH)

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))
    sids = [s for s in ann if s in gt and ann[s] is not None and s in calib]

    # Configs: each sensor combination
    CONFIGS = {
        "baseline": {"pitch": False, "compass": None, "radius_km": None},
        "pitch_only": {"pitch": True, "compass": None, "radius_km": None},
        "compass_15": {"pitch": False, "compass": 15.0, "radius_km": None},
        "compass_30": {"pitch": False, "compass": 30.0, "radius_km": None},
        "radius_5km": {"pitch": False, "compass": None, "radius_km": 5.0},
        "radius_10km": {"pitch": False, "compass": None, "radius_km": 10.0},
        "pitch+compass": {"pitch": True, "compass": 15.0, "radius_km": None},
        "pitch+radius": {"pitch": True, "compass": None, "radius_km": 5.0},
        "compass+radius": {"pitch": False, "compass": 15.0, "radius_km": 5.0},
        "ALL THREE": {"pitch": True, "compass": 15.0, "radius_km": 5.0},
    }
    cfg_names = list(CONFIGS.keys())

    results = {
        c: {"errs": [], "ranks": [], "top1": 0, "top5_lt500": 0, "true_vp_in_top5": 0}
        for c in cfg_names
    }

    t0 = time.time()
    for si, sid in enumerate(sids):
        ts = time.time()
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp_true = int(g["closest_viewpoint_id"])
        tilt = np.array(g["cam_R_tilt"])
        dp_calib = float(calib[sid].get("delta_pitch_deg", 0.0))
        dth_calib = float(calib[sid].get("delta_heading_deg", 0.0))

        # Radius mask (vectorized haversine)
        dist_km = haversine_km(tlat, tlon, vp_lat, vp_lon)

        # Profile with calibrated pitch
        mask = mask_from_ann(ann[sid])
        if mask is None:
            continue
        R_cal = Rx(dp_calib) @ tilt
        pr = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=R_cal, bin_deg=0.5
        )
        if not pr["ok"]:
            continue
        profile = pr["profile"]
        M = len(profile)
        qv, qd = _feature_bundle(profile)
        qvz, qdz = qv - qv.mean(), qd - qd.mean()
        qvn, qdn = np.linalg.norm(qvz), np.linalg.norm(qdz)

        # Also extract with RAW pitch (no calibration) for baseline
        pr_raw = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=tilt, bin_deg=0.5
        )
        if not pr_raw["ok"]:
            continue
        prof_raw = pr_raw["profile"]
        M_raw = len(prof_raw)
        qv_r, qd_r = _feature_bundle(prof_raw)
        qvz_r, qdz_r = qv_r - qv_r.mean(), qd_r - qd_r.mean()
        qvn_r, qdn_r = np.linalg.norm(qvz_r), np.linalg.norm(qdz_r)

        # Scan: compute NCC for both calibrated and raw profiles in one pass
        scores_cal = np.zeros(N_vps, dtype=np.float64)
        offsets_cal = np.zeros(N_vps, dtype=np.int32)
        scores_raw = np.zeros(N_vps, dtype=np.float64)
        offsets_raw = np.zeros(N_vps, dtype=np.int32)
        cs = 0
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            chunk = decode_horizon_column(
                batch.to_pandas()["raw_horizon_deg"].to_numpy()
            )
            N = len(chunk)
            v, d1 = feature_bundle_matrix(chunk)
            ve_cal = np.concatenate([v, v[:, : M - 1]], axis=1)
            de_cal = np.concatenate([d1, d1[:, : M - 1]], axis=1)
            comb_cal = 0.5 * _pearson_ncc_batch(
                ve_cal, qvz, qvn
            ) + 0.5 * _pearson_ncc_batch(de_cal, qdz, qdn)
            scores_cal[cs : cs + N] = comb_cal.max(axis=1)
            offsets_cal[cs : cs + N] = comb_cal.argmax(axis=1)

            ve_raw = np.concatenate([v, v[:, : M_raw - 1]], axis=1)
            de_raw = np.concatenate([d1, d1[:, : M_raw - 1]], axis=1)
            comb_raw = 0.5 * _pearson_ncc_batch(
                ve_raw, qvz_r, qvn_r
            ) + 0.5 * _pearson_ncc_batch(de_raw, qdz_r, qdn_r)
            scores_raw[cs : cs + N] = comb_raw.max(axis=1)
            offsets_raw[cs : cs + N] = comb_raw.argmax(axis=1)
            cs += N

        # Compass mask
        bins = np.arange(720)
        exp_bin_cal = (dth_calib / 0.5) % 720
        compass_masks = {}
        for tol in [15.0, 30.0]:
            tb = tol / 0.5
            cd = np.minimum((bins - exp_bin_cal) % 720, (exp_bin_cal - bins) % 720)
            compass_masks[tol] = cd <= tb

        def eval_config(name, cfg):
            if cfg["pitch"]:
                sc = scores_cal.copy()
                off = offsets_cal.copy()
            else:
                sc = scores_raw.copy()
                off = offsets_raw.copy()

            if cfg["compass"] is not None:
                mask_arr = compass_masks[cfg["compass"]]
                sc_masked = np.where(mask_arr[off], sc, -np.inf)
            else:
                sc_masked = sc

            if cfg["radius_km"] is not None:
                sc_masked = np.where(dist_km <= cfg["radius_km"], sc_masked, -np.inf)

            bv = int(np.argmax(sc_masked))
            e = geodesic((vp_lat[bv], vp_lon[bv]), (tlat, tlon)).meters
            rank = int((sc_masked > sc_masked[vp_true]).sum())
            r_true = float(sc_masked[vp_true])

            # Top-5 within mask
            finite_mask = np.isfinite(sc_masked)
            n_finite = finite_mask.sum()
            top5_count = min(5, n_finite)
            if top5_count > 0:
                top5_vps = np.argpartition(sc_masked[finite_mask], -top5_count)[
                    -top5_count:
                ]
                top5_in_true = (
                    vp_true in [i for i, v in enumerate(finite_mask) if v][:n_finite]
                )
                # more direct check
                top5_vps_abs = np.where(finite_mask)[0][
                    np.argsort(-sc_masked[finite_mask])[:5]
                ]
                top5_in_top5 = vp_true in top5_vps_abs
            else:
                top5_in_top5 = False

            results[name]["errs"].append(e)
            results[name]["ranks"].append(rank)
            if e < CORRECT_M:
                results[name]["top1"] += 1
            if top5_in_top5:
                results[name]["true_vp_in_top5"] += 1

        for name, cfg in CONFIGS.items():
            eval_config(name, cfg)

        # Print per-sample
        be = eval_config  # just to show one key config
        e_base = results["baseline"]["errs"][-1]
        e_triple = results["ALL THREE"]["errs"][-1]
        r_triple = results["ALL THREE"]["ranks"][-1]
        in5 = (
            "IN-TOP5"
            if results["ALL THREE"]["true_vp_in_top5"]
            > (si if si > 0 else 0) - sum(1 for s in cfg_names if s != "ALL THREE")
            else ""
        )
        print(
            f"[{si + 1:2d}] {sid[:20]:<20}"
            f"  base={e_base / 1000:6.1f}km"
            f"  p={results['pitch_only']['errs'][-1] / 1000:6.1f}"
            f"  c15={results['compass_15']['errs'][-1] / 1000:6.1f}"
            f"  r5={results['radius_5km']['errs'][-1] / 1000:6.1f}"
            f"  p+c={results['pitch+compass']['errs'][-1] / 1000:6.1f}"
            f"  ALL={e_triple / 1000:6.1f}km r{r_triple:>6}"
            f"  ({time.time() - ts:.0f}s)",
            flush=True,
        )

    # ---- Summary table ----
    print(f"\n{'=' * 110}")
    print(
        f"{'config':<18} {'N':>3} {'top1@500m':>10} {'top1@5km':>9} {'median':>8} {'med_rank':>9} {'in_top5':>8}"
    )
    print("-" * 80)
    for name in cfg_names:
        r = results[name]
        e = np.array(r["errs"])
        rk = np.array(r["ranks"])
        N = len(e)
        t1 = sum(x < CORRECT_M for x in e)
        t5 = sum(x < 5000 for x in e)
        print(
            f"{name:<18} {N:3d} {t1:4d}/{N:<4d} {t5:4d}/{N:<4d}"
            f" {np.median(e) / 1000:6.1f}km {int(np.median(rk)):>9d}"
            f" {r['true_vp_in_top5']:3d}/{N:<4d}"
        )
    print(f"\nTotal: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

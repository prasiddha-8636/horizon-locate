#!/usr/bin/env python
"""High-precision GSV evaluation + failure taxonomy on the 18 annotated samples.

Protocol:
  - Profile extracted with CALIBRATED pitch (delta_pitch_deg from
    calibrated_ground_truth.json) — simulates an app with IMU pitch.
  - Azimuth prior: circular search centered on the calibrated heading
    (delta_heading_deg) at tolerances ±15°, ±30°, and unconstrained.
    NOTE: the calibrated heading is derived from the true VP, so these
    bounded columns are a CEILING (calibrated-compass) reading; the
    unconstrained column is the honest no-compass result.
  - One scan pass per sample computes all configs (masked-window bests).

Output: console report table + JSON in data/eval/high_precision_gsv_report.json
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
OUT_JSON = os.path.join(ROOT, "data/eval/high_precision_gsv_report.json")
W, H = 1080, 720
TOLS = [15.0, 30.0, None]  # None = unconstrained
RELIEF_THRESHOLD = 2.5  # valid-mountain subset: profile std >= this
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
    return mask, ycol


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()
    pf = pq.ParquetFile(DB_PATH)
    first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
    L = int(360.0 / BIN_DEG)

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))
    sids = [s for s in ann if s in gt and ann[s] is not None and s in calib]

    def label(tol):
        return "unconstrained" if tol is None else f"±{tol:.0f}°"

    configs = [label(t) for t in TOLS]
    samples = {}

    t0 = time.time()
    for si, sid in enumerate(sids):
        ts = time.time()
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp_true = int(g["closest_viewpoint_id"])
        tilt = np.array(g["cam_R_tilt"])
        dp = float(calib[sid].get("delta_pitch_deg", 0.0))
        dth = float(calib[sid].get("delta_heading_deg", 0.0))
        r_cal = float(calib[sid].get("r_calibrated", 0.0))
        r_base = float(calib[sid].get("r_baseline", 0.0))

        mask, ycol = mask_from_ann(ann[sid])
        if mask is None:
            print(f"[{si + 1}] {sid} MASK_FAIL", flush=True)
            continue
        pr = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=Rx(dp) @ tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            print(f"[{si + 1}] {sid} PROFILE_FAIL", flush=True)
            continue
        profile = pr["profile"]
        M = len(profile)
        qv, qd = _feature_bundle(profile)
        qvz, qdz = qv - qv.mean(), qd - qd.mean()
        qvn, qdn = np.linalg.norm(qvz), np.linalg.norm(qdz)

        # window-mask for each tolerance (circular, in bins)
        bins = np.arange(L)
        exp_bin = (dth / BIN_DEG) % L
        masks = {}
        for tol in TOLS:
            if tol is None:
                masks[label(tol)] = None
            else:
                tb = tol / BIN_DEG
                cd = np.minimum((bins - exp_bin) % L, (exp_bin - bins) % L)
                masks[label(tol)] = cd <= tb

        scores = {c: np.zeros(len(lon), dtype=np.float64) for c in configs}
        offs = {c: np.zeros(len(lon), dtype=np.int32) for c in configs}
        cs = 0
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            chunk = decode_horizon_column(
                batch.to_pandas()["raw_horizon_deg"].to_numpy()
            )
            N = len(chunk)
            v, d1 = feature_bundle_matrix(chunk)
            ve = np.concatenate([v, v[:, : M - 1]], axis=1)
            de = np.concatenate([d1, d1[:, : M - 1]], axis=1)
            cv = _pearson_ncc_batch(ve, qvz, qvn)
            cd = _pearson_ncc_batch(de, qdz, qdn)
            comb = 0.5 * cv + 0.5 * cd  # (N, L)

            for c in configs:
                if masks[c] is None:
                    cm = comb
                else:
                    cm = np.where(masks[c][np.newaxis, :], comb, -np.inf)
                off = np.argmax(cm, axis=1)
                scores[c][cs : cs + N] = cm[np.arange(N), off]
                offs[c][cs : cs + N] = off
            cs += N

        # per-config results
        sample = {
            "sid": sid,
            "profile_std": float(profile.std()),
            "profile_range": float(profile.max() - profile.min()),
            "grad_mean": float(np.abs(np.gradient(profile)).mean()),
            "median_boundary_row": float(np.median(ycol)),
            "delta_pitch": dp,
            "delta_heading": dth,
            "r_calibrated": r_cal,
            "r_baseline": r_base,
            "vp_true": vp_true,
            "configs": {},
        }
        for c in configs:
            sc = scores[c]
            top5 = np.argpartition(-sc, 5)[:5]
            top5_errs = sorted(
                geodesic((lat[i], lon[i]), (tlat, tlon)).meters for i in top5
            )
            bv = int(np.argmax(sc))
            e1 = geodesic((lat[bv], lon[bv]), (tlat, tlon)).meters
            rank = int((sc > sc[vp_true]).sum())
            sample["configs"][c] = {
                "top1_vp": bv,
                "top1_err_m": float(e1),
                "top5_errs_m": [float(x) for x in top5_errs],
                "top5_min_err_m": float(min(top5_errs)),
                "rank_true": int(rank),
                "r_true": float(sc[vp_true]),
                "r_best": float(sc[bv]),
            }
        samples[sid] = sample
        s0 = sample["configs"]["unconstrained"]
        line = (
            f"[{si + 1:2d}] {sid[:22]:<22} std={sample['profile_std']:5.2f}"
            f"  unc: {s0['top1_err_m'] / 1000:6.1f}km r{s0['rank_true']:>6}"
            f"  ±15: {sample['configs']['±15°']['top1_err_m'] / 1000:6.1f}km"
            f"  ±30: {sample['configs']['±30°']['top1_err_m'] / 1000:6.1f}km"
            f"  r@true={s0['r_true']:.2f}/{s0['r_best']:.2f}  ({time.time() - ts:.0f}s)"
        )
        print(line, flush=True)

    # ---- aggregate report ----
    def agg(subset):
        out = {}
        for c in configs:
            errs = [samples[s]["configs"][c]["top1_err_m"] for s in subset]
            top5 = [samples[s]["configs"][c]["top5_min_err_m"] for s in subset]
            ranks = [samples[s]["configs"][c]["rank_true"] for s in subset]
            out[c] = {
                "N": len(subset),
                "top1_acc_lt500m": sum(e < CORRECT_M for e in errs),
                "top5_acc_lt500m": sum(e < CORRECT_M for e in top5),
                "top1_acc_lt5km": sum(e < 5000 for e in errs),
                "median_err_m": float(np.median(errs)),
                "median_rank_true": int(np.median(ranks)),
            }
        return out

    all_sids = list(samples.keys())
    valid_sids = [s for s in all_sids if samples[s]["profile_std"] >= RELIEF_THRESHOLD]

    def print_table(title, subset):
        a = agg(subset)
        print(f"\n=== {title} (N={len(subset)}) ===")
        print(
            f"{'config':<16} {'top1@500m':>10} {'top5@500m':>10} {'top1@5km':>9} {'med err':>9} {'med rank':>9}"
        )
        for c in configs:
            r = a[c]
            print(
                f"{c:<16} {r['top1_acc_lt500m']:4d}/{r['N']:<4d}"
                f" {r['top5_acc_lt500m']:4d}/{r['N']:<4d}"
                f" {r['top1_acc_lt5km']:4d}/{r['N']:<4d}"
                f" {r['median_err_m'] / 1000:7.1f}km {r['median_rank_true']:>9d}"
            )
        return a

    a_all = print_table("ALL samples (calibrated pitch + heading prior)", all_sids)
    a_valid = print_table(
        f"VALID MOUNTAIN subset (std>={RELIEF_THRESHOLD:.1f}°)", valid_sids
    )

    # ---- failure taxonomy ----
    print(f"\n=== FAILURE TAXONOMY (per non-Top-1 sample, config: ±15°) ===")
    tax_cfg = "±15°"
    print(
        f"{'sid':<24} {'reason':<44} {'std':>5} {'r_true':>6} {'r_best':>6} {'top1km':>7}"
    )
    tax_rows = []
    for s in sorted(
        samples.values(), key=lambda x: x["configs"][tax_cfg]["top1_err_m"]
    ):
        c = s["configs"][tax_cfg]
        if c["top1_err_m"] < CORRECT_M:
            reason = "TOP-1 HIT"
        else:
            if s["profile_std"] < RELIEF_THRESHOLD:
                reason = "LOW RELIEF (std < 2.5°)"
            elif c["r_true"] < 0.55:
                reason = "WEAK true-VP match after calibration (r<0.55)"
            elif c["r_best"] - c["r_true"] > 0.10:
                reason = "IMPOSTER shape mimicry (r_best - r_true > 0.10)"
            elif abs(s["delta_heading"]) > 100:
                reason = "LARGE heading prior (calib Δθ>100°)"
            else:
                reason = "AMBIENT ambiguity (calibrated, still lost)"
            tax_rows.append((s["sid"], reason))
        print(
            f"{s['sid'][:24]:<24} {reason:<44} {s['profile_std']:5.2f}"
            f" {c['r_true']:6.3f} {c['r_best']:6.3f} {c['top1_err_m'] / 1000:7.1f}"
        )

    json.dump(
        {
            "configs": configs,
            "all": a_all,
            "valid": a_valid,
            "samples": samples,
            "taxonomy": tax_rows,
        },
        open(OUT_JSON, "w"),
        indent=2,
    )
    print(f"\nReport saved → {OUT_JSON}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

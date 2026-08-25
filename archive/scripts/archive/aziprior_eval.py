#!/usr/bin/env python
"""Azimuth-prior scan: restrict each VP's shift window to the metadata
crop heading (true_heading_deg) ± tolerance. Honest prior (no GT peek):
the heading was chosen at crop time by maximizing DB correlation at the
nearest VP (skyline_crop.py). Compares against the perfect-azimuth ceiling
(median 9.4 km) and the free-shift baseline (13.4 km)."""

import json, os, sys, pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, feature_bundle_matrix, _pearson_ncc_batch
from src.horizon_format import decode_horizon_column

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
W, H = 1080, 720
RES_FILE = os.path.join(CACHE_DIR, "aziprior_results.pkl")
TOLS = [20, 30, 45, 60, 90]


def mask_from_points(points):
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


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()

    pf = pq.ParquetFile(DB_PATH)
    first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [
        sid
        for sid in ann
        if sid in gt
        and ann[sid] is not None
        and os.path.exists(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))
    ]

    done = {}
    if os.path.exists(RES_FILE):
        done = pickle.load(open(RES_FILE, "rb"))
        print(f"resuming: {len(done)}", flush=True)

    variants = [f"az{t}" for t in TOLS] + ["ncc"]
    results = {v: {"errs": [], "ranks": [], "n": 0} for v in variants}

    for si, sid in enumerate(sids):
        if sid in done:
            print(f"[{si + 1}] {sid}: cached", flush=True)
            continue
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp_true = int(g["closest_viewpoint_id"])
        hdg = float(g.get("true_heading_deg", np.nan))
        if not np.isfinite(hdg):
            print(f"[{si + 1}] {sid}: no heading", flush=True)
            continue
        fov = g["fov_y_deg"]
        tilt = np.array(g["cam_R_tilt"])
        mask = mask_from_points(ann[sid])
        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            print(f"[{si + 1}] {sid}: profile fail", flush=True)
            continue
        profile = pr["profile"]
        M = len(profile)
        qv, qd = _feature_bundle(profile)
        qvz = qv - qv.mean()
        qdz = qd - qd.mean()
        L = int(360.0 / BIN_DEG)

        bins = np.arange(L)
        exp_bin = (hdg / BIN_DEG) % L
        masks = {}
        for t in TOLS:
            tol_bins = t / BIN_DEG
            cd = np.minimum((bins - exp_bin) % L, (exp_bin - bins) % L)
            masks[t] = cd <= tol_bins

        scores = {t: np.zeros(len(lon), dtype=np.float64) for t in TOLS}
        ncc_all = np.zeros(len(lon), dtype=np.float64)
        cs = 0
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            chunk = decode_horizon_column(
                batch.to_pandas()["raw_horizon_deg"].to_numpy()
            )
            v, d1 = feature_bundle_matrix(chunk)
            N = len(chunk)
            ve = np.concatenate([v, v[:, : M - 1]], axis=1)
            de = np.concatenate([d1, d1[:, : M - 1]], axis=1)
            cv = _pearson_ncc_batch(ve, qvz, np.linalg.norm(qvz))
            cd = _pearson_ncc_batch(de, qdz, np.linalg.norm(qdz))
            comb = 0.5 * cv + 0.5 * cd
            ncc_all[cs : cs + N] = comb.max(axis=1)
            for t in TOLS:
                cm = np.where(masks[t][np.newaxis, :], comb, -np.inf)
                scores[t][cs : cs + N] = cm.max(axis=1)
            cs += N

        line = f"[{si + 1}] {sid[:18]:<18} hdg={hdg:+7.1f}"
        for t in TOLS:
            bv = int(np.argmax(scores[t]))
            e = geodesic((lat[bv], lon[bv]), (tlat, tlon)).meters
            rank = int((scores[t] > scores[t][vp_true]).sum())
            results[f"az{t}"]["errs"].append(e)
            results[f"az{t}"]["ranks"].append(rank)
            line += f" az{t}: {e / 1000:6.1f}km r{rank:>5}"
        bv = int(np.argmax(ncc_all))
        e = geodesic((lat[bv], lon[bv]), (tlat, tlon)).meters
        rank = int((ncc_all > ncc_all[vp_true]).sum())
        results["ncc"]["errs"].append(e)
        results["ncc"]["ranks"].append(rank)
        line += f" ncc: {e / 1000:6.1f}km r{rank:>5}"
        print(line, flush=True)

        done[sid] = True
        pickle.dump(done, open(RES_FILE, "wb"))

    print("\n" + "=" * 90, flush=True)
    print(
        f"{'variant':<8} {'N':>4} {'top1@500m':>12} {'median':>10} {'<1km':>5} {'<5km':>5} {'<10km':>6} {'med_rank':>9}",
        flush=True,
    )
    for v in variants:
        r = results[v]
        if not r["errs"]:
            continue
        errs = np.array(r["errs"])
        med = np.median(errs)
        t1 = sum(x < 500 for x in errs)
        lt1 = sum(x < 1000 for x in errs)
        lt5 = sum(x < 5000 for x in errs)
        lt10 = sum(x < 10000 for x in errs)
        mr = int(np.median(r["ranks"]))
        print(
            f"{v:<8} {len(errs):4d} {t1:5d}/{len(errs):<5d} {med / 1000:8.1f}km {lt1:5d} {lt5:5d} {lt10:6d} {mr:9d}",
            flush=True,
        )


if __name__ == "__main__":
    main()

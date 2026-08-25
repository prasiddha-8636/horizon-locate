#!/usr/bin/env python
"""Test absolute elevation penalty on 17 annotated GSV samples.

Optimized: NCC matrix computed once per chunk; elevation penalty
applied per gamma as a post-hoc scalar adjustment per VP.
"""

import json, os, sys, time
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
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
W, H = 1080, 720
GAMMAS = [0.0, 0.02, 0.05, 0.10]
MAX_DIFF = 10.0


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
        s
        for s in ann
        if s in gt
        and ann[s] is not None
        and os.path.exists(os.path.join(IMAGES_DIR, f"{s}.png"))
    ]

    print(f"VPs: {len(lon)}, BIN_DEG: {BIN_DEG}, samples: {len(sids)}")
    L = int(360.0 / BIN_DEG)
    print(f"Gammas: {GAMMAS}, max_diff: {MAX_DIFF}°, L={L}\n")

    results = {g: {"errs": [], "ranks": [], "true_rs": []} for g in GAMMAS}

    t0 = time.time()
    for si, sid in enumerate(sids):
        ts = time.time()
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp_true = int(g["closest_viewpoint_id"])
        fov = g["fov_y_deg"]
        tilt = np.array(g["cam_R_tilt"])

        mask = mask_from_points(ann[sid])
        if mask is None:
            continue
        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            continue
        profile = pr["profile"]
        M = len(profile)
        query_mean = float(np.mean(profile))
        qv, qd = _feature_bundle(profile)
        qvz = qv - qv.mean()
        qdz = qd - qd.mean()
        qvn = np.linalg.norm(qvz)
        qdn = np.linalg.norm(qdz)

        N_total = len(lon)
        combined_all = np.zeros(N_total, dtype=np.float64)
        offset_all = np.zeros(N_total, dtype=np.int32)
        win_mean_all = np.zeros(N_total, dtype=np.float64)
        cs = 0

        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            raw_chunk = batch.to_pandas()["raw_horizon_deg"].to_numpy()
            chunk_deg = decode_horizon_column(raw_chunk)
            N = len(chunk_deg)
            v, d1 = feature_bundle_matrix(chunk_deg)
            ve = np.concatenate([v, v[:, : M - 1]], axis=1)
            de = np.concatenate([d1, d1[:, : M - 1]], axis=1)
            cv = _pearson_ncc_batch(ve, qvz, qvn)
            cd = _pearson_ncc_batch(de, qdz, qdn)
            comb = 0.5 * cv + 0.5 * cd  # (N, L)

            best_off = np.argmax(comb, axis=1)
            best_cor = comb[np.arange(N), best_off]

            # window mean of raw elevation
            db_ext = np.concatenate([chunk_deg, chunk_deg[:, : M - 1]], axis=1)
            cum = np.concatenate(
                [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext, axis=1)], axis=1
            )
            wm = (cum[:, M : M + L] - cum[:, :L]) / M
            matched_means = wm[np.arange(N), best_off]

            combined_all[cs : cs + N] = best_cor
            offset_all[cs : cs + N] = best_off
            win_mean_all[cs : cs + N] = matched_means
            cs += N

        elev_diff = np.abs(query_mean - win_mean_all)
        true_elev_diff = elev_diff[vp_true]

        for gamma in GAMMAS:
            penalty = gamma * elev_diff
            penalty = np.where(elev_diff <= MAX_DIFF, penalty, np.inf)
            scores = combined_all - penalty
            bv = int(np.argmax(scores))
            e = geodesic((lat[bv], lon[bv]), (tlat, tlon)).meters
            rank = int((scores > scores[vp_true]).sum())
            true_r = float(scores[vp_true])
            results[gamma]["errs"].append(e)
            results[gamma]["ranks"].append(rank)
            results[gamma]["true_rs"].append(true_r)

        base_e = geodesic(
            (lat[int(np.argmax(combined_all))], lon[int(np.argmax(combined_all))]),
            (tlat, tlon),
        ).meters
        best_g = max(GAMMAS[1:], key=lambda ga: results[ga]["true_rs"][-1])
        best_e = geodesic(
            (
                lat[
                    int(
                        np.argmax(
                            combined_all
                            - np.where(
                                elev_diff <= MAX_DIFF, best_g * elev_diff, np.inf
                            )
                        )
                    )
                ],
                lon[
                    int(
                        np.argmax(
                            combined_all
                            - np.where(
                                elev_diff <= MAX_DIFF, best_g * elev_diff, np.inf
                            )
                        )
                    )
                ],
            ),
            (tlat, tlon),
        ).meters

        elapsed = time.time() - ts
        print(
            f"[{si + 1:2d}] {sid[:22]:<22}"
            f"  base: {base_e / 1000:6.1f}km"
            f"  g={best_g:.2f}: {best_e / 1000:6.1f}km"
            f"  Δelev@true={true_elev_diff:.1f}°"
            f"  ({elapsed:.0f}s)"
        )

    print(f"\n{'=' * 100}")
    print(
        f"{'gamma':>6} {'N':>3} {'top1@500m':>10} {'median':>8} {'<1km':>5} {'<5km':>5} {'<10km':>6} {'med_rank':>9} {'med_true_r':>10}"
    )
    print("-" * 80)
    for gamma in GAMMAS:
        r = results[gamma]
        e = np.array(r["errs"])
        ranks = np.array(r["ranks"])
        tr = np.array(r["true_rs"])
        t1 = sum(x < 500 for x in e)
        lt1 = sum(x < 1000 for x in e)
        lt5 = sum(x < 5000 for x in e)
        lt10 = sum(x < 10000 for x in e)
        print(
            f"{gamma:6.2f} {len(e):3d} {t1:5d}/{len(e):<4d}"
            f" {np.median(e) / 1000:6.1f}km {lt1:5d} {lt5:5d} {lt10:6d}"
            f" {int(np.median(ranks)):>9d} {np.median(tr):10.4f}"
        )

    print(f"\nTotal: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

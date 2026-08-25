#!/usr/bin/env python
"""2D Chamfer Stage-2 re-ranker for GSV skyline matching.

Stage 1: NCC over 1.3M VPs returns Top-5.
Stage 2: For each candidate, render the DB horizon as a full-resolution
         binary edge image (1080x720), aligned to the query by the NCC shift,
         then compute Chamfer distance against the query annotation edge.
         Lower Chamfer = better 2D geometric fit → re-rank.
"""

import json
import os
import sys
import time

import cv2
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
BIN_DEG = 0.5
L = 720


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


def horizon_to_edge_image(elev_deg, shift_bins, profile_len):
    """Render a 720-bin DB horizon as a (H, W) binary edge image.

    Maps azimuth bins → image columns (after applying shift) and
    elevation angles → image rows. Only the profile_len central columns
    are active (matching the query FOV).
    """
    img = np.zeros((H, W), dtype=np.uint8)
    start_col = (W - profile_len) // 2
    for i in range(profile_len):
        az_bin = (i + shift_bins) % L
        elev = float(elev_deg[az_bin])
        row = int(np.clip(H - 1 - elev / 90.0 * H, 0, H - 1))
        col = start_col + i
        if 0 <= col < W:
            img[row, col] = 255
    return img


def chamfer_distance(img_q, img_db):
    """Symmetric Chamfer distance between two binary edge images."""
    _, img_q = cv2.threshold(img_q, 1, 255, cv2.THRESH_BINARY)
    _, img_db = cv2.threshold(img_db, 1, 255, cv2.THRESH_BINARY)
    dt_q = cv2.distanceTransform(255 - img_q, cv2.DIST_L2, 5)
    dt_d = cv2.distanceTransform(255 - img_db, cv2.DIST_L2, 5)
    valid_q = img_q > 0
    valid_d = img_db > 0
    if not valid_q.any() or not valid_d.any():
        return np.inf
    d_q = dt_q[valid_d].mean()
    d_d = dt_d[valid_q].mean()
    return 0.5 * (d_q + d_d)


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()
    N_vps = len(lon)
    pf = pq.ParquetFile(DB_PATH)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))
    sids = [s for s in ann if s in gt and ann[s] is not None and s in calib]

    # Preload all DB horizons for fast lookup
    print("Preloading DB horizons...", flush=True)
    t_preload = time.time()
    all_horizons = np.zeros((N_vps, L), dtype=np.uint8)
    for rg in range(pf.num_row_groups):
        b = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
        raw = b["raw_horizon_deg"].to_numpy()
        decoded = decode_horizon_column(raw)
        start = int(rg_starts[rg])
        all_horizons[start : start + len(decoded)] = np.array(decoded, dtype=np.uint8)
    print(f"Preloaded {N_vps} horizons in {time.time() - t_preload:.0f}s", flush=True)

    # Query annotation edge image (static per sample)
    results = {}
    t0 = time.time()

    for si, sid in enumerate(sids):
        ts = time.time()
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp_true = int(g["closest_viewpoint_id"])
        tilt = np.array(g["cam_R_tilt"])
        dp = float(calib[sid].get("delta_pitch_deg", 0.0))
        dth = float(calib[sid].get("delta_heading_deg", 0.0))

        mask, ycol = mask_from_ann(ann[sid])
        if mask is None:
            continue
        pr = extract_elevation_profile(
            mask, fov_y_deg=g["fov_y_deg"], r_tilt=Rx(dp) @ tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            continue
        profile = pr["profile"]
        M = len(profile)
        start_az = pr.get("start_az", 0.0)
        qv, qd = _feature_bundle(profile)
        qvz, qdz = qv - qv.mean(), qd - qd.mean()
        qvn, qdn = np.linalg.norm(qvz), np.linalg.norm(qdz)

        # Stage-1: NCC scan
        scores = np.zeros(N_vps, dtype=np.float64)
        offsets = np.zeros(N_vps, dtype=np.int32)
        cs = 0
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            raw = batch.to_pandas()["raw_horizon_deg"].to_numpy()
            chunk = decode_horizon_column(raw)
            N = len(chunk)
            v, d1 = feature_bundle_matrix(chunk)
            ve = np.concatenate([v, v[:, : M - 1]], axis=1)
            de = np.concatenate([d1, d1[:, : M - 1]], axis=1)
            cv_s = _pearson_ncc_batch(ve, qvz, qvn)
            cd_s = _pearson_ncc_batch(de, qdz, qdn)
            comb = 0.5 * cv_s + 0.5 * cd_s
            scores[cs : cs + N] = comb.max(axis=1)
            offsets[cs : cs + N] = comb.argmax(axis=1)
            cs += N

        top5_idx = np.argpartition(-scores, 5)[:5]
        top5_idx = top5_idx[np.argsort(-scores[top5_idx])]

        # Query edge from annotation
        img_q = np.zeros((H, W), dtype=np.uint8)
        for x in range(W):
            img_q[max(0, min(H - 1, int(ycol[x]))), x] = 255

        # Stage-2: Chamfer for each Top-5 candidate
        chamfer_scores = np.zeros(5, dtype=np.float64)
        ncc_scores = np.zeros(5, dtype=np.float64)
        for rank, vp in enumerate(top5_idx):
            h = all_horizons[vp]
            shift = int(offsets[vp])
            img_db = horizon_to_edge_image(h, shift, M)
            chamfer_scores[rank] = chamfer_distance(img_q, img_db)
            ncc_scores[rank] = scores[vp]

        # Re-rank: combine NCC and Chamfer (lower Chamfer = better)
        # Normalize Chamfer to [0,1] range
        ch_min, ch_max = chamfer_scores.min(), chamfer_scores.max()
        ch_range = ch_max - ch_min if ch_max > ch_min else 1.0
        ch_norm = (chamfer_scores - ch_min) / ch_range
        combined = 0.5 * ncc_scores / ncc_scores.max() + 0.5 * (1.0 - ch_norm)
        rerank_order = np.argsort(-combined)
        top5_reranked = top5_idx[rerank_order]

        # Compute errors
        def err_m(vp):
            return geodesic((lat[vp], lon[vp]), (tlat, tlon)).meters

        s1_vp = top5_idx[0]
        s2_vp = top5_reranked[0]
        e1 = err_m(s1_vp)
        e2 = err_m(s2_vp)
        e_true_vp = err_m(vp_true)

        s1_top1 = e1 < CORRECT_M
        s2_top1 = e2 < CORRECT_M
        promoted = "PROMOTED" if s2_vp != s1_vp else "UNCHANGED"
        improved = "BETTER" if e2 < e1 else ("WORSE" if e2 > e1 else "SAME")

        print(
            f"[{si + 1:2d}] {sid[:20]:<20} s1={e1 / 1000:5.1f}km "
            f"s2={e2 / 1000:5.1f}km r@true={scores[vp_true]:.3f} "
            f"ch@true={chamfer_scores[0] if top5_idx[0] == vp_true else 'N/A':>6} "
            f"{promoted} {improved} ({time.time() - ts:.0f}s)",
            flush=True,
        )

        results[sid] = {
            "stage1_top1_err": float(e1),
            "stage2_top1_err": float(e2),
            "promoted": s2_vp != s1_vp,
            "improved": e2 < e1,
            "r_true": float(scores[vp_true]),
            "true_vp_err": float(e_true_vp),
        }

    # Summary
    errs1 = [r["stage1_top1_err"] for r in results.values()]
    errs2 = [r["stage2_top1_err"] for r in results.values()]
    t1_s1 = sum(e < CORRECT_M for e in errs1)
    t1_s2 = sum(e < CORRECT_M for e in errs2)
    promoted = sum(r["promoted"] for r in results.values())
    better = sum(r["improved"] for r in results.values())
    worse = sum(not r["improved"] and r["promoted"] for r in results.values())

    print(f"\n{'=' * 70}")
    print(
        f"Stage 1 (NCC only):  top1@500m = {t1_s1}/{len(results)}  "
        f"median = {np.median(errs1) / 1000:.2f}km"
    )
    print(
        f"Stage 2 (Chamfer):   top1@500m = {t1_s2}/{len(results)}  "
        f"median = {np.median(errs2) / 1000:.2f}km"
    )
    print(f"Promoted: {promoted}/{len(results)}  Better: {better}  Worse: {worse}")
    print(f"Total: {time.time() - t0:.0f}s")

    json.dump(
        results,
        open(os.path.join(ROOT, "data/eval/chamfer_rerank.json"), "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()

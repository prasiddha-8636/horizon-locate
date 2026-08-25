#!/usr/bin/env python
"""Evaluate matching with spatial radius prior (20km / 50km).

Optimized: for each saliency config, ONE full-DB scan per sample,
then filter by radius post-hoc. ~34 scans total.

Usage:
  python scripts/radius_eval.py [--limit N] [--configs NAME1,NAME2,...]
"""

import json, os, sys, time, argparse
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from scipy.spatial import cKDTree
from geopy.distance import geodesic
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile, evaluate_skyline_quality
from src.matching import (
    feature_bundle_matrix,
    _pearson_ncc_batch,
    _feature_bundle,
)
from src.horizon_format import decode_horizon_column
from scripts.annotate_gsv import dark_channel_prior_dehaze

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
W, H = 1080, 720


def latlon_to_ecef(lat_deg, lon_deg):
    R = 6371000.0
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    return np.column_stack(
        [R * np.cos(phi) * np.cos(lam), R * np.cos(phi) * np.sin(lam), R * np.sin(phi)]
    )


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


def full_db_scan_both(profile, bin_deg, lon, lat, chunk_rows=4000):
    """Single DB pass: compute both baseline and saliency corr arrays.

    Returns (corr_baseline, corr_saliency, global_idx_array).
    """
    from src.matching import ncc_scores, feature_bundle_matrix

    pf = pq.ParquetFile(DB_PATH)
    all_c = {k: [] for k in ("baseline", "saliency")}
    all_idx = []
    cs = 0
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        dbv, dbd = feature_bundle_matrix(chunk)
        # baseline
        c_base, _ = ncc_scores(dbv, dbd, profile, bin_deg)
        # saliency
        c_sal, _ = ncc_scores(dbv, dbd, profile, bin_deg, saliency_alpha=2.0)
        all_c["baseline"].extend(c_base.tolist())
        all_c["saliency"].extend(c_sal.tolist())
        all_idx.extend(range(cs, cs + len(chunk)))
        cs += len(chunk)
    return (
        np.array(all_c["baseline"]),
        np.array(all_c["saliency"]),
        np.array(all_idx, dtype=np.int64),
    )


def fb_at_best(profile, horizon):
    qv, qd = _feature_bundle(profile)
    qv = qv - qv.mean()
    qd = qd - qd.mean()
    dbv, dbd = feature_bundle_matrix(horizon[None, :])
    L = len(profile)
    ext_v = np.concatenate([dbv, dbv[:, : L - 1]], axis=1)
    ext_d = np.concatenate([dbd, dbd[:, : L - 1]], axis=1)
    comb = 0.5 * _pearson_ncc_batch(
        ext_v, qv, np.linalg.norm(qv)
    ) + 0.5 * _pearson_ncc_batch(ext_d, qd, np.linalg.norm(qd))
    return float(comb.max())


def fetch_one_horizon(pf, vp_idx, rg_starts):
    """Fetch a single horizon using cumulative row-group starts."""
    rg = int(np.searchsorted(rg_starts, vp_idx, side="right") - 1)
    pos = vp_idx - rg_starts[rg]
    b = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
    return decode_horizon_column([b["raw_horizon_deg"].iloc[pos]])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()
    ecef = latlon_to_ecef(lat, lon)
    tree = cKDTree(ecef)
    NVPS = len(lon)

    pf = pq.ParquetFile(DB_PATH)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [sid for sid in ann if sid in gt and ann[sid] is not None]
    if args.limit:
        sids = sids[: args.limit]

    radii = [("full", 999999), ("50km", 50000), ("20km", 20000)]
    scan_configs = [("baseline", 0.0), ("saliency", 2.0)]

    results = {}
    for sn, _ in scan_configs:
        for rn, _ in radii:
            results[(sn, rn)] = {"errs": [], "top1_ok": 0, "top5_ok": 0, "n": 0}
    results["gate"] = {"passed": 0, "total": 0, "rejected": []}

    print(f"Config: {NVPS} VPs, bin_deg={BIN_DEG}, {len(sids)} samples")
    print(f"Scans per sample: {len(scan_configs)} | Radii: {[r for r, _ in radii]}\n")

    for si, sid in enumerate(sids):
        t0 = time.time()
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        tl, tn = g["true_lat"], g["true_lon"]
        fov = g["fov_y_deg"]
        r_tilt = np.array(g["cam_R_tilt"])

        mask = mask_from_points(ann[sid])
        if mask is None:
            continue

        img = np.array(
            Image.open(os.path.join(IMAGES_DIR, f"{sid}.png")).convert("RGB")
        )
        dehazed = dark_channel_prior_dehaze(img)

        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            continue

        profile = pr["profile"]
        passed, _, reason = evaluate_skyline_quality(dehazed, mask, profile)
        results["gate"]["total"] += 1
        if not passed:
            results["gate"]["rejected"].append((sid, reason))
            print(f"  [{si + 1}/{len(sids)}] {sid}: REJECTED ({reason})")
            continue
        results["gate"]["passed"] += 1

        q_ecef = latlon_to_ecef(np.array([tl]), np.array([tn]))[0]
        cand_mask = np.zeros(NVPS, dtype=bool)
        for ci in tree.query_ball_point(q_ecef, r=50000):
            cand_mask[ci] = True
        cand_mask[vp] = True
        n_50k = int(cand_mask.sum())
        cand_mask_20k = np.zeros(NVPS, dtype=bool)
        for ci in tree.query_ball_point(q_ecef, r=20000):
            cand_mask_20k[ci] = True
        cand_mask_20k[vp] = True
        n_20k = int(cand_mask_20k.sum())

        fb_true = fb_at_best(profile, fetch_one_horizon(pf, vp, rg_starts))

        line = f"  [{si + 1}/{len(sids)}] {sid}: fb_true={fb_true:.3f} (50k:{n_50k} 20k:{n_20k} cands) "

        corr_base, corr_sal, idxs = full_db_scan_both(profile, BIN_DEG, lon, lat)

        for sn, corr in (("baseline", corr_base), ("saliency", corr_sal)):
            for rn, rm in radii:
                mask_r = (
                    cand_mask_20k
                    if rn == "20km"
                    else cand_mask
                    if rn == "50km"
                    else np.ones(NVPS, dtype=bool)
                )
                mask_r[vp] = True

                visible = mask_r[idxs]
                vis_corr = corr[visible]
                vis_idx = idxs[visible]

                top5_rel = np.argsort(-vis_corr)[:5]
                top5_idx = vis_idx[top5_rel]
                top5_corr = vis_corr[top5_rel]

                best_idx = top5_idx[0]
                best_corr = float(top5_corr[0])
                err = geodesic((tl, tn), (lat[best_idx], lon[best_idx])).meters
                top5_errs = [
                    geodesic((tl, tn), (lat[i], lon[i])).meters for i in top5_idx
                ]

                r = results[(sn, rn)]
                r["errs"].append(err)
                r["n"] += 1
                if err < 500.0:
                    r["top1_ok"] += 1
                if any(e < 500.0 for e in top5_errs):
                    r["top5_ok"] += 1

            line += f"{sn[:3]}={err / 1000:.0f}km "

        print(line, flush=True)

    print(f"\n{'=' * 90}")
    print("RESULTS")
    print(f"{'=' * 90}")

    qg = results["gate"]
    print(f"Quality Gate: {qg['passed']}/{qg['total']} passed", end="")
    if qg["rejected"]:
        print(f" (rejected: {[r for _, r in qg['rejected']]})")
    else:
        print()

    print(
        f"\n{'config':<20} {'N':>4} {'top1@500m':>12} {'top5@500m':>12} {'median':>10} {'<1km':>5} {'<5km':>5} {'<10km':>6} {'<20km':>6}"
    )
    print("-" * 85)
    for sn, _ in scan_configs:
        for rn, _ in radii:
            r = results[(sn, rn)]
            n = r["n"]
            if n == 0:
                continue
            errs = np.array(r["errs"])
            med = np.median(errs)
            lt1 = sum(e < 1000 for e in errs)
            lt5 = sum(e < 5000 for e in errs)
            lt10 = sum(e < 10000 for e in errs)
            lt20 = sum(e < 20000 for e in errs)
            print(
                f"{sn}_{rn:<12} {n:4d} {r['top1_ok']:5d}/{n:<5d} {r['top5_ok']:5d}/{n:<5d} "
                f"{med / 1000:8.1f}km {lt1:5d} {lt5:5d} {lt10:6d} {lt20:6d}"
            )


if __name__ == "__main__":
    main()

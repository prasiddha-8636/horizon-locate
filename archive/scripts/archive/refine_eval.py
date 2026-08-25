#!/usr/bin/env python
"""Evaluate refinements on 17 hand-annotated GSV samples:
  1. Affine A·DB+b fit + impossible-scale gate (A in [0.6, 1.6])
  2. Multi-pitch camera tilt sweep (±6 deg, 2 deg steps) on top candidates
  3. Tree/cloud column filtering (boundary-gradient keep mask)

Per sample: one full-DB baseline scan (cached to disk), then cheap
refinements on the top-K candidates.
"""

import json, os, sys, time, argparse
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from geopy.distance import geodesic
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import (
    extract_elevation_profile,
    evaluate_skyline_quality,
    compute_column_keep_mask,
)
from src.matching import (
    ncc_scores,
    feature_bundle_matrix,
    fit_affine_scale_offset,
    affine_scale_ok,
)
from src.horizon_format import decode_horizon_column
from scripts.annotate_gsv import dark_channel_prior_dehaze

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
W, H = 1080, 720
TOP_K = 200
TILTS_DEG = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
COL_GRAD_THRESH = 8.0


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


def fetch_horizons(vp_idxs, rg_starts, pf):
    out = {}
    # group by row group
    rgs = {}
    for i, v in enumerate(vp_idxs):
        rg = int(np.searchsorted(rg_starts, v, side="right") - 1)
        rgs.setdefault(rg, []).append((i, v))
    for rg, items in rgs.items():
        b = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
        base = int(rg_starts[rg])
        for i, v in items:
            h = b["raw_horizon_deg"].iloc[v - base]
            out[i] = decode_horizon_column([h])[0]
    return np.stack([out[i] for i in range(len(vp_idxs))])


def Rx(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def scan_sample(profile, bin_deg, sid):
    """Full-DB baseline scan, cached. Returns (corr, idxs)."""
    cache = os.path.join(CACHE_DIR, f"{sid}_corr.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["corr"], z["idxs"]
    pf = pq.ParquetFile(DB_PATH)
    all_c = []
    all_idx = []
    cs = 0
    for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        dbv, dbd = feature_bundle_matrix(chunk)
        c, _ = ncc_scores(dbv, dbd, profile, bin_deg)
        all_c.extend(c.tolist())
        all_idx.extend(range(cs, cs + len(chunk)))
        cs += len(chunk)
    corr, idxs = np.array(all_c), np.array(all_idx, dtype=np.int64)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, corr=corr, idxs=idxs)
    return corr, idxs


def ncc_one(profile, horizon, bin_deg):
    dbv, dbd = feature_bundle_matrix(horizon[None, :])
    c, off = ncc_scores(dbv, dbd, profile, bin_deg)
    return float(c[0]), int(off[0])


def err_at(vp, lon, lat, tl, tn):
    return geodesic((tl, tn), (lat[vp], lon[vp])).meters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()

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

    variants = ["baseline", "affine_gate", "affine_rmse", "tilt", "colfilter"]
    results = {v: {"errs": [], "top1_ok": 0, "n": 0} for v in variants}

    print(
        f"Config: {len(lon)} VPs, bin_deg={BIN_DEG}, {len(sids)} samples, TOP_K={TOP_K}"
    )
    print(f"Variants: {variants}")
    print(f"Tilts: {TILTS_DEG} deg | col grad thr: {COL_GRAD_THRESH}\n")

    hdr = f"{'sid':<20}" + "".join(f" {v[:10]:>12}" for v in variants)
    print(hdr)
    print("-" * (20 + 12 * len(variants)))

    for si, sid in enumerate(sids):
        t0 = time.time()
        g = gt[sid]
        vp_true = int(g["closest_viewpoint_id"])
        tl, tn = g["true_lat"], g["true_lon"]
        fov = g["fov_y_deg"]
        base_tilt = np.array(g["cam_R_tilt"])

        mask = mask_from_points(ann[sid])
        if mask is None:
            continue
        img = np.array(
            Image.open(os.path.join(IMAGES_DIR, f"{sid}.png")).convert("RGB")
        )
        dehazed = dark_channel_prior_dehaze(img)

        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=base_tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            print(f"  {sid}: profile FAILED ({pr['status']})")
            continue
        profile = pr["profile"]

        passed, _, reason = evaluate_skyline_quality(dehazed, mask, profile)
        if not passed:
            print(f"  {sid}: REJECTED by quality gate ({reason})")
            continue

        corr, idxs = scan_sample(profile, BIN_DEG, sid)
        top_rel = np.argsort(-corr)[:TOP_K]
        top_idx = idxs[top_rel]
        top_corr = corr[top_rel]
        top_hor = fetch_horizons(top_idx, rg_starts, pf)

        # precompute ncc + affine on top-K
        info = []  # (idx, ncc, A, b, rmse, best_offset)
        for k in range(TOP_K):
            c1, off = ncc_one(profile, top_hor[k], BIN_DEG)
            A, b, rmse = fit_affine_scale_offset(top_hor[k], profile, off)
            info.append((int(top_idx[k]), c1, A, b, rmse, off))

        out = {}

        # baseline: best NCC overall
        bi = top_rel[0]
        out["baseline"] = int(idxs[bi])

        # affine_gate: best NCC among plausible A
        surv = [(c1, vi) for vi, c1, A, b, rmse, off in info if affine_scale_ok(A)]
        out["affine_gate"] = surv[0][1] if surv else out["baseline"]

        # affine_rmse: lowest fit residual
        best_rmse = min(info, key=lambda t: t[4])
        out["affine_rmse"] = best_rmse[0]

        # tilt sweep: re-extract at each tilt, NCC vs top-K
        best_tilt_vp, best_tilt_c = out["baseline"], top_corr[0]
        for d in TILTS_DEG:
            prt = extract_elevation_profile(
                mask, fov_y_deg=fov, r_tilt=base_tilt @ Rx(d), bin_deg=BIN_DEG
            )
            if not prt["ok"]:
                continue
            for k in range(min(50, TOP_K)):
                c2, _ = ncc_one(prt["profile"], top_hor[k], BIN_DEG)
                if c2 > best_tilt_c:
                    best_tilt_c, best_tilt_vp = c2, int(top_idx[k])
        out["tilt"] = best_tilt_vp

        # colfilter: profile with unreliable columns removed
        keep, _ = compute_column_keep_mask(
            dehazed, mask, gradient_threshold=COL_GRAD_THRESH
        )
        pcf = extract_elevation_profile(
            mask,
            fov_y_deg=fov,
            r_tilt=base_tilt,
            bin_deg=BIN_DEG,
            column_keep_mask=keep,
        )
        if pcf["ok"]:
            best_cf, best_cf_vp = top_corr[0], out["baseline"]
            for k in range(TOP_K):
                c3, _ = ncc_one(pcf["profile"], top_hor[k], BIN_DEG)
                if c3 > best_cf:
                    best_cf, best_cf_vp = c3, int(top_idx[k])
            out["colfilter"] = best_cf_vp
        else:
            out["colfilter"] = out["baseline"]

        line = f"  [{si + 1}] {sid:<16}"
        for v in variants:
            e = err_at(out[v], lon, lat, tl, tn)
            results[v]["errs"].append(e)
            results[v]["n"] += 1
            if e < 500.0:
                results[v]["top1_ok"] += 1
            line += f" {e / 1000:10.1f}km"
        print(line + f"  ({time.time() - t0:.0f}s)", flush=True)

    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(
        f"\n{'variant':<14} {'N':>4} {'top1@500m':>12} {'median':>10} {'<1km':>5} {'<5km':>5} {'<10km':>6} {'<20km':>6}"
    )
    print("-" * 65)
    for v in variants:
        r = results[v]
        if r["n"] == 0:
            continue
        errs = np.array(r["errs"])
        med = np.median(errs)
        lt1 = sum(e < 1000 for e in errs)
        lt5 = sum(e < 5000 for e in errs)
        lt10 = sum(e < 10000 for e in errs)
        lt20 = sum(e < 20000 for e in errs)
        print(
            f"{v:<14} {r['n']:4d} {r['top1_ok']:5d}/{r['n']:<5d} {med / 1000:8.1f}km {lt1:5d} {lt5:5d} {lt10:6d} {lt20:6d}"
        )


if __name__ == "__main__":
    main()

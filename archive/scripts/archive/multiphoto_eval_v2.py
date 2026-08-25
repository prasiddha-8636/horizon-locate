#!/usr/bin/env python
"""Multi-Photo Evaluator V6 — Streaming coarse→fine→DTW per crop.

Loads DB in batches for coarse scan, then loads only top candidates for DTW.

Usage:
  python scripts/multiphoto_eval_v2.py
"""

import heapq
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon_format import decode_horizon_column
from matching import fft_prefilter, _feature_bundle, fastdtw, euclidean
from query_profile import extract_elevation_profile

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
OUT_JSON = ROOT / "data" / "street_view" / "eval_v6_results.json"

W, H_img = 1080, 720

# Coarse scan: stride=10, keep top-50
COARSE_STRIDE = 10
COARSE_TOP = 50
# Fine refinement: load neighborhood of top-5, stride=1
FINE_NEIGHBORHOOD = 15  # ±15 around each top-5 coarse hit
FINE_TOP = 20
# DTW
DTW_WINDOW = 15


def mask_from_points(points):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.unique(xs).size < 2:
        return None
    cols = np.arange(W, dtype=np.float64)
    rows = np.interp(cols, xs, ys)
    rows = np.clip(np.rint(rows), 1, H_img - 1).astype(int)
    rr = np.arange(H_img)[:, None]
    sky = rr < rows[None, :]
    return np.where(sky, 0, 255).astype(np.uint8)


def extract_crop_profiles(crops, gt_data, bin_deg):
    profiles = []
    for c in crops:
        mask = mask_from_points(c["points"])
        if mask is None:
            continue
        fov_y = c.get("fov_y_deg", 65.0)
        sid = c.get("sid", "")
        gt_entry = gt_data.get(sid) or {}
        r_tilt = c.get("cam_R_tilt") or gt_entry.get("cam_R_tilt")
        if r_tilt is not None:
            r_tilt = np.array(r_tilt)
        skyline_rows = np.full(W, H_img - 1, dtype=np.int32)
        for col in range(W):
            sky_rows = np.where(mask[:, col] == 0)[0]
            if len(sky_rows) > 0:
                skyline_rows[col] = sky_rows[-1]
        unclipped_cols = (skyline_rows > 2) & (skyline_rows < H_img - 2)
        res = extract_elevation_profile(
            mask, fov_y_deg=fov_y, r_tilt=r_tilt, bin_deg=bin_deg,
            column_keep_mask=unclipped_cols, azim_frame="camera",
        )
        if not res["ok"]:
            continue
        profiles.append({
            "heading": c.get("heading_deg", 0.0),
            "profile": res["profile"],
        })
    return profiles


def coarse_scan_streaming(query_profile, pq_file, bin_deg):
    """Streaming coarse scan (stride=10) → top-50 candidates."""
    heap = []
    chunk_start = 0
    for batch in pq_file.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        idxs = np.arange(0, len(chunk), COARSE_STRIDE)
        if len(idxs) == 0:
            chunk_start += len(chunk)
            continue
        corr, _ = fft_prefilter(chunk[idxs], query_profile, bin_deg)
        for k in range(len(idxs)):
            vp_idx = chunk_start + idxs[k]
            c_val = float(corr[k])
            if len(heap) < COARSE_TOP:
                heapq.heappush(heap, (c_val, vp_idx))
            elif c_val > heap[0][0]:
                heapq.heapreplace(heap, (c_val, vp_idx))
        chunk_start += len(chunk)
    heap.sort(key=lambda x: -x[0])
    return [(h[1], h[0]) for h in heap]


def load_rows(pq_file, vp_indices):
    """Load specific rows from parquet by VP index."""
    needed = set(vp_indices.tolist())
    result = {}
    chunk_start = 0
    for batch in pq_file.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        for k in range(len(chunk)):
            vp_idx = chunk_start + k
            if vp_idx in needed:
                result[vp_idx] = chunk[k]
        chunk_start += len(chunk)
        if len(result) >= len(needed):
            break
    return result


def fine_refine_and_dtw(query_profile, coarse_hits, pq_file, bin_deg, n_bins):
    """Fine refinement + DTW on top candidates."""
    # Get neighborhoods around top-5 coarse hits
    fine_indices = set()
    for vp_idx, _ in coarse_hits[:5]:
        for offset in range(-FINE_NEIGHBORHOOD, FINE_NEIGHBORHOOD + 1):
            candidate = vp_idx + offset
            if candidate >= 0:
                fine_indices.add(candidate)
    fine_indices = np.array(sorted(fine_indices), dtype=np.int64)

    if len(fine_indices) == 0:
        return []

    # Load these rows
    rows = load_rows(pq_file, fine_indices)
    if not rows:
        return []

    # Fine NCC
    fine_db = np.array([rows.get(vp, np.zeros(n_bins)) for vp in fine_indices], dtype=np.float64)
    fine_corr, fine_offsets = fft_prefilter(fine_db, query_profile, bin_deg)

    # Top candidates from fine scan
    top_fine = np.argsort(-fine_corr)[:FINE_TOP]

    # DTW on top candidates
    query_val, query_d1 = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T
    m = len(query_profile)

    candidates = []
    for idx in top_fine:
        vp_idx = fine_indices[idx]
        if fine_corr[idx] < 0.2:
            continue
        offset = int(fine_offsets[idx])
        horizon = rows.get(vp_idx, np.zeros(n_bins))
        windowed = horizon[np.arange(offset, offset + m) % n_bins]
        db_val, db_d1 = _feature_bundle(windowed)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(query_features, db_features, radius=DTW_WINDOW, dist=euclidean)
        dtw_norm = dtw_cost / max(len(query_features), 1)
        score = float(fine_corr[idx]) - 0.01 * dtw_norm
        candidates.append({
            "vp_idx": int(vp_idx),
            "score": score,
            "fft_corr": float(fine_corr[idx]),
            "dtw_norm": dtw_norm,
        })

    candidates.sort(key=lambda c: -c["score"])
    return candidates


def match_single_crop(profile, heading, pq_file, bin_deg, n_bins):
    """Full coarse→fine→DTW pipeline for a single crop."""
    t0 = time.time()

    # Step 1: Coarse scan (streaming)
    coarse = coarse_scan_streaming(profile, pq_file, bin_deg)

    # Step 2: Fine refinement + DTW
    candidates = fine_refine_and_dtw(profile, coarse, pq_file, bin_deg, n_bins)

    elapsed = time.time() - t0
    return candidates, elapsed


def main():
    print("=" * 90)
    print("MULTI-PHOTO EVALUATOR V6 — Streaming coarse→fine→DTW per crop")
    print("=" * 90)
    print()

    t_total = time.time()

    pq_file = pq.ParquetFile(str(DB_PATH))
    meta_db = pq.read_table(str(DB_PATH), columns=["lon", "lat"])
    lat_arr = meta_db.column("lat").to_pandas().values
    lon_arr = meta_db.column("lon").to_pandas().values

    batch0 = next(pq_file.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    decoded0 = decode_horizon_column(batch0.to_pandas()["raw_horizon_deg"].to_numpy())
    bin_deg = 360.0 / decoded0.shape[1]
    n_bins = decoded0.shape[1]

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    with open(ANNOT_FILE) as f:
        annots = json.load(f).get("annotations", {})

    panos = {}
    for sid, points in annots.items():
        meta_p = CROPS_DIR / f"{sid}.json"
        if not meta_p.exists():
            continue
        with open(meta_p) as f:
            meta = json.load(f)
        pid = meta.get("pano_id")
        if not pid:
            continue
        if pid not in panos:
            panos[pid] = []
        meta["sid"] = sid
        meta["points"] = points
        panos[pid].append(meta)

    multi_panos = {k: v for k, v in panos.items() if len(v) >= 2}
    print(f"Found {len(multi_panos)} multi-crop panoramas")
    print()

    results = []
    for pid, crops in multi_panos.items():
        t_pano = time.time()
        sid0 = crops[0]["sid"]
        gt_entry = gt_data.get(sid0) or gt_data.get(pid) or {}
        true_vp = gt_entry.get("closest_viewpoint_id")
        true_lat = gt_entry.get("true_lat") or gt_entry.get("lat")
        true_lon = gt_entry.get("true_lon") or gt_entry.get("lon")
        if true_vp is None or true_lat is None:
            continue
        true_vp = int(true_vp)

        crop_profiles = extract_crop_profiles(crops, gt_data, bin_deg)
        n_crops = len(crop_profiles)
        if n_crops < 2:
            print(f"  [REJECT] {pid[:20]:20s} only {n_crops} crops")
            continue

        # Match each crop independently
        crop_candidates = []
        crop_times = []
        for cp in crop_profiles:
            cands, t = match_single_crop(cp["profile"], cp["heading"], pq_file, bin_deg, n_bins)
            crop_candidates.append(cands)
            crop_times.append(t)

        # Intersection: VPs in ALL crops' top-10
        crop_top_vps = []
        for cands in crop_candidates:
            top_vps = set(c["vp_idx"] for c in cands[:10])
            crop_top_vps.append(top_vps)

        common = crop_top_vps[0]
        for s in crop_top_vps[1:]:
            common = common & s

        # Determine match
        if common:
            # Best from first crop's candidates
            best = None
            for c in crop_candidates[0]:
                if c["vp_idx"] in common:
                    best = c
                    break
            if best:
                match_vp = best["vp_idx"]
                match_err = geodesic(
                    (true_lat, true_lon), (lat_arr[match_vp], lon_arr[match_vp])
                ).meters
                status = "MATCH"
            else:
                match_err = float("inf")
                status = "REJECT"
        else:
            match_err = float("inf")
            status = "REJECT"

        # Single-crop best
        sc_errs = []
        for cands in crop_candidates:
            if cands:
                vp = cands[0]["vp_idx"]
                sc_err = geodesic(
                    (true_lat, true_lon), (lat_arr[vp], lon_arr[vp])
                ).meters
                sc_errs.append(sc_err)
        best_sc_err = min(sc_errs) if sc_errs else float("inf")

        # True VP ranks
        true_ranks = []
        for cands in crop_candidates:
            rank = -1
            for i, c in enumerate(cands):
                if c["vp_idx"] == true_vp:
                    rank = i
                    break
            true_ranks.append(rank)

        elapsed = time.time() - t_pano
        is_correct = (match_err < 1000) if status == "MATCH" else False
        tag = "HIT " if is_correct else ("MISS" if status == "MATCH" else "REJECT")
        print(
            f"  [{tag}] {pid[:20]:20s} "
            f"match={match_err:>8.0f} "
            f"1crop={best_sc_err:>8.0f} "
            f"ranks={true_ranks} "
            f"vote={'Y' if common else 'N'} "
            f"[{elapsed:.0f}s]"
        )

        results.append({
            "pano_id": pid, "n_crops": n_crops, "status": status,
            "match_err_m": float(match_err) if status == "MATCH" else None,
            "sc_err_m": float(best_sc_err),
            "true_ranks": true_ranks,
            "n_common": len(common),
        })

    # =========================================================================
    # REPORT
    # =========================================================================
    n_total = len(results)
    matched = [r for r in results if r["status"] == "MATCH"]
    rejected = [r for r in results if r["status"] == "REJECT"]

    print()
    print("=" * 90)
    print("EVALUATION REPORT")
    print("=" * 90)

    print()
    print("1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"   Total multi-crop panoramas:  {n_total}")
    print(f"   Matched:                     {len(matched)} ({len(matched)/max(n_total,1)*100:.0f}%)")
    print(f"   Rejected:                    {len(rejected)} ({len(rejected)/max(n_total,1)*100:.0f}%)")

    print()
    print("2. CONFIDENCE GUARANTEE")
    print("-" * 40)
    if matched:
        correct = sum(1 for r in matched if r["match_err_m"] is not None and r["match_err_m"] < 1000)
        print(f"   Matches reported:     {len(matched)}")
        print(f"   Correct (<1km):       {correct}/{len(matched)} ({correct/len(matched)*100:.0f}%)")
        print(f"   FALSE POSITIVES:      {len(matched) - correct}")
        if correct == len(matched):
            print(f"   ✓ 100% CONFIDENCE — Zero false positives")
        else:
            print(f"   ✗ {100 - correct/len(matched)*100:.0f}% false positive rate")
    else:
        print(f"   No matches attempted")

    print()
    print("3. ACCURACY ON MATCHED PANOS")
    print("-" * 40)
    if matched:
        errs = np.array([r["match_err_m"] for r in matched])
        print(f"   Median: {np.median(errs):.0f}m, Mean: {np.mean(errs):.0f}m")
        for thr in [100, 500, 1000, 5000, 10000]:
            c = int((errs < thr).sum())
            print(f"   <{thr:>5}m: {c}/{len(matched)} ({c/len(matched)*100:.0f}%)")
    else:
        print("   No matches")

    print()
    print("4. SINGLE-CROP REFERENCE")
    print("-" * 40)
    all_sc = np.array([r["sc_err_m"] for r in results])
    print(f"   Median: {np.median(all_sc):.0f}m, Mean: {np.mean(all_sc):.0f}m")
    for thr in [100, 500, 1000, 5000, 10000]:
        c = int((all_sc < thr).sum())
        print(f"   <{thr:>5}m: {c}/{n_total} ({c/n_total*100:.0f}%)")

    print()
    print("5. COVERAGE vs ACCURACY")
    print("-" * 40)
    n_correct = sum(1 for r in matched if r["match_err_m"] is not None and r["match_err_m"] < 1000)
    print(f"   Matched & correct: {n_correct}/{n_total} ({n_correct/max(n_total,1)*100:.0f}%)")
    print(f"   Rejected:          {len(rejected)}/{n_total} ({len(rejected)/max(n_total,1)*100:.0f}%)")

    print()
    print("=" * 90)
    print("PER-SAMPLE DETAIL")
    print("=" * 90)
    for r in sorted(results, key=lambda x: (
        0 if x["status"] == "MATCH" else 1,
        x["match_err_m"] if x["match_err_m"] is not None else float("inf"),
    )):
        if r["status"] == "MATCH":
            tag = "HIT " if r["match_err_m"] < 1000 else "MISS"
            print(f"  [{tag}] {r['pano_id'][:20]:20s} match={r['match_err_m']:>8.0f} "
                  f"1crop={r['sc_err_m']:>8.0f} ranks={r['true_ranks']} common={r['n_common']}")
        else:
            print(f"  [REJECT] {r['pano_id'][:20]:20s} "
                  f"1crop={r['sc_err_m']:>8.0f} ranks={r['true_ranks']}")

    report = {"results": results, "summary": {
        "n_total": n_total, "n_matched": len(matched), "n_rejected": len(rejected),
    }}
    os.makedirs(OUT_JSON.parent, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")
    print(f"Total time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()

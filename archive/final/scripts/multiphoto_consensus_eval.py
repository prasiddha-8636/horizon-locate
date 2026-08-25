#!/usr/bin/env python
"""Multi-Photo Consensus Re-Ranking Evaluator (single-pass optimized).

Two-stage approach in one streaming pass:
  Stage 1: Fused wide-FOV profile -> collect top-N candidates during scan.
  Stage 2: Re-rank candidates by per-crop consensus.

Key optimization: DB horizons for top-N are stored during the Stage 1 pass,
avoiding a second read.

Usage:
  python scripts/multiphoto_consensus_eval.py
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

from horizon_format import decode_horizon_column, decode_horizon_uint8
from matching import fft_prefilter
from query_profile import extract_elevation_profile

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
OUT_JSON = ROOT / "data" / "street_view" / "consensus_eval_results.json"

W, H = 1080, 720
STRIDE = 12
TOP_N = 600  # keep 600 candidates (margin above 500)


def mask_from_points(points):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.unique(xs).size < 2:
        return None
    cols = np.arange(W, dtype=np.float64)
    rows = np.interp(cols, xs, ys)
    rows = np.clip(np.rint(rows), 1, H - 1).astype(int)
    rr = np.arange(H)[:, None]
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
        skyline_rows = np.full(W, H - 1, dtype=np.int32)
        for col in range(W):
            sky_rows = np.where(mask[:, col] == 0)[0]
            if len(sky_rows) > 0:
                skyline_rows[col] = sky_rows[-1]
        unclipped_cols = (skyline_rows > 2) & (skyline_rows < H - 2)
        res = extract_elevation_profile(
            mask, fov_y_deg=fov_y, r_tilt=r_tilt, bin_deg=bin_deg,
            column_keep_mask=unclipped_cols, azim_frame="camera",
        )
        if not res["ok"]:
            continue
        profiles.append((c.get("heading_deg", 0.0), res["profile"], c))
    return profiles


def fuse_and_calibrate(crop_profiles, true_horizon, n_bins, bin_deg):
    joint = np.full(n_bins, np.nan, dtype=np.float64)
    for heading, prof, c in crop_profiles:
        m = len(prof)
        center_bin = int(round((heading % 360.0) / bin_deg))
        half_m = m // 2
        for i in range(m):
            bin_idx = (center_bin - half_m + i) % n_bins
            if not np.isnan(prof[i]):
                joint[bin_idx] = prof[i]

    valid = ~np.isnan(joint)
    if valid.sum() < 30:
        return None, 0.0, 0

    all_bins = np.arange(n_bins)
    fused = np.interp(all_bins, all_bins[valid], joint[valid])

    best_corr = -np.inf
    best_dp = 0.0
    for dp in np.arange(-20.0, 20.5, 0.5):
        corr, _ = fft_prefilter(true_horizon[None, :], fused + dp, bin_deg=bin_deg)
        if float(corr[0]) > best_corr:
            best_corr = float(corr[0])
            best_dp = dp

    return fused + best_dp, best_dp, int(valid.sum() * bin_deg)


def single_pass_scan(fused_cal, pq_file, bin_deg, n_bins):
    """Single streaming pass: collect top-N candidates with their DB horizons.

    Uses a heap to keep the top-N by correlation. At the end, returns
    (correlations, vp_indices, db_horizons) arrays.
    """
    # Min-heap: (corr, vp_idx, db_horizon) — keep top-N highest corr
    heap = []
    chunk_start = 0

    for batch in pq_file.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        sub = chunk[::STRIDE]
        corr, offsets = fft_prefilter(sub, fused_cal, bin_deg)
        n_sub = len(sub)

        for k in range(n_sub):
            vp_idx = chunk_start + k * STRIDE
            c_val = float(corr[k])
            if len(heap) < TOP_N:
                heapq.heappush(heap, (c_val, vp_idx, chunk[k * STRIDE]))
            elif c_val > heap[0][0]:
                heapq.heapreplace(heap, (c_val, vp_idx, chunk[k * STRIDE]))

        chunk_start += len(chunk)

    # Sort by correlation descending
    heap.sort(key=lambda x: -x[0])
    corrs = np.array([h[0] for h in heap], dtype=np.float64)
    vp_idxs = np.array([h[1] for h in heap], dtype=np.int64)
    db_horiz = np.array([h[2] for h in heap], dtype=np.float64)

    return corrs, vp_idxs, db_horiz


def compute_crop_scores(crop_queries, db_horizons, n_bins):
    """Compute per-crop correlation scores against all candidates."""
    n_cands = db_horizons.shape[0]
    n_crops = len(crop_queries)
    crop_scores = np.zeros((n_crops, n_cands), dtype=np.float64)
    for ci, (heading, prof_clean, prof_zm, prof_norm) in enumerate(crop_queries):
        m = len(prof_clean)
        center_bin = int(round((heading % 360.0) / 0.5))
        half_m = m // 2
        bin_indices = (np.arange(m) - half_m + center_bin) % n_bins
        db_windowed = db_horizons[:, bin_indices]
        if prof_norm < 1e-12:
            continue
        db_mean = db_windowed.mean(axis=1)
        db_zm = db_windowed - db_mean[:, None]
        db_norm = np.linalg.norm(db_zm, axis=1)
        db_norm = np.maximum(db_norm, 1e-12)
        crop_scores[ci, :] = (db_zm @ prof_zm) / (db_norm * prof_norm)
    return crop_scores


# ── Scoring variants ──────────────────────────────────────────────

def variant_current(crop_scores, fusion_corrs, n_crops, **_):
    """Original (broken) consensus scoring."""
    agree_thr = 0.3
    n_agree = (crop_scores > agree_thr).sum(axis=0)
    agree_ratio = n_agree / max(n_crops, 1)
    crop_mean = crop_scores.mean(axis=0)
    crop_min = crop_scores.min(axis=0)
    fusion_bonus = np.maximum(fusion_corrs - 0.3, 0) * 2.0
    score = agree_ratio * np.maximum(crop_mean, 0) * (1.0 + fusion_bonus)
    penalty = np.where(crop_min < -0.2, 0.5, 1.0)
    score *= penalty
    return score, {"n_agree": n_agree, "agree_ratio": agree_ratio,
                  "crop_mean": crop_mean, "crop_min": crop_min}


def variant_fusion_dominant(crop_scores, fusion_corrs, n_crops, **_):
    """Fusion correlation is primary; agreement is a multiplier."""
    agree_thr = 0.3
    n_agree = (crop_scores > agree_thr).sum(axis=0)
    agree_ratio = n_agree / max(n_crops, 1)
    crop_mean = crop_scores.mean(axis=0)
    crop_min = crop_scores.min(axis=0)
    # Require at least 1 crop to agree, then rank by fusion corr
    score = np.maximum(fusion_corrs, 0) * np.maximum(agree_ratio, 0.01)
    score *= np.maximum(crop_mean, 0.01)
    return score, {"n_agree": n_agree, "agree_ratio": agree_ratio,
                  "crop_mean": crop_mean, "crop_min": crop_min}


def variant_fusion_strong(crop_scores, fusion_corrs, n_crops, **_):
    """Fusion squared dominates; agreement as soft gate."""
    agree_thr = 0.3
    n_agree = (crop_scores > agree_thr).sum(axis=0)
    agree_ratio = n_agree / max(n_crops, 1)
    crop_mean = crop_scores.mean(axis=0)
    crop_min = crop_scores.min(axis=0)
    # Fusion^2 makes high-fusion candidates dominate
    score = np.maximum(fusion_corrs, 0) ** 2 * np.maximum(agree_ratio, 0.01)
    score *= np.maximum(crop_mean, 0.01)
    return score, {"n_agree": n_agree, "agree_ratio": agree_ratio,
                  "crop_mean": crop_mean, "crop_min": crop_min}


def variant_high_threshold(crop_scores, fusion_corrs, n_crops, **_):
    """Raise agreement threshold to 0.6 to filter low-quality matches."""
    agree_thr = 0.6
    n_agree = (crop_scores > agree_thr).sum(axis=0)
    agree_ratio = n_agree / max(n_crops, 1)
    crop_mean = crop_scores.mean(axis=0)
    crop_min = crop_scores.min(axis=0)
    fusion_bonus = np.maximum(fusion_corrs - 0.3, 0) * 2.0
    score = agree_ratio * np.maximum(crop_mean, 0) * (1.0 + fusion_bonus)
    penalty = np.where(crop_min < -0.2, 0.5, 1.0)
    score *= penalty
    return score, {"n_agree": n_agree, "agree_ratio": agree_ratio,
                  "crop_mean": crop_mean, "crop_min": crop_min}


def variant_filter_then_rank(crop_scores, fusion_corrs, n_crops, **_):
    """Require ≥2 crops to agree (thr=0.5), then rank by fusion corr."""
    agree_thr = 0.5
    n_agree = (crop_scores > agree_thr).sum(axis=0)
    agree_ratio = n_agree / max(n_crops, 1)
    crop_mean = crop_scores.mean(axis=0)
    crop_min = crop_scores.min(axis=0)
    # Hard gate: need at least 2/3 agreeing at 0.5
    min_agree = max(2, n_crops - 1)  # at least N-1 of N crops
    mask = n_agree >= min_agree
    score = np.where(mask, fusion_corrs, -999.0)
    return score, {"n_agree": n_agree, "agree_ratio": agree_ratio,
                  "crop_mean": crop_mean, "crop_min": crop_min}


def variant_geometric(crop_scores, fusion_corrs, n_crops, **_):
    """Geometric mean of fusion corr and crop-mean corr."""
    agree_thr = 0.3
    n_agree = (crop_scores > agree_thr).sum(axis=0)
    agree_ratio = n_agree / max(n_crops, 1)
    crop_mean = crop_scores.mean(axis=0)
    crop_min = crop_scores.min(axis=0)
    f = np.maximum(fusion_corrs, 0.001)
    c = np.maximum(crop_mean, 0.001)
    score = np.sqrt(f * c) * agree_ratio
    return score, {"n_agree": n_agree, "agree_ratio": agree_ratio,
                  "crop_mean": crop_mean, "crop_min": crop_min}


VARIANTS = {
    "current": variant_current,
    "fusion_dom": variant_fusion_dominant,
    "fusion_strong": variant_fusion_strong,
    "high_thr": variant_high_threshold,
    "filter_rank": variant_filter_then_rank,
    "geometric": variant_geometric,
}


def consensus_rerank_all(crop_queries, db_horizons, fusion_corrs, n_bins):
    """Run all scoring variants, return best candidate index per variant."""
    n_cands = len(fusion_corrs)
    n_crops = len(crop_queries)
    if n_cands == 0 or n_crops == 0:
        return {name: (0, 0.0, {}) for name in VARIANTS}

    crop_scores = compute_crop_scores(crop_queries, db_horizons, n_bins)

    results = {}
    for name, fn in VARIANTS.items():
        score, meta = fn(crop_scores, fusion_corrs, n_crops)
        best_idx = int(np.argmax(score))
        n_agree = meta["n_agree"]
        agree_ratio = meta["agree_ratio"]
        crop_mean = meta["crop_mean"]
        crop_min = meta["crop_min"]
        details = {
            "consensus_score": float(score[best_idx]),
            "n_agree": int(n_agree[best_idx]),
            "agree_ratio": float(agree_ratio[best_idx]),
            "crop_mean_corr": float(crop_mean[best_idx]),
            "crop_min_corr": float(crop_min[best_idx]),
            "per_crop_corrs": [float(crop_scores[ci, best_idx]) for ci in range(n_crops)],
        }
        results[name] = (best_idx, score[best_idx], details)
    return results


def main():
    print("=" * 70)
    print("Multi-Photo Consensus Re-Ranking Evaluator (single-pass)")
    print("=" * 70)
    t_total = time.time()

    pq_file = pq.ParquetFile(str(DB_PATH))
    meta_db = pq.read_table(str(DB_PATH), columns=["lon", "lat"])
    lat_arr = meta_db.column("lat").to_pandas().values
    lon_arr = meta_db.column("lon").to_pandas().values

    _first = next(pq_file.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    bin_deg = 360.0 / len(_first.to_pandas()["raw_horizon_deg"].iloc[0])
    n_bins = int(round(360.0 / bin_deg))

    # Pre-compute row-group starts for random access (true VP only)
    sizes = [pq_file.metadata.row_group(i).num_rows for i in range(pq_file.num_row_groups)]
    rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]

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

        # Fetch true VP horizon (single row-group read)
        rg = int(np.searchsorted(rg_starts, true_vp, side="right") - 1)
        pos = true_vp - rg_starts[rg]
        batch = pq_file.read_row_group(rg, columns=["raw_horizon_deg"])
        true_horizon = decode_horizon_uint8(batch.to_pandas()["raw_horizon_deg"].iloc[pos])

        # Extract crop profiles
        crop_profiles = extract_crop_profiles(crops, gt_data, bin_deg)
        if len(crop_profiles) < 2:
            continue

        # Fuse + calibrate pitch
        fused_cal, dp, cov_deg = fuse_and_calibrate(crop_profiles, true_horizon, n_bins, bin_deg)
        if fused_cal is None:
            continue

        # Prepare per-crop query features
        crop_queries = []
        for heading, prof, c_meta in crop_profiles:
            prof_f = np.asarray(prof, dtype=np.float64)
            prof_clean = np.nan_to_num(prof_f, nan=0.0)
            prof_zm = prof_clean - prof_clean.mean()
            prof_norm = np.linalg.norm(prof_zm)
            crop_queries.append((heading, prof_clean, prof_zm, prof_norm))

        # Single-pass scan: collect top-N with DB horizons
        fusion_corrs, vp_idxs, db_horiz = single_pass_scan(fused_cal, pq_file, bin_deg, n_bins)

        # Consensus re-ranking — all variants
        variant_results = consensus_rerank_all(
            crop_queries, db_horiz, fusion_corrs, n_bins
        )

        # Baseline = fusion-only top-1
        baseline_vp = int(vp_idxs[0])
        baseline_err = geodesic(
            (true_lat, true_lon), (lat_arr[baseline_vp], lon_arr[baseline_vp])
        ).meters

        # True VP rank in fusion
        true_rank = -1
        for i in range(len(vp_idxs)):
            if vp_idxs[i] == true_vp:
                true_rank = i
                break
        in_top5 = true_vp in vp_idxs[:5]
        in_top50 = true_vp in vp_idxs[:50]

        r = {
            "pano_id": pid,
            "n_crops": len(crop_profiles),
            "coverage_deg": cov_deg,
            "pitch_offset": float(dp),
            "baseline_err_m": float(baseline_err),
            "true_rank_fusion": true_rank,
            "in_top5": in_top5,
            "in_top50": in_top50,
        }
        for vname, (v_idx, v_score, v_details) in variant_results.items():
            v_vp = int(vp_idxs[v_idx])
            v_err = geodesic(
                (true_lat, true_lon), (lat_arr[v_vp], lon_arr[v_vp])
            ).meters
            r[f"{vname}_err_m"] = float(v_err)
            r[f"{vname}_details"] = v_details
        results.append(r)

        elapsed = time.time() - t_pano
        base_s = f"{baseline_err:8.0f}m"
        var_errs = " ".join(
            f"{vname}={r[f'{vname}_err_m']:>8.0f}m"
            for vname in VARIANTS
        )
        print(
            f"  {pid[:20]:20s} base={base_s} {var_errs} rank={true_rank:>4d} "
            f"fov={cov_deg:.0f} [{elapsed:.1f}s]"
        )

    # --- Summary ---
    n = len(results)
    base_errs = np.array([r["baseline_err_m"] for r in results])

    print(f"\n{'=' * 80}")
    print("VARIANT COMPARISON")
    print(f"{'=' * 80}")

    # Table header
    col_w = 10
    header = f"{'Metric':<24s} {'Baseline':>{col_w}s}"
    for vname in VARIANTS:
        header += f" {vname:>{col_w}s}"
    print(header)
    print("-" * (24 + col_w * (1 + len(VARIANTS))))

    print(f"{'N panos':<24s} {n:>{col_w}d}")
    print(f"{'Median error':<24s} {np.median(base_errs):>{col_w - 1}.0f}m")
    for vname in VARIANTS:
        v_errs = np.array([r[f"{vname}_err_m"] for r in results])
        print(f"{'':24s} {np.median(v_errs):>{col_w - 1}.0f}m", end="")
    print()

    print(f"{'Mean error':<24s} {np.mean(base_errs):>{col_w - 1}.0f}m", end="")
    for vname in VARIANTS:
        v_errs = np.array([r[f"{vname}_err_m"] for r in results])
        print(f" {np.mean(v_errs):>{col_w - 1}.0f}m", end="")
    print()

    for thr in [100, 500, 1000, 2000, 5000, 10000]:
        bc = int((base_errs < thr).sum())
        line = f"  <{thr:>5}m:           {bc:>3d}/{n} ({bc / n * 100:4.1f}%)"
        for vname in VARIANTS:
            v_errs = np.array([r[f"{vname}_err_m"] for r in results])
            vc = int((v_errs < thr).sum())
            line += f"  {vc:>3d}/{n} ({vc / n * 100:4.1f}%)"
        print(line)

    in5 = sum(1 for r in results if r["in_top5"])
    in50 = sum(1 for r in results if r["in_top50"])
    print(f"\nTrue VP in fusion top-5:  {in5}/{n} ({in5 / n * 100:.1f}%)")
    print(f"True VP in fusion top-50: {in50}/{n} ({in50 / n * 100:.1f}%)")

    # Per-sample detail — show all variants
    print(f"\n{'-' * 80}")
    print("Per-sample (sorted by best variant error):")
    print(f"{'-' * 80}")
    # Find best variant per sample
    for r in sorted(results, key=lambda x: min(x[f"{v}_err_m"] for v in VARIANTS)):
        base = r["baseline_err_m"]
        parts = []
        for vname in VARIANTS:
            ve = r[f"{vname}_err_m"]
            marker = "HIT" if ve < 1000 else "   "
            parts.append(f"{vname[:8]:>8s}={ve:>8.0f}m {marker}")
        print(
            f"  {r['pano_id'][:20]:20s} base={base:>8.0f}m  "
            f"rank={r['true_rank_fusion']:>4d} fov={r['coverage_deg']:.0f}"
        )
        for p in parts:
            print(f"    {p}")

    report = {"results": results, "n_total": n, "variants": list(VARIANTS.keys())}
    os.makedirs(OUT_JSON.parent, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")
    print(f"Total time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()

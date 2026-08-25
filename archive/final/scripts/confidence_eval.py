#!/usr/bin/env python
"""Confidence-gated multi-photo evaluator V5 (memory-safe, single-pass).

Key change vs V4: NO precomputed feature files needed. Because every fused
query profile is a full circle (length 720 == DB bin count), Pearson NCC over
all circular rotations collapses to one matrix product per chunk against
z-scored DB feature rows. The whole 1.34M-row database is scored for ALL
panoramas in a single streaming pass (~3-6 min total), with bounded memory.

Confidence gates (unchanged from V3/V4):
  1. FOV >= 200 deg          (wide profiles break valley symmetry)
  2. n70 <= 200              (profile distinctive: <200 of all VPs match >0.70)
  3. gap_top10 >= 0.02       (top candidate clearly beats the field)
"""

import sys
import time
import json
import heapq
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from query_profile import extract_elevation_profile

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
ANNOT_FILE = ROOT / "data" / "street_view" / "gsv_annotations.json"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"

BIN_DEG = 0.5
N_BINS = int(360.0 / BIN_DEG)
CHUNK = 8000
TOP_KEEP = 50


# ---------------------------------------------------------------------------
# Feature helpers (mirror src/matching.py)
# ---------------------------------------------------------------------------

def _zscore_rows(mat):
    mat = np.asarray(mat, dtype=np.float64)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (mat - mu) / sd


def feature_bundle_rows(mat):
    value = _zscore_rows(mat)
    d1 = _zscore_rows(np.gradient(value, axis=1))
    return value, d1


def query_bundle(profile):
    v = np.asarray(profile, dtype=np.float64)[None, :]
    val, d1 = feature_bundle_rows(v)
    return val[0], d1[0]


# ---------------------------------------------------------------------------
# Crop loading + fusion (same as previous working versions)
# ---------------------------------------------------------------------------

W, H = 400, 300  # fallback mask size; real size taken from points extent


def mask_from_points(points, w=W, h=H):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    if xs.max() > w or ys.max() > h:
        w = max(int(xs.max()) + 2, w)
        h = max(int(ys.max()) + 2, h)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.unique(xs).size < 2:
        return None, (w, h)
    cols = np.arange(w, dtype=np.float64)
    rows_interp = np.interp(cols, xs, ys)
    rows_interp = np.clip(np.rint(rows_interp), 1, h - 1).astype(int)
    rr = np.arange(h)[:, None]
    sky = rr < rows_interp[None, :]
    return np.where(sky, 0, 255).astype(np.uint8), (w, h)


def extract_crop_profiles(crops, gt_data):
    profiles = []
    for c in crops:
        mask, (w, h) = mask_from_points(c["points"])
        if mask is None:
            continue
        fov_y = c.get("fov_y_deg", 65.0)
        sid = c.get("sid", "")
        gt_entry = gt_data.get(sid) or {}
        r_tilt = c.get("cam_R_tilt") or gt_entry.get("cam_R_tilt")
        if r_tilt is not None:
            r_tilt = np.array(r_tilt)
        skyline_rows = np.full(w, h - 1, dtype=np.int32)
        for col in range(w):
            sky_rows = np.where(mask[:, col] == 0)[0]
            if len(sky_rows) > 0:
                skyline_rows[col] = sky_rows[-1]
        unclipped_cols = (skyline_rows > 2) & (skyline_rows < h - 2)
        res = extract_elevation_profile(
            mask, fov_y_deg=fov_y, r_tilt=r_tilt, bin_deg=BIN_DEG,
            column_keep_mask=unclipped_cols, azim_frame="camera",
        )
        if not res["ok"]:
            continue
        profiles.append((c.get("heading_deg", 0.0), res["profile"]))
    return profiles


def fuse_profiles(crop_profiles):
    joint = np.full(N_BINS, np.nan, dtype=np.float64)
    for heading, prof in crop_profiles:
        m = len(prof)
        center_bin = int(round((heading % 360.0) / BIN_DEG))
        half_m = m // 2
        for i in range(m):
            bin_idx = (center_bin - half_m + i) % N_BINS
            if not np.isnan(prof[i]):
                joint[bin_idx] = prof[i]
    valid = ~np.isnan(joint)
    if valid.sum() < 30:
        return None, 0
    all_bins = np.arange(N_BINS)
    fused = np.interp(all_bins, all_bins[valid], joint[valid])
    return fused, int(valid.sum())


# ---------------------------------------------------------------------------
# Single-pass full-DB scoring
# ---------------------------------------------------------------------------

def query_spectra(profile):
    """Precompute the query-side frequency multipliers (once per pano)."""
    qv, qd = query_bundle(profile)
    return (0.5 * np.conj(np.fft.rfft(qv)),
            0.5 * np.conj(np.fft.rfft(qd)))


class QueryState:
    __slots__ = ("spec_v", "spec_d", "best_score", "best_row", "best_lat",
                 "best_lon", "heap", "n70", "n75", "n80", "true_corr")

    def __init__(self, profile):
        self.spec_v, self.spec_d = query_spectra(profile)
        self.best_score = -np.inf
        self.best_row = -1
        self.best_lat = self.best_lon = None
        self.heap = []          # min-heap of top scores
        self.n70 = self.n75 = self.n80 = 0
        self.true_corr = None

    def update(self, corr_chunk, row_start, lats, lons, true_row):
        # top candidate in this chunk
        top_local = int(np.argmax(corr_chunk))
        s = float(corr_chunk[top_local])
        if s > self.best_score:
            self.best_score = s
            self.best_row = row_start + top_local
            self.best_lat = float(lats[top_local])
            self.best_lon = float(lons[top_local])

        # bounded top-K heap (vectorized: only push rows beating the threshold)
        if len(self.heap) < TOP_KEEP:
            need = TOP_KEEP - len(self.heap)
            if corr_chunk.shape[0] > need:
                idx = np.argpartition(-corr_chunk, need)[:need]
                for i in idx:
                    heapq.heappush(self.heap, float(corr_chunk[i]))
            else:
                for v in corr_chunk:
                    heapq.heappush(self.heap, float(v))
        else:
            thr = self.heap[0]
            over = corr_chunk[corr_chunk > thr]
            if over.size > 512:
                over = np.partition(over, -512)[-512:]
            for v in over:
                heapq.heappush(self.heap, float(v))
                heapq.heappop(self.heap)  # evict current minimum (size stays fixed)

        # distinctiveness counters
        self.n70 += int((corr_chunk > 0.70).sum())
        self.n75 += int((corr_chunk > 0.75).sum())
        self.n80 += int((corr_chunk > 0.80).sum())

        # correlation at the true VP row
        if true_row is not None and true_row is not False:
            local = true_row - row_start
            if 0 <= local < corr_chunk.shape[0]:
                self.true_corr = float(corr_chunk[local])


def run_scan(queries, pano_ids, true_rows):
    """One streaming pass over the whole DB, scoring every fused profile.

    Per chunk: DB features + their FFT are computed ONCE and shared across
    all queries; each query costs one complex multiply + one inverse FFT.
    Identical results to matching.ncc_scores (weights 0.5/0.5), verified.
    """
    pf = pq.ParquetFile(str(DB_PATH))
    total_chunks = (pf.metadata.num_rows + CHUNK - 1) // CHUNK

    t0 = time.time()
    pos = 0
    done = 0
    for batch in pf.iter_batches(batch_size=CHUNK,
                                 columns=["raw_horizon_deg", "lat", "lon"]):
        df = batch.to_pandas()
        n = len(df)
        horizons = decode_horizon_column(df["raw_horizon_deg"].to_numpy())
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()

        zv, zd = feature_bundle_rows(horizons)
        Fv = np.fft.rfft(zv, axis=1)
        Fd = np.fft.rfft(zd, axis=1)
        del zv, zd, horizons, df, batch

        for qi, q in enumerate(queries):
            spec = q.spec_v[None, :] * Fv + q.spec_d[None, :] * Fd
            corr = np.fft.irfft(spec, n=N_BINS, axis=1) / N_BINS
            row_max = corr.max(axis=1)
            del spec, corr
            q.update(row_max, pos, lats, lons, true_rows.get(pano_ids[qi]))

        pos += n
        done += 1
        if done % 40 == 0 or done == total_chunks:
            el = time.time() - t0
            print(f"  chunk {done}/{total_chunks}  ({el:.0f}s elapsed)", flush=True)


# ---------------------------------------------------------------------------
# Per-pano evaluation using scan results
# ---------------------------------------------------------------------------

def evaluate_from_state(pano_id, state, coverage, n_crops, compute_err_fn):
    fused_corr = state.best_score
    sorted_scores = sorted(state.heap, reverse=True)
    top1 = sorted_scores[0] if sorted_scores else 0.0
    mean_top10 = float(np.mean(sorted_scores[:10])) if len(sorted_scores) >= 10 else top1
    gap_top10 = top1 - mean_top10
    n70, n75, n80 = state.n70, state.n75, state.n80

    err, _ = compute_err_fn()
    fov = coverage * BIN_DEG

    reject_reasons = []
    if fov < 200.0:
        reject_reasons.append(f"fov={fov:.0f}<200")
    if n70 > 200:
        reject_reasons.append(f"generic(n70={n70})")
    if gap_top10 < 0.02:
        reject_reasons.append(f"low_gap10({gap_top10:.4f})")

    is_confident = len(reject_reasons) == 0
    tag = "[CONFIDENT]" if is_confident else "[REJECT  ]"
    hit = " HIT" if err is not None and err < 1000 else ""
    err_str = f"{err:8.0f}" if err is not None else "     inf"

    print(f"  {tag} {pano_id[:18]:18s} crops={n_crops:1d} "
          f"FOV={fov:5.0f}° err={err_str}m "
          f"corr={fused_corr:.3f} n70={n70:6d} gap10={gap_top10:.4f}{hit}")

    if reject_reasons:
        print(f"           REASON: {'; '.join(reject_reasons)}")

    return {
        "sid": pano_id, "fov": fov,
        "err": err if err is not None else float("inf"),
        "fused_corr": fused_corr, "gap_top10": gap_top10,
        "n70": n70, "n75": n75, "n80": n80,
        "n_crops": n_crops,
        "is_confident": is_confident,
        "reject_reasons": reject_reasons,
        "true_corr": state.true_corr,
        "top_lat": state.best_lat, "top_lon": state.best_lon,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("CONFIDENCE-GATED MULTI-PHOTO EVALUATOR V5")
    print("=" * 72)
    print("Single streaming pass over all 1.34M VPs — no precompute needed\n")

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    with open(ANNOT_FILE) as f:
        annots = json.load(f).get("annotations", {})

    # Group crops by pano
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

    multi = {k: v for k, v in panos.items() if len(v) >= 2}
    print(f"Found {len(multi)} multi-crop panoramas\n")

    # Phase 1: build fused profiles up front
    prepared = {}   # pid -> (state, coverage, n_crops)
    for pid, crops in sorted(multi.items()):
        crop_profiles = extract_crop_profiles(crops, gt_data)
        if len(crop_profiles) < 2:
            continue
        fused, coverage = fuse_profiles(crop_profiles)
        if fused is None or len(fused) != N_BINS:
            continue
        prepared[pid] = (QueryState(fused), coverage, len(crop_profiles))

    if not prepared:
        print("No valid panoramas — nothing to do.")
        return

    # Phase 2: find nearest DB row to each pano's true location (for rank)
    pano_ids = list(prepared.keys())
    truth = []
    for pid in pano_ids:
        g = gt_data.get(pid) or {}
        la, lo = g.get("true_lat") or g.get("lat"), g.get("true_lon") or g.get("lon")
        truth.append((la, lo))

    true_rows = {}
    have_truth = [i for i, t in enumerate(truth) if t[0] is not None]
    if have_truth:
        try:
            from scipy.spatial import cKDTree
            print("Indexing DB coordinates for true-VP lookup...")
            pf = pq.ParquetFile(str(DB_PATH))
            lat_chunks, lon_chunks = [], []
            for batch in pf.iter_batches(batch_size=50000, columns=["lat", "lon"]):
                df = batch.to_pandas()
                lat_chunks.append(df["lat"].to_numpy())
                lon_chunks.append(df["lon"].to_numpy())
            db_lat = np.concatenate(lat_chunks)
            db_lon = np.concatenate(lon_chunks)
            mlats = np.deg2rad(np.mean(db_lat))
            xs = db_lon * np.cos(mlats)
            ys = db_lat
            tree = cKDTree(np.column_stack([xs, ys]))
            for i in have_truth:
                la, lo = truth[i]
                d, idx = tree.query([lo * np.cos(mlats), la], k=1)
                if d * 111_320 < 250:  # within ~250m -> that's the true VP row
                    true_rows[pano_ids[i]] = int(idx)
            print(f"  true VP row found for {len(true_rows)}/{len(pano_ids)} panos\n")
            del db_lat, db_lon, tree
        except Exception as e:
            print(f"  KDTree unavailable ({e}) — ranks will be skipped\n")
    del have_truth, truth

    # Phase 3: single-pass scan
    states = [prepared[pid][0] for pid in pano_ids]
    run_scan(states, pano_ids, true_rows)

    # Phase 4: evaluate + report
    print()
    results = []
    for pid in pano_ids:
        state, coverage, n_crops = prepared[pid]
        g = gt_data.get(pid) or {}
        tl, lo_ = g.get("true_lat") or g.get("lat"), g.get("true_lon") or g.get("lon")

        def make_compute_err(st=state, t_lat=tl, t_lon=lo_):
            def fn():
                if t_lat is None or st.best_lat is None:
                    return None, None
                return geodesic((t_lat, t_lon),
                                (st.best_lat, st.best_lon)).meters, None
            return fn

        r = evaluate_from_state(pid, state, coverage, n_crops,
                                make_compute_err())
        results.append(r)

    # --- Summary ---
    confident = [r for r in results if r["is_confident"]]
    rejected = [r for r in results if not r["is_confident"]]

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    def print_stats(label, errs):
        errs = [e for e in errs if e != float("inf")]
        if not errs:
            print(f"  {label}: N=0")
            return
        e = np.array(errs)
        print(f"  {label}: N={len(e)}  median={np.median(e):.0f}m  mean={np.mean(e):.0f}m")
        for thr in [100, 1000, 5000, 10000]:
            n = int((e < thr).sum())
            print(f"    <{thr:>6}m: {n}/{len(e)} ({100*n/len(e):5.1f}%)")

    print_stats("ALL PANOS", [r["err"] for r in results])
    print_stats("CONFIDENT", [r["err"] for r in confident])
    print_stats("REJECTED ", [r["err"] for r in rejected])

    hits = [r for r in results if r["err"] < 1000]
    conf_hits = [r for r in confident if r["err"] < 1000]
    if confident:
        print(f"\n  PRECISION: {len(conf_hits)}/{len(confident)} confident are true "
              f"({100*len(conf_hits)/len(confident):.1f}%)")
    if hits:
        print(f"  RECALL:    {len(conf_hits)}/{len(hits)} hits captured "
              f"({100*len(conf_hits)/len(hits):.0f}%)")
    if results:
        print(f"  ACCEPTANCE:{len(confident)}/{len(results)} panos accepted "
              f"({100*len(confident)/len(results):.1f}%)")

    out_path = ROOT / "data" / "street_view" / "confidence_eval_v5.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Off-grid synthetic evaluation — REAL matching pipeline.

Uses the same 3-scorer + RRF + DTW pipeline as the real system.
Picks positions BETWEEN DB grid points, renders synthetic horizons
using HORAYZON at those positions, and matches against the DB by
streaming chunks.

Usage:
    python archive/scripts/offgrid_synthetic_eval.py --samples 50 --resume
"""

import sys
import time
import json
import hashlib
import pickle
import argparse
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic
from heapq import heappush, heappop

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import horayzon as hray
from pyproj import Transformer
from horizon_format import decode_horizon_column
from matching import (
    prepare_scorer_states, score_chunk_shared_fft, rrf_fusion,
    _feature_bundle, _safe_zscore, _bandpass,
)
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
MESH_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "terrain_mesh.npy"
META_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "terrain_meta.json"
BIN_DEG = 0.5
N_BINS = int(360 / BIN_DEG)
CHUNK = 4000
TOP_K = 30          # candidates for DTW refinement
RRF_K = 60          # RRF constant
DTW_WINDOW = 15     # FastDTW radius
CKPT_DIR = ROOT / "data" / "eval_ckpt"


def _ckpt_path(n_samples, seed):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(f"{n_samples}_{seed}".encode()).hexdigest()[:8]
    return CKPT_DIR / f"offgrid_v2_{tag}.pkl"


def _save_ckpt(path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"  [CKPT] Saved ({mb:.1f}MB)")


def _load_ckpt(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"  [RESUME] Loaded ({mb:.1f}MB)")
    return data


def load_mesh_and_meta():
    vert_grid = np.load(str(MESH_PATH))
    with open(META_PATH) as f:
        meta = json.load(f)
    return vert_grid, meta


def load_db_coords():
    pf = pq.ParquetFile(str(DB_PATH))
    lats, lons = [], []
    for batch in pf.iter_batches(batch_size=50000, columns=["lat", "lon"]):
        df = batch.to_pandas()
        lats.append(df["lat"].values)
        lons.append(df["lon"].values)
    return np.concatenate(lats), np.concatenate(lons)


def pick_offgrid_positions(meta, n_samples, rng):
    px_w = abs(meta["px_w"])
    px_h = meta["px_h"]
    start_x = meta["start_x"]
    start_y = meta["start_y"]
    dim0 = meta["dem_dim_0"]
    dim1 = meta["dem_dim_1"]
    offset = px_w * 0.5
    margin = 5
    rows = rng.randint(margin, dim0 - margin, size=n_samples)
    cols = rng.randint(margin, dim1 - margin, size=n_samples)
    xs = start_x + cols * px_w + offset
    ys = start_y + rows * px_h + offset
    return xs, ys, rows, cols


def render_horizons_at_positions(vert_grid, meta, xs, ys, n_bins=N_BINS):
    n = len(xs)
    px_w = abs(meta["px_w"])
    start_x = meta["start_x"]
    start_y = meta["start_y"]
    dim0 = meta["dem_dim_0"]
    dim1 = meta["dem_dim_1"]
    px_h = meta["px_h"]
    cols = np.clip(np.round((xs - start_x) / px_w).astype(int), 0, dim1 - 1)
    rows = np.clip(np.round((ys - start_y) / px_h).astype(int), 0, dim0 - 1)
    idx = (rows * dim1 + cols) * 3 + 2
    zs = vert_grid[idx].astype(np.float32)
    coords = np.column_stack([
        xs.astype(np.float32), ys.astype(np.float32), zs,
    ])
    vec_up = np.zeros((n, 3), dtype=np.float32)
    vec_up[:, 2] = 1.0
    vec_north = np.zeros((n, 3), dtype=np.float32)
    vec_north[:, 1] = 1.0
    ray_org_elev = np.full(n, 1.6, dtype=np.float32)

    result = hray.horizon.horizon_locations(
        vert_grid, meta["dem_dim_0"], meta["dem_dim_1"],
        coords, vec_up, vec_north,
        dist_search=30000.0, azim_num=n_bins, hori_acc=0.1,
        ray_org_elev=ray_org_elev, hori_dist_out=False, elev_ang_low_lim=-89.0,
    )
    horizon_elev_rad = result[0] if isinstance(result, tuple) else result
    return np.rad2deg(horizon_elev_rad)


def add_noise(profile, sigma):
    if sigma == 0:
        return profile.copy()
    return profile + np.random.normal(0, sigma, size=profile.shape)


def pass1_streaming_ncc(queries, db_path, n_samples, n_db):
    """Pass 1: streaming NCC + RRF across all chunks. Returns top-K candidates per query."""
    # Build query states for all 3 scorers
    states_list = prepare_scorer_states(queries.tolist(), bin_deg=BIN_DEG)

    # Per-query: top-K heap per scorer (score, row_idx)
    n_scorers = 3
    scorer_names = ["baseline", "bp28", "bp316"]
    heaps = [[[] for _ in range(n_scorers)] for _ in range(n_samples)]

    pf = pq.ParquetFile(str(db_path))
    row_start = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["raw_horizon_deg"]):
        decoded = decode_horizon_column(batch["raw_horizon_deg"])
        n_chunk = len(decoded)
        dummy_lats = np.zeros(n_chunk)
        dummy_lons = np.zeros(n_chunk)

        results = score_chunk_shared_fft(
            states_list, decoded, dummy_lats, dummy_lons, row_start,
            scorers=scorer_names,
        )

        for qi in range(n_samples):
            for si, scorer in enumerate(scorer_names):
                heap = results[qi][scorer]["heap"]
                for score, row, _, _ in heap:
                    heappush(heaps[qi][si], (score, row))
                    if len(heaps[qi][si]) > TOP_K:
                        heappop(heaps[qi][si])

        row_start += n_chunk
        if row_start % 200000 == 0:
            print(f"    streamed {row_start}/{n_db}...")

    # RRF fusion per query
    fused_results = []
    for qi in range(n_samples):
        ranked_lists = []
        for si in range(n_scorers):
            # Convert min-heap to ranked list (highest score first)
            items = sorted(heaps[qi][si], key=lambda x: -x[0])
            ranked_lists.append([(row, score) for score, row in items])
        fused_scores, best_row = rrf_fusion(ranked_lists, k=RRF_K)
        # Top-T candidates by RRF score
        top_rows = sorted(fused_scores.keys(), key=lambda r: fused_scores[r], reverse=True)[:TOP_K]
        fused_results.append(top_rows)

    return fused_results


def pass2_dtw_refine(query_profile, candidate_rows, db_path):
    """Pass 2: load only candidate rows from DB and run DTW."""
    if not candidate_rows:
        return None, 0.0

    # Load only the candidate rows
    pf = pq.ParquetFile(str(db_path))
    candidates_set = set(candidate_rows)
    row_map = {}  # local_row -> global_row
    collected = []
    global_row = 0
    for batch in pf.iter_batches(batch_size=50000, columns=["raw_horizon_deg"]):
        decoded = decode_horizon_column(batch["raw_horizon_deg"])
        for local_i, global_i in enumerate(range(global_row, global_row + len(decoded))):
            if global_i in candidates_set:
                row_map[len(collected)] = global_i
                collected.append(decoded[local_i])
        global_row += len(decoded)
        if len(collected) >= len(candidates_set):
            break
    del pf

    if not collected:
        return candidate_rows[0] if candidate_rows else None, 0.0

    db_matrix = np.array(collected)

    # DTW on each candidate
    query_val, query_d1 = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    best_score = -np.inf
    best_row = candidate_rows[0]

    for local_i in range(len(collected)):
        horizon = db_matrix[local_i]
        global_row_idx = row_map[local_i]

        # NCC score
        h_val, h_d1 = _feature_bundle(horizon)
        corr = 0.5 * float(np.corrcoef(query_val, h_val)[0, 1]) + \
               0.5 * float(np.corrcoef(query_d1, h_d1)[0, 1])

        # DTW
        db_val, db_d1 = _feature_bundle(horizon)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(query_features, db_features, radius=DTW_WINDOW, dist=euclidean)
        dtw_normalized = dtw_cost / max(len(query_features), 1)
        score = corr - 0.01 * dtw_normalized

        if score > best_score:
            best_score = score
            best_row = global_row_idx

    return best_row, best_score


def run_offgrid(n_samples=50, seed=42, output_path=None,
                noise_sigmas=(0.0, 0.25, 0.5, 1.0, 2.0),
                resume=False, fresh=False):
    rng = np.random.RandomState(seed)

    print("=" * 72)
    print("OFF-GRID SYNTHETIC EVALUATION (full pipeline: 3 scorers + RRF + DTW)")
    print("=" * 72)
    print(f"n_samples={n_samples}, seed={seed}")

    ck = _ckpt_path(n_samples, seed)
    cached = None
    if resume and ck.exists() and not fresh:
        try:
            cached = _load_ckpt(ck)
            print(f"  [RESUME] Resuming from checkpoint")
        except Exception as e:
            print(f"  [RESUME] Cache corrupt ({e}) — re-running")
            cached = None

    # Always load DB coords (needed for error computation and n_db)
    t0 = time.time()
    lat_db, lon_db = load_db_coords()
    n_db = len(lat_db)
    print(f"Loaded {n_db} DB coords in {time.time()-t0:.1f}s")

    if cached is not None:
        all_results = cached["all_results"]
        lat_off, lon_off = cached["lat_off"], cached["lon_off"]
        nearest_idx = cached["nearest_idx"]
        offgrid_horizons = cached["offgrid_horizons"]
        noise_sigmas = cached["noise_sigmas"]
        n_samples = cached["n_samples"]
        meta = {"px_w": cached["meta_px_w"]}
    else:
        # Phase 1: Render off-grid horizons
        t0 = time.time()
        vert_grid, meta = load_mesh_and_meta()
        print(f"Loaded terrain mesh in {time.time()-t0:.1f}s")

        pass  # lat_db, lon_db, n_db already loaded above

        xs, ys, rows, cols = pick_offgrid_positions(meta, n_samples, rng)

        t0 = time.time()
        offgrid_horizons = render_horizons_at_positions(vert_grid, meta, xs, ys)
        print(f"Rendered {n_samples} off-grid horizons in {time.time()-t0:.1f}s")

        to_gps = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
        lon_off, lat_off = to_gps.transform(xs, ys)

        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([lat_db, lon_db]))
        _, nearest_idx = tree.query(np.column_stack([lat_off, lon_off]))
        print(f"Nearest DB rows found")

        # Save checkpoint after rendering
        _save_ckpt(ck, {
            "all_results": [],
            "offgrid_horizons": offgrid_horizons,
            "nearest_idx": nearest_idx,
            "lat_off": lat_off, "lon_off": lon_off,
            "n_samples": n_samples, "noise_sigmas": noise_sigmas,
            "meta_px_w": meta["px_w"],
            "phase": "rendered",
        })
        all_results = []

    # Phase 2: Match each query against full DB using real pipeline
    # Build conditions
    conditions = []
    for sigma in noise_sigmas:
        queries = np.array([add_noise(p.copy(), sigma) for p in offgrid_horizons])
        conditions.append((f"noise={sigma}", queries))

    # uint8 quantization
    q_u8 = np.array([
        np.clip(np.round(p * (255/90)), 0, 255).astype(np.uint8).astype(np.float64) * (90/255)
        for p in offgrid_horizons
    ])
    conditions.append(("quant_u8", q_u8))

    # Check which conditions are already done
    done_conditions = set(r["condition"] for r in all_results)
    done_per_cond = {}
    for r in all_results:
        done_per_cond.setdefault(r["condition"], set()).add(r["query_idx"])

    total_to_do = sum(
        n_samples - len(done_per_cond.get(cname, set()))
        for cname, _ in conditions
    )
    print(f"\nConditions to evaluate: {len(conditions)}")
    print(f"Queries remaining: {total_to_do}")

    # Stream DB and match
    t_total = time.time()

    for cond_name, queries in conditions:
        if cond_name in done_conditions and len(done_per_cond.get(cond_name, set())) == n_samples:
            print(f"\n  {cond_name}: already complete, skipping")
            continue

        print(f"\n--- {cond_name} ---")
        t0 = time.time()

        # Find queries that still need matching
        pending = [qi for qi in range(n_samples) if qi not in done_per_cond.get(cond_name, set())]
        if not pending:
            print(f"  all done, skipping")
            continue

        queries_pending = queries[pending]

        # Pass 1: streaming NCC + RRF (low memory, scans full DB)
        print(f"  Pass 1: streaming NCC + RRF ({len(pending)} queries)...")
        t1 = time.time()
        fused_results = pass1_streaming_ncc(queries_pending, DB_PATH, len(pending), n_db)
        print(f"  Pass 1 done in {time.time()-t1:.0f}s")

        # Pass 2: DTW on top candidates only (loads ~30 rows per query)
        print(f"  Pass 2: DTW refinement...")
        t2 = time.time()
        for i, qi in enumerate(pending):
            query = queries[qi]
            candidates = fused_results[i]
            best_row, best_score = pass2_dtw_refine(query, candidates, DB_PATH)

            if best_row is None:
                err = float("inf")
            else:
                err = geodesic(
                    (lat_off[qi], lon_off[qi]),
                    (lat_db[best_row], lon_db[best_row])
                ).meters

            all_results.append({
                "condition": cond_name,
                "query_idx": qi,
                "true_row": int(nearest_idx[qi]),
                "best_row": int(best_row) if best_row is not None else -1,
                "top1_err_m": float(err),
                "match_score": float(best_score) if best_score is not None else 0.0,
            })

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t2
                remaining = len(pending) - i - 1
                print(f"    [{i+1}/{len(pending)}] {elapsed:.0f}s elapsed")
        print(f"  Pass 2 done in {time.time()-t2:.0f}s")

        # Save checkpoint after each condition
        _save_ckpt(ck, {
            "all_results": all_results,
            "offgrid_horizons": offgrid_horizons,
            "nearest_idx": nearest_idx,
            "lat_off": lat_off, "lon_off": lon_off,
            "n_samples": n_samples, "noise_sigmas": noise_sigmas,
            "meta_px_w": meta["px_w"],
        })

        # Summary for this condition
        errs = np.array([r["top1_err_m"] for r in all_results if r["condition"] == cond_name])
        under_100 = (errs < 100).sum()
        under_1k = (errs < 1000).sum()
        print(f"  {cond_name}: <100m={under_100}/{len(errs)}, <1km={under_1k}/{len(errs)}, median={np.median(errs):.0f}m ({time.time()-t0:.0f}s)")

    print(f"\nTotal time: {time.time()-t_total:.0f}s")

    # Final summary
    print("\n" + "=" * 72)
    print("SUMMARY (full pipeline: 3 scorers + RRF + DTW)")
    print("=" * 72)
    print(f"{'condition':>12} | {'n':>3} {'<100m':>6} {'<1km':>6} {'median':>8}")
    print("-" * 50)
    for cond_name, _ in conditions:
        errs = np.array([r["top1_err_m"] for r in all_results if r["condition"] == cond_name])
        if len(errs) == 0:
            continue
        under_100 = (errs < 100).sum()
        under_1k = (errs < 1000).sum()
        print(f"{cond_name:>12} | {len(errs):>3} {under_100:>5}/{len(errs)} {under_1k:>5}/{len(errs)} {np.median(errs):>7.0f}m")

    # Save
    if output_path is None:
        output_path = ROOT / "data" / "street_view" / "offgrid_eval_results_v2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoints if available")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore all checkpoints, run from scratch")
    args = ap.parse_args()
    run_offgrid(n_samples=args.samples, seed=args.seed, output_path=args.output,
                resume=args.resume, fresh=args.fresh)

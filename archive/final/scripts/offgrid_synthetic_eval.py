#!/usr/bin/env python
"""Off-grid synthetic evaluation (memory-safe streaming version).

Tests matcher accuracy under controlled degradation, WITHOUT loading the
whole horizon DB into RAM.

Design notes
------------
* All queries are full-circle profiles (length = N_BINS = 720). For M == L,
  Pearson NCC over all circular rotations reduces to
      corr(db, q) = <z(db), z(q)> / L
  i.e. a single matrix product between z-scored rows. This lets us score
  hundreds of query variants against a stride-subsampled DB in seconds.
* The DB is subsampled by `--stride` (default 10 -> ~134k VPs, ~300m grid).
  Only the strided rows are materialised: 134k x 720 float32 x 2 features
  ~= 770 MB, well within a 7 GB machine.
* Pitch is NOT swept: adding a constant or circularly rolling the query
  profile leaves max-over-offsets Pearson NCC unchanged (z-score removes
  constants; rolls just relabel offsets). The degradations that DO matter
  are additive noise and quantisation — those are swept here.

Usage
-----
    python scripts/offgrid_synthetic_eval.py --samples 40
    python scripts/offgrid_synthetic_eval.py --fov-sweep
"""

import sys
import time
import json
import argparse
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from horizon_format import decode_horizon_column

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
BIN_DEG = 0.5


# ---------------------------------------------------------------------------
# Feature helpers (mirror src/matching.py exactly)
# ---------------------------------------------------------------------------

def _zscore_rows(mat):
    """Z-score each row; constant rows -> zeros."""
    mat = np.asarray(mat, dtype=np.float64)
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (mat - mu) / sd


def feature_bundle(mat):
    """Return (value, d1) z-scored feature pair for a batch of profiles."""
    value = _zscore_rows(mat)
    d1 = _zscore_rows(np.gradient(value, axis=1))
    return value, d1


def query_bundle(profile):
    """Feature bundle for a single query profile -> two unit-std vectors."""
    v, d = feature_bundle(np.asarray(profile, dtype=np.float64)[None, :])
    return v[0], d[0]


def build_db_spectra(Z_val, Z_d1):
    """Precompute DB frequency-domain feature spectra (shared by all queries)."""
    return np.fft.rfft(Z_val.astype(np.float64), axis=1), \
           np.fft.rfft(Z_d1.astype(np.float64), axis=1)


def fft_score_all(F_val, F_d1, profiles, n_bins):
    """Max-over-shift Pearson NCC of every profile against every DB row.

    Mathematically identical to matching.ncc_scores(weights=(0.5, 0.5))
    for full-circle queries (verified to 1e-8), but shares the DB FFT across
    all queries and folds value+d1 into ONE inverse FFT per query.
    """
    out = np.empty((len(profiles), F_val.shape[0]), dtype=np.float32)
    for qi, prof in enumerate(profiles):
        qv, qd = query_bundle(prof)
        spec = (0.5 * F_val * np.conj(np.fft.rfft(qv))[None, :]
                + 0.5 * F_d1 * np.conj(np.fft.rfft(qd))[None, :])
        numer = np.fft.irfft(spec, n=n_bins, axis=1)
        out[qi] = (numer / n_bins).max(axis=1)
    return out


# ---------------------------------------------------------------------------
# DB loading (streaming, stride-subsampled)
# ---------------------------------------------------------------------------

def load_strided_db(stride):
    """Stream the parquet once; return strided (horizons, lat, lon)."""
    pf = pq.ParquetFile(str(DB_PATH))
    h_chunks, lat_chunks, lon_chunks = [], [], []
    pos = 0
    for batch in pf.iter_batches(batch_size=8000,
                                 columns=["raw_horizon_deg", "lat", "lon"]):
        df = batch.to_pandas()
        n = len(df)
        sel = np.arange(pos, pos + n) % stride == 0
        if sel.any():
            dec = decode_horizon_column(df["raw_horizon_deg"].to_numpy())
            h_chunks.append(dec[sel])
            lat_chunks.append(df["lat"].to_numpy()[sel])
            lon_chunks.append(df["lon"].to_numpy()[sel])
        pos += n
        if pos % 200000 < 8000:
            print(f"  streamed {pos} rows...", flush=True)
    return (np.concatenate(h_chunks),
            np.concatenate(lat_chunks),
            np.concatenate(lon_chunks))


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------

UINT8_STEP = 90.0 / 255.0  # DB storage quantisation (deg per level)


def add_noise(profile, sigma):
    return profile + np.random.normal(0.0, sigma, len(profile))


def quantize_u8(profile):
    return np.round(profile / UINT8_STEP) * UINT8_STEP


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_offgrid(n_samples=40, stride=10, seed=42, output_path=None,
                noise_sigmas=(0.0, 0.25, 0.5, 1.0, 2.0)):
    rng = np.random.RandomState(seed)

    print("=" * 72)
    print("OFF-GRID SYNTHETIC EVALUATION (streaming)")
    print("=" * 72)
    print(f"stride={stride}, n_samples={n_samples}")

    t0 = time.time()
    H_sub, lat_sub, lon_sub = load_strided_db(stride)
    n_rows, n_bins = H_sub.shape
    print(f"Loaded {n_rows} strided VPs x {n_bins} bins "
          f"in {time.time()-t0:.1f}s (grid ~{30*stride:.0f}m)")

    t0 = time.time()
    Z_val, Z_d1 = feature_bundle(H_sub)
    Z_val = Z_val.astype(np.float32)
    Z_d1 = Z_d1.astype(np.float32)
    print(f"Features: {Z_val.nbytes/1e6:.0f} MB + {Z_d1.nbytes/1e6:.0f} MB")
    F_val, F_d1 = build_db_spectra(Z_val, Z_d1)
    del Z_val, Z_d1
    print(f"DB spectra ready in {time.time()-t0:.1f}s")

    # --- Pick test locations (rows of the strided grid) ---
    test_idx = rng.choice(n_rows, size=min(n_samples, n_rows), replace=False)
    base_profiles = H_sub[test_idx]

    # Nearest-neighbour grid floor for each test point
    print(f"\nSweeping noise sigma in {list(noise_sigmas)} + uint8 quantisation")
    all_results = []

    conditions = [(f"noise={s}", lambda p, s=s: add_noise(p, s))
                  for s in noise_sigmas]
    conditions.append(("quant_u8", quantize_u8))

    for cond_name, perturb in conditions:
        t0 = time.time()
        queries_raw = [perturb(p.copy()) for p in base_profiles]

        C = fft_score_all(F_val, F_d1, queries_raw, n_bins)

        errs, ranks, n70s = [], [], []
        for qi, ti in enumerate(test_idx):
            row = C[qi]
            top = int(np.argmax(row))
            errs.append(geodesic((lat_sub[ti], lon_sub[ti]),
                                 (lat_sub[top], lon_sub[top])).meters)
            ranks.append(int(np.sum(row > row[ti])))
            n70s.append(int(np.sum(row > 0.70)))

        errs = np.array(errs)
        ranks = np.array(ranks)
        dt = time.time() - t0

        print(f"\n--- {cond_name} --- ({dt:.1f}s)")
        print(f"  Top-1 vs true grid point: median={np.median(errs):.0f}m "
              f"mean={np.mean(errs):.0f}m")
        print(f"    exact-hit(rank0): {np.mean(ranks == 0):.1%}   "
              f"<30m: {np.mean(errs < 30):.1%}   <100m: {np.mean(errs < 100):.1%}   "
              f"<1km: {np.mean(errs < 1000):.1%}")
        print(f"  True VP: median_rank={np.median(ranks):.0f}  "
              f"median_n70={int(np.median(n70s))}")

        for qi, ti in enumerate(test_idx):
            all_results.append({
                "condition": cond_name,
                "true_row": int(ti),
                "top1_err_m": float(errs[qi]),
                "true_rank": int(ranks[qi]),
                "n70": int(n70s[qi]),
            })

    # --- Summary table ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'condition':>12} | {'median':>8} {'rank0':>6} {'<100m':>6} {'<1km':>6} {'med_n70':>7}")
    print("-" * 60)
    conds = []
    for r in all_results:
        if r["condition"] not in conds:
            conds.append(r["condition"])
    for c in conds:
        sub = [r for r in all_results if r["condition"] == c]
        e = np.array([r["top1_err_m"] for r in sub])
        rk = np.array([r["true_rank"] for r in sub])
        n70 = np.array([r["n70"] for r in sub])
        print(f"{c:>12} | {np.median(e):>7.0f}m {np.mean(rk==0):>6.1%} "
              f"{np.mean(e<100):>6.1%} {np.mean(e<1000):>6.1%} "
              f"{int(np.median(n70)):>7d}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {output_path}")

    return all_results


def run_fov_sweep(n_samples=40, stride=10, seed=42):
    """Accuracy vs angular coverage. Uses src/matching.ncc_scores (max over
    shifts) chunked over DB rows so RAM stays bounded."""
    sys.path.insert(0, str(ROOT / "src"))
    from matching import ncc_scores, feature_bundle_matrix

    rng = np.random.RandomState(seed)
    print("=" * 72)
    print("FOV SWEEP (streaming)")
    print("=" * 72)

    H_sub, lat_sub, lon_sub = load_strided_db(stride)
    n_rows, n_bins = H_sub.shape
    test_idx = rng.choice(n_rows, size=min(n_samples, n_rows), replace=False)

    print(f"\n{'FOV':>5} | {'median':>9} {'<100m':>6} {'<1km':>6} {'<10km':>6}")
    print("-" * 44)

    for fov_deg in [30, 65, 90, 120, 180, 240, 300, 360]:
        w = max(8, int(fov_deg / BIN_DEG))
        start = (n_bins - w) // 2
        sl = slice(start, start + w)
        db_crop = np.ascontiguousarray(H_sub[:, sl])
        q_profiles = np.ascontiguousarray(H_sub[test_idx][:, sl])

        best_corr = np.full(len(test_idx), -np.inf)
        best_row = np.full(len(test_idx), -1, dtype=np.int64)

        CH = 20000
        for s0 in range(0, n_rows, CH):
            s1 = min(s0 + CH, n_rows)
            zv, zd = feature_bundle_matrix(db_crop[s0:s1])
            for qi, qp in enumerate(q_profiles):
                c, _ = ncc_scores(zv, zd, qp, bin_deg=BIN_DEG)
                j = int(np.argmax(c))
                if c[j] > best_corr[qi]:
                    best_corr[qi] = c[j]
                    best_row[qi] = s0 + j

        errs = np.array([
            geodesic((lat_sub[ti], lon_sub[ti]),
                     (lat_sub[r], lon_sub[r])).meters
            for ti, r in zip(test_idx, best_row)
        ])
        print(f"{fov_deg:>4}° | {np.median(errs):>8.0f}m "
              f"{np.mean(errs < 100):>6.1%} {np.mean(errs < 1000):>6.1%} "
              f"{np.mean(errs < 10000):>6.1%}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Off-grid synthetic evaluation")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10,
                    help="DB row subsample factor (10 -> ~300m grid)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--fov-sweep", action="store_true")
    args = ap.parse_args()

    out = args.output or str(ROOT / "data" / "street_view" / "offgrid_eval_results.json")

    if args.fov_sweep:
        run_fov_sweep(n_samples=args.samples, stride=args.stride, seed=args.seed)
    else:
        run_offgrid(n_samples=args.samples, stride=args.stride,
                    seed=args.seed, output_path=out)

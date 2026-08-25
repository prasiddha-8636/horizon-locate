#!/usr/bin/env python3
"""Noise robustness: n=100, memory-efficient.

Loads DB profiles one at a time. For each reference, builds a mini-DB
with the true match + N random distractors.
"""
import sys, json, time, argparse, pickle
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
OUT_JSON = ROOT / "data" / "street_view" / "noise_robustness_100.json"
CKPT_DIR = ROOT / "data" / "eval_ckpt"
N_BINS = 720
TOP_KEEP = 50


def zr(m):
    m = np.asarray(m, dtype=np.float64)
    mu = m.mean(axis=-1, keepdims=True)
    sd = m.std(axis=-1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (m - mu) / sd


def dog(mat, s1, s2):
    return (gaussian_filter1d(mat, s1, axis=1, mode="wrap")
            - gaussian_filter1d(mat, s2, axis=1, mode="wrap"))


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_profile(idx):
    """Load a single profile by global row index."""
    pf = pq.ParquetFile(str(DB_PATH))
    row_scan = 0
    for batch in pf.iter_batches(batch_size=50000, columns=["raw_horizon_deg"]):
        decoded = batch.to_pandas()["raw_horizon_deg"].to_numpy()
        n = len(decoded)
        if idx < row_scan + n:
            del pf
            return np.array(decoded[idx - row_scan], dtype=np.float64)
        row_scan += n
    del pf
    return None


def match_mini_db(query, mini_profiles, mini_lats, mini_lons, true_idx):
    """Match query against mini-DB using 3-scorer RRF."""
    q64 = np.asarray(query, dtype=np.float64)
    qv = zr(q64)
    qd = zr(np.gradient(qv))

    mini_matrix = np.array(mini_profiles)
    n_rows = len(mini_matrix)

    zv = zr(mini_matrix)
    zd = zr(np.gradient(zv, axis=1))
    Fv = np.fft.rfft(zv, axis=1)
    Fd = np.fft.rfft(zd, axis=1)
    Fb28 = np.fft.rfft(zr(dog(mini_matrix, 2.0, 8.0)), axis=1)
    Fb316 = np.fft.rfft(zr(dog(mini_matrix, 3.0, 16.0)), axis=1)

    spec_v_base = 0.5 * np.conj(np.fft.rfft(qv))
    spec_d_base = 0.5 * np.conj(np.fft.rfft(qd))
    spec_v_bp28 = np.conj(np.fft.rfft(zr(dog(q64[None, :], 2.0, 8.0))[0]))
    spec_v_bp316 = np.conj(np.fft.rfft(zr(dog(q64[None, :], 3.0, 16.0))[0]))

    cb = (np.fft.irfft(spec_v_base[None, :] * Fv, n=N_BINS, axis=1)
          + np.fft.irfft(spec_d_base[None, :] * Fd, n=N_BINS, axis=1)) / N_BINS
    s1 = cb.max(axis=1)

    c2 = np.fft.irfft(spec_v_bp28[None, :] * Fb28, n=N_BINS, axis=1) / N_BINS
    s2 = c2.max(axis=1)

    c3 = np.fft.irfft(spec_v_bp316[None, :] * Fb316, n=N_BINS, axis=1) / N_BINS
    s3 = c3.max(axis=1)

    k = 60
    rrf_scores = np.zeros(n_rows)
    for scores in [s1, s2, s3]:
        rl = np.argsort(-scores)
        for rank, row in enumerate(rl):
            rrf_scores[row] += 1.0 / (k + rank)

    best_row = int(np.argmax(rrf_scores))
    err = float(haversine(mini_lats[true_idx], mini_lons[true_idx],
                          mini_lats[best_row], mini_lons[best_row]) * 1000)

    # Rank of true match in scorer 1
    rank = int(np.where(np.argsort(-s1) == true_idx)[0][0])
    return err, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--distractors", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Load lat/lon only (lightweight)
    print("Loading DB coordinates...")
    pf0 = pq.ParquetFile(str(DB_PATH))
    n_db = pf0.metadata.num_rows
    del pf0
    meta_ll = pq.read_table(DB_PATH, columns=["lat", "lon"])
    all_lats = meta_ll.column("lat").to_pandas().values
    all_lons = meta_ll.column("lon").to_pandas().values
    print(f"  {n_db:,} viewpoints")

    # Pick reference indices with good terrain
    print("Selecting reference profiles with sufficient terrain relief...")
    ref_indices = []
    attempts = 0
    while len(ref_indices) < args.samples and attempts < n_db:
        idx = np.random.randint(0, n_db)
        if idx in set(ref_indices):
            continue
        prof = load_profile(idx)
        if prof is not None and np.std(prof) > 2.0 and np.max(prof) > 3.0:
            ref_indices.append(idx)
        attempts += 1
    ref_indices.sort()
    print(f"  Selected {len(ref_indices)} profiles ({attempts} attempts)")

    # Load reference profiles into memory (each is 720 × 8 bytes = 5.6 KB)
    ref_profiles = {}
    for idx in ref_indices:
        ref_profiles[idx] = load_profile(idx)

    # Noise levels
    noise_levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    conditions = [(f"noise={n}", n) for n in noise_levels] + [("quant_u8", -1)]

    # Check resume
    ckpt_path = CKPT_DIR / "noise100.pkl"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cached = {}
    if args.resume and ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            cached = pickle.load(f)
        print(f"  Resumed {len(cached)} cached conditions")

    results = {}
    total = len(conditions) * len(ref_indices)
    done = 0
    t0 = time.time()

    for cond_name, noise_sigma in conditions:
        if cond_name in cached:
            results[cond_name] = cached[cond_name]
            done += len(ref_indices)
            continue

        print(f"\n--- {cond_name} ---")
        cond_results = []

        for i, ref_idx in enumerate(ref_indices):
            ref_prof = ref_profiles[ref_idx]

            # Corrupt profile
            if noise_sigma < 0:  # quant_u8
                noisy = np.round(ref_prof * 2).clip(0, 255).astype(np.float32) / 2.0
            elif noise_sigma > 0:
                noisy = ref_prof + np.random.normal(0, noise_sigma, ref_prof.shape)
            else:
                noisy = ref_prof.copy()

            # Build mini-DB: true match + random distractors
            distractor_pool = np.random.choice(n_db, size=args.distractors + 10, replace=False)
            distractor_pool = distractor_pool[distractor_pool != ref_idx][:args.distractors]

            mini_profiles = [ref_prof]  # true match first
            mini_lats = [all_lats[ref_idx]]
            mini_lons = [all_lons[ref_idx]]

            for d_idx in distractor_pool:
                d_prof = load_profile(d_idx)
                if d_prof is not None:
                    mini_profiles.append(d_prof)
                    mini_lats.append(all_lats[d_idx])
                    mini_lons.append(all_lons[d_idx])

            mini_lats = np.array(mini_lats)
            mini_lons = np.array(mini_lons)

            err, rank = match_mini_db(noisy, mini_profiles, mini_lats, mini_lons, 0)

            cond_results.append({
                "ref_idx": int(ref_idx),
                "error_m": err,
                "rank": rank,
                "n_distractors": len(mini_profiles) - 1,
            })

            done += 1
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                eta = el / max(done, 1) * (total - done)
                errs_sofar = [r["error_m"] for r in cond_results]
                print(f"  [{i+1}/{len(ref_indices)}] {el:.0f}s elapsed, ~{eta:.0f}s left, "
                      f"<1km={sum(e<1000 for e in errs_sofar)}/{len(errs_sofar)}")

        results[cond_name] = cond_results

        # Save checkpoint
        with open(ckpt_path, "wb") as f:
            pickle.dump(results, f)

        # Summary
        errs = [r["error_m"] for r in cond_results]
        n100 = sum(e < 100 for e in errs)
        n1k = sum(e < 1000 for e in errs)
        print(f"  n={len(errs)}, <100m={n100}/{len(errs)} ({100*n100/len(errs):.1f}%), "
              f"<1km={n1k}/{len(errs)} ({100*n1k/len(errs):.1f}%), "
              f"median={np.median(errs):.0f}m")

    # Final summary
    print("\n" + "=" * 70)
    print(f"NOISE ROBUSTNESS (n={len(ref_indices)}, {args.distractors} distractors)")
    print("=" * 70)
    for cond_name, _ in conditions:
        if cond_name not in results:
            continue
        errs = [r["error_m"] for r in results[cond_name]]
        n100 = sum(e < 100 for e in errs)
        n1k = sum(e < 1000 for e in errs)
        print(f"  {cond_name:12s}: n={len(errs):3d}  "
              f"<100m={n100:3d} ({100*n100/len(errs):5.1f}%)  "
              f"<1km={n1k:3d} ({100*n1k/len(errs):5.1f}%)  "
              f"median={np.median(errs):8.0f}m")

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {"n_samples": len(ref_indices), "n_distractors": args.distractors,
           "seed": args.seed, "results": results}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2,
                   default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Idea 3a: DB-side spatial-neighborhood aggregation on cached NCC scores.

For each VP, score = max/mean of corr over VPs within radius r (m).
Neighbors stored as CSR (flat int32 + indptr) to avoid OOM from Python
list-of-lists at large radii; per-sample aggregation via reduceat.

Radii: [30, 60, 90, 120] m.
"""

import json, os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
RADII = [30, 60, 90, 120]
CHUNK = 100_000


def latlon_to_ecef(lat_deg, lon_deg):
    R = 6371000.0
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    return np.column_stack(
        [R * np.cos(phi) * np.cos(lam), R * np.cos(phi) * np.sin(lam), R * np.sin(phi)]
    )


def build_csr(tree, ecef, r):
    """Return (flat int32 neighbor ids, indptr int64) for all VPs within r m."""
    n = len(ecef)
    counts = np.zeros(n, dtype=np.int64)
    flats = []
    pos = 0
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        for nlist in tree.query_ball_point(ecef[s:e], r, workers=-1):
            counts[pos] = len(nlist)
            if len(nlist):
                flats.append(np.asarray(nlist, dtype=np.int32))
            pos += 1
    flat = np.concatenate(flats) if flats else np.zeros(0, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return flat, indptr


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()
    ecef = latlon_to_ecef(lat, lon)
    tree = cKDTree(ecef)
    N = len(lon)
    print(f"VPs: {N}")

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [
        sid
        for sid in ann
        if sid in gt
        and ann[sid] is not None
        and os.path.exists(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))
    ]
    print(f"samples: {len(sids)}")

    corrs = {
        sid: np.load(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))["corr"] for sid in sids
    }

    variants = [f"r{r}m_{op}" for r in RADII for op in ("max", "mean")]
    results = {v: {"errs": [], "n": 0} for v in variants}
    best_by_sample = {sid: {} for sid in sids}

    print(f"\n{'sid':<20}" + "".join(f" {v:>11}" for v in variants))
    for r in RADII:
        flat, indptr = build_csr(tree, ecef, r)
        lens = np.diff(indptr).astype(np.float64)
        print(f"  r={r}m CSR: {len(flat)} edges ({len(flat) / 1e6:.1f}M)", flush=True)
        for sid in sids:
            corr = corrs[sid]
            seg = corr[flat]
            agg_max = np.maximum.reduceat(seg, indptr[:-1])
            agg_mean = np.add.reduceat(seg, indptr[:-1]) / np.where(lens > 0, lens, 1)
            agg_max[lens == 0] = -np.inf
            agg_mean[lens == 0] = np.nan
            for op, agg in (("max", agg_max), ("mean", agg_mean)):
                v = f"r{r}m_{op}"
                bv = int(np.nanargmax(agg))
                e = geodesic(
                    (lat[bv], lon[bv]),
                    (np.array(gt[sid]["true_lat"]), np.array(gt[sid]["true_lon"])),
                ).meters
                results[v]["errs"].append(e)
                results[v]["n"] += 1
                best_by_sample[sid][v] = (bv, float(agg[bv]), e)
        del flat, indptr
        gc_count = None
        print(f"  r={r}m done", flush=True)

    print("\n" + "=" * 100)
    for sid in sids:
        line = f"{sid:<20}"
        for v in variants:
            e = best_by_sample[sid][v][2]
            line += f" {e / 1000:10.1f}km"
        print(line, flush=True)

    print("\n" + "=" * 100)
    print(
        f"{'variant':<14} {'N':>4} {'top1@500m':>12} {'median':>10} {'<1km':>5} {'<5km':>5} {'<10km':>6} {'<20km':>6}"
    )
    print("-" * 75)
    for v in variants:
        r = results[v]
        errs = np.array(r["errs"])
        med = np.median(errs)
        t1 = sum(e < 500 for e in errs)
        lt1 = sum(e < 1000 for e in errs)
        lt5 = sum(e < 5000 for e in errs)
        lt10 = sum(e < 10000 for e in errs)
        lt20 = sum(e < 20000 for e in errs)
        print(
            f"{v:<14} {r['n']:4d} {t1:5d}/{r['n']:<5d} {med / 1000:8.1f}km {lt1:5d} {lt5:5d} {lt10:6d} {lt20:6d}"
        )

    import pickle

    with open(os.path.join(CACHE_DIR, "idea3a_best_by_sample.pkl"), "wb") as f:
        pickle.dump(best_by_sample, f)
    print("\nsaved idea3a_best_by_sample.pkl")


if __name__ == "__main__":
    main()

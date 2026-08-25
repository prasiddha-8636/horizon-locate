#!/usr/bin/env python
"""GSV metric improvement eval: bandpass NCC + rank fusion on FUSED profiles.

Combines the two strongest proven signals, which have never been tested together:
  * Wide-FOV fusion (breaks valley symmetry; 87.7% true VP in top-5)
  * Bandpass NCC bp(2,8)/bp(3,16) (collapses imposter effect on single crops)

Scorers compared per panorama (one streaming DB pass, shared DB FFTs):
  S0  baseline : value+d1 feature bundle (current production)
  S1  bp(2,8)  : plain Pearson on DoG-bandpassed horizons
  S2  bp(3,16) : plain Pearson, broader band
  S3  RRF      : reciprocal-rank fusion of S0/S1/S2 top-50 lists
  ORACLE       : best of the four (upper bound)

Usage:
  python scripts/gsv_improve_eval.py [--stride 2]
"""

import sys
import json
import time
import heapq
import argparse
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from horizon_format import decode_horizon_column
from calibrate_and_eval_multiphoto import fuse_pano

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
OUT_JSON = ROOT / "data" / "street_view" / "gsv_improve_eval_results.json"

BIN_DEG = 0.5
N_BINS = int(360 / BIN_DEG)
CHUNK = 8000
TOP_KEEP = 50

BASE_SCORERS = ["baseline", "bp28", "bp316"]


def zr(m):
    m = np.asarray(m, dtype=np.float64)
    mu = m.mean(axis=-1, keepdims=True)
    sd = m.std(axis=-1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (m - mu) / sd


def dog(mat, s1, s2):
    return (gaussian_filter1d(mat, s1, axis=1, mode="wrap")
            - gaussian_filter1d(mat, s2, axis=1, mode="wrap"))


class ScorerState:
    """Best hit + bounded top-K heap for one pano under one scorer."""

    __slots__ = ("spec_v", "spec_d", "best_score", "best_row",
                 "best_lat", "best_lon", "heap")

    def __init__(self, spec_v=None, spec_d=None):
        self.spec_v = spec_v          # conj(rfft(query value feature)) * weight
        self.spec_d = spec_d          # same for d1 channel; None for bandpass
        self.best_score = -np.inf
        self.best_row = -1
        self.best_lat = self.best_lon = None
        self.heap = []

    def update(self, corr, row_start, lats, lons):
        j = int(np.argmax(corr))
        s = float(corr[j])
        if s > self.best_score:
            self.best_score = s
            self.best_row = row_start + j
            self.best_lat = float(lats[j])
            self.best_lon = float(lons[j])

        thr = self.heap[0][0] if len(self.heap) >= TOP_KEEP else -np.inf
        over = np.where(corr > thr)[0]
        if over.size > 256:
            keep = np.argpartition(-corr[over], min(255, over.size - 1))[:256]
            over = over[keep]
        for i in over:
            item = (float(corr[i]), row_start + int(i),
                    float(lats[i]), float(lons[i]))
            if len(self.heap) < TOP_KEEP:
                heapq.heappush(self.heap, item)
            elif item[0] > self.heap[0][0]:
                heapq.heapreplace(self.heap, item)


def build_states(profile):
    """Per-scorer frequency-domain query multipliers."""
    q64 = np.asarray(profile, dtype=np.float64)
    qv = zr(q64)
    qd = zr(np.gradient(qv))

    baseline = ScorerState(
        spec_v=0.5 * np.conj(np.fft.rfft(qv)),
        spec_d=0.5 * np.conj(np.fft.rfft(qd)),
    )
    states = {"baseline": baseline}

    for name, (s1, s2) in [("bp28", (2.0, 8.0)), ("bp316", (3.0, 16.0))]:
        qb = zr(dog(q64[None, :], s1, s2))[0]
        states[name] = ScorerState(
            spec_v=np.conj(np.fft.rfft(qb)), spec_d=None)
    return states


def score_chunk(states_by_pano, H_chunk, lats, lons, row_start):
    """Score every pano under every scorer against one DB chunk."""
    zv = zr(H_chunk)
    zd = zr(np.gradient(zv, axis=1))
    Fv = np.fft.rfft(zv, axis=1)
    Fd = np.fft.rfft(zd, axis=1)
    del zv, zd

    Fb28 = np.fft.rfft(zr(dog(H_chunk, 2.0, 8.0)), axis=1)
    Fb316 = np.fft.rfft(zr(dog(H_chunk, 3.0, 16.0)), axis=1)

    for sts in states_by_pano:
        cb = (np.fft.irfft(sts["baseline"].spec_v[None, :] * Fv,
                           n=N_BINS, axis=1)
              + np.fft.irfft(sts["baseline"].spec_d[None, :] * Fd,
                             n=N_BINS, axis=1)) / N_BINS
        sts["baseline"].update(cb.max(axis=1), row_start, lats, lons)
        del cb

        c1 = np.fft.irfft(sts["bp28"].spec_v[None, :] * Fb28,
                          n=N_BINS, axis=1) / N_BINS
        sts["bp28"].update(c1.max(axis=1), row_start, lats, lons)
        del c1

        c2 = np.fft.irfft(sts["bp316"].spec_v[None, :] * Fb316,
                          n=N_BINS, axis=1) / N_BINS
        sts["bp316"].update(c2.max(axis=1), row_start, lats, lons)
        del c2


# ---------------------------------------------------------------------------
# Post-pass: errors, ranks, RRF fusion
# ---------------------------------------------------------------------------

def heap_err(state, true_lat, true_lon):
    if state.best_lat is None or true_lat is None:
        return float("inf")
    return geodesic((true_lat, true_lon),
                    (state.best_lat, state.best_lon)).meters


def rrf_top1(sts, k=60):
    """Reciprocal-rank fusion over the three top-K heaps."""
    scores = {}
    for name in BASE_SCORERS:
        ranked = sorted(sts[name].heap, key=lambda x: -x[0])
        for rank, (_, row, _, _) in enumerate(ranked):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
    # map best lat/lon per row from any heap entry
    latlon = {}
    for name in BASE_SCORERS:
        for _, row, lat, lon in sts[name].heap:
            if row not in latlon:
                latlon[row] = (lat, lon)
    if not scores:
        return None, float("inf"), {}
    best_row = max(scores, key=scores.get)
    votes = sum(1 for name in BASE_SCORERS
                if any(e[1] == best_row for e in sts[name].heap))
    lat, lon = latlon.get(best_row, (None, None))
    return (lat, lon), votes, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    stride = args.stride

    print("=" * 76)
    print("GSV IMPROVEMENT EVAL — fused profiles x {baseline, bp28, bp316, RRF}")
    print("=" * 76)
    print(f"DB stride={stride}\n")

    with open(GT_FILE) as f:
        gt_data = json.load(f)
    annots = json.loads(ANNOT_FILE.read_text()).get("annotations", {})

    panos = {}
    for sid, points in annots.items():
        meta_p = CROPS_DIR / f"{sid}.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        pid = meta.get("pano_id")
        if not pid:
            continue
        meta["sid"] = sid
        meta["points"] = points
        panos.setdefault(pid, []).append(meta)
    multi = {k: v for k, v in panos.items() if len(v) >= 2}
    print(f"Found {len(multi)} multi-crop panoramas")

    prepared = {}
    for pid, crops in sorted(multi.items()):
        fused, cov = fuse_pano(crops, gt_data, bin_deg=BIN_DEG)
        if fused is None or len(fused) != N_BINS:
            continue
        prepared[pid] = {"states": build_states(fused), "cov": cov}
    pano_ids = list(prepared.keys())

    # --- true VP rows ---
    truth = {}
    for pid in pano_ids:
        g = gt_data.get(pid) or {}
        la = g.get("true_lat") or g.get("lat")
        lo = g.get("true_lon") or g.get("lon")
        if la is not None:
            truth[pid] = (la, lo)
    try:
        from scipy.spatial import cKDTree
        pf = pq.ParquetFile(str(DB_PATH))
        las, los = [], []
        for b in pf.iter_batches(batch_size=50000, columns=["lat", "lon"]):
            df = b.to_pandas()
            las.append(df["lat"].to_numpy())
            los.append(df["lon"].to_numpy())
        db_lat, db_lon = np.concatenate(las), np.concatenate(los)
        mcos = np.cos(np.deg2rad(db_lat.mean()))
        tree = cKDTree(np.column_stack([db_lon * mcos, db_lat]))
        for pid, (la, lo) in truth.items():
            d, idx = tree.query([lo * mcos, la], k=1)
            if d * 111_320 < 250:
                prepared[pid]["true_row"] = int(idx)
            else:
                prepared[pid]["true_row"] = None
        del db_lat, db_lon, tree
        n_tr = sum(1 for p in pano_ids if prepared[p].get("true_row"))
        print(f"True VP row resolved for {n_tr}/{len(pano_ids)} panos\n")
    except Exception as e:
        print(f"KDTree failed ({e}); ranks skipped\n")
        for pid in pano_ids:
            prepared[pid]["true_row"] = None

    # --- streaming pass ---
    pf = pq.ParquetFile(str(DB_PATH))
    total_chunks = (pf.metadata.num_rows + CHUNK - 1) // CHUNK
    states_list = [prepared[p]["states"] for p in pano_ids]
    t0 = time.time()
    pos = done = 0

    for batch in pf.iter_batches(batch_size=CHUNK,
                                 columns=["raw_horizon_deg", "lat", "lon"]):
        df = batch.to_pandas()
        n = len(df)
        sel = slice(None) if stride == 1 else slice(None, None, stride)
        Hc = decode_horizon_column(df["raw_horizon_deg"].to_numpy())[sel]
        lats = df["lat"].to_numpy()[sel]
        lons = df["lon"].to_numpy()[sel]
        del df, batch

        score_chunk(states_list, Hc, lats, lons, pos)
        pos += n
        done += 1
        if done % 25 == 0 or done == total_chunks:
            el = time.time() - t0
            eta = el / done * (total_chunks - done)
            print(f"  chunk {done}/{total_chunks}  {el:.0f}s elapsed, "
                  f"~{eta:.0f}s left", flush=True)

    # --- evaluate ---
    print("\nEvaluating...\n")
    results = []
    for pid in pano_ids:
        P = prepared[pid]
        tl, lo_ = truth.get(pid, (None, None))
        errs = {name: heap_err(P["states"][name], tl, lo_)
                for name in BASE_SCORERS}
        (rlat, rlon), rrf_votes, _ = rrf_top1(P["states"])
        if rlat is not None and tl is not None:
            errs["rrf"] = geodesic((tl, lo_), (rlat, rlon)).meters
        else:
            errs["rrf"] = float("inf")

        ranks = {}
        tr = P.get("true_row")
        for name in BASE_SCORERS:
            if tr is None:
                ranks[name] = -1
            else:
                ranked = sorted(P["states"][name].heap, key=lambda x: -x[0])
                ranks[name] = next(
                    (r for r, e in enumerate(ranked) if e[1] == tr), TOP_KEEP)

        results.append({
            "pano_id": pid, "coverage_deg": P["cov"],
            "errs": errs, "ranks": ranks,
            "rrf_votes": int(rrf_votes),
            "top1": {name: [P["states"][name].best_lat,
                             P["states"][name].best_lon]
                     for name in BASE_SCORERS},
            "true_in_top50": {name: bool(ranks[name] < TOP_KEEP)
                              for name in BASE_SCORERS},
        })

        def fmt(e):
            return f"{e:8.0f}" if e != float("inf") else "     inf"

        print(f"  {pid[:20]:20s} FOV={P['cov']:4.0f}°  "
              f"base={fmt(errs['baseline'])}m(bp28={fmt(errs['bp28'])}m) "
              f"(bp316={fmt(errs['bp316'])}m) rrf={fmt(errs['rrf'])}m "
              f"votes={rrf_votes}")

    oracle = [min(r["errs"].values()) for r in results]

    def table(label, get_err):
        es = np.array([get_err(r) for r in results], dtype=float)
        es = es[np.isfinite(es)]
        if es.size == 0:
            return f"{label:<12} N=0"
        return (f"{label:<12} median={np.median(es)/1000:6.1f}km  "
                f"<100m={np.mean(es < 100):5.1%}  <1km={np.mean(es < 1000):5.1%}  "
                f"<5km={np.mean(es < 5000):5.1%}  <10km={np.mean(es < 10000):5.1%}")

    print("\n" + "=" * 76)
    print("RESULTS (all multi-crop panos)")
    print("=" * 76)
    for name in SCORERS_LIST:
        print(table(name.upper(), lambda r, n=name: r["errs"][n]))
    print(table("oracle", lambda r: min(r["errs"].values())))

    wide = [r for r in results if r["coverage_deg"] >= 200]
    if wide:
        print("\n--- Wide-FOV subset (>=200°) ---")
        for name in SCORERS_LIST + ["oracle"]:
            get = ((lambda r, n=name: r["errs"][n]) if name != "oracle"
                   else (lambda r: min(r["errs"].values())))
            es = np.array([get(r) for r in wide], dtype=float)
            es = es[np.isfinite(es)]
            if es.size:
                print(f"{name.upper():<12} median={np.median(es)/1000:6.1f}km  "
                      f"<100m={np.mean(es < 100):5.1%}  "
                      f"<1km={np.mean(es < 1000):5.1%}  "
                      f"<10km={np.mean(es < 10000):5.1%}")

    # --- confidence gate: cross-scorer consensus -----------------------
    # A match is CONFIDENT when all three scorers' top-1 predictions land
    # within D of each other (max pairwise geodesic distance).
    print("\n" + "=" * 76)
    print("CONFIDENCE GATE: cross-scorer top-1 consensus")
    print("=" * 76)

    def consensus_dist(r):
        pts = [tuple(r["top1"][n]) for n in BASE_SCORERS]
        if any(p[0] is None for p in pts):
            return float("inf")
        return max(geodesic(pts[i], pts[j]).meters
                   for i in range(3) for j in range(i + 1, 3))

    print(f"{'threshold':>10} {'N accept':>9} {'<100m':>12} {'<1km':>12} "
          f"{'median':>9}   (of all panos)")
    for thr_km in (0.5, 1.0, 2.0, 5.0):
        acc = [r for r in results if consensus_dist(r) <= thr_km * 1000]
        if not acc:
            print(f"{thr_km:>8}km {0:>9}   --")
            continue
        es = np.array([r["errs"]["rrf"] for r in acc], dtype=float)
        es = es[np.isfinite(es)]
        hits100 = int(np.sum(es < 100))
        hits1k = int(np.sum(es < 1000))
        med = np.median(es) / 1000 if es.size else float("nan")
        print(f"{thr_km:>8}km {len(acc):>9} {hits100:>6}/{len(acc):<3d} "
              f"{hits1k:>5}/{len(acc):<3d} {med:>7.1f}km")

    out = {
        "stride": stride,
        "results": [
            {"pano_id": r["pano_id"], "coverage_deg": r["coverage_deg"],
             **{f"err_{k}": v for k, v in r["errs"].items()},
             **{f"rank_{k}": v for k, v in r["ranks"].items()},
             **{f"top1_{k}": v for k, v in r["top1"].items()},
             "rrf_votes": r["rrf_votes"]}
            for r in results
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


SCORERS_LIST = ["baseline", "bp28", "bp316", "rrf"]

if __name__ == "__main__":
    main()

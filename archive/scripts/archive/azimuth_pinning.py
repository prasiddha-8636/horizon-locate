#!/usr/bin/env python
"""Can azimuth-pinning help?

For each annotated sample:
  s_true = azimuth shift aligning the profile to the DB at the TRUE VP
           (ground-truth peek — this is a CEILING test, not honest eval).
  - baseline:        best-over-shifts NCC per VP (cached) — current matcher
  - shift0:          NCC at fixed shift 0 (perfect GSV heading)
  - fixed_s_true:    NCC at the true azimuth shift only (ceiling if solar
                     compass pinned the azimuth exactly)

If fixed_s_true >> baseline for true VP rank, azimuth DOF is the lever and
solar pinning could help. If not, Idea 4 is dead regardless of solar.
"""

import json, os, sys, pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, feature_bundle_matrix
from src.horizon_format import decode_horizon_column

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
W, H = 1080, 720
RES_FILE = os.path.join(CACHE_DIR, "idea4_diag_results.pkl")


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


def fixed_shift_scan(q_val, q_d1, M, L, shifts, pf, decode_chunks):
    """NCC at fixed shifts only, O(N·M) per feature instead of O(N·L·logL).

    Returns dict shift -> (N,) score array.
    """
    qv = q_val - q_val.mean()
    qd = q_d1 - q_d1.mean()
    qv_n = np.linalg.norm(qv)
    qd_n = np.linalg.norm(qd)
    scores = {s: np.zeros(0, dtype=np.float64) for s in shifts}
    for v, d1 in decode_chunks:
        N = len(v)
        ve = np.concatenate([v, v[:, : M - 1]], axis=1)
        de = np.concatenate([d1, d1[:, : M - 1]], axis=1)
        cum = np.concatenate(
            [np.zeros((N, 1), dtype=np.float64), np.cumsum(ve, axis=1)], axis=1
        )
        cum_sq = np.concatenate(
            [np.zeros((N, 1), dtype=np.float64), np.cumsum(ve**2, axis=1)], axis=1
        )
        win_sum = cum[:, M : M + L] - cum[:, :L]
        win_sq = cum_sq[:, M : M + L] - cum_sq[:, :L]
        win_var = win_sq - win_sum**2 / M
        win_norm = np.sqrt(np.maximum(win_var, 0.0))
        for s in shifts:
            num_v = ve[:, s : s + M] @ qv
            denom_v = qv_n * win_norm[:, s]
            ncc_v = num_v / np.maximum(denom_v, 1e-12)
            d1_win = de[:, s : s + M]
            d1m = d1_win - d1_win.mean(axis=1, keepdims=True)
            win_norm_d1 = np.linalg.norm(d1m, axis=1)
            num_d1 = d1m @ qd
            denom_d1 = qd_n * win_norm_d1
            ncc_d1 = num_d1 / np.maximum(denom_d1, 1e-12)
            scores[s] = np.concatenate([scores[s], 0.5 * ncc_v + 0.5 * ncc_d1])
    return scores


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()

    pf = pq.ParquetFile(DB_PATH)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
    L = int(360.0 / BIN_DEG)

    def decode_chunks():
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            chunk = decode_horizon_column(
                batch.to_pandas()["raw_horizon_deg"].to_numpy()
            )
            yield feature_bundle_matrix(chunk)

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [
        sid
        for sid in ann
        if sid in gt
        and ann[sid] is not None
        and os.path.exists(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))
    ]

    print(f"VPs: {len(lon)}, bin_deg={BIN_DEG}, samples: {len(sids)}", flush=True)

    done = {}
    if os.path.exists(RES_FILE):
        done = pickle.load(open(RES_FILE, "rb"))
        print(f"resuming: {len(done)} samples already done", flush=True)

    rows = []
    for si, sid in enumerate(sids):
        if sid in done:
            print(f"[{si + 1}] {sid}: cached", flush=True)
            rows.append(done[sid])
            continue
        g = gt[sid]
        vp_true = int(g["closest_viewpoint_id"])
        tlat, tlon = g["true_lat"], g["true_lon"]
        fov = g["fov_y_deg"]
        base_tilt = np.array(g["cam_R_tilt"])

        mask = mask_from_points(ann[sid])
        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=base_tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            continue
        profile = pr["profile"]
        M = len(profile)
        q_val, q_d1 = _feature_bundle(profile)

        # s_true from the true VP horizon
        h_true = fetch_horizons([vp_true], rg_starts, pf)[0]
        vv, dd = feature_bundle_matrix(h_true[None, :])
        s_true = 0
        best_c = -np.inf
        qv = q_val - q_val.mean()
        qd = q_d1 - q_d1.mean()
        for s in range(L):
            win = np.roll(h_true, -s)[:M]
            # full feature at this shift
            wv, wd = _feature_bundle(win)
            c = 0.5 * _corr(wv, q_val) + 0.5 * _corr(wd, q_d1)
            if c > best_c:
                best_c, s_true = c, s

        # fixed-shift scans (fast path: only shifts {0, s_true})
        scores = fixed_shift_scan(q_val, q_d1, M, L, (0, s_true), pf, decode_chunks())
        base_corr = np.load(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))["corr"]

        def err_at(scores_arr):
            bv = int(np.argmax(scores_arr))
            return geodesic((lat[bv], lon[bv]), (tlat, tlon)).meters, bv

        e_b, _ = err_at(base_corr)
        e_0, _ = err_at(scores[0])
        e_s, _ = err_at(scores[s_true])
        e_true = geodesic((lat[vp_true], lon[vp_true]), (tlat, tlon)).meters
        r_b = int((base_corr > base_corr[vp_true]).sum())
        r_0 = int((scores[0] > scores[0][vp_true]).sum())
        r_s = int((scores[s_true] > scores[s_true][vp_true]).sum())

        shift_deg = (s_true % L) * BIN_DEG
        if shift_deg > 180:
            shift_deg -= 360
        print(
            f"[{si + 1}] {sid:<20} s_true={shift_deg:+7.1f}° "
            f"| base e={e_b / 1000:6.1f}km rank={r_b:>6} | shift0 e={e_0 / 1000:6.1f}km rank={r_0:>6} "
            f"| ceil e={e_s / 1000:6.1f}km rank={r_s:>6} | trueVP={e_true / 1000:6.1f}km",
            flush=True,
        )

        rows.append((sid, shift_deg, e_b, r_b, e_0, r_0, e_s, r_s, e_true))
        done[sid] = rows[-1]
        pickle.dump(done, open(RES_FILE, "wb"))
        print(f"  saved {sid}", flush=True)

    print("\n" + "=" * 110, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 110, flush=True)
    rows = np.array(rows, dtype=object)
    for lbl, ri in (("baseline", 2), ("shift0", 4), ("ceiling_s_true", 6)):
        e = rows[:, ri].astype(float)
        rank = rows[:, ri + 1].astype(int)
        med = np.median(e)
        t1 = sum(x < 500 for x in e)
        lt1 = sum(x < 1000 for x in e)
        lt5 = sum(x < 5000 for x in e)
        lt10 = sum(x < 10000 for x in e)
        print(
            f"{lbl:<16} median={med / 1000:6.1f}km top1@500m={t1}/{len(e)} "
            f"<1km={lt1} <5km={lt5} <10km={lt10} median_rank={np.median(rank):.0f}",
            flush=True,
        )

    shifts = rows[:, 1].astype(float)
    print(
        f"\n|s_true| dist (deg): med={np.median(np.abs(shifts)):.1f} "
        f"max={np.max(np.abs(shifts)):.1f} (>20°: {(np.abs(shifts) > 20).sum()}/{len(shifts)})",
        flush=True,
    )

    with open(os.path.join(CACHE_DIR, "idea4_diag.pkl"), "wb") as f:
        pickle.dump(
            {
                "rows": [
                    dict(
                        zip(
                            [
                                "sid",
                                "s_true_deg",
                                "e_base",
                                "rank_base",
                                "e_shift0",
                                "rank_shift0",
                                "e_ceil",
                                "rank_ceil",
                                "e_true",
                            ],
                            r,
                        )
                    )
                    for r in rows
                ]
            },
            f,
        )
    print("saved idea4_diag.pkl", flush=True)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


if __name__ == "__main__":
    main()

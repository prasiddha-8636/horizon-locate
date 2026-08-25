#!/usr/bin/env python
"""GSV eval against the full skyline DB.

Canonical harness: correct cumulative-row-group fetch (never `vp // 4096`), streaming
DB chunks via `iter_batches`, no compass/expected-offset gating (matching is
azimuth-shift-invariant; per-pano column-0 rotation is unknown).

Per sample:
  - extract profile from U-Net mask: fov_y from GT, r_tilt=cam_R_tilt, bin_deg=0.5
  - true-VP FB_best (best circular offset) — mask-quality metric
  - coarse scan over the full DB (spatial `stride`) -> top-5 by FB
  - geodesic error of best match

Usage:
  python scripts/gsv_eval.py --limit 50 --stride 12
  python scripts/gsv_eval.py --stride 12                # all 1808
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.matching import fft_prefilter
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8, decode_horizon_column

DB_PATH = ROOT / "notebooks/02_SkylineDatabase/output/skyline_db.parquet"
GT_PATH = ROOT / "data/street_view/ground_truth.json"
MASKS_DIR = ROOT / "data/street_view/masks"

_pf = None
_rg_starts = None


def _db():
    global _pf, _rg_starts
    if _pf is None:
        _pf = pq.ParquetFile(str(DB_PATH))
        sizes = [_pf.metadata.row_group(i).num_rows for i in range(_pf.num_row_groups)]
        _rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    return _pf, _rg_starts


def fetch_horizon(vp_idx):
    pf, rg_starts = _db()
    rg = int(np.searchsorted(rg_starts, vp_idx, side="right") - 1)
    pos = vp_idx - rg_starts[rg]
    batch = pf.read_row_group(rg, columns=["raw_horizon_deg"])
    return decode_horizon_uint8(batch.to_pandas()["raw_horizon_deg"].iloc[pos])


def fb_at_best(profile, horizon):
    from src.matching import _feature_bundle, _pearson_ncc_batch, feature_bundle_matrix

    qv, qd = _feature_bundle(profile)
    qv = qv - qv.mean()
    qd = qd - qd.mean()
    dbv, dbd = feature_bundle_matrix(horizon[None, :])
    L = len(profile)
    ext_v = np.concatenate([dbv, dbv[:, : L - 1]], axis=1)
    ext_d = np.concatenate([dbd, dbd[:, : L - 1]], axis=1)
    comb = 0.5 * _pearson_ncc_batch(
        ext_v, qv, np.linalg.norm(qv)
    ) + 0.5 * _pearson_ncc_batch(ext_d, qd, np.linalg.norm(qd))
    best = int(comb.argmax())
    return float(comb.max()), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    ap.add_argument("--stride", type=int, default=12, help="spatial VP stride")
    ap.add_argument(
        "--out", type=str, default=str(ROOT / "data/street_view/gsv_eval.csv")
    )
    args = ap.parse_args()

    print("Loading ground truth.", flush=True)
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    sids = list(gt_data.keys())
    if args.limit:
        sids = sids[: args.limit]
    print(f"Total samples: {len(sids)}", flush=True)

    print("Loading DB metadata (lon/lat).", flush=True)
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon_arr, lat_arr = meta["lon"].to_numpy(), meta["lat"].to_numpy()
    nv = len(lon_arr)
    del meta

    first = next(
        pq.ParquetFile(str(DB_PATH)).iter_batches(
            batch_size=1, columns=["raw_horizon_deg"]
        )
    )
    bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
    print(f"bin_deg={bin_deg}  n_vps={nv}  stride={args.stride}", flush=True)

    rows = []
    t0 = time.time()
    n_prof_fail = 0
    for i, sid in enumerate(sids):
        g = gt_data[sid]
        mask_path = MASKS_DIR / f"{sid}.png"
        if not mask_path.exists():
            n_prof_fail += 1
            continue

        fov = g.get("fov_y_deg", 65.0)
        r_tilt = np.array(g["cam_R_tilt"]) if g.get("cam_R_tilt") else None
        pr = extract_elevation_profile(
            str(mask_path), fov_y_deg=fov, r_tilt=r_tilt, bin_deg=bin_deg
        )
        if not pr["ok"]:
            n_prof_fail += 1
            rows.append({"sid": sid, "status": pr["status"], "error_m": float("nan")})
            continue
        profile = pr["profile"]
        tl, tn = g["true_lat"], g["true_lon"]

        vp = int(g.get("closest_viewpoint_id", -1))
        fb_true, _ = (
            fb_at_best(profile, fetch_horizon(vp)) if vp >= 0 else (float("nan"), 0)
        )

        best = None
        best_corr = -np.inf
        pf = pq.ParquetFile(str(DB_PATH))
        chunk_start = 0
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            chunk = decode_horizon_column(
                batch.to_pandas()["raw_horizon_deg"].to_numpy()
            )
            stride_rows = chunk[:: args.stride]
            corr, offsets = fft_prefilter(stride_rows, profile, bin_deg)
            # select by MAX CORRELATION (the honest matcher), never by distance to truth
            k = int(np.argmax(corr))
            if corr[k] > best_corr:
                best_corr = float(corr[k])
                global_idx = chunk_start + k * args.stride
                best = {
                    "row_index": int(global_idx),
                    "error_m": float(
                        geodesic(
                            (tl, tn), (lat_arr[global_idx], lon_arr[global_idx])
                        ).meters
                    ),
                    "fft_corr": best_corr,
                }
            chunk_start += len(chunk)
            del chunk, corr, offsets

        if best is None:
            n_prof_fail += 1
            rows.append({"sid": sid, "status": "NO_MATCH", "error_m": float("nan")})
            continue

        rows.append(
            {
                "sid": sid,
                "status": "OK",
                "error_m": best["error_m"],
                "fb_true_vp": fb_true,
                "best_corr": best["fft_corr"],
                "best_vp": best["row_index"],
            }
        )
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(sids)} done, {el / (i + 1):.1f}s/sample", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    ok = df[df["status"] == "OK"]
    print(f"\n=== GSV EVAL ===", flush=True)
    print(
        f"Samples: {len(sids)}  OK: {len(ok)}  profile/other fail: {n_prof_fail}",
        flush=True,
    )
    if len(ok):
        e = ok["error_m"].to_numpy()
        print(f"Median error: {np.median(e):.0f} m", flush=True)
        print(f"Mean error: {np.mean(e):.0f} m", flush=True)
        for d in (100, 200, 500, 1000, 2000, 5000):
            print(f"  top-1<{d}m: {np.mean(e < d) * 100:.1f}%", flush=True)
        if "fb_true_vp" in ok and ok["fb_true_vp"].notna().any():
            fb = ok["fb_true_vp"].to_numpy()
            fb = fb[~np.isnan(fb)]
            print(
                f"True-VP FB_best: median {np.median(fb):.3f}  p25 {np.percentile(fb, 25):.3f}  p75 {np.percentile(fb, 75):.3f}",
                flush=True,
            )
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()

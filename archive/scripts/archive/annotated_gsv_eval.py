#!/usr/bin/env python
"""Non-circular GSV validation with hand-annotated skylines.

For each annotated sample: points -> skyline mask -> elevation profile ->
honest matcher (max-corr, stride 12) -> true-VP rank + best-match error.
Optional pitch sweep absorbs the unresolved cam_R_tilt vs crop_pitch geometry.

Usage:
  python scripts/annotated_gsv_eval.py
"""

import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import fft_prefilter
from src.horizon_format import decode_horizon_column
from scripts.gsv_eval import fetch_horizon, fb_at_best, DB_PATH

_meta = pq.read_table(str(DB_PATH), columns=["lon", "lat"])
lat_arr = _meta.column("lat").to_pandas().values
lon_arr = _meta.column("lon").to_pandas().values
NVPS = len(lat_arr)
_first = next(
    pq.ParquetFile(str(DB_PATH)).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(_first.to_pandas()["raw_horizon_deg"].iloc[0])

GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
W, H = 1080, 720
PITCHES = [0.0, -5.0, -3.0, 3.0, 5.0]
STRIDE = 12
SWEEP_STRIDE = 50


def r_pitch(delta_deg):
    cp, sp = np.cos(np.radians(delta_deg)), np.sin(np.radians(delta_deg))
    return np.array([[1.0, 0, 0], [0, cp, -sp], [0, sp, cp]])


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


def best_match(profile, stride):
    """Max-corr scan (honest matcher). Returns (error_m, best_corr)."""
    pf = pq.ParquetFile(str(DB_PATH))
    best_corr = -np.inf
    best_idx = -1
    chunk_start = 0
    for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        stride_rows = chunk[::stride]
        corr, _ = fft_prefilter(stride_rows, profile, BIN_DEG)
        k = int(np.argmax(corr))
        if corr[k] > best_corr:
            best_corr = float(corr[k])
            best_idx = chunk_start + k * stride
        chunk_start += len(chunk)
        del chunk, corr
    return best_idx, best_corr


def main():
    with open(GT_FILE) as f:
        gt = json.load(f)
    if not os.path.exists(ANNOT_FILE):
        print(f"No annotations file at {ANNOT_FILE} — run annotate_gsv.py first.")
        return
    with open(ANNOT_FILE) as f:
        data = json.load(f)
    annots = data.get("annotations", {})
    print(f"Annotated samples: {len(annots)}")

    rows_out = []
    t0 = time.time()
    for i, (sid, points) in enumerate(annots.items()):
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        tl, tn = g["true_lat"], g["true_lon"]
        fov = g.get("fov_y_deg", 65.0)
        r_tilt = np.array(g["cam_R_tilt"])

        mask = mask_from_points(points)
        if mask is None:
            print(f"  {sid}: invalid annotation (<2 unique x)")
            continue

        # headline: pitch 0, stride 12
        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            print(f"  {sid}: profile FAIL {pr['status']}")
            continue
        profile = pr["profile"]
        fb_true, _ = fb_at_best(profile, fetch_horizon(vp))
        idx, corr = best_match(profile, STRIDE)
        err0 = geodesic((tl, tn), (lat_arr[idx], lon_arr[idx])).meters
        n_sampled = int(np.ceil(NVPS / STRIDE))

        # rank: fraction of sampled VPs beating the true VP
        pf = pq.ParquetFile(str(DB_PATH))
        n_better = 0
        chunk_start = 0
        for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            chunk = decode_horizon_column(
                batch.to_pandas()["raw_horizon_deg"].to_numpy()
            )
            sub = chunk[::STRIDE]
            c, _ = fft_prefilter(sub, profile, BIN_DEG)
            n_better += int(np.sum(c > fb_true))
            chunk_start += len(chunk)
            del chunk, c
        rank0 = n_better

        # pitch sweep at stride 50
        best_err = err0
        best_pitch = 0.0
        for dp in PITCHES:
            if dp == 0.0:
                continue
            r_tilt_p = r_pitch(dp) @ r_tilt
            prp = extract_elevation_profile(
                mask, fov_y_deg=fov, r_tilt=r_tilt_p, bin_deg=BIN_DEG
            )
            if not prp["ok"]:
                continue
            idxp, _ = best_match(prp["profile"], SWEEP_STRIDE)
            ep = geodesic((tl, tn), (lat_arr[idxp], lon_arr[idxp])).meters
            if ep < best_err:
                best_err = ep
                best_pitch = dp

        rows_out.append(
            {
                "sid": sid,
                "err0_m": float(err0),
                "corr0": float(corr),
                "fb_true": float(fb_true),
                "rank0": int(rank0),
                "n_sampled": n_sampled,
                "best_err_m": float(best_err),
                "best_pitch": float(best_pitch),
            }
        )
        print(
            f"  {i + 1}/{len(annots)} {sid[:22]:24s} err0={err0:7.0f}m "
            f"fb_true={fb_true:.3f} rank={rank0:6d}/{n_sampled} "
            f"| best(err)={best_err:7.0f}m @pitch{best_pitch:+.0f}  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    if not rows_out:
        print("No usable annotations.")
        return
    e0 = np.array([r["err0_m"] for r in rows_out])
    eb = np.array([r["best_err_m"] for r in rows_out])
    fb = np.array([r["fb_true"] for r in rows_out])
    rk = np.array([r["rank0"] for r in rows_out])
    ns = rows_out[0]["n_sampled"]

    print("\n=== HAND-ANNOTATED SKYLINE (honest matcher) ===")
    print(f"n={len(rows_out)}")
    print(
        f"pitch0: median err {np.median(e0):.0f}m  <1km {(e0 < 1000).mean() * 100:.0f}%  "
        f"<5km {(e0 < 5000).mean() * 100:.0f}%"
    )
    print(
        f"best-pitch: median err {np.median(eb):.0f}m  <1km {(eb < 1000).mean() * 100:.0f}%"
    )
    print(f"true-VP FB median {np.median(fb):.3f}")
    print(
        f"true-VP rank: median {np.median(rk):.0f}/{ns} "
        f"({np.median(rk) / ns * 100:.1f}%)  best {rk.min()}"
    )


if __name__ == "__main__":
    main()

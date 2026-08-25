#!/usr/bin/env python
"""Auto-calibrate camera heading+pitch for 17 annotated GSV samples.

Grid-search pitch adjustment Δp ∈ [-12°, +12°] at 1.0° steps.
At each Δp, re-extract profile with R_new = Rx(Δp) @ cam_R_tilt, then
find the best circular azimuth shift against the true VP's DB horizon.

Output: data/street_view/calibrated_ground_truth.json
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import (
    _feature_bundle,
    feature_bundle_matrix,
    _pearson_ncc_batch,
)

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
OUT_FILE = os.path.join(ROOT, "data/street_view/calibrated_ground_truth.json")
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")

PITCH_RANGE = np.arange(-12.0, 12.5, 1.0)
BIN_DEG = 0.5
W, H = 1080, 720


def Rx(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def mask_from_ann(points):
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


def best_ncc_shift(profile, db_horizon):
    """Compute best NCC correlation over all circular shifts.
    Returns (best_shift_bins, best_corr)."""
    M = len(profile)
    qv, qd = _feature_bundle(profile)
    vv, dd = feature_bundle_matrix(db_horizon[None, :])
    ve = np.concatenate([vv, vv[:, : M - 1]], axis=1)
    de = np.concatenate([dd, dd[:, : M - 1]], axis=1)
    qvz = qv - qv.mean()
    qdz = qd - qd.mean()
    qv_norm = np.linalg.norm(qvz)
    qd_norm = np.linalg.norm(qdz)
    if qv_norm < 1e-12 or qd_norm < 1e-12:
        return 0, 0.0
    cv = _pearson_ncc_batch(ve, qvz, qv_norm)[0]
    cd = _pearson_ncc_batch(de, qdz, qd_norm)[0]
    comb = 0.5 * cv + 0.5 * cd
    s = int(np.argmax(comb))
    return s, float(comb[s])


def main():
    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [s for s in ann if s in gt and ann[s] is not None]

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(DB_PATH)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    DB_BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
    L = int(360.0 / DB_BIN_DEG)

    def fetch_db_horizon(vp_idx):
        rg = int(np.searchsorted(rg_starts, vp_idx, side="right") - 1)
        base = int(rg_starts[rg])
        b = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
        from src.horizon_format import decode_horizon_column

        return decode_horizon_column([b["raw_horizon_deg"].iloc[vp_idx - base]])[0]

    from PIL import Image

    calib = {}
    print(f"{'sid':<22} {'Δp*':>6} {'Δθ*':>8} {'r*':>8} {'baseline_r':>10}  notes")
    print("-" * 80)

    for sid in sids:
        g = gt[sid]
        vp_true = int(g["closest_viewpoint_id"])
        cam_R_tilt = np.array(g["cam_R_tilt"])

        ann_pts = ann[sid]
        mask = mask_from_ann(ann_pts)
        if mask is None:
            print(f"{sid:<22}  SKIP: mask failed")
            continue

        try:
            img = np.array(
                Image.open(os.path.join(IMAGES_DIR, f"{sid}.png")).convert("L")
            )
        except Exception:
            print(f"{sid:<22}  SKIP: image missing")
            continue

        db_h = fetch_db_horizon(vp_true)

        best_dp, best_ds, best_r = 0.0, 0, 0.0
        baseline_r = 0.0

        for dp in PITCH_RANGE:
            R_new = Rx(dp) @ cam_R_tilt
            pr = extract_elevation_profile(
                mask, fov_y_deg=g["fov_y_deg"], r_tilt=R_new, bin_deg=BIN_DEG
            )
            if not pr["ok"] or pr["profile"] is None:
                continue
            prof = pr["profile"]
            ds, r = best_ncc_shift(prof, db_h)
            if dp == 0.0:
                baseline_r = r
            if r > best_r:
                best_r, best_dp, best_ds = r, float(dp), ds

        shift_deg = (best_ds * BIN_DEG) % 360
        if shift_deg > 180:
            shift_deg -= 360

        improved = (
            "IMPROVED"
            if best_r > baseline_r + 0.005
            else ("same" if abs(best_r - baseline_r) < 0.005 else "WORSE")
        )

        calib[sid] = {
            "delta_pitch_deg": best_dp,
            "delta_heading_bins": best_ds,
            "delta_heading_deg": shift_deg,
            "r_calibrated": round(best_r, 6),
            "r_baseline": round(baseline_r, 6),
            "improvement": improved,
        }

        print(
            f"{sid:<22} {best_dp:+6.1f} {shift_deg:+8.1f} {best_r:8.4f} "
            f"{baseline_r:10.4f}  {improved}"
        )

    with open(OUT_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved {len(calib)} samples → {OUT_FILE}")

    r_cal = np.array([v["r_calibrated"] for v in calib.values()])
    r_base = np.array([v["r_baseline"] for v in calib.values()])
    print(f"\n=== SUMMARY ===")
    print(
        f"r@calibrated  median={np.median(r_cal):.4f}  min={r_cal.min():.4f}  max={r_cal.max():.4f}"
    )
    print(
        f"r@baseline    median={np.median(r_base):.4f}  min={r_base.min():.4f}  max={r_base.max():.4f}"
    )
    print(f"improved: {(r_cal > r_base + 0.005).sum()}/{len(calib)}")
    print(
        f"median Δp* = {np.median([v['delta_pitch_deg'] for v in calib.values()]):+.1f}°"
    )
    print(
        f"median |Δθ*| = {np.median([abs(v['delta_heading_deg']) for v in calib.values()]):.1f}°"
    )


if __name__ == "__main__":
    main()

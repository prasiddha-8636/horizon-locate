#!/usr/bin/env python
"""Hand-Annotated Multi-Photo Perspective Fusion Evaluator.

Fuses multiple hand-annotated perspective crops from the same panorama
(e.g., Heading 0° + Heading 90° + Heading 180°) into a wide-FOV joint profile,
then runs full-DB streaming scan (skyline_db.parquet) to measure multi-photo accuracy!
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from geopy.distance import geodesic
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon_format import decode_horizon_column
from matching import fft_prefilter
from query_profile import extract_elevation_profile

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"

W, H = 1080, 720
STRIDE = 12


def mask_from_points(points):
    """Convert point annotations to binary sky mask (sky=0, terrain=255)."""
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


def load_multi_photo_annotations():
    """Group hand-annotated crops by panorama ID."""
    if not ANNOT_FILE.exists():
        print(f"Error: Annotations file not found: {ANNOT_FILE}")
        return {}

    with open(ANNOT_FILE) as f:
        annot_data = json.load(f)
    annots = annot_data.get("annotations", {})

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

    # Keep panos with at least 2 annotated crops
    return {k: v for k, v in panos.items() if len(v) >= 2}


def fuse_annotated_joint_profile(crop_list, bin_deg=0.5):
    """Fuse multiple perspective crop profiles into a continuous 360-degree array."""
    n_bins = int(round(360.0 / bin_deg))
    joint_profile = np.full(n_bins, np.nan, dtype=np.float32)

    for c in crop_list:
        mask = mask_from_points(c["points"])
        if mask is None:
            continue
        fov_y = c.get("fov_y_deg", 65.0)
        heading = c.get("heading_deg", 0.0)

        # Ignore columns clipped at top frame edge (r <= 2px)
        skyline_rows = np.full(W, H - 1, dtype=np.int32)
        for col in range(W):
            sky_rows = np.where(mask[:, col] == 0)[0]
            if len(sky_rows) > 0:
                skyline_rows[col] = sky_rows[-1]

        unclipped_cols = (skyline_rows > 2) & (skyline_rows < H - 2)

        res = extract_elevation_profile(
            mask, fov_y_deg=fov_y, bin_deg=bin_deg, column_keep_mask=unclipped_cols
        )
        if not res["ok"]:
            continue

        prof = res["profile"]
        m = len(prof)
        center_bin = int(round((heading % 360.0) / bin_deg))
        half_m = m // 2

        for i in range(m):
            bin_idx = (center_bin - half_m + i) % n_bins
            if not np.isnan(prof[i]):
                joint_profile[bin_idx] = prof[i]

    valid_mask = ~np.isnan(joint_profile)
    if valid_mask.sum() < 30:
        return None, 0.0

    valid_bins = int(valid_mask.sum())
    coverage_deg = valid_bins * bin_deg

    # Smooth interpolation across NaN gap bins (preserves continuous profile without cliff steps)
    all_bins = np.arange(n_bins)
    valid_idx = all_bins[valid_mask]
    valid_vals = joint_profile[valid_mask]

    fused = np.interp(all_bins, valid_idx, valid_vals)
    return fused, coverage_deg

def best_match_full_db(profile, bin_deg, stride=12):
    """Full DB streaming scan for best viewpoint match."""
    pf = pq.ParquetFile(str(DB_PATH))
    best_corr = -np.inf
    best_idx = -1
    chunk_start = 0

    for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        sub = chunk[::stride]
        corr, _ = fft_prefilter(sub, profile, bin_deg)
        k = int(np.argmax(corr))
        if corr[k] > best_corr:
            best_corr = float(corr[k])
            best_idx = chunk_start + k * stride
        chunk_start += len(chunk)

    return best_idx, best_corr


def main():
    print("Multi-Photo Hand-Annotated Perspective Fusion Evaluator")
    print("=" * 65)

    if not DB_PATH.exists():
        print(f"Error: DB file not found: {DB_PATH}")
        return

    meta_db = pq.read_table(str(DB_PATH), columns=["lon", "lat"])
    lat_arr = meta_db.column("lat").to_pandas().values
    lon_arr = meta_db.column("lon").to_pandas().values

    _first = next(pq.ParquetFile(str(DB_PATH)).iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    bin_deg = 360.0 / len(_first.to_pandas()["raw_horizon_deg"].iloc[0])

    pano_groups = load_multi_photo_annotations()
    print(f"Found {len(pano_groups)} panos with >= 2 hand-annotated crops.")

    if not pano_groups:
        print("\nTo annotate multi-photo crops:")
        print("  python scripts/annotate_gsv.py --multi-only")
        return

    results = []
    t0 = time.time()

    with open(GT_FILE) as f:
        gt_data = json.load(f)

        for pid, crops in pano_groups.items():
            ref_meta = crops[0]
            sid = ref_meta["sid"]

            gt_entry = gt_data.get(sid) or gt_data.get(pid) or {}
            true_lat = gt_entry.get("true_lat") or gt_entry.get("lat") or ref_meta.get("lat")
            true_lon = gt_entry.get("true_lon") or gt_entry.get("lon") or ref_meta.get("lon")

            if true_lat is None or true_lon is None:
                print(f"Pano {pid}: Ground truth lat/lon missing.")
                continue

            joint_prof, cov_deg = fuse_annotated_joint_profile(crops, bin_deg=bin_deg)
            if joint_prof is None:
                print(f"Pano {pid}: Joint profile generation failed.")
                continue

            best_idx, best_corr = best_match_full_db(joint_prof, bin_deg=bin_deg, stride=STRIDE)
            err_m = geodesic((true_lat, true_lon), (lat_arr[best_idx], lon_arr[best_idx])).meters

            results.append({
                "pano_id": pid,
                "n_crops": len(crops),
                "coverage_deg": cov_deg,
                "err_m": err_m,
                "best_corr": best_corr,
            })

            print(
                f"Pano {pid[:12]}: {len(crops)} crops | "
                f"FOV: {cov_deg:.0f}° | "
                f"Err: {err_m:6.0f}m | "
                f"Corr: {best_corr:.3f} "
                f"[{time.time() - t0:.0f}s]"
            )

    if results:
        errs = np.array([r["err_m"] for r in results])
        print("\n" + "=" * 65)
        print("MULTI-PHOTO FUSION RESULTS")
        print("=" * 65)
        print(f"Evaluated Panos: {len(results)}")
        print(f"Median Error:    {np.median(errs):.1f} meters")
        print(f"Top-1 < 500m:    {(errs < 500).mean() * 100:.1f}%")
        print(f"Top-1 < 1.0 km:  {(errs < 1000).mean() * 100:.1f}%")
        print(f"Top-1 < 5.0 km:  {(errs < 5000).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()

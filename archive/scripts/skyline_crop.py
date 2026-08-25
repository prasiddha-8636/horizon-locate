#!/usr/bin/env python
"""Skyline-guided pano cropping.

For each panorama, find the heading with the best skyline (max elevation-angle
relief within the crop FOV) using the horizon DB at the nearest viewpoint, then
crop at that heading with pitch chosen to center the ridge.

Writes:
  data/street_view/images/{id}.png        — new crops
  data/street_view/crop_quality.json      — per-pano heading/metrics
  data/street_view/ground_truth.json      — updated true_heading_deg = best heading

Skips panos whose best skyline is too weak (no clear ridge), recording them in
crop_quality.json as "skipped".
"""

import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
from src.evaluation import _fetch_rows
from src.streetview_utils import slice_perspective

trans = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)

PANOS_DIR = ROOT / "data/street_view/panos"
IMAGES_DIR = ROOT / "data/street_view/images"
GT_PATH = ROOT / "data/street_view/ground_truth.json"
DB_PATH = ROOT / "notebooks/02_SkylineDatabase/output/skyline_db.parquet"

FOV_Y_DEG = 65.0
OUT_W, OUT_H = 1080, 720
MIN_RELIEF_DEG = 3.0  # min std of elevation angles within FOV → clear skyline
MIN_MAX_ELEV_DEG = 1.0  # min peak elevation within FOV → ridge above eye level


def load_horizon_db():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    lon_arr = meta["lon"].to_numpy()
    lat_arr = meta["lat"].to_numpy()
    # True UTM metric coords for KDTree (projection must match the query exactly,
    # otherwise the cos(lat)-scaled lon offsets the nearest VP by ~1.25 km)
    e, n = trans.transform(lon_arr, lat_arr)
    tree = cKDTree(np.stack([e, n], axis=-1))
    return lon_arr, lat_arr, tree


def skyline_score_profile(horizon, fov_deg=65.0):
    """Sliding-window skyline quality over all headings.

    Returns (quality[360], best_heading, best_quality, max_elev_window).
    quality[az] = std of horizon elevation within FOV centered at az.
    """
    n = len(horizon)
    step = 360.0 / n
    half = int(round(fov_deg / 2 / step))
    # Pad horizon cyclically
    padded = np.concatenate([horizon[-half:], horizon, horizon[:half]])
    # Sliding std via cumulative sum of values and squares
    c = np.cumsum(np.insert(padded, 0, 0.0))
    c2 = np.cumsum(np.insert(padded**2, 0, 0.0))
    win = 2 * half + 1
    sums = c[win:] - c[:-win]
    sums2 = c2[win:] - c2[:-win]
    means = sums / win
    vars_ = sums2 / win - means**2
    stds = np.sqrt(np.maximum(vars_, 0.0))
    # max elevation in each window
    # sliding max via simple approach (small n)
    maxes = np.zeros(n)
    hz = horizon
    for i in range(n):
        idxs = (np.arange(win) + i - half) % n
        maxes[i] = hz[idxs].max()
    quality = stds[:n]
    best_az = int(np.argmax(quality))
    return quality, best_az, float(quality[best_az]), float(maxes[best_az])


def heading_pitch_for_pano(horizon):
    """Pick heading/pitch maximizing skyline visibility."""
    quality, best_az, best_q, best_max = skyline_score_profile(horizon, FOV_Y_DEG)
    if best_q < MIN_RELIEF_DEG or best_max < MIN_MAX_ELEV_DEG:
        return None, best_az, best_q, best_max
    # Pitch: center the ridge. horizon[best_az] is the elevation at the chosen
    # heading (1° bins). Use mean elevation within the window as the pitch.
    n = len(horizon)
    step = 360.0 / n
    half = int(round(FOV_Y_DEG / 2 / step))
    window = np.roll(horizon, -best_az)[: 2 * half + 1]
    pitch_deg = float(np.median(window))
    # Clamp to reasonable range so we don't point at the ground/sky
    pitch_deg = float(np.clip(pitch_deg, -10.0, 25.0))
    return pitch_deg, best_az, best_q, best_max


def main():
    gt = json.loads(GT_PATH.read_text())
    print(f"Panos in GT: {len(gt)}", flush=True)

    lon_arr, lat_arr, tree = load_horizon_db()
    print(f"DB viewpoints: {len(lon_arr)}", flush=True)

    # Query points for nearest viewpoint (same UTM projection as the tree)
    qe, qn = trans.transform(
        np.array([v["true_lon"] for v in gt.values()]),
        np.array([v["true_lat"] for v in gt.values()]),
    )
    dists, idxs = tree.query(np.stack([qe, qn], axis=-1), k=1)
    print(f"Nearest viewpoints found (mean dist {dists.mean():.1f} m)", flush=True)

    # Fetch horizons for all nearest viewpoints in one batched call
    unique_idx = np.unique(idxs)
    fetched = _fetch_rows(str(DB_PATH), unique_idx)
    horizons = {idx: np.asarray(fetched[idx], dtype=np.float64) for idx in unique_idx}
    print(f"Horizons fetched: {len(horizons)}", flush=True)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    quality_records = {}
    t0 = time.time()
    n_ok = 0
    n_skip = 0

    for i, (sid, v) in enumerate(gt.items()):
        pano_path = PANOS_DIR / f"{sid}.jpg"
        if not pano_path.exists():
            quality_records[sid] = {"status": "no_pano"}
            continue

        horizon = horizons[int(idxs[i])]
        pitch_deg, best_az, best_q, best_max = heading_pitch_for_pano(horizon)
        quality_records[sid] = {
            "best_heading": best_az,
            "best_quality": round(best_q, 2),
            "best_max_elev": round(best_max, 2),
            "nearest_vp": int(idxs[i]),
            "nearest_dist_m": float(dists[i]),
        }

        if pitch_deg is None:
            quality_records[sid]["status"] = "skip_low_relief"
            n_skip += 1
            continue

        crop = slice_perspective(
            str(pano_path),
            heading_deg=float(best_az),
            pitch_deg=pitch_deg,
            roll_deg=0.0,
            fov_y_deg=FOV_Y_DEG,
            out_w=OUT_W,
            out_h=OUT_H,
        )
        crop.save(IMAGES_DIR / f"{sid}.png")
        quality_records[sid]["status"] = "ok"
        quality_records[sid]["pitch_deg"] = round(pitch_deg, 2)
        # Update GT heading to the actual crop heading
        v["true_heading_deg"] = float(best_az)
        # cam_R_tilt = pitch-only rotation (matches the crop's pitch)
        p = np.radians(pitch_deg)
        v["cam_R_tilt"] = [
            [1.0, 0.0, 0.0],
            [0.0, float(np.cos(p)), float(-np.sin(p))],
            [0.0, float(np.sin(p)), float(np.cos(p))],
        ]
        n_ok += 1

        if (i + 1) % 200 == 0:
            print(
                f"  {i + 1}/{len(gt)}: ok={n_ok} skip={n_skip} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    GT_PATH.write_text(json.dumps(gt, indent=2))
    (ROOT / "data/street_view/crop_quality.json").write_text(
        json.dumps(quality_records, indent=2)
    )
    print(
        f"\nDone: ok={n_ok}, skip={n_skip}, elapsed {time.time() - t0:.0f}s", flush=True
    )


if __name__ == "__main__":
    main()

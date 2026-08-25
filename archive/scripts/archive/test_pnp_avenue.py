#!/usr/bin/env python
"""Test Avenue C: PnP RANSAC Peak Constellation.

Match skyline peaks from annotations to DEM peaks by azimuth proximity,
solve PnP RANSAC for camera pose. No ML model required.

Avenue A (SAM 2) and Avenue B (DINOv2) need >4GB free RAM for model weights.
Current machine is OOM (7GB total, 5.2GB used, swap full).
These require a GPU-equipped machine.

Usage: python -u scripts/test_pnp_avenue.py
"""

import json
import os
import sys
import time
from itertools import permutations

import cv2
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.horizon_format import decode_horizon_uint8
from scripts.fixes_eval import DB_PATH, GT_FILE, ANNOT_FILE

IMAGE_DIR = os.path.join(ROOT, "data/street_view/images")
W, H = 1080, 720


def load_geometry():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    return (
        meta["lon"].to_numpy(),
        meta["lat"].to_numpy(),
        meta["elevation_m"].to_numpy(),
    )


def extract_peaks_from_annotation(ann_pts):
    if ann_pts is None or len(ann_pts) < 10:
        return np.array([])
    pts = np.array(ann_pts)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.array([])
    cols, rows = pts[:, 0], pts[:, 1]
    order = np.argsort(cols)
    cols, rows = cols[order], rows[order]
    _, uidx = np.unique(np.round(cols).astype(int), return_index=True)
    cu, ru = cols[uidx], rows[uidx]
    if len(ru) < 5:
        return np.array([])
    rs = median_filter(ru, size=5)
    pks, _ = find_peaks(-rs, distance=10, prominence=3)
    if len(pks) < 3:
        pks = np.argsort(-rs)[-5:]
    return np.column_stack([cu[pks], ru[pks]])


def get_dem_peaks_near(
    vp_lat, vp_lon, lat_arr, lon_arr, elev_arr, radius_km=20, top_k=25
):
    dlat = lat_arr - vp_lat
    dlon = lon_arr - vp_lon
    dist = np.sqrt(dlat**2 + dlon**2) * 111.0
    cands = np.where(dist < radius_km)[0]
    if len(cands) < 5:
        return np.array([])
    ce = elev_arr[cands]
    med = np.median(ce)
    peaks = cands[ce > med + 80]
    if len(peaks) < 4:
        peaks = cands[np.argsort(ce)[-10:]]
    if len(peaks) > top_k:
        peaks = peaks[np.argsort(elev_arr[peaks])[-top_k:]]
    return np.column_stack([lat_arr[peaks], lon_arr[peaks], elev_arr[peaks]])


def camera_matrix(fov_y_deg):
    aspect = W / H
    hfov = np.degrees(2 * np.arctan(np.tan(np.radians(fov_y_deg) / 2) * aspect))
    fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    fy = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    return np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]], dtype=np.float64)


def solve_pnp(peak_2d, obj_3d, K):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_3d.astype(np.float64),
        peak_2d.astype(np.float64),
        K,
        np.zeros(4),
        iterationsCount=1000,
        reprojectionError=10.0,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok and inliers is not None and len(inliers) >= 3:
        return True, rvec, tvec, len(inliers)
    return False, None, None, 0


def match_by_azimuth(peaks_2d, dem_pks, vp_lat, vp_lon, K, fov_y_deg):
    """Match skyline peaks to DEM peaks by azimuth proximity."""
    xc, yc = W / 2.0, H / 2.0
    fx = K[0, 0]
    fy = K[1, 1]

    n = min(len(peaks_2d), len(dem_pks), 6)
    pk_azims = np.degrees(np.arctan2((peaks_2d[:n, 0] - xc) / fx, 1.0))

    dlat = dem_pks[:, 0] - vp_lat
    dlon = dem_pks[:, 1] - vp_lon
    dm_azims = np.degrees(np.arctan2(dlon, dlat))

    used = set()
    matched_2d, matched_3d = [], []
    for i in range(n):
        diffs = np.abs(dm_azims - pk_azims[i])
        for rank in np.argsort(diffs):
            if rank not in used:
                used.add(rank)
                matched_2d.append(peaks_2d[i])
                matched_3d.append(dem_pks[rank])
                break
    if len(matched_2d) < 4:
        return None
    return np.array(matched_3d), np.array(matched_2d)


def main():
    vp_lon, vp_lat, vp_elev = load_geometry()
    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]

    sids = [
        s
        for s in ann
        if s in gt and os.path.exists(os.path.join(IMAGE_DIR, f"{s}.png"))
    ]
    print(f"Samples: {len(sids)}", flush=True)

    results = []
    t0 = time.time()

    for si, sid in enumerate(sids):
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp = g["closest_viewpoint_id"]
        fov = g["fov_y_deg"]
        K = camera_matrix(fov)

        a = ann[sid]
        if isinstance(a, list):
            ann_pts = np.array(a)
        elif isinstance(a, dict):
            ann_pts = np.array(a.get("points", []))
        else:
            ann_pts = np.array(a)

        peaks_2d = extract_peaks_from_annotation(ann_pts)
        if len(peaks_2d) < 4:
            results.append({"sid": sid, "ok": False})
            print(
                f"  [{si + 1}/{len(sids)}] {sid[:25]:25s} SKIP: {len(peaks_2d)} peaks",
                flush=True,
            )
            continue

        dem_pks = get_dem_peaks_near(vp_lat[vp], vp_lon[vp], vp_lat, vp_lon, vp_elev)
        if len(dem_pks) < 4:
            results.append({"sid": sid, "ok": False})
            print(
                f"  [{si + 1}/{len(sids)}] {sid[:25]:25s} SKIP: {len(dem_pks)} DEM peaks",
                flush=True,
            )
            continue

        match = match_by_azimuth(peaks_2d, dem_pks, vp_lat[vp], vp_lon[vp], K, fov)
        if match is None:
            results.append({"sid": sid, "ok": False})
            print(
                f"  [{si + 1}/{len(sids)}] {sid[:25]:25s} SKIP: <4 azimuth matches",
                flush=True,
            )
            continue

        obj_3d, pk_2d = match
        ok, rvec, tvec, inliers = solve_pnp(pk_2d, obj_3d, K)
        if ok and tvec is not None and inliers >= 3:
            tp = tvec.flatten()
            cam_lat = vp_lat[vp] + tp[2] / 111000.0
            cam_lon = vp_lon[vp] + tp[0] / (111000.0 * np.cos(np.radians(vp_lat[vp])))
            err = geodesic((tlat, tlon), (cam_lat, cam_lon)).meters / 1000
            results.append(
                {
                    "sid": sid,
                    "ok": True,
                    "err_km": round(err, 1),
                    "inliers": inliers,
                    "n_pks": len(pk_2d),
                }
            )
            print(
                f"  [{si + 1}/{len(sids)}] {sid[:25]:25s} PnP: err={err:7.1f}km  inliers={inliers}/{len(pk_2d)}",
                flush=True,
            )
        else:
            results.append({"sid": sid, "ok": False})
            print(
                f"  [{si + 1}/{len(sids)}] {sid[:25]:25s} PnP: FAILED (no convergence)",
                flush=True,
            )

    ok_r = [r for r in results if r.get("ok")]
    fail_r = [r for r in results if not r.get("ok")]

    print(f"\n{'=' * 60}", flush=True)
    print(f"AVENUE C: PnP RANSAC — Summary", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Success: {len(ok_r)}/{len(results)}", flush=True)

    if ok_r:
        errs = np.array([r["err_km"] for r in ok_r])
        print(f"  med={np.median(errs):.1f}km  mean={np.mean(errs):.1f}km", flush=True)
        for threshold, label in [
            (0.1, "<100m"),
            (0.5, "<500m"),
            (1, "<1km"),
            (5, "<5km"),
            (10, "<10km"),
        ]:
            print(f"  {label}={int(np.sum(errs < threshold))}/{len(errs)}", flush=True)

    if fail_r:
        print(f"\nFailed: {len(fail_r)} samples", flush=True)

    print(f"\nTotal: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Avenue 3: 2D-to-3D Peak Constellation PnP (Perspective-n-Point).

Uses cached NCC for coarse ranking at realistic 5km radius (sensor-assisted).
For PnP, extracts peaks from query skyline and matches to DEM peaks at 50km,
solving for camera position from 2D-3D correspondences.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, _pearson_ncc_batch, feature_bundle_matrix
from src.horizon_format import decode_horizon_column

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
CALIB_FILE = os.path.join(ROOT, "data/street_view/calibrated_ground_truth.json")
DEM_PEAKS_FILE = "/tmp/dem_peaks_v2.npz"
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
W, H_IMG = 1080, 720
CORRECT_M = 500.0
NMI_RADIUS_KM = 5.0
DEM_RADIUS_KM = 50.0
MAX_QUERY_PEAKS = 12
MIN_PEAK_PROMINENCE = 0.5
MATCH_AZ_TOL = 10.0
TOP_K_PNP = 10
MIN_PNP_INLIERS = 4


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
        return None, None
    mask = np.zeros((H_IMG, W), dtype=np.uint8)
    ycol = np.full(W, H_IMG, dtype=np.int64)
    ii = 0
    for x in range(W):
        while ii < len(xs) - 1 and xs[ii + 1] <= x:
            ii += 1
        if xs[ii] <= x <= xs[-1]:
            ycol[x] = int(np.interp(x, xs, ys))
    for x in range(W):
        mask[min(H_IMG - 1, max(0, int(ycol[x]))) :, x] = 255
    return mask, ycol


def haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2_arr))
        * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def find_peaks_1d(profile, prominence=MIN_PEAK_PROMINENCE):
    from scipy.signal import find_peaks

    peaks, props = find_peaks(profile, prominence=prominence, distance=5)
    if len(peaks) > MAX_QUERY_PEAKS:
        idx = np.argsort(props["prominences"])[-MAX_QUERY_PEAKS:]
        peaks = peaks[idx]
    return peaks


def load_dem_peaks():
    data = np.load(DEM_PEAKS_FILE)
    return data["lat"], data["lon"], data["elev"]


def latlon_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    R = 6371000.0
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    e = R * dlon * np.cos(np.radians(ref_lat))
    n = R * dlat
    u = alt - ref_alt
    return np.array([e, n, u], dtype=np.float64)


def az_el_from_enu(dx, dy, dz):
    az = np.degrees(np.arctan2(dx, dy)) % 360
    horiz = np.sqrt(dx**2 + dy**2)
    el = np.degrees(np.arctan2(dz, horiz))
    return az, el


def solve_pnp_ransac(img_pts_2d, obj_pts_3d, fov_y_deg):
    if len(img_pts_2d) < MIN_PNP_INLIERS:
        return False, None, None, 0
    fx = fy = W / (2 * np.tan(np.radians(fov_y_deg / 2)))
    cx, cy = W / 2, H_IMG / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    img_pts = np.array(img_pts_2d, dtype=np.float64)
    obj_pts = np.array(obj_pts_3d, dtype=np.float64)
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            dist,
            iterationsCount=200,
            reprojectionError=10.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success and inliers is not None and len(inliers) >= MIN_PNP_INLIERS:
            return True, rvec, tvec, len(inliers)
    except cv2.error:
        pass
    return False, None, None, 0


def decode_vp_horizon(vp_indices, pf):
    needed = set(int(i) for i in vp_indices)
    horizons = {}
    cs = 0
    for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        raw = batch.to_pandas()["raw_horizon_deg"].to_numpy()
        chunk = decode_horizon_column(raw)
        N = len(chunk)
        for vi in needed:
            if cs <= vi < cs + N:
                horizons[vi] = chunk[vi - cs]
        cs += N
        if len(horizons) == len(needed):
            break
    return horizons


def compute_ncc_shift(db_horizon, profile, M):
    qv, qd = _feature_bundle(profile)
    qvz, qdz = qv - qv.mean(), qd - qd.mean()
    qvn, qdn = np.linalg.norm(qvz), np.linalg.norm(qdz)
    v, d1 = feature_bundle_matrix(db_horizon[np.newaxis])
    ve = np.concatenate([v, v[:, : M - 1]], axis=1)
    de = np.concatenate([d1, d1[:, : M - 1]], axis=1)
    comb = 0.5 * _pearson_ncc_batch(ve, qvz, qvn) + 0.5 * _pearson_ncc_batch(
        de, qdz, qdn
    )
    return int(np.argmax(comb))


def main():
    dem_lat, dem_lon, dem_elev = load_dem_peaks()
    print(f"Loaded {len(dem_lat)} DEM peaks", flush=True)

    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    vp_lon = meta["lon"].to_numpy()
    vp_lat = meta["lat"].to_numpy()
    vp_elev = meta["elevation_m"].to_numpy()
    N_vps = len(vp_lon)

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))
    sids = [s for s in ann if s in gt and ann[s] is not None and s in calib]

    pf = pq.ParquetFile(DB_PATH)

    results = {
        "ncc": [],
        "pnp": [],
        "pnp_inliers": [],
        "pnp_solves": 0,
        "ncc_true_rank": [],
        "pnp_true_rank": [],
    }

    t0 = time.time()
    for si, sid in enumerate(sids):
        ts = time.time()
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        tilt = np.array(g["cam_R_tilt"])
        dp_calib = float(calib[sid].get("delta_pitch_deg", 0.0))
        fov_y = g["fov_y_deg"]

        mask, ycol = mask_from_ann(ann[sid])
        if mask is None:
            continue

        R_cal = Rx(dp_calib) @ tilt
        pr = extract_elevation_profile(mask, fov_y_deg=fov_y, r_tilt=R_cal, bin_deg=0.5)
        if not pr["ok"]:
            continue
        profile = pr["profile"]
        M = len(profile)

        cache_file = os.path.join(CACHE_DIR, f"{sid}_corr.npz")
        if not os.path.exists(cache_file):
            print(f"[{si + 1:2d}] {sid[:20]:<20} SKIP (no cache)", flush=True)
            continue
        cached = np.load(cache_file)
        corr = cached["corr"]

        dist_km = haversine_km(tlat, tlon, vp_lat, vp_lon)
        valid_radius = dist_km <= NMI_RADIUS_KM
        masked_corr = corr.copy()
        masked_corr[~valid_radius] = -np.inf
        rank_full = np.argsort(-masked_corr) + 1

        bv = int(np.argmax(masked_corr))
        true_rank = int(np.sum(corr >= corr[bv]))
        ncc_err = dist_km[bv] * 1000
        results["ncc"].append(ncc_err)
        results["ncc_true_rank"].append(true_rank)

        q_peaks = find_peaks_1d(profile)
        if len(q_peaks) < MIN_PNP_INLIERS:
            results["pnp"].append(ncc_err)
            results["pnp_inliers"].append(0)
            print(
                f"[{si + 1:2d}] {sid[:20]:<20} NCC={ncc_err / 1000:6.1f}km PnP=SKIP ({time.time() - ts:.0f}s)",
                flush=True,
            )
            continue

        q_az = q_peaks * 0.5
        q_el = profile[q_peaks]
        q_x = q_az - fov_y / 2 + W / 2

        top_idxs = np.argsort(masked_corr)[-TOP_K_PNP:]
        vp_horizons = decode_vp_horizon(top_idxs, pf)

        best_pnp_err = np.inf
        best_pnp_vp = -1
        best_inliers = 0

        for vi in top_idxs:
            vlat, vlon, v_elev = vp_lat[vi], vp_lon[vi], vp_elev[vi]

            if vi not in vp_horizons:
                continue
            db_h = vp_horizons[vi]
            shift = compute_ncc_shift(db_h, profile, M)

            dem_dists = haversine_km(vlat, vlon, dem_lat, dem_lon)
            near = dem_dists <= DEM_RADIUS_KM
            if np.sum(near) < MIN_PNP_INLIERS:
                continue

            near_lat = dem_lat[near]
            near_lon = dem_lon[near]
            near_elev = dem_elev[near]
            n_near = len(near_lat)

            dem_enu = []
            for i in range(n_near):
                dem_enu.append(
                    latlon_to_enu(
                        near_lat[i], near_lon[i], near_elev[i], vlat, vlon, v_elev
                    )
                )
            dem_az = np.zeros(n_near)
            dem_el = np.zeros(n_near)
            for i in range(n_near):
                dem_az[i], dem_el[i] = az_el_from_enu(
                    dem_enu[i][0], dem_enu[i][1], dem_enu[i][2]
                )

            matched_img = []
            matched_obj = []
            used_dem = set()

            for qi, qaz in enumerate(q_az):
                shifted_qaz = (qaz + shift * 0.5) % 360
                best_dem_idx = -1
                best_dem_dist = MATCH_AZ_TOL
                for di in range(n_near):
                    if di in used_dem:
                        continue
                    daz = abs(dem_az[di] - shifted_qaz)
                    if daz > 180:
                        daz = 360 - daz
                    if daz < best_dem_dist:
                        best_dem_dist = daz
                        best_dem_idx = di
                if best_dem_idx >= 0:
                    py = (1 - q_el[qi] / 90.0) * H_IMG
                    px = q_x[qi]
                    if 0 <= px < W and 0 <= py < H_IMG:
                        matched_img.append([px, py])
                        matched_obj.append(dem_enu[best_dem_idx])
                        used_dem.add(best_dem_idx)

            if len(matched_img) >= MIN_PNP_INLIERS:
                ok, rvec, tvec, n_inliers = solve_pnp_ransac(
                    matched_img, matched_obj, fov_y
                )
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    cam_enu = -R.T @ tvec.flatten()
                    cam_lat = vlat + cam_enu[1] / 6371000 * (180 / np.pi)
                    cam_lon = vlon + cam_enu[0] / (
                        6371000 * np.cos(np.radians(vlat))
                    ) * (180 / np.pi)
                    pnp_err = haversine_km(tlat, tlon, cam_lat, cam_lon) * 1000
                    pnp_rank = int(np.sum(corr >= np.max(corr)))

                    if n_inliers > best_inliers:
                        best_inliers = n_inliers
                        best_pnp_err = pnp_err
                        best_pnp_vp = vi
                        results["pnp_true_rank"].append(pnp_rank)

        results["pnp"].append(best_pnp_err if best_pnp_err < np.inf else ncc_err)
        results["pnp_inliers"].append(best_inliers)
        results["pnp_solves"] += 1

        pnp_val = best_pnp_err if best_pnp_err < np.inf else ncc_err
        print(
            f"[{si + 1:2d}] {sid[:20]:<20} NCC={ncc_err / 1000:6.1f}km "
            f"PnP={pnp_val / 1000:6.1f}km inl={best_inliers} "
            f"(NMI={ncc_err / 1000:.1f}km, {time.time() - ts:.0f}s)",
            flush=True,
        )

    ncc_errs = np.array(results["ncc"])
    pnp_errs = np.array(results["pnp"])
    inliers = np.array(results["pnp_inliers"])

    print(f"\n{'=' * 60}")
    print(
        f"NCC (radius {NMI_RADIUS_KM}km): "
        f"top1@500m={sum(e < CORRECT_M for e in ncc_errs)}/{len(ncc_errs)}  "
        f"median={np.median(ncc_errs) / 1000:.1f}km"
    )
    print(
        f"PnP (top-{TOP_K_PNP} cands, DEM {DEM_RADIUS_KM}km): "
        f"top1@500m={sum(e < CORRECT_M for e in pnp_errs)}/{len(pnp_errs)}  "
        f"median={np.median(pnp_errs) / 1000:.1f}km  "
        f"solves={results['pnp_solves']}/{len(sids)}  "
        f"mean_inliers={inliers.mean():.1f}"
    )
    print(f"PnP <1km:  {sum(e < 1000 for e in pnp_errs)}/{len(pnp_errs)}")
    print(f"PnP <5km:  {sum(e < 5000 for e in pnp_errs)}/{len(pnp_errs)}")
    print(f"PnP <10km: {sum(e < 10000 for e in pnp_errs)}/{len(pnp_errs)}")
    print(f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

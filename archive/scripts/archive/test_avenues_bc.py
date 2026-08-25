#!/usr/bin/env python
"""Test Avenue B (DINOv2) and Avenue C (PnP RANSAC) on 17 GSV samples.

Avenue B: For each query, encode the photo + ~700 candidate DB horizon
silhouettes with DINOv2, compare cosine similarity. Tests whether 2D
vision-transformer features can discriminate true VP from imposters.

Avenue C: Extract skyline peak vertices from annotations, match to DEM
peaks by azimuth, solve PnP RANSAC for camera pose.

Usage: python -u scripts/test_avenues_bc.py
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from geopy.distance import geodesic
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8
from src.matching import _feature_bundle_matrix, _pearson_ncc_batch
from scripts.fixes_eval import (
    Rx,
    mask_from_ann,
    DB_PATH,
    GT_FILE,
    ANNOT_FILE,
    CALIB_FILE,
)

BIN_DEG = 0.5
IMAGE_DIR = os.path.join(ROOT, "data/street_view/images")
MASK_DIR = os.path.join(ROOT, "data/street_view/masks")
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")


def load_geometry():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    return (
        meta["lon"].to_numpy(),
        meta["lat"].to_numpy(),
        meta["elevation_m"].to_numpy(),
    )


def horizon_to_silhouette_image(horizon, W=224, H=224):
    L = len(horizon)
    img = np.zeros((H, W), dtype=np.uint8)
    for c in range(W):
        col_az = (c / W) * 360.0
        h_idx = int((col_az % 360.0) / (360.0 / L)) % L
        elev = horizon[h_idx]
        row = int(H * (1.0 - (elev + 10.0) / 100.0))
        row = max(0, min(H - 1, row))
        img[row:, c] = 255
    return img


def cosine_similarity(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return An @ Bn.T


def fetch_horizons_batch(idx_list):
    _pf = pq.ParquetFile(DB_PATH)
    sizes = [_pf.metadata.row_group(i).num_rows for i in range(_pf.num_row_groups)]
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    groups = {}
    for vi in idx_list:
        rg = int(np.searchsorted(starts, vi, side="right") - 1)
        groups.setdefault(rg, []).append(vi)
    out = {}
    for rg, vis in groups.items():
        raw = (
            _pf.read_row_group(rg, columns=["raw_horizon_deg"])
            .to_pandas()["raw_horizon_deg"]
            .to_numpy()
        )
        for vi in vis:
            out[vi] = decode_horizon_uint8(raw[vi - starts[rg]])
    return out


def extract_peaks_from_annotation(ann_pts, W=1080):
    if ann_pts is None or len(ann_pts) < 10:
        return np.array([])
    pts = np.array(ann_pts)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.array([])

    cols = pts[:, 0]
    rows = pts[:, 1]
    sorted_idx = np.argsort(cols)
    cols, rows = cols[sorted_idx], rows[sorted_idx]

    _, unique_idx = np.unique(np.round(cols).astype(int), return_index=True)
    cols_u, rows_u = cols[unique_idx], rows[unique_idx]

    rows_smooth = median_filter(rows_u, size=5)
    inv_rows = -rows_smooth
    peaks, _ = find_peaks(inv_rows, distance=10, prominence=3)
    if len(peaks) < 3:
        peaks = np.argsort(inv_rows)[-5:]
    return np.column_stack([cols_u[peaks], rows_u[peaks]])


def get_dem_peaks_near(
    vp_lat, vp_lon, lat_arr, lon_arr, elev_arr, radius_km=20, top_k=25
):
    dlat = lat_arr - vp_lat
    dlon = lon_arr - vp_lon
    dist = np.sqrt(dlat**2 + dlon**2) * 111.0
    mask = dist < radius_km
    candidates = np.where(mask)[0]
    if len(candidates) < 5:
        return np.array([])
    cand_elev = elev_arr[candidates]
    local_med = np.median(cand_elev)
    peaks = candidates[cand_elev > local_med + 80]
    if len(peaks) < 4:
        peaks = candidates[np.argsort(cand_elev)[-10:]]
    if len(peaks) > top_k:
        peaks = peaks[np.argsort(elev_arr[peaks])[-top_k:]]
    return np.column_stack([lat_arr[peaks], lon_arr[peaks], elev_arr[peaks]])


def solve_pnp_ransac(peak_2d, obj_points, fov_y_deg, W, H):
    if len(peak_2d) < 4 or len(obj_points) < 4:
        return False, None, None, 0

    aspect = W / H
    hfov = np.degrees(2 * np.arctan(np.tan(np.radians(fov_y_deg) / 2) * aspect))
    focal_x = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
    focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    K = np.array(
        [[focal_x, 0, W / 2.0], [0, focal_y, H / 2.0], [0, 0, 1.0]], dtype=np.float64
    )

    n = min(len(peak_2d), len(obj_points))
    pts_2d = peak_2d[:n].astype(np.float64)
    pts_3d = obj_points[:n].astype(np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d,
        pts_2d,
        K,
        np.zeros(4),
        iterationsCount=500,
        reprojectionError=8.0,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok and inliers is not None:
        return True, rvec, tvec, len(inliers)
    return False, None, None, 0


def main():
    vp_lon, vp_lat, vp_elev = load_geometry()
    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))

    sids = [
        s
        for s in ann
        if s in gt and os.path.exists(os.path.join(IMAGE_DIR, f"{s}.png"))
    ]
    print(f"Samples: {len(sids)}", flush=True)

    # =====================================================================
    # AVENUE B: DINOv2 Feature Matching
    # =====================================================================
    print("\n" + "=" * 60, flush=True)
    print("AVENUE B: DINOv2 Feature Matching", flush=True)
    print("=" * 60, flush=True)

    from torchvision import transforms as T

    dino_model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
    )
    dino_model.eval()
    dino_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def encode_images(images):
        from PIL import Image as PILImage

        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = PILImage.fromarray(img).convert("RGB")
            tensors.append(dino_transform(img))
        x = torch.stack(tensors)
        with torch.no_grad():
            feat = dino_model(x)
        return feat.cpu().numpy()

    from PIL import Image as PILImage

    dino_results = []
    t0 = time.time()
    for si, sid in enumerate(sids):
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp = g["closest_viewpoint_id"]

        q_img = PILImage.open(os.path.join(IMAGE_DIR, f"{sid}.png")).convert("RGB")
        q_feat = encode_images([q_img])

        cache_path = os.path.join(CACHE_DIR, f"{sid}_corr.npz")
        if not os.path.exists(cache_path):
            print(f"  {sid}: no corr cache, skip", flush=True)
            continue
        corr = np.load(cache_path)["corr"]
        top500 = np.argsort(corr)[-500:]
        dlat = vp_lat - tlat
        dlon = vp_lon - tlon
        approx_km = np.sqrt(dlat**2 + dlon**2) * 111.0
        near200 = np.argsort(approx_km)[:200]
        cand = np.unique(np.concatenate([top500, [vp], near200])).astype(int)

        hdict = fetch_horizons_batch(list(cand))
        cand_h = [hdict[int(c)] for c in cand if int(c) in hdict]
        cand_valid = [int(c) for c in cand if int(c) in hdict]
        if len(cand_h) == 0:
            continue

        sil_imgs = [horizon_to_silhouette_image(h) for h in cand_h]
        batch_feats = []
        for bi in range(0, len(sil_imgs), 32):
            batch_feats.append(encode_images(sil_imgs[bi : bi + 32]))
        db_feats = np.concatenate(batch_feats, axis=0)
        cand_valid = np.array(cand_valid[: len(db_feats)])

        sims = cosine_similarity(q_feat, db_feats)[0]
        top5_i = np.argsort(sims)[-5:][::-1]
        top5_vps = cand_valid[top5_i]
        top1_err = (
            geodesic((tlat, tlon), (vp_lat[top5_vps[0]], vp_lon[top5_vps[0]])).meters
            / 1000
        )

        true_i = np.where(cand_valid == vp)[0]
        true_rank = int(np.sum(sims > sims[true_i[0]])) + 1 if len(true_i) > 0 else -1
        true_sim = float(sims[true_i[0]]) if len(true_i) > 0 else -1

        dino_results.append(
            {
                "sid": sid,
                "top1_err_km": round(top1_err, 1),
                "true_rank": true_rank,
                "sim_true": round(true_sim, 4),
                "n_cand": len(cand_valid),
            }
        )
        print(
            f"  [{si + 1}/{len(sids)}] {sid[:20]:20s} top1={top1_err:6.1f}km  true_rank={true_rank:5d}  "
            f"sim_true={true_sim:.4f}  n_cand={len(cand_valid)}",
            flush=True,
        )

    if dino_results:
        de = np.array([r["top1_err_km"] for r in dino_results])
        dr = np.array([r["true_rank"] for r in dino_results if r["true_rank"] > 0])
        print(
            f"\nDINOv2 silhouette-matching on {len(dino_results)} candidates:",
            flush=True,
        )
        print(
            f"  med_err={np.median(de):.1f}km  <1km={int(np.sum(de < 1))}/{len(de)}  "
            f"<5km={int(np.sum(de < 5))}/{len(de)}  <10km={int(np.sum(de < 10))}/{len(de)}",
            flush=True,
        )
        print(
            f"  true-VP rank: med={int(np.median(dr)):d}  mean={np.mean(dr):.0f}",
            flush=True,
        )

    # =====================================================================
    # AVENUE C: PnP RANSAC
    # =====================================================================
    print("\n" + "=" * 60, flush=True)
    print("AVENUE C: PnP RANSAC Peak Constellation", flush=True)
    print("=" * 60, flush=True)

    pnp_results = []
    for si, sid in enumerate(sids):
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp = g["closest_viewpoint_id"]
        fov = g["fov_y_deg"]
        W, H = 1080, 720

        a = ann[sid]
        if isinstance(a, list):
            ann_pts = np.array(a)
        elif isinstance(a, dict):
            ann_pts = np.array(a.get("points", []))
        else:
            ann_pts = np.array(a)

        peaks_2d = extract_peaks_from_annotation(ann_pts, W)
        if len(peaks_2d) < 4:
            pnp_results.append(
                {"sid": sid, "ok": False, "reason": f"peaks={len(peaks_2d)}"}
            )
            print(
                f"  [{si + 1}] {sid[:20]:20s} SKIP: {len(peaks_2d)} peaks", flush=True
            )
            continue

        aspect = W / H
        hfov = np.degrees(2 * np.arctan(np.tan(np.radians(fov) / 2) * aspect))
        fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
        fy = H / (2.0 * np.tan(np.radians(fov) / 2.0))
        xc, yc = W / 2.0, H / 2.0
        azims = np.degrees(np.arctan2((peaks_2d[:, 0] - xc) / fx, 1.0))

        dem_pks = get_dem_peaks_near(vp_lat[vp], vp_lon[vp], vp_lat, vp_lon, vp_elev)
        if len(dem_pks) < 4:
            pnp_results.append(
                {"sid": sid, "ok": False, "reason": f"dem_pks={len(dem_pks)}"}
            )
            print(
                f"  [{si + 1}] {sid[:20]:20s} SKIP: {len(dem_pks)} DEM peaks",
                flush=True,
            )
            continue

        best_err = float("inf")
        best_ok = False
        best_inliers = 0

        from itertools import combinations

        n_pks = min(len(peaks_2d), len(dem_pks), 8)
        for combo in combinations(range(len(dem_pks)), n_pks):
            obj = dem_pks[list(combo)][:n_pks]
            ok, rvec, tvec, inliers = solve_pnp_ransac(peaks_2d[:n_pks], obj, fov, W, H)
            if ok and inliers >= 3 and tvec is not None:
                tp = tvec.flatten()
                cam_lat = vp_lat[vp] + tp[2] / 111000.0
                cam_lon = vp_lon[vp] + tp[0] / (
                    111000.0 * np.cos(np.radians(vp_lat[vp]))
                )
                err = geodesic((tlat, tlon), (cam_lat, cam_lon)).meters / 1000
                if inliers > best_inliers or (
                    inliers == best_inliers and err < best_err
                ):
                    best_err = err
                    best_ok = True
                    best_inliers = inliers

        if best_ok:
            pnp_results.append(
                {
                    "sid": sid,
                    "ok": True,
                    "err_km": round(best_err, 1),
                    "inliers": best_inliers,
                    "n_pks": n_pks,
                    "n_dem": len(dem_pks),
                }
            )
            print(
                f"  [{si + 1}] {sid[:20]:20s} PnP: err={best_err:6.1f}km  inliers={best_inliers}/{n_pks}",
                flush=True,
            )
        else:
            pnp_results.append({"sid": sid, "ok": False, "reason": "no_convergence"})
            print(f"  [{si + 1}] {sid[:20]:20s} PnP: FAILED", flush=True)

    pnp_ok = [r for r in pnp_results if r.get("ok")]
    if pnp_ok:
        pe = np.array([r["err_km"] for r in pnp_ok])
        print(f"\nPnP success: {len(pnp_ok)}/{len(pnp_results)}", flush=True)
        print(
            f"  med={np.median(pe):.1f}km  <1km={int(np.sum(pe < 1))}/{len(pe)}  "
            f"<5km={int(np.sum(pe < 5))}/{len(pe)}  <10km={int(np.sum(pe < 10))}/{len(pe)}",
            flush=True,
        )
    else:
        print(f"\nPnP: 0/{len(pnp_results)} succeeded", flush=True)

    del dino_model
    print(f"\nTotal: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

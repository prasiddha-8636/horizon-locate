#!/usr/bin/env python
"""Quick test: 3 panos, stride=20 per crop, intersection voting."""
import heapq
import json
import time
import numpy as np
import pyarrow.parquet as pq
import sys
from pathlib import Path
from geopy.distance import geodesic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from horizon_format import decode_horizon_column
from query_profile import extract_elevation_profile

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
W, H_img = 1080, 720
TOP_K = 50
STRIDE = 20

with open(GT_FILE) as f:
    gt = json.load(f)
with open(ANNOT_FILE) as f:
    annots = json.load(f).get("annotations", {})

pq_file = pq.ParquetFile(str(DB_PATH))
meta = pq.read_table(str(DB_PATH), columns=["lon", "lat"])
lat_arr = meta.column("lat").to_pandas().values
lon_arr = meta.column("lon").to_pandas().values
batch0 = next(pq_file.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
decoded0 = decode_horizon_column(batch0.to_pandas()["raw_horizon_deg"].to_numpy())
n_bins = decoded0.shape[1]
bin_deg = 360.0 / n_bins

# Build pano map
panos = {}
for sid, points in annots.items():
    mp = CROPS_DIR / f"{sid}.json"
    if not mp.exists():
        continue
    with open(mp) as f:
        m = json.load(f)
    pid = m.get("pano_id")
    if not pid:
        continue
    if pid not in panos:
        panos[pid] = []
    m["sid"] = sid
    m["points"] = points
    panos[pid].append(m)

# Test 3 panos
test_ids = ["-yiHVpEf_kKTG9YGJ-dzsg", "--WyciZkeyJi1pLXhEO8BQ", "061cCYn21HxjaJ7h2mZ5ag"]

for pid in test_ids:
    if pid not in panos:
        continue
    crops = panos[pid]
    g = gt.get(crops[0]["sid"]) or gt.get(pid) or {}
    true_vp = int(g.get("closest_viewpoint_id", -1))
    true_lat, true_lon = g.get("true_lat"), g.get("true_lon")
    if true_vp < 0 or not true_lat:
        continue

    t0 = time.time()

    # Extract all crop profiles
    profiles = []
    for c in crops:
        pts = c["points"]
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if np.unique(xs).size < 2:
            continue
        cols = np.arange(W, dtype=np.float64)
        ri = np.clip(np.rint(np.interp(cols, xs, ys)), 1, H_img - 1).astype(int)
        mask = np.where(np.arange(H_img)[:, None] < ri[None, :], 0, 255).astype(np.uint8)
        fov_y = c.get("fov_y_deg", 65.0)
        r_tilt = c.get("cam_R_tilt") or g.get("cam_R_tilt")
        if r_tilt is not None:
            r_tilt = np.array(r_tilt)
        sr = np.full(W, H_img - 1, dtype=np.int32)
        for col in range(W):
            s = np.where(mask[:, col] == 0)[0]
            if len(s) > 0:
                sr[col] = s[-1]
        uc = (sr > 2) & (sr < H_img - 2)
        res = extract_elevation_profile(
            mask, fov_y_deg=fov_y, r_tilt=r_tilt, bin_deg=bin_deg,
            column_keep_mask=uc, azim_frame="camera",
        )
        if res["ok"]:
            profiles.append((c.get("heading_deg", 0.0), res["profile"]))

    if len(profiles) < 2:
        print(f"  {pid}: only {len(profiles)} crops, skip")
        continue

    # Per-crop scan with stride=20
    crop_top_vps = []
    for hi, prof in profiles:
        m_len = len(prof)
        center_bin = int(round((hi % 360.0) / bin_deg))
        half_m = m_len // 2
        bin_idx = (np.arange(m_len) - half_m + center_bin) % n_bins
        pf = np.asarray(prof, dtype=np.float64)
        pzm = pf - pf.mean()
        pnorm = np.linalg.norm(pzm)
        if pnorm < 1e-12:
            crop_top_vps.append(set())
            continue
        heap = []
        cs = 0
        for batch in pq_file.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            ch = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
            idxs = np.arange(0, len(ch), STRIDE)
            if len(idxs) == 0:
                cs += len(ch)
                continue
            dw = ch[idxs][:, bin_idx]
            dm = dw.mean(axis=1)
            dz = dw - dm[:, None]
            dn = np.linalg.norm(dz, axis=1)
            dn = np.maximum(dn, 1e-12)
            ncc = (dz @ pzm) / (dn * pnorm)
            for k in range(len(idxs)):
                vi = cs + idxs[k]
                cv = float(ncc[k])
                if len(heap) < TOP_K:
                    heapq.heappush(heap, (cv, vi))
                elif cv > heap[0][0]:
                    heapq.heapreplace(heap, (cv, vi))
            cs += len(ch)
        heap.sort(key=lambda x: -x[0])
        crop_top_vps.append(set(h[1] for h in heap[:10]))

    # Intersection
    common = crop_top_vps[0]
    for s in crop_top_vps[1:]:
        common = common & s

    # Match error
    if common:
        match_vp = list(common)[0]
        match_err = geodesic(
            (true_lat, true_lon), (lat_arr[match_vp], lon_arr[match_vp])
        ).meters
    else:
        match_err = float("inf")

    # True VP rank in each crop
    true_ranks = []
    for hi, prof in profiles:
        m_len = len(prof)
        center_bin = int(round((hi % 360.0) / bin_deg))
        half_m = m_len // 2
        bin_idx = (np.arange(m_len) - half_m + center_bin) % n_bins
        pf = np.asarray(prof, dtype=np.float64)
        pzm = pf - pf.mean()
        pnorm = np.linalg.norm(pzm)
        if pnorm < 1e-12:
            true_ranks.append(-1)
            continue
        rank = -1
        n = 0
        cs = 0
        for batch in pq_file.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            ch = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
            idxs = np.arange(0, len(ch), STRIDE)
            if len(idxs) == 0:
                cs += len(ch)
                continue
            dw = ch[idxs][:, bin_idx]
            dm = dw.mean(axis=1)
            dz = dw - dm[:, None]
            dn = np.linalg.norm(dz, axis=1)
            dn = np.maximum(dn, 1e-12)
            ncc = (dz @ pzm) / (dn * pnorm)
            for k in range(len(idxs)):
                vi = cs + idxs[k]
                if vi == true_vp:
                    # Count how many scanned VPs have higher NCC
                    rank = n + int(np.sum(ncc[:k] > ncc[k]))
                    break
                n += 1
            cs += len(ch)
            if rank >= 0:
                break
        true_ranks.append(rank)

    elapsed = time.time() - t0
    tag = "MATCH" if match_err < 1000 else "NO_VOTE"
    err_str = f"{match_err:.0f}" if match_err < 1e6 else "inf"
    print(
        f"  [{tag:7s}] {pid[:24]:24s} "
        f"match={err_str:>8s} "
        f"ranks={true_ranks} "
        f"common={len(common)} "
        f"[{elapsed:.0f}s]"
    )

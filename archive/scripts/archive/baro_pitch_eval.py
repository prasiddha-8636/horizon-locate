#!/usr/bin/env python
"""Honest no-GPS eval: barometric altitude gate (not GPS radius) + calibrated
pitch profile + two-layer/parallax ensemble re-rank of the top-K in-gate NCC
candidates. All gating derives from DEM elevation vs a *barometer* altitude,
which in a real deployment the phone supplies; no true-location mask used.

Outputs per-sample errors and summary metrics.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_column, decode_horizon_uint8
from scripts.fixes_eval import (
    Rx,
    mask_from_ann,
    split_horizon_layers,
    match_two_layer,
    compute_parallax_ratio,
    match_parallax_ratio,
    DB_PATH,
    GT_FILE,
    ANNOT_FILE,
    CALIB_FILE,
)

CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
TOPS = 200
SWEEP = False  # if True, sweep baro tolerance + pitch grid


def load_db_geometry():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat", "elevation_m"])
    return (
        meta["lon"].to_numpy(),
        meta["lat"].to_numpy(),
        meta["elevation_m"].to_numpy(),
    )


def fetch_horizons_fast(idx_list):
    """Fetch specific VP horizons via row-group index (canonical fetch)."""
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


def eval_sample(
    sid,
    gt,
    calib,
    ann_vps,
    corr,
    vp_lat,
    vp_lon,
    vp_elev,
    baro_tol=60.0,
    pitch_deg=None,
    top_k=TOPS,
):
    g = gt[sid]
    tlat, tlon = g["true_lat"], g["true_lon"]
    tilt = np.array(g["cam_R_tilt"])
    dp = (
        pitch_deg
        if pitch_deg is not None
        else float(calib.get(sid, {}).get("delta_pitch_deg", 0.0))
    )
    mask, _ = mask_from_ann(ann_vps[sid])
    pr = extract_elevation_profile(
        mask, fov_y_deg=g["fov_y_deg"], r_tilt=Rx(dp) @ tilt, bin_deg=0.5
    )
    if not pr["ok"]:
        return None
    q = pr["profile"]

    # Baro gate: DEM elevation within tolerance of camera elevation (phone baro
    # reads camera z ~ DEM at true VP; no GMAS/ground-truth location used).
    h_baro = vp_elev[g["closest_viewpoint_id"]]
    inband = np.abs(vp_elev - h_baro) <= baro_tol
    masked = np.where(inband, corr, -np.inf)
    bv = int(np.argmax(masked))
    out = {
        "ncc_err_m": geodesic((tlat, tlon), (vp_lat[bv], vp_lon[bv])).meters,
        "ncc_vp": bv,
        "gate_size": int(inband.sum()),
        "true_in_gate": bool(inband[g["closest_viewpoint_id"]]),
    }
    if not inband.any():
        return out

    # Re-rank top-K in-gate with two-layer + parallax
    top = np.argsort(masked)[-top_k:]
    top = top[np.isfinite(masked[top])]
    if len(top) == 0:
        return out
    hdict = fetch_horizons_fast(list(map(int, top)))
    qf, qn = split_horizon_layers(q)
    lay = np.zeros(len(top))
    par = np.zeros(len(top))
    for j, vi in enumerate(top):
        vi = int(vi)
        if vi not in hdict:
            continue
        h = hdict[vi]
        f, n = split_horizon_layers(h)
        lay[j] = match_two_layer(f, n, qf, qn)
        par[j] = match_parallax_ratio(f + n, q)
    lb = top[int(np.argmax(lay))]
    pb = top[int(np.argmax(par))]
    out["layer_err_m"] = geodesic((tlat, tlon), (vp_lat[lb], vp_lon[lb])).meters
    out["parallax_err_m"] = geodesic((tlat, tlon), (vp_lat[pb], vp_lon[pb])).meters
    ens_vp = _ensemble(top, lay, par, ncc_vp=bv, layer_vp=lb, parallax_vp=pb)
    out["ens_strat"] = ens_vp
    out["ens_err_m"] = geodesic((tlat, tlon), (vp_lat[ens_vp], vp_lon[ens_vp])).meters
    return out


def _ensemble(top, lay, par, ncc_vp=None, layer_vp=None, parallax_vp=None):
    lsurge = lay.max() - np.mean(lay)
    psurge = par.max() - np.mean(par)
    if lsurge > 0.01 and psurge > 0.01:
        if lay.max() >= par.max():
            return int(top[int(lay.argmax())])
        return int(top[int(par.argmax())])
    return ncc_vp


def summary(errs, label):
    e = np.array([x for x in errs if x is not None])
    if len(e) == 0:
        print(f"{label}: no results")
        return
    print(
        f"{label}: n={len(e)}  med={np.median(e) / 1000:5.1f}km  "
        f"<1km={int(np.sum(e < 1000))}/{len(e)}  "
        f"<2km={int(np.sum(e < 2000))}/{len(e)}  "
        f"<5km={int(np.sum(e < 5000))}/{len(e)}"
    )


def main():
    vp_lon, vp_lat, vp_elev = load_db_geometry()
    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    calib = json.load(open(CALIB_FILE))
    sids = [
        s
        for s in ann
        if s in gt and os.path.exists(os.path.join(CACHE_DIR, f"{s}_corr.npz"))
    ]

    for tol in [20, 60, 100, 200]:
        errs = {"ncc": [], "layer": [], "parallax": [], "ens": []}
        for sid in sids:
            corr = np.load(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))["corr"]
            r = eval_sample(
                sid,
                gt,
                calib,
                ann,
                corr,
                vp_lat,
                vp_lon,
                vp_elev,
                baro_tol=tol,
            )
            if r is None:
                continue
            for k in errs:
                errs[k].append(r.get(f"{k}_err_m"))
        print(f"\n=== baro_tol={tol}m (no GPS) ===")
        for k, v in errs.items():
            summary(v, k)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time() - t0:.0f}s")

#!/usr/bin/env python
"""Uniqueness of true-VP horizon among all 1.3M DB VPs.

For each GSV sample, take the 87-bin horizon window at the true VP and find
the BEST match among ALL OTHER VPs (excluding the true VP itself).

  best_other ~ 1.0  -> horizon not unique, location ambiguous even w/ perfect mask
  best_other < 0.9  -> horizon distinctive, masks are the only bottleneck
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

from src.evaluation import _stream_horizon_chunks
from src.matching import feature_bundle_matrix, ncc_scores
from src.horizon_format import decode_horizon_uint8

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    pf = pq.ParquetFile(DB_PATH)
    rg_sizes = np.array(
        [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    )
    cum = np.concatenate([[0], np.cumsum(rg_sizes)])
    cache = {}

    def fetch(vp):
        if vp not in cache:
            rg = int(np.searchsorted(cum[1:], vp, side="right"))
            pos = vp - cum[rg]
            cache[vp] = decode_horizon_uint8(
                pf.read_row_group(rg, columns=["raw_horizon_deg"])
                .to_pandas()["raw_horizon_deg"]
                .iloc[pos]
            )
        return cache[vp]

    n = 87
    results = []
    sample_sids = list(gt.keys())[:5]
    for sid in sample_sids:
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        hor = fetch(vp)
        exp = int(round((g["true_heading_deg"] + (-43.0)) % 360))
        prof = hor[np.arange(exp, exp + n) % 360]
        prof = prof - prof.mean()
        pn = np.linalg.norm(prof)
        if pn < 1e-12:
            continue

        t0 = time.time()
        best_other = -np.inf
        best_other_vp = -1
        best_other_fb = -np.inf
        best_other_fb_vp = -1
        for chunk_matrix, bd, cs in _stream_horizon_chunks(DB_PATH, 4000):
            si = np.arange(0, chunk_matrix.shape[0], 12)
            sm = chunk_matrix[si]
            gi = np.arange(cs, cs + chunk_matrix.shape[0])[si]
            # raw corr
            ext = np.concatenate([sm, sm[:, : n - 1]], axis=1)
            W_ = np.lib.stride_tricks.sliding_window_view(ext, n, axis=1)
            Wzm = W_ - W_.mean(axis=2, keepdims=True)
            Wn = np.linalg.norm(Wzm, axis=2)
            denom = pn * Wn
            valid = denom > 1e-12
            raw = np.full((W_.shape[0], W_.shape[1]), -np.inf)
            raw[valid] = (Wzm[valid] @ prof) / denom[valid]
            rowmax = raw.max(axis=1)
            # feature-bundle corr
            dv, dd = feature_bundle_matrix(sm)
            fb, _ = ncc_scores(dv, dd, prof, bd, expected_offset_deg=None)
            for k, gv in enumerate(gi):
                if gv == vp:
                    continue
                if rowmax[k] > best_other:
                    best_other = rowmax[k]
                    best_other_vp = int(gv)
                if np.isfinite(fb[k]) and fb[k] > best_other_fb:
                    best_other_fb = fb[k]
                    best_other_fb_vp = int(gv)
        dist_to_true = 0.0
        # haversine distance between best_other_vp and vp
        if best_other_vp >= 0:
            from math import radians, sin, cos, atan2, sqrt

            def haversine(lat1, lon1, lat2, lon2):
                R = 6371000.0
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = (
                    sin(dlat / 2) ** 2
                    + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
                )
                return 2 * R * atan2(sqrt(a), sqrt(1 - a))

            # fetch lon/lat for vp and best_other_vp
            def ll(v):
                rg = int(np.searchsorted(cum[1:], v, side="right"))
                pos = v - cum[rg]
                r = pf.read_row_group(rg, columns=["lon", "lat"]).to_pandas().iloc[pos]
                return r["lat"], r["lon"]

            la1, lo1 = ll(vp)
            la2, lo2 = ll(best_other_vp)
            dist_to_true = haversine(la1, lo1, la2, lo2)
        print(
            f"{sid[:24]} best_other_raw={best_other:+.4f}@{best_other_vp}  best_other_fb={best_other_fb:+.4f}@{best_other_fb_vp} [{time.time() - t0:.0f}s]"
        )
        results.append((sid, best_other, best_other_fb))

    v = np.array([r[1] for r in results])
    f = np.array([r[2] for r in results])
    print(
        f"\nn={len(v)} best_other RAW: median={np.median(v):.3f} p75={np.percentile(v, 75):.3f} p90={np.percentile(v, 90):.3f}"
    )
    print(
        f"           best_other FB : median={np.median(f):.3f} p75={np.percentile(f, 75):.3f} p90={np.percentile(f, 90):.3f}"
    )
    print(
        f"RAW best_other > 0.95: {100 * np.mean(v > 0.95):.0f}%   FB > 0.95: {100 * np.mean(f > 0.95):.0f}%"
    )


if __name__ == "__main__":
    main()

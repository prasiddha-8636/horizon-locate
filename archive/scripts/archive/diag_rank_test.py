#!/usr/bin/env python
"""Full-DB rank test: raw correlation vs feature-bundle, no compass."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

from src.query_profile import extract_elevation_profile
from src.evaluation import load_db_metadata, _stream_horizon_chunks
from src.matching import feature_bundle_matrix, ncc_scores

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
MASKS_DIR = ROOT / "data" / "street_view" / "masks"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"

SIDS = ["EZ15hr7Ojg3ito6i3BY6YA", "CIHM0ogKEICAgIC6rYfGdw", "5LDxSllCwStxt7aJ35F2Ww"]


first = next(
    pq.ParquetFile(DB_PATH).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
def main():
    with open(GT_PATH) as f:
        gt = json.load(f)
    lon, lat, elev_m, n_vp = load_db_metadata(DB_PATH)

    profiles = {}
    for sid in SIDS:
        g = gt[sid]
        pr = extract_elevation_profile(
            str(MASKS_DIR / f"{sid}.png"),
            fov_y_deg=g["fov_y_deg"],
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=BIN_DEG,
        )
        if pr["ok"]:
            profiles[sid] = (pr["profile"], int(g["closest_viewpoint_id"]), g)
        print(f"{sid[:24]} vp={int(g['closest_viewpoint_id'])} profile_ok={pr['ok']}")

    for sid, (prof, vp, g) in profiles.items():
        t0 = time.time()
        # raw correlation, no compass, stride 12
        best_raw = np.full(n_vp, -np.inf)
        best_fb = np.full(n_vp, -np.inf)
        n = len(prof)
        prof_zm = prof - prof.mean()
        pn = np.linalg.norm(prof_zm)
        for chunk_matrix, bd, cs in _stream_horizon_chunks(DB_PATH, 4000):
            si = np.arange(0, chunk_matrix.shape[0], 12)
            sm = chunk_matrix[si]
            # raw corr
            ext = np.concatenate([sm, sm[:, : n - 1]], axis=1)
            W = np.lib.stride_tricks.sliding_window_view(ext, n, axis=1)
            W_zm = W - W.mean(axis=2, keepdims=True)
            Wn = np.linalg.norm(W_zm, axis=2)
            denom = pn * Wn
            valid = denom > 1e-12
            raw = np.full((W.shape[0], W.shape[1]), -np.inf)
            raw[valid] = (W_zm[valid] @ prof_zm) / denom[valid]
            best_raw_row = raw.max(axis=1)
            gi = np.arange(cs, cs + chunk_matrix.shape[0])[si]
            best_raw[gi] = best_raw_row
            # feature bundle
            dv, dd = feature_bundle_matrix(sm)
            corr, _ = ncc_scores(dv, dd, prof, bd, expected_offset_deg=None)
            best_fb[gi] = corr
        dt = time.time() - t0
        rank_raw = int(np.sum(best_raw > best_raw[vp])) + 1
        rank_fb = int(np.sum(best_fb > best_fb[vp])) + 1
        print(
            f"\n{sid[:24]} vp={vp} raw_corr={best_raw[vp]:.3f} fb_corr={best_fb[vp]:.3f}"
        )
        print(f"  RAW rank: {rank_raw}/{n_vp}   FB rank: {rank_fb}/{n_vp}  ({dt:.0f}s)")
        top5r = np.argsort(best_raw)[-5:][::-1]
        top5f = np.argsort(best_fb)[-5:][::-1]
        print(
            "  top-5 RAW:",
            [int(i) for i in top5r],
            [f"{best_raw[i]:.3f}" for i in top5r],
        )
        print(
            "  top-5 FB :",
            [int(i) for i in top5f],
            [f"{best_fb[i]:.3f}" for i in top5f],
        )


if __name__ == "__main__":
    main()

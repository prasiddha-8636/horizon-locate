#!/usr/bin/env python
"""GSV geometry check: build ideal profile from DB horizon on the crop's
azimuth grid. If ideal profile matches DB at expected offset -> geometry is
right and the mask is the problem. If not -> crop projection is broken.
"""

import sys, json, os
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import feature_bundle_matrix, ncc_scores
from src.horizon_format import decode_horizon_uint8

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")


def fetch_horizon(vp_idx):
    pf = pq.ParquetFile(DB)
    rg = int(vp_idx) // 4096
    pos = int(vp_idx) % 4096
    return decode_horizon_uint8(
        pf.read_row_group(rg, columns=["raw_horizon_deg"])
        .to_pandas()["raw_horizon_deg"]
        .iloc[pos]
    )


def ideal_profile(horizon, heading, start_az, n_bins, bin_deg=1.0):
    """DB-predicted elevation on the mask profile's bin grid.
    profile bin k -> camera azimuth start_az + k -> world azimuth heading + start_az + k."""
    azs = start_az + bin_deg * np.arange(n_bins)
    world = np.round((heading + azs) / bin_deg).astype(int) % int(round(360.0 / bin_deg))
    return horizon[world]


first = next(
    pq.ParquetFile(DB_PATH).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
def main():
    with open(os.path.join(ROOT, "data/street_view/ground_truth.json")) as f:
        gt = json.load(f)
    sids = list(gt.keys())[:10]
    print(
        f"{'sid':<20} {'geom_corr':>9} {'mask_corr':>9} {'exp_off':>7} {'ideal_off':>9} {'mask_off':>8} {'delta':>5}"
    )
    for sid in sids:
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        horizon = fetch_horizon(vp)
        heading = g["true_heading_deg"]
        fov_y = g.get("fov_y_deg", 65.0)
        W, H = 1080, 720
        aspect = W / H

        pr = extract_elevation_profile(
            os.path.join(ROOT, f"data/street_view/masks/{sid}.png"),
            fov_y_deg=fov_y,
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=BIN_DEG,
        )
        if not pr["ok"]:
            print(f"{sid:<20} PROFILE_FAIL")
            continue
        mask_prof = pr["profile"]
        start_az = pr["start_az"]
        exp_off = (heading + start_az) % 360.0

        ideal = ideal_profile(horizon, heading, start_az, len(mask_prof))

        db_val, db_d1 = feature_bundle_matrix(horizon[None, :])
        c_ideal, o_ideal = ncc_scores(db_val, db_d1, ideal, 1.0, weights=(0.5, 0.5))
        c_mask, o_mask = ncc_scores(db_val, db_d1, mask_prof, 1.0, weights=(0.5, 0.5))
        o_ideal, o_mask = int(o_ideal[0]), int(o_mask[0])
        delta = (o_ideal - exp_off) % 360
        delta = min(delta, 360 - delta)
        print(
            f"{sid:<20} {float(c_ideal[0]):>9.3f} {float(c_mask[0]):>9.3f} "
            f"{exp_off:>7.0f} {o_ideal:>9} {o_mask:>8} {delta:>5}"
        )


if __name__ == "__main__":
    main()

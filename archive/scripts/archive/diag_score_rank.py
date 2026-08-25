#!/usr/bin/env python
"""Score-competition diagnostic: GT VP score vs random-VP score distribution.

For a few GSV samples, compute NCC against a large random sample of DB VPs
(with compass+altimeter filters) and report where the GT VP ranks.
"""

import sys, json, os
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.matching import feature_bundle_matrix, ncc_scores
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8
from geopy.distance import geodesic

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
N_VP = 1338650
RNG = np.random.default_rng(0)


def fetch_rows(indices):
    """Fetch horizons at arbitrary VP indices (row-group indexed)."""
    pf = pq.ParquetFile(DB)
    out = {}
    groups = {}
    for idx in indices:
        groups.setdefault(int(idx) // 4096, []).append(int(idx))
    for rg, idxs in groups.items():
        batch = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
        for idx in idxs:
            out[idx] = np.asarray(
                decode_horizon_uint8(batch["raw_horizon_deg"].iloc[idx % 4096])
            )
    return out


first = next(
    pq.ParquetFile(DB_PATH).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
def main():
    with open(os.path.join(ROOT, "data/street_view/ground_truth.json")) as f:
        gt = json.load(f)
    meta = pq.read_table(DB, columns=["lon", "lat", "elevation_m"])
    lon_arr = meta.column("lon").to_pandas().values
    lat_arr = meta.column("lat").to_pandas().values
    elev_arr = meta.column("elevation_m").to_pandas().values

    sample_idx = RNG.choice(N_VP, size=5000, replace=False)
    sample_elev = elev_arr[sample_idx]
    sample_lat = lat_arr[sample_idx]
    sample_lon = lon_arr[sample_idx]
    horizons = fetch_rows(sample_idx)

    sids = list(gt.keys())[:8]
    print(
        f"{'sid':<22} {'tol':>4} {'gtvp_score':>10} {'max_sample':>10} {'p99':>6} {'gtvp_pctile':>12} {'gtvp_rank':>9}"
    )
    for sid in sids:
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        pr = extract_elevation_profile(
            os.path.join(ROOT, f"data/street_view/masks/{sid}.png"),
            fov_y_deg=g.get("fov_y_deg", 65.0),
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=BIN_DEG,
        )
        if not pr["ok"]:
            print(f"{sid:<22} PROFILE_FAIL")
            continue
        prof = pr["profile"]
        exp_off = (g["true_heading_deg"] + pr["start_az"]) % 360.0
        elev = g["eye_z_m"]

        for tol in [10.0, 20.0]:
            # GT VP score
            gt_h = horizons.get(vp)
            if gt_h is None:
                gt_h = fetch_rows([vp])[vp]
            db_val, db_d1 = feature_bundle_matrix(np.stack([gt_h]))
            c_gt, _ = ncc_scores(
                db_val,
                db_d1,
                prof,
                1.0,
                weights=(0.5, 0.5),
                expected_offset_deg=exp_off,
                tolerance_deg=tol,
            )
            gt_score = float(c_gt[0])

            # sample scores
            arr = np.stack([horizons[i] for i in sample_idx])
            db_val, db_d1 = feature_bundle_matrix(arr)
            c_samp, _ = ncc_scores(
                db_val,
                db_d1,
                prof,
                1.0,
                weights=(0.5, 0.5),
                expected_offset_deg=exp_off,
                tolerance_deg=tol,
            )
            ok_alt = np.abs(sample_elev - elev) <= 200.0
            c_samp = np.where(ok_alt, c_samp, -np.inf)
            n_valid = int(np.isfinite(c_samp).sum())
            finite = c_samp[c_samp > -np.inf]
            max_s = float(finite.max()) if len(finite) else -np.inf
            p99 = float(np.percentile(finite, 99)) if len(finite) else -np.inf
            pctile = 100 * np.mean(finite >= gt_score) if len(finite) else 1.0
            rank = int((finite >= gt_score).sum()) + 1 if len(finite) else 0
            print(
                f"{sid:<22} {tol:>4.0f} {gt_score:>10.3f} {max_s:>10.3f} "
                f"{p99:>6.3f} {pctile:>11.1f}% {rank:>7}/{n_valid}"
            )


if __name__ == "__main__":
    main()

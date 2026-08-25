#!/usr/bin/env python
"""Test exact-offset scoring: correlation at ONLY the expected offset.

Hypothesis: best-of-41-offsets inflates random-VP scores. Scoring at the exact
expected offset should make the GT VP discriminative.
"""

import sys, json, os
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.matching import feature_bundle_matrix, ncc_scores
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
N_VP = 1338650
RNG = np.random.default_rng(0)


def ncc_fixed_offset(horizon, profile, offset_bin):
    """Correlation at a single fixed offset (no max over window)."""
    wind = horizon[np.arange(offset_bin, offset_bin + len(profile)) % 360]
    p = np.asarray(profile, dtype=np.float64)
    w = np.asarray(wind, dtype=np.float64)
    if p.std() < 1e-9 or w.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(p, w)[0, 1])


first = next(
    pq.ParquetFile(DB_PATH).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
def main():
    with open(os.path.join(ROOT, "data/street_view/ground_truth.json")) as f:
        gt = json.load(f)
    meta = pq.read_table(DB, columns=["lon", "lat", "elevation_m"])
    elev_arr = meta.column("elevation_m").to_pandas().values

    sample_idx = RNG.choice(N_VP, size=4000, replace=False)
    sample_elev = elev_arr[sample_idx]

    # fetch all horizons in one streaming pass
    pf = pq.ParquetFile(DB)
    horizons = {}
    groups = {}
    for idx in sample_idx:
        groups.setdefault(int(idx) // 4096, []).append(int(idx))
    for rg, idxs in groups.items():
        batch = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
        for idx in idxs:
            horizons[idx] = np.asarray(
                decode_horizon_uint8(batch["raw_horizon_deg"].iloc[idx % 4096])
            )

    sids = list(gt.keys())[:12]
    print(
        f"{'sid':<22} {'gtvp':>7} {'max_samp':>8} {'p99':>6} {'pctile_gt':>9} {'worse_frac':>10}"
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
        exp_bin = int(round(exp_off))
        elev = g["eye_z_m"]

        gt_h = horizons.get(vp)
        if gt_h is None:
            gt_h = decode_horizon_uint8(
                pf.read_row_group(vp // 4096, columns=["raw_horizon_deg"])
                .to_pandas()["raw_horizon_deg"]
                .iloc[vp % 4096]
            )
        gt_score = ncc_fixed_offset(gt_h, prof, exp_bin)

        arr = np.stack([horizons[i] for i in sample_idx])
        scores = np.array(
            [ncc_fixed_offset(arr[j], prof, exp_bin) for j in range(len(sample_idx))]
        )
        ok_alt = np.abs(sample_elev - elev) <= 200.0
        scores = np.where(ok_alt, scores, -np.inf)
        finite = scores[scores > -np.inf]
        p99 = np.percentile(finite, 99)
        pctile = 100 * np.mean(finite >= gt_score)
        worse = 100 * np.mean(finite <= gt_score)
        print(
            f"{sid:<22} {gt_score:>7.3f} {float(finite.max()):>8.3f} "
            f"{p99:>6.3f} {pctile:>8.1f}% {worse:>9.1f}%"
        )


if __name__ == "__main__":
    main()

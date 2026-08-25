#!/usr/bin/env python
"""Focused diagnostic: profile vs GT-VP horizon agreement (no DB scan).

For each sample, NCC between extracted profile and the DB horizon at the
GT viewpoint across all 360 offsets. Isolates:
  - convention correctness (peak offset vs expected offset)
  - profile quality (peak score)
  - systematic vertical bias (mean elevation diff)
"""

import json, sys, os
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.matching import feature_bundle_matrix, ncc_scores
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8

DB = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_PATH = os.path.join(ROOT, "data/synthetic_dataset/ground_truth.json")
MASK_DIR = os.path.join(ROOT, "data/synthetic_dataset/masks")
PRED_DIR = os.path.join(ROOT, "data/synthetic_dataset/predicted_masks")

with open(GT_PATH) as f:
    gt = json.load(f)

pf0 = pq.ParquetFile(DB)
first = next(pf0.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
del pf0


def fetch_horizon(vp_idx):
    pf = pq.ParquetFile(DB)
    rg = int(vp_idx) // 4096
    pos = int(vp_idx) % 4096
    batch = pf.read_row_group(rg, columns=["raw_horizon_deg"])
    return decode_horizon_uint8(batch.to_pandas()["raw_horizon_deg"].iloc[pos])


def ncc_all_offsets(horizon, profile):
    """NCC of profile vs horizon at ALL offsets (no compass mask)."""
    db_val, db_d1 = feature_bundle_matrix(horizon[None, :])
    corr, offs = ncc_scores(
        db_val,
        db_d1,
        profile,
        bin_deg,
        weights=(0.5, 0.5),
        expected_offset_deg=None,
        tolerance_deg=None,
    )
    return float(corr[0]), float(offs[0])


def analyze(tag, mask_dir, sids):
    print(f"=== {tag} ===")
    rows = []
    for sid in sids:
        g = gt[sid]
        pr = extract_elevation_profile(
            os.path.join(mask_dir, f"sample_{int(sid):04d}.png"),
            fov_y_deg=g["fov_y_deg"],
            bin_deg=bin_deg,
        )
        if not pr["ok"]:
            rows.append((sid, "PROFILE_FAIL", pr["status"]))
            continue
        horizon = fetch_horizon(int(g["closest_viewpoint_id"]))
        score, off = ncc_all_offsets(horizon, pr["profile"])
        off = int(off)
        exp_off = (g["true_heading_deg"] + pr["start_az"]) % 360.0
        off_err = np.degrees(np.angle(np.exp(1j * np.radians(off - exp_off)))) % 360
        off_err = min(off_err, 360 - off_err)
        prof = pr["profile"]
        # vertical bias: mean(profile) vs mean(horizon over matched window)
        wind = horizon[np.arange(off, off + len(prof)) % 360]
        bias = np.mean(prof - wind)
        std = np.std(prof)
        rows.append((sid, "OK", score, off_err, bias, std, len(prof)))

    oks = [r for r in rows if r[1] == "OK"]
    print(f"  profile ok: {len(oks)}/{len(rows)}")
    if not oks:
        print("  (all failed)")
        return
    scores = np.array([r[2] for r in oks])
    offs = np.array([r[3] for r in oks])
    biases = np.array([r[4] for r in oks])
    stds = np.array([r[5] for r in oks])
    print(
        f"  peak NCC @ GT-VP: median={np.median(scores):.3f} p25={np.percentile(scores, 25):.3f}"
    )
    print(
        f"  offset err vs expected: median={np.median(offs):.1f}deg p90={np.percentile(offs, 90):.1f}deg"
    )
    print(f"  vertical bias: median={np.median(biases):+.2f}deg")
    print(f"  profile std: median={np.median(stds):.2f}deg")
    print()


sids = list(gt.keys())[:30]
print(f"Testing {len(sids)} synthetic samples (bin={bin_deg})\n")
analyze("GT render mask", MASK_DIR, sids)
analyze("predicted mask", PRED_DIR, sids)

#!/usr/bin/env python
"""Phase 2: decompose GSV profile-vs-horizon failure.

For every GSV sample, correlate the mask profile against the true-VP horizon
at ALL 360 azimuth offsets.

  best_corr high + expected_corr high  -> sample OK
  best_corr high + expected_corr low   -> pano rotation / azimuth shift
  best_corr low                        -> mask/profile quality problem
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
MASKS_DIR = ROOT / "data" / "street_view" / "masks"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"


def stream_horizons(vp_set):
    """Yield {vp: horizon} for the requested VPs by streaming the DB once."""
    pf = pq.ParquetFile(DB_PATH)
    found = {}
    need = set(int(v) for v in vp_set)
    for rg in range(pf.num_row_groups):
        cols = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
        # local index = global index - row offset. We need to know the offset.
        rg_sizes = np.array(
            [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
        )
        offset = int(np.sum(rg_sizes[:rg]))
        for pos, h in enumerate(cols["raw_horizon_deg"]):
            h = decode_horizon_uint8(h)
            gv = offset + pos
            if gv in need:
                found[gv] = np.asarray(h, dtype=np.float64)
                need.discard(gv)
        if not need:
            break
    return found


def best_offset_corr(prof, hor):
    """Best correlation of prof against hor at any circular offset (vectorized)."""
    n = len(prof)
    L = len(hor)
    prof_zm = prof - prof.mean()
    pnorm = np.linalg.norm(prof_zm)
    if pnorm < 1e-12:
        return 0.0, 0
    ext = np.concatenate([hor, hor[: n - 1]])
    W = np.lib.stride_tricks.sliding_window_view(ext, n)
    W_zm = W - W.mean(axis=1, keepdims=True)
    Wn = np.linalg.norm(W_zm, axis=1)
    denom = pnorm * Wn
    valid = denom > 1e-12
    corr = np.full(L, -np.inf)
    corr[valid] = (W_zm[valid] @ prof_zm) / denom[valid]
    best = int(np.argmax(corr))
    return float(corr[best]), best


first = next(
    pq.ParquetFile(DB_PATH).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
def main():
    t0 = time.time()
    with open(GT_PATH) as f:
        gt = json.load(f)

    profiles = {}
    needed_vps = set()
    for sid, g in gt.items():
        mask_path = MASKS_DIR / f"{sid}.png"
        if not mask_path.exists():
            continue
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        pr = extract_elevation_profile(
            str(mask_path),
            fov_y_deg=g["fov_y_deg"],
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=BIN_DEG,
        )
        if not pr["ok"]:
            continue
        profiles[sid] = {
            "prof": pr["profile"],
            "sa": pr["start_az"],
            "hdg": g["true_heading_deg"],
            "vp": vp,
        }
        needed_vps.add(vp)
    print(f"profiles extracted: {len(profiles)} in {time.time() - t0:.0f}s", flush=True)

    horizons = stream_horizons(needed_vps)
    print(f"horizons fetched: {len(horizons)} in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for sid, info in profiles.items():
        prof = info["prof"]
        n = len(prof)
        exp = int(round((info["hdg"] + info["sa"]) % 360))
        hor = horizons[info["vp"]]
        w_exp = hor[np.arange(exp, exp + n) % 360]
        exp_corr = float(np.corrcoef(prof, w_exp)[0, 1])
        best_corr, best_off = best_offset_corr(prof, hor)
        rev_corr = float(np.corrcoef(prof[::-1], w_exp)[0, 1])
        off_delta = (best_off - exp) % 360
        off_delta = min(off_delta, 360 - off_delta)
        rows.append((sid, exp_corr, best_corr, rev_corr, off_delta, n))

    r = np.array([(x[1], x[2], x[3], x[4]) for x in rows])
    print(f"Samples: {len(rows)}")
    print(
        f"  exp_corr : median={np.median(r[:, 0]):+.3f} p25={np.percentile(r[:, 0], 25):+.3f} p75={np.percentile(r[:, 0], 75):+.3f}"
    )
    print(
        f"  best_corr: median={np.median(r[:, 1]):+.3f} p25={np.percentile(r[:, 1], 25):+.3f} p75={np.percentile(r[:, 1], 75):+.3f}"
    )
    print(f"  rev_corr : median={np.median(r[:, 2]):+.3f}")
    print()

    good = (r[:, 1] >= 0.9) & (r[:, 0] >= 0.9)
    rotated = (r[:, 1] >= 0.9) & (r[:, 0] < 0.9)
    maskq = r[:, 1] < 0.9
    print(
        f"OK          (best>=.9 & exp>=.9): {good.sum():4d} ({100 * good.mean():.0f}%)"
    )
    print(
        f"ROTATED     (best>=.9 & exp<.9 ): {rotated.sum():4d} ({100 * rotated.mean():.0f}%)  median off_delta={np.median(r[rotated, 3]):.0f}deg"
    )
    print(
        f"MASK QUALITY(best<.9           ): {maskq.sum():4d} ({100 * maskq.mean():.0f}%)"
    )
    print()

    goodb = r[:, 1] >= 0.9
    deltas = r[goodb, 3]
    if len(deltas):
        print(
            f"offset delta (best>=.9): median={np.median(deltas):.0f} p25={np.percentile(deltas, 25):.0f} p75={np.percentile(deltas, 75):.0f}"
        )
        for bucket in [(0, 5), (5, 20), (20, 60), (60, 181)]:
            m = (deltas >= bucket[0]) & (deltas < bucket[1])
            print(
                f"  delta {bucket[0]:3d}-{bucket[1]:3d} deg: {m.sum():4d} ({100 * m.mean():.0f}%)"
            )

    for label, mask in [("OK", good), ("ROTATED", rotated), ("MASK_QUALITY", maskq)]:
        idxs = np.where(mask)[0][:4]
        if len(idxs):
            print(f"\n{label} examples:")
            for i in idxs:
                x = rows[i]
                print(
                    f"  {x[0][:24]:24s} exp={x[1]:+.3f} best={x[2]:+.3f} rev={x[3]:+.3f} d_off={x[4]:3.0f} n={x[5]}"
                )


if __name__ == "__main__":
    main()

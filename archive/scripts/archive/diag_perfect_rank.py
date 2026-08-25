#!/usr/bin/env python
"""Full-DB rank test with the PERFECT profile (DB horizon at true VP).

Answers: is a perfect 87-bin skyline distinctive enough to rank the true VP #1
among 1.3M VPs?  If not, no mask improvement can fix matching.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

from src.evaluation import load_db_metadata, _stream_horizon_chunks
from src.horizon_format import decode_horizon_uint8

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)
    lon, lat, elev_m, n_vp = load_db_metadata(DB_PATH)

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

    n = 87  # fixed profile width for all samples (from GSV fov)
    results = []
    for sid in list(gt.keys())[:40]:
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        hor = fetch(vp)
        # perfect profile = horizon at true VP over the expected window
        exp = int(round((g["true_heading_deg"] + (-43.0)) % 360))
        prof = hor[np.arange(exp, exp + n) % 360]
        prof = prof - prof.mean()
        pn = np.linalg.norm(prof)
        if pn < 1e-12:
            continue
        t0 = time.time()
        best = np.full(n_vp, -np.inf)
        for chunk_matrix, bd, cs in _stream_horizon_chunks(DB_PATH, 4000):
            si = np.arange(0, chunk_matrix.shape[0], 12)
            sm = chunk_matrix[si]
            ext = np.concatenate([sm, sm[:, : n - 1]], axis=1)
            W_ = np.lib.stride_tricks.sliding_window_view(ext, n, axis=1)
            Wzm = W_ - W_.mean(axis=2, keepdims=True)
            Wn = np.linalg.norm(Wzm, axis=2)
            denom = pn * Wn
            valid = denom > 1e-12
            raw = np.full((W_.shape[0], W_.shape[1]), -np.inf)
            raw[valid] = (Wzm[valid] @ prof) / denom[valid]
            gi = np.arange(cs, cs + chunk_matrix.shape[0])[si]
            best[gi] = raw.max(axis=1)
        # exact score at true vp (may be off the stride grid)
        vp_row = cum[np.searchsorted(cum[1:], vp, side="right")]
        # true-vp window corr:
        vp_win = hor[np.arange(exp, exp + n) % 360]
        vp_corr = float(
            np.dot(prof, vp_win - vp_win.mean())
            / (pn * np.linalg.norm(vp_win - vp_win.mean()))
        )
        # only count VPs actually scored (stride grid)
        scored = np.isfinite(best)
        # but also VPs with vp%12 in {0,4,8}; true vp might not be scored -> approximate:
        # compare true-vp corr against the distribution of scored maxima
        above = np.sum(best > vp_corr)
        print(
            f"{sid[:24]} true_vp_corr={vp_corr:+.3f} scored={scored.sum():7d} above_true={above:6d} ({time.time() - t0:.0f}s)"
        )
        results.append((sid, vp_corr, above, int(scored.sum())))

    print(
        f"\nMedian true-VP perfect-profile corr: {np.median([r[1] for r in results]):+.3f}"
    )


if __name__ == "__main__":
    main()

"""Re-validate synthetic profiles against corrected VP mapping."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8

GT_PATH = ROOT / "data" / "synthetic_dataset" / "ground_truth.json"
MASKS_DIR = ROOT / "data" / "synthetic_dataset" / "masks"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"


def fetch_horizon(vp):
    pf = pq.ParquetFile(DB_PATH)
    rg_sizes = np.array(
        [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    )
    cum = np.concatenate([[0], np.cumsum(rg_sizes)])
    rg = int(np.searchsorted(cum[1:], vp, side="right"))
    pos = vp - cum[rg]
    return decode_horizon_uint8(
        pf.read_row_group(rg, columns=["raw_horizon_deg"])
        .to_pandas()["raw_horizon_deg"]
        .iloc[pos]
    )


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    cf, cr, med = [], [], []
    n_ok = 0
    for sid in sorted(gt.keys()):
        g = gt[sid]
        mask_path = MASKS_DIR / f"sample_{int(sid):04d}.png"
        if not os.path.exists(mask_path):
            continue
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        pr = extract_elevation_profile(
            str(mask_path),
            fov_y_deg=g["fov_y_deg"],
            r_tilt=np.array(g["cam_R_tilt"]),
            bin_deg=1.0,
        )
        if not pr["ok"]:
            continue
        prof = pr["profile"]
        sa = pr["start_az"]
        n = len(prof)
        exp = int(round((g["true_heading_deg"] + sa) % 360))
        hor = fetch_horizon(vp)
        w = hor[np.arange(exp, exp + n) % 360]
        cf.append(np.corrcoef(prof, w)[0, 1])
        cr.append(np.corrcoef(prof[::-1], w)[0, 1])
        n_ok += 1

    cf = np.array(cf)
    cr = np.array(cr)
    print(f"Validated {n_ok} synthetic samples (corrected VPs)")
    print(
        f"  FWD: median={np.median(cf):+.3f} mean={np.mean(cf):+.3f} p10={np.percentile(cf, 10):+.3f}"
    )
    print(f"  REV: median={np.median(cr):+.3f}")
    print(f"  FWD better: {100 * np.mean(cf >= cr):.0f}%")
    print(f"  Samples with FWD > 0.95: {100 * np.mean(cf > 0.95):.0f}%")


if __name__ == "__main__":
    main()

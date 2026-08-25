#!/usr/bin/env python
"""Verify fixed slice_perspective: regenerate crops, re-segment, compare.

Checks that regenerated crops + fresh U-Net masks produce profiles that
match the correct-VP horizon at the expected offset (heading + start_az).
"""

import sys, os, json, tempfile
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.streetview_utils import slice_perspective
from src.segmentation import load_segmentation_model, segment_image
from src.query_profile import extract_elevation_profile
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


def main():
    model = load_segmentation_model(
        os.path.join(ROOT, "data/sky_segmentation_unet_model.pth"), "cpu"
    )
    gt = json.load(open(os.path.join(ROOT, "data/street_view/ground_truth.json")))
    crop = json.load(open(os.path.join(ROOT, "data/street_view/crop_quality.json")))

    tmp = tempfile.mkdtemp()
    sids = list(gt.keys())[:10]
    print(f"{'sid':<22} {'corr':>6} {'status'}")
    for sid in sids:
        g = gt[sid]
        c = crop[sid]
        heading = c["best_heading"]
        pitch = c["pitch_deg"]
        pano = os.path.join(ROOT, f"data/street_view/panos/{sid}.jpg")
        img = slice_perspective(
            pano,
            heading_deg=heading,
            pitch_deg=pitch,
            roll_deg=0.0,
            fov_y_deg=65.0,
            out_w=1080,
            out_h=720,
        )
        img_path = os.path.join(tmp, f"{sid}.png")
        img.save(img_path)
        mask_path = os.path.join(tmp, f"{sid}_mask.png")
        res = segment_image(model, img_path, mask_path, "cpu", tta=True)
        if not res["ok"]:
            print(f"{sid:<22} SEG_FAIL {res['status']}")
            continue
        pr = extract_elevation_profile(
            mask_path, fov_y_deg=65.0, r_tilt=np.array(g["cam_R_tilt"]), bin_deg=1.0
        )
        if not pr["ok"]:
            print(f"{sid:<22} PROFILE_FAIL")
            continue
        vp = int(c["nearest_vp"])
        hor = fetch_horizon(vp)
        exp = int(round((heading + pr["start_az"]) % 360))
        w = hor[np.arange(exp, exp + len(pr["profile"])) % 360]
        cf = np.corrcoef(pr["profile"], w)[0, 1]
        cr = np.corrcoef(pr["profile"][::-1], w)[0, 1]
        print(f"{sid:<22} {max(cf, cr):>6.3f} {'FWD' if cf >= cr else 'REV'}")

    print(f"\ntemp crops in {tmp}")


if __name__ == "__main__":
    main()

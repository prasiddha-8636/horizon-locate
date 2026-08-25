#!/usr/bin/env python
"""Regenerate GSV crops with the fixed slice_perspective geometry, re-segment,
and update the ground-truth VP mapping.

Backs up originals to data/street_view/backup_old/ first (unless --no-backup).
Writes:
  data/street_view/images/{sid}.png   — geometry-fixed crops
  data/street_view/masks/{sid}.png    — fresh U-Net masks
  data/street_view/ground_truth.json  — closest_viewpoint_id fixed to crop VP
  data/street_view/regenerate_report.json — per-sample verification
"""

import sys, os, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq

from src.streetview_utils import slice_perspective
from src.segmentation import load_segmentation_model, segment_image
from src.query_profile import extract_elevation_profile
from src.horizon_format import decode_horizon_uint8

GT_PATH = ROOT / "data/street_view/ground_truth.json"
CROP_PATH = ROOT / "data/street_view/crop_quality.json"
PANOS_DIR = ROOT / "data/street_view/panos"
IMAGES_DIR = ROOT / "data/street_view/images"
MASKS_DIR = ROOT / "data/street_view/masks"
DB_PATH = ROOT / "notebooks/02_SkylineDatabase/output/skyline_db.parquet"
MODEL_PATH = ROOT / "data/sky_segmentation_unet_model.pth"
FOV_Y = 65.0
OUT_W, OUT_H = 1080, 720


_DB_PF = None
_DB_CUM = None


def _get_db():
    global _DB_PF, _DB_CUM
    if _DB_PF is None:
        _DB_PF = pq.ParquetFile(str(DB_PATH))
        sizes = [
            _DB_PF.metadata.row_group(i).num_rows for i in range(_DB_PF.num_row_groups)
        ]
        _DB_CUM = np.concatenate([[0], np.cumsum(sizes)])
    return _DB_PF, _DB_CUM


def _fetch_horizon(vp_idx):
    pf, cum = _get_db()
    vp = int(vp_idx)
    rg = int(np.searchsorted(cum[1:], vp, side="right"))
    pos = vp - cum[rg]
    return decode_horizon_uint8(
        pf.read_row_group(rg, columns=["raw_horizon_deg"])
        .to_pandas()["raw_horizon_deg"]
        .iloc[pos]
    )


def _verify(sid, gt_info, heading, nearest_vp):
    """Extract profile from the fresh mask and compare vs DB horizon at the
    expected offset (heading + start_az)."""
    try:
        pr = extract_elevation_profile(
            str(MASKS_DIR / f"{sid}.png"),
            fov_y_deg=FOV_Y,
            r_tilt=np.array(gt_info["cam_R_tilt"]),
            bin_deg=1.0,
        )
    except Exception:
        return None
    if not pr["ok"]:
        return {"status": "PROFILE_FAIL", "reason": pr["status"]}
    prof = pr["profile"]
    hor = _fetch_horizon(nearest_vp)
    exp = int(round((heading + pr["start_az"]) % 360))
    w = hor[np.arange(exp, exp + len(prof)) % 360]
    cf = np.corrcoef(prof, w)[0, 1]
    cr = np.corrcoef(prof[::-1], w)[0, 1]
    return {
        "status": "OK",
        "corr_fwd": round(float(cf), 3),
        "corr_rev": round(float(cr), 3),
        "corr": round(float(max(cf, cr)), 3),
        "direction": "FWD" if cf >= cr else "REV",
        "profile_len": len(prof),
        "start_az": float(pr["start_az"]),
    }


def _process_one(args):
    sid, g, c, model = args
    out = {"sid": sid}
    try:
        img = slice_perspective(
            str(PANOS_DIR / f"{sid}.jpg"),
            heading_deg=c["best_heading"],
            pitch_deg=c["pitch_deg"],
            roll_deg=0.0,
            fov_y_deg=FOV_Y,
            out_w=OUT_W,
            out_h=OUT_H,
        )
        img.save(IMAGES_DIR / f"{sid}.png")
        seg = segment_image(
            model,
            str(IMAGES_DIR / f"{sid}.png"),
            str(MASKS_DIR / f"{sid}.png"),
            "cpu",
            tta=True,
        )
        out["seg_status"] = seg["status"]
        out["verify"] = _verify(sid, g, c["best_heading"], int(c["nearest_vp"]))
    except Exception as e:
        out["error"] = str(e)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    args = ap.parse_args()

    gt = json.loads(GT_PATH.read_text())

    if not args.limit:
        bak = ROOT / "data/street_view/backup_old"
        if not bak.exists():
            bak.mkdir(parents=True, exist_ok=True)
            for sub in ("images", "masks"):
                (bak / sub).mkdir(exist_ok=True)
                for p in (IMAGES_DIR.parent / sub).glob("*"):
                    p.rename(bak / sub / p.name)
            GT_PATH.rename(bak / "ground_truth.json")
            print(f"Backed up old data to {bak}", flush=True)

    crop = json.loads(CROP_PATH.read_text())
    sids = list(gt.keys())
    if args.limit:
        sids = sids[: args.limit]

    args_list = []
    for sid in sids:
        g = gt[sid]
        c = crop.get(sid, {})
        if "best_heading" not in c:
            continue
        args_list.append((sid, g, c))

    print(f"Processing {len(args_list)} samples single-process", flush=True)
    model = load_segmentation_model(str(MODEL_PATH), "cpu")
    t0 = time.time()
    results = []
    for i, (sid, g, c) in enumerate(args_list):
        out = _process_one((sid, g, c, model))
        results.append(out)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(args_list)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"Done in {time.time() - t0:.0f}s", flush=True)

    report = {r["sid"]: r for r in results}
    ok = [r for r in results if r.get("verify") and r["verify"].get("status") == "OK"]
    corrs = np.array([r["verify"]["corr"] for r in ok])
    fwd = np.mean([r["verify"]["direction"] == "FWD" for r in ok])
    print(
        f"seg OK: {sum(1 for r in results if r.get('seg_status') == 'OK')}/{len(results)}"
    )
    print(f"verified: {len(ok)}/{len(results)}")
    if len(corrs):
        print(
            f"corr: median={np.median(corrs):.3f} mean={np.mean(corrs):.3f} "
            f">0.7: {100 * np.mean(corrs > 0.7):.0f}% >0.5: {100 * np.mean(corrs > 0.5):.0f}%"
        )
        print(f"FWD fraction: {100 * fwd:.0f}%")

    # Fix GT VP mapping from crop_quality (the stored closest_viewpoint_id is stale)
    n_fixed = 0
    for sid in sids:
        if sid in crop:
            gt[sid]["closest_viewpoint_id"] = int(crop[sid]["nearest_vp"])
            gt[sid]["closest_viewpoint_dist_m"] = round(
                float(crop[sid]["nearest_dist_m"]), 3
            )
            n_fixed += 1
    (ROOT / "data/street_view/ground_truth.json").write_text(json.dumps(gt, indent=2))
    (ROOT / "data/street_view/regenerate_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"Updated GT VP mapping for {n_fixed} samples", flush=True)


if __name__ == "__main__":
    main()

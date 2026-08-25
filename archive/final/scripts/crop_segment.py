#!/usr/bin/env python
"""Crop all Street View panos to perspective views, then segment to masks.

Run locally. Resumable via /tmp/eval_study/crops_done.json and masks_done.json.
"""

import sys
import os
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.streetview_utils import slice_perspective

PANOS_DIR = ROOT / "data/street_view/panos"
IMAGES_DIR = ROOT / "data/street_view/images"
MASKS_DIR = ROOT / "data/street_view/masks"
GT_PATH = ROOT / "data/street_view/ground_truth.json"
CKPT_DIR = Path("/tmp/eval_study")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_gt():
    import json as _json

    with open(GT_PATH) as f:
        return _json.load(f)


def crop_all(gt):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "crops_done.json"
    done = set(json.loads(ckpt_path.read_text())) if ckpt_path.exists() else set()

    need = [sid for sid in gt if sid not in done]
    print(f"Crop: {len(gt)} total, {len(done)} done, {len(need)} need", flush=True)
    t0 = time.time()
    for i, sid in enumerate(need):
        v = gt[sid]
        pano_path = PANOS_DIR / f"{sid}.jpg"
        if not pano_path.exists():
            continue
        R = np.array(v["cam_R_tilt"])
        pitch = float(np.degrees(np.arcsin(np.clip(-R[2, 1], -1, 1))))
        roll = float(np.degrees(np.arctan2(R[2, 0], R[2, 2])))
        crop = slice_perspective(
            str(pano_path),
            heading_deg=v["true_heading_deg"],
            pitch_deg=pitch,
            roll_deg=roll,
            fov_y_deg=v["fov_y_deg"],
            out_w=1080,
            out_h=720,
        )
        crop.save(IMAGES_DIR / f"{sid}.png")
        done.add(sid)
        if (i + 1) % 200 == 0:
            ckpt_path.write_text(json.dumps(list(done)))
            print(f"  Crop {i + 1}/{len(need)}: {time.time() - t0:.0f}s", flush=True)
    ckpt_path.write_text(json.dumps(list(done)))
    print(f"Crop done: {len(done)}/{len(gt)} in {time.time() - t0:.0f}s", flush=True)


def segment_all():
    from src.segmentation import load_segmentation_model, segment_image
    import torch

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "masks_done.json"
    done = set(json.loads(ckpt_path.read_text())) if ckpt_path.exists() else set()

    images = sorted(IMAGES_DIR.glob("*.png"))
    need = [p for p in images if p.stem not in done]
    print(
        f"Segment: {len(images)} images, {len(done)} done, {len(need)} need", flush=True
    )
    if not need:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_segmentation_model(
        str(ROOT / "data/sky_segmentation_unet_model.pth"), device
    )
    print(f"Segmentation device: {device}", flush=True)

    t0 = time.time()
    for i, img_path in enumerate(need):
        mask_path = MASKS_DIR / img_path.name
        if not mask_path.exists():
            segment_image(model, str(img_path), str(mask_path), device)
        done.add(img_path.stem)
        if (i + 1) % 200 == 0:
            ckpt_path.write_text(json.dumps(list(done)))
            print(f"  Segment {i + 1}/{len(need)}: {time.time() - t0:.0f}s", flush=True)
    ckpt_path.write_text(json.dumps(list(done)))
    print(f"Segment done: {len(done)} masks in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    gt = load_gt()
    if mode in ("all", "crop"):
        crop_all(gt)
    if mode in ("all", "segment"):
        segment_all()

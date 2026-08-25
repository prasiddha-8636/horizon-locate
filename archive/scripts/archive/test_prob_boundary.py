#!/usr/bin/env python
"""Save U-Net probability maps and test threshold-based boundary extraction."""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from src.segmentation import load_segmentation_model
from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle
from src.horizon_format import decode_horizon_uint8

GT_PATH = ROOT / "data" / "street_view" / "ground_truth.json"
IMAGES_DIR = ROOT / "data" / "street_view" / "images"
MODEL_PATH = ROOT / "data" / "sky_segmentation_unet_model.pth"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"

PROB_DIR = ROOT / "data" / "street_view" / "prob_maps"
PROB_DIR.mkdir(parents=True, exist_ok=True)


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


def boundary_from_prob(prob, thresh=0.5):
    """Sky is prob > thresh (model outputs P(sky)). Boundary = topmost sky row."""
    H, W = prob.shape
    sky = (prob > thresh).astype(np.uint8)
    b = np.full(W, H - 1, dtype=np.float32)
    for c in range(W):
        rows = np.where(sky[:, c] == 1)[0]
        if len(rows) > 0:
            b[c] = rows[0]
    return b


def profile_from_boundary(b, H, W, fov_y, r_tilt, bin_deg=1.0):
    aspect = W / H
    hfov_deg = np.degrees(2 * np.arctan(np.tan(np.radians(fov_y) / 2) * aspect))
    fx = W / (2 * np.tan(np.radians(hfov_deg) / 2))
    fy = H / (2 * np.tan(np.radians(fov_y) / 2))
    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    rays = np.vstack([(cols - x_c) / fx, (y_c - b) / fy, -np.ones(W)])
    rays /= np.linalg.norm(rays, axis=0)
    azim_cam = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))
    if r_tilt is not None:
        rays = np.asarray(r_tilt) @ rays
    elev = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))
    order = np.argsort(azim_cam)
    azim_cam, elev = azim_cam[order], elev[order]
    sa = np.ceil(azim_cam[0] / bin_deg) * bin_deg
    ea = np.floor(azim_cam[-1] / bin_deg) * bin_deg
    grid = np.arange(sa, ea + 1e-6, bin_deg)
    return np.interp(grid, azim_cam, elev), sa


def fb_score(p, h):
    pv, pd = _feature_bundle(p)
    hv, hd = _feature_bundle(h)
    c0 = float(np.corrcoef(pv, hv)[0, 1]) if np.std(pv) > 0 and np.std(hv) > 0 else 0.0
    c1 = float(np.corrcoef(pd, hd)[0, 1]) if np.std(pd) > 0 and np.std(hd) > 0 else 0.0
    return 0.5 * (c0 + c1)


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    model = load_segmentation_model(str(MODEL_PATH), "cpu")
    import torch

    transform = __import__(
        "torchvision.transforms", fromlist=["Compose", "ToTensor", "Normalize"]
    ).Compose(
        [
            __import__(
                "torchvision.transforms", fromlist=["ToTensor", "Normalize"]
            ).ToTensor(),
            __import__(
                "torchvision.transforms", fromlist=["ToTensor", "Normalize"]
            ).Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    import torchvision.transforms as T
    from src.segmentation import _prepare_inference_image

    sids = list(gt.keys())[:50]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = {t: [] for t in thresholds}
    results["binary"] = []
    t0 = time.time()

    for sid in sids:
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        if vp < 0:
            continue
        img_path = IMAGES_DIR / f"{sid}.png"
        if not os.path.exists(img_path):
            continue

        # run U-Net to get probability map
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        padded, crop_info = _prepare_inference_image(img, input_size=256)
        tensor = transform(Image.fromarray(padded)).unsqueeze(0)
        with torch.no_grad():
            raw = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
        pad_l, pad_t, rw, rh = crop_info
        prob_small = raw[pad_t : pad_t + rh, pad_l : pad_l + rw]
        prob = (
            np.array(img.resize((W, H), Image.BILINEAR)).mean(axis=2) / 255.0
        )  # fallback: use grayscale
        # Use U-Net prob directly (it's 256x256, resize to 720x1080)
        from PIL import Image as PILImage

        prob_full = (
            np.array(
                PILImage.fromarray((prob_small * 255).astype(np.uint8)).resize(
                    (W, H), PILImage.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )

        hor = fetch_horizon(vp)
        r_tilt = np.array(g["cam_R_tilt"])
        fov = g["fov_y_deg"]

        # binary mask (current approach)
        from src.segmentation import segment_image
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_path = f.name
        seg = segment_image(model, img_path, mask_path, "cpu", tta=True)
        if seg["status"] != "OK":
            os.unlink(mask_path)
            continue

        pr = extract_elevation_profile(
            mask_path, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=1.0
        )
        os.unlink(mask_path)
        if not pr["ok"]:
            continue
        prof_orig = pr["profile"]
        sa = pr["start_az"]
        n = len(prof_orig)
        exp = int(round((g["true_heading_deg"] + sa) % 360))
        w = hor[np.arange(exp, exp + n) % 360]

        fb_orig = fb_score(prof_orig, w)
        results["binary"].append(fb_orig)

        for thresh in thresholds:
            b = boundary_from_prob(prob_full, thresh=thresh)
            prof_new, sa2 = profile_from_boundary(b, H, W, fov, r_tilt, 1.0)
            n2 = len(prof_new)
            exp2 = int(round((g["true_heading_deg"] + sa2) % 360))
            w2 = hor[np.arange(exp2, exp2 + n2) % 360]
            fb_new = fb_score(prof_new, w2)
            results[thresh].append(fb_new)

        if len(results["binary"]) % 10 == 0:
            print(
                f"  {len(results['binary'])} samples done ({time.time() - t0:.0f}s)",
                flush=True,
            )

    print(f"\nThreshold sweep ({len(results['binary'])} samples):")
    print(f"{'threshold':>10} {'median_fb':>10} {'mean_fb':>10} {'vs_binary':>10}")
    binary_m = np.median(results["binary"])
    for t in thresholds:
        v = np.array(results[t])
        delta = np.median(v) - binary_m
        print(f"{t:>10.1f} {np.median(v):>10.3f} {np.mean(v):>10.3f} {delta:>+10.3f}")
    print(
        f"{'binary':>10} {binary_m:>10.3f} {np.mean(results['binary']):>10.3f} {'---':>10}"
    )


if __name__ == "__main__":
    main()

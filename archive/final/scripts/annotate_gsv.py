#!/usr/bin/env python
"""Zero-Labor AI Skyline Server (Direct-Scribble Contour Engine).

Fixes:
  - Multi-Directory Image Resolver: Finds crops in both data/street_view/gsv_crops/ and images/
  - Direct-Scribble Path Replacement: Your hand-drawn scribble directly shapes 
    the skyline (snapping to nearby edges) and NEVER produces straight horizontal lines.
  - Storm-Cloud Resistant Feature Map: Uses CLAHE + Multi-Scale Sobel Texture 
    gradients that easily separate dark storm clouds from dark mountain shadows.
  - Smooth Boundary Blending: Seamlessly merges scribble updates with existing line.

Usage:
  python scripts/annotate_gsv.py [--port 8787] [--host 127.0.0.1] [--limit 1000] [--multi-only]
"""

import argparse
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
CROPS_DIR = os.path.join(ROOT, "data/street_view/gsv_crops")
MASKS_DIR = os.path.join(ROOT, "data/street_view/masks")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")

H, W = 720, 1080
HIST_CAP = 25

STATE = {
    "sids": [],
    "annotations": {},
    "skipped": {},
    "lines": {},
    "hist": {},
    "lock": threading.Lock(),
}


def find_image_path(sid):
    """Find image path in gsv_crops/ or images/ with any supported image extension."""
    for folder in [CROPS_DIR, IMAGES_DIR]:
        if not os.path.exists(folder):
            continue
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
            p = os.path.join(folder, sid + ext)
            if os.path.exists(p):
                return p
    return None


def safe_load_json(file_path, default_value):
    """Loads a JSON file safely, recovering automatically if empty or corrupted."""
    if not os.path.exists(file_path):
        return default_value
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default_value
            return json.loads(content)
    except Exception as e:
        print(f"Warning: Could not parse {file_path} ({e}). Starting fresh.")
        return default_value


def load_state(limit, multi_only=False):
    """Safely loads ground truth and annotations, preserving existing annotations."""
    gt = safe_load_json(GT_FILE, {})

    annot_data = safe_load_json(ANNOT_FILE, {"annotations": {}, "skipped": {}})
    STATE["annotations"] = annot_data.get("annotations", {})
    STATE["skipped"] = annot_data.get("skipped", {})

    all_sids = []

    # Scan gsv_crops directory
    if os.path.exists(CROPS_DIR):
        for f in sorted(os.listdir(CROPS_DIR)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                stem = Path(f).stem
                if stem not in all_sids:
                    all_sids.append(stem)

    # Scan images directory
    if os.path.exists(IMAGES_DIR):
        for f in sorted(os.listdir(IMAGES_DIR)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                stem = Path(f).stem
                if stem not in all_sids:
                    all_sids.append(stem)

    # Add ground truth keys if missing
    for k in gt.keys():
        if k not in all_sids:
            all_sids.append(k)

    # Group by pano_id
    pano_map = {}
    if os.path.exists(CROPS_DIR):
        for f in sorted(os.listdir(CROPS_DIR)):
            if f.endswith(".json") and not f.startswith("crop_") and not f.startswith("deleted_") and f != "annotations.json":
                meta = safe_load_json(os.path.join(CROPS_DIR, f), {})
                pid = meta.get("pano_id")
                stem = meta.get("filename", "").replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
                if pid and stem:
                    pano_map.setdefault(pid, []).append(stem)

    if multi_only and pano_map:
        multi_sids = []
        for pid, stems in pano_map.items():
            if len(stems) >= 2:
                for s in stems:
                    if s in all_sids and s not in multi_sids:
                        multi_sids.append(s)
        sids = multi_sids if multi_sids else all_sids
    else:
        sids = all_sids

    # Unannotated sids first so dashboard queue starts with fresh crops
    unannotated = [s for s in sids if s not in STATE["annotations"] and s not in STATE["skipped"]]
    annotated = [s for s in sids if s in STATE["annotations"] or s in STATE["skipped"]]
    final_sids = unannotated + annotated

    if limit and limit < len(final_sids):
        final_sids = final_sids[:limit]

    STATE["sids"] = final_sids


def save_state():
    """Atomic save: writes to temp file first to prevent JSON corruption."""
    os.makedirs(os.path.dirname(ANNOT_FILE), exist_ok=True)
    tmp_file = ANNOT_FILE + ".tmp"
    payload = {"annotations": STATE["annotations"], "skipped": STATE["skipped"]}
    with open(tmp_file, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp_file, ANNOT_FILE)


def dark_channel_prior_dehaze(img_rgb, omega=0.80, patch_size=15):
    """He et al. Dark Channel Prior Dehazing to sharpen distant hazy peaks."""
    img_float = img_rgb.astype(np.float64) / 255.0
    min_channel = np.min(img_float, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    flat_dark = dark_channel.ravel()
    flat_img = img_float.reshape(-1, 3)
    num_pixels = len(flat_dark)
    top_num = max(1, int(num_pixels * 0.001))
    indices = np.argpartition(flat_dark, -top_num)[-top_num:]
    A = np.mean(flat_img[indices], axis=0)
    A = np.maximum(A, 0.1)

    normalized = img_float / A
    min_norm = np.min(normalized, axis=2)
    dark_norm = cv2.erode(min_norm, kernel)
    transmission = 1.0 - omega * dark_norm
    transmission = np.clip(transmission, 0.1, 1.0)

    dehazed = (img_float - A) / transmission[:, :, None] + A
    dehazed = np.clip(dehazed * 255.0, 0, 255).astype(np.uint8)
    return dehazed


def compute_enhanced_skyline_feature_map(img_rgb):
    """Robust Texture & Gradient Feature Map resilient against storm clouds."""
    H_sub, W_sub, _ = img_rgb.shape

    dehazed = dark_channel_prior_dehaze(img_rgb, omega=0.80)
    smoothed = cv2.bilateralFilter(dehazed, d=5, sigmaColor=35, sigmaSpace=35)

    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(gray).astype(np.float64)
    b_clahe = clahe.apply(lab[:, :, 2]).astype(np.float64)

    # Color & Luminance vertical gradients
    db = np.maximum(0, b_clahe[2:, :] - b_clahe[:-2, :])
    dl = np.maximum(0, l_clahe[:-2, :] - l_clahe[2:, :])

    canny = cv2.Canny(smoothed, 20, 80).astype(np.float64) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g_mag = np.sqrt(gx**2 + gy**2)
    g_norm = g_mag / (g_mag.max() + 1e-6)

    pad_db = np.zeros((H_sub, W_sub), dtype=np.float64)
    pad_db[1:-1, :] = db

    pad_dl = np.zeros((H_sub, W_sub), dtype=np.float64)
    pad_dl[1:-1, :] = dl

    raw_signal = (pad_db * 2.0 + pad_dl * 1.0) * (1.0 + 2.0 * canny) + 1.5 * g_norm

    # TOP-DOWN FIRST-EDGE SUPPRESSION
    cum_signal = np.cumsum(raw_signal, axis=0)
    prev_cum = cum_signal - raw_signal
    top_first_suppression = np.exp(-1.5 * np.maximum(0.0, prev_cum / (raw_signal.max() + 1e-6)))

    topmost_map = raw_signal * top_first_suppression
    if topmost_map.max() > 0:
        topmost_map /= topmost_map.max()

    return topmost_map, dehazed, b_clahe


def extract_pure_image_baseline_skyline(img_rgb):
    """Automatic Base Skyline Extractor supporting steep vertical cliffs."""
    H_img, W_img, _ = img_rgb.shape
    feature_map, _, _ = compute_enhanced_skyline_feature_map(img_rgb)

    cost = 1.0 - feature_map

    dp = np.full((H_img, W_img), 1e7, dtype=np.float32)
    backtrack = np.zeros((H_img, W_img), dtype=np.int32)
    dp[:, 0] = cost[:, 0]

    max_step = 25
    steps = np.arange(-max_step, max_step + 1)
    step_costs = 0.02 * (np.abs(steps) ** 1.4)

    for c in range(1, W_img):
        for dr, step_cost in zip(steps, step_costs):
            if dr >= 0:
                prev = dp[: H_img - dr, c - 1] + step_cost
                curr_target = cost[dr:, c] + prev
                mask = curr_target < dp[dr:, c]
                dp[dr:, c] = np.where(mask, curr_target, dp[dr:, c])
                backtrack[dr:, c] = np.where(mask, np.arange(H_img - dr), backtrack[dr:, c])
            else:
                abs_dr = -dr
                prev = dp[abs_dr:, c - 1] + step_cost
                curr_target = cost[: H_img - abs_dr, c] + prev
                mask = curr_target < dp[: H_img - abs_dr, c]
                dp[: H_img - abs_dr, c] = np.where(mask, curr_target, dp[: H_img - abs_dr, c])
                backtrack[: H_img - abs_dr, c] = np.where(
                    mask, np.arange(abs_dr, H_img), backtrack[: H_img - abs_dr, c]
                )

    best_last_r = np.argmin(dp[:, W_img - 1])
    base_line = np.zeros(W_img, dtype=np.float32)
    base_line[-1] = best_last_r
    for c in range(W_img - 1, 0, -1):
        base_line[c - 1] = backtrack[int(base_line[c]), c]

    base_line_smooth = gaussian_filter1d(base_line, sigma=1.0)
    return np.clip(np.round(base_line_smooth), 0, H_img - 1).astype(np.int32)


def load_line(sid):
    p_img = find_image_path(sid)
    if p_img and os.path.exists(p_img):
        try:
            img_rgb = np.array(Image.open(p_img).convert("RGB"))
            return extract_pure_image_baseline_skyline(img_rgb)
        except Exception as e:
            print(f"Warning: Could not extract image base skyline for {sid}: {e}")

    p_mask = os.path.join(MASKS_DIR, sid + ".png")
    line = np.full(W, H - 1, dtype=np.int32)
    if os.path.exists(p_mask):
        try:
            m = np.array(Image.open(p_mask).convert("L"))
            if m[:10, :].mean() > m[-10:, :].mean():
                m = 255 - m
            mask = (m >= 128).astype(np.uint8) * 255
            for c in range(W):
                r = np.where(mask[:, c] == 255)[0]
                if len(r):
                    line[c] = r[0]
        except Exception as e:
            print(f"Warning: Could not load mask for {sid}: {e}")

    return line


def line_for(sid):
    if sid not in STATE["lines"]:
        STATE["lines"][sid] = load_line(sid)
    return STATE["lines"][sid]


def push_hist(sid):
    hist = STATE["hist"].setdefault(sid, [])
    hist.append(line_for(sid).copy())
    if len(hist) > HIST_CAP:
        del hist[0]


def pop_hist(sid):
    hist = STATE["hist"].get(sid, [])
    if hist:
        restored_line = hist.pop()
        set_line_and_sync_mask(sid, restored_line)
        return True
    return False

def magic_scribble_snap(img_rgb, current_line, points, radius=30):
    """Direct-Scribble Contour Engine: Snaps locally to physical edges along your drawn path."""
    H_img, W_img, _ = img_rgb.shape
    if not points or len(points) < 2:
        return current_line.copy()

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Sort stroke by x coordinate
    sort_idx = np.argsort(xs)
    xs = xs[sort_idx]
    ys = ys[sort_idx]

    c_min = max(0, int(np.min(xs)))
    c_max = min(W_img - 1, int(np.max(xs)))

    if c_max <= c_min + 2:
        return current_line.copy()

    # Interpolate target scribble curve for all columns in stroke range
    scribble_cols = np.arange(c_min, c_max + 1)
    scribble_y = np.interp(scribble_cols, xs, ys)

    # Compute robust feature map
    feature_map, _, _ = compute_enhanced_skyline_feature_map(img_rgb)

    # Local edge-snapping along the user's stroke (Wide search band for easy snapping)
    snapped_y = scribble_y.copy()
    search_radius = max(45, int(radius * 1.5))

    for idx, col in enumerate(scribble_cols):
        target_y = int(scribble_y[idx])
        y_start = max(0, target_y - search_radius)
        y_end = min(H_img - 1, target_y + search_radius)

        edge_strip = feature_map[y_start : y_end + 1, col]
        if len(edge_strip) > 0 and edge_strip.max() > 0.08:
            local_rows = np.arange(y_start, y_end + 1)
            dist_penalty = 0.001 * ((local_rows - target_y) ** 2)
            score = edge_strip - dist_penalty
            best_local_y = local_rows[np.argmax(score)]
            snapped_y[idx] = best_local_y

    # Smooth the snapped stroke
    snapped_y_smooth = gaussian_filter1d(snapped_y, sigma=1.2)

    # Replace line across scribble span
    updated_line = current_line.copy()
    updated_line[c_min : c_max + 1] = np.clip(
        np.round(snapped_y_smooth), 0, H_img - 1
    ).astype(np.int32)

    # Smooth full path transition
    updated_line = gaussian_filter1d(updated_line.astype(np.float64), sigma=0.8)
    return np.clip(np.round(updated_line), 0, H_img - 1).astype(np.int32)

STATE["masks"] = {}

def mask_for(sid):
    """Retrieve 2D binary mask array (0=sky, 255=terrain)."""
    if sid not in STATE["masks"]:
        line = line_for(sid)
        rr = np.arange(H)[:, None]
        sky = rr < line[None, :]
        STATE["masks"][sid] = np.where(sky, 0, 255).astype(np.uint8)
    return STATE["masks"][sid]

def mask_to_line(mask2d):
    """Extract skyline boundary line from 2D mask."""
    line = np.zeros(W, dtype=np.int32)
    for c in range(W):
        sky_rows = np.where(mask2d[:, c] == 0)[0]
        line[c] = sky_rows[-1] if len(sky_rows) > 0 else 0
    return line

def set_line_and_sync_mask(sid, new_line):
    """Update line and regenerate 2D mask overlay to stay in 100% sync."""
    STATE["lines"][sid] = new_line
    rr = np.arange(H)[:, None]
    sky = rr < new_line[None, :]
    STATE["masks"][sid] = np.where(sky, 0, 255).astype(np.uint8)

def set_mask_and_sync_line(sid, new_mask2d):
    """Update 2D mask and extract skyline line to stay in 100% sync."""
    STATE["masks"][sid] = new_mask2d
    STATE["lines"][sid] = mask_to_line(new_mask2d)

def apply_2d_manual_brush(mask2d, points, tool, radius):
    """Paint 2D filled circle directly onto the image mask."""
    if not points:
        return mask2d.copy()

    H_img, W_img = mask2d.shape
    updated_mask = mask2d.copy()
    fill_val = 0 if tool == "sky_brush" else 255

    for px, py in points:
        cx = int(round(px))
        cy = int(round(py))
        R = max(3, int(radius))

        y_min = max(0, cy - R)
        y_max = min(H_img - 1, cy + R)
        x_min = max(0, cx - R)
        x_max = min(W_img - 1, cx + R)

        if x_max < x_min or y_max < y_min:
            continue

        yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        circle_mask = dist_sq <= R**2

        updated_mask[y_min : y_max + 1, x_min : x_max + 1][circle_mask] = fill_val

    return updated_mask

# def apply_point_constraint(img_rgb, current_line, click_x, click_y, label, radius=35):
#     """Re-evaluate skyline locally around click_x, forcing click_y as sky or terrain."""
#     H_img, W_img, _ = img_rgb.shape
#     click_x = int(np.clip(click_x, 0, W_img - 1))
#     click_y = int(np.clip(click_y, 0, H_img - 1))

#     r_col = max(15, int(radius))
#     c_min = max(0, click_x - r_col)
#     c_max = min(W_img - 1, click_x + r_col)

#     feature_map, _, _ = compute_enhanced_skyline_feature_map(img_rgb)

#     local_cols = np.arange(c_min, c_max + 1)
#     local_target = current_line[c_min : c_max + 1].copy().astype(np.float64)

#     target_y_at_click = max(click_y, current_line[click_x]) if label == "sky" else min(click_y, current_line[click_x])

#     for idx, col in enumerate(local_cols):
#         dist = abs(col - click_x)
#         w = max(0.0, 1.0 - (dist / float(r_col)))
#         local_target[idx] = (1.0 - w) * local_target[idx] + w * target_y_at_click

#     updated_local = local_target.copy()
#     for idx, col in enumerate(local_cols):
#         ty = int(local_target[idx])
#         y0 = max(0, ty - 12)
#         y1 = min(H_img - 1, ty + 12)
#         strip = feature_map[y0 : y1 + 1, col]
#         if len(strip) > 0 and strip.max() > 0.10:
#             best_y = y0 + np.argmax(strip)
#             updated_local[idx] = best_y

#     if label == "sky":
#         updated_local[click_x - c_min] = max(updated_local[click_x - c_min], click_y)
#     else:
#         updated_local[click_x - c_min] = min(updated_local[click_x - c_min], click_y)

#     snapped_smooth = gaussian_filter1d(updated_local, sigma=1.0)
#     updated_line = current_line.copy()
#     updated_line[c_min : c_max + 1] = np.clip(np.round(snapped_smooth), 0, H_img - 1).astype(np.int32)
#     return updated_line

def apply_manual_paint_brush(current_line, points, tool, radius):
    """Pure manual blob painting (zero edge-snapping, zero smoothing)."""
    if not points:
        return current_line.copy()

    H_img, W_img = H, W
    updated_line = current_line.copy()

    for px, py in points:
        cx = int(np.clip(px, 0, W_img - 1))
        cy = int(np.clip(py, 0, H_img - 1))
        R = max(3, int(radius))

        x_min = max(0, cx - R)
        x_max = min(W_img - 1, cx + R)

        for x in range(x_min, x_max + 1):
            dy = int(np.sqrt(max(0, R**2 - (x - cx)**2)))
            if tool == "sky_brush":
                target_y = int(np.clip(cy + dy, 0, H_img - 1))
                updated_line[x] = max(updated_line[x], target_y)
            else:  # terrain_brush
                target_y = int(np.clip(cy - dy, 0, H_img - 1))
                updated_line[x] = min(updated_line[x], target_y)

    return updated_line

# def apply_scribble_brush(sid, points, tool, radius):
#     """Applies Magic Scribble Guide."""
#     push_hist(sid)
#     line = line_for(sid)

#     img_path = find_image_path(sid)
#     if img_path and os.path.exists(img_path):
#         img_rgb = np.array(Image.open(img_path).convert("RGB"))
#         STATE["lines"][sid] = magic_scribble_snap(
#             img_rgb,
#             line,
#             points=points,
#             radius=radius,
#         )

def apply_scribble_brush(sid, points, tool, radius):
    """Applies Scribble Guide, Sky Point, or Terrain Point constraint."""
    push_hist(sid)
    line = line_for(sid)

    img_path = os.path.join(IMAGES_DIR, sid + ".png")
    crops_dir = os.path.join(ROOT, "data/street_view/gsv_crops")
    if not os.path.exists(img_path) and os.path.exists(crops_dir):
        p_crop = os.path.join(crops_dir, sid + ".png")
        if os.path.exists(p_crop):
            img_path = p_crop

    if os.path.exists(img_path):
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask2d = mask_for(sid)
        if tool in ("sky_brush", "terrain_brush"):
            updated_mask = apply_2d_manual_brush(
                mask2d, points=points, tool=tool, radius=radius
            )
            set_mask_and_sync_line(sid, updated_mask)
        else:
            new_line = magic_scribble_snap(
                img_rgb, line, points=points, radius=radius
            )
            set_line_and_sync_mask(sid, new_line)
            
def line_to_points(line, step=1):
    return [[int(c), int(line[c])] for c in range(0, W, step)]


def render_view_mode_png(sid, view_mode, line):
    """Serves requested inspection layer: dehazed photo, LAB b*, CLAHE edge, or overlay tint."""
    img_path = find_image_path(sid)
    if not img_path or not os.path.exists(img_path):
        return ""

    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    if view_mode == "original":
        return ""

    if view_mode == "dehazed":
        dehazed = dark_channel_prior_dehaze(img_rgb, omega=0.80)
        buf = io.BytesIO()
        Image.fromarray(dehazed).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    elif view_mode == "lab":
        _, _, b_clahe = compute_enhanced_skyline_feature_map(img_rgb)
        b_vis = cv2.applyColorMap(b_clahe.astype(np.uint8), cv2.COLORMAP_JET)
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(b_vis, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    elif view_mode == "clahe":
        feature_map, _, _ = compute_enhanced_skyline_feature_map(img_rgb)
        feat_vis = (feature_map * 255.0).astype(np.uint8)
        feat_vis = cv2.applyColorMap(feat_vis, cv2.COLORMAP_MAGMA)
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(feat_vis, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # Default mask overlay tint
    mask2d = mask_for(sid)
    sky = mask2d == 0

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[sky, 0:3] = (30, 110, 255)    # Blue Sky
    rgba[sky, 3] = 85
    rgba[~sky, 0:3] = (255, 140, 20)  # Orange Terrain
    rgba[~sky, 3] = 60

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def sample_payload(sid, view_mode="tint"):
    done = len(STATE["annotations"]) + len(STATE["skipped"])
    total = len(STATE["sids"])
    idx = next((i for i, s in enumerate(STATE["sids"]) if s == sid), 0) + 1
    line = line_for(sid)
    return {
        "sid": sid,
        "img": "/api/img/" + sid + ".png",
        "points": line_to_points(line, step=1),
        "tint": render_view_mode_png(sid, view_mode, line),
        "total": total,
        "done": done,
        "idx": idx,
    }


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Direct-Scribble AI Skyline Refinement</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 14px; background: #111; color: #eee; max-width: 1100px; }
  #status { font-size: 18px; font-weight: 600; margin-bottom: 2px; }
  #sub { font-size: 12px; color: #888; margin-bottom: 8px; }
  #barwrap { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  #bar { flex: 1; height: 8px; background: #222; border-radius: 4px; overflow: hidden; }
  #barfill { height: 100%; background: #38bdf8; width: 0%; transition: width .2s; }
  #barprog { font-size: 13px; color: #aaa; white-space: nowrap; }
  #toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
  .btn { padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer;
         background: #222; color: #ddd; border: 1px solid #444; border-radius: 6px; }
  .btn.active { border-color: #38bdf8; background: #0284c7; color: #fff; }
  .btn.primary { background: #16a34a; border-color: #16a34a; color: #fff; }
  .view-btn.active { border-color: #f59e0b; background: #b45309; color: #fff; }
  #tip { font-size: 13px; color: #7dd3fc; margin-bottom: 8px; min-height: 18px; }
  #imgwrap { position: relative; display: inline-block; line-height: 0; user-select: none; }
  #img { display: block; max-width: 1080px; height: auto; border-radius: 4px; }
  #tint { position: absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  svg { position: absolute; top:0; left:0; width:100%; height:100%; touch-action: none; }
</style></head>
<body>
  <div id="status">Loading AI Engine…</div>
  <div id="sub"></div>
  <div id="barwrap"><div id="bar"><div id="barfill"></div></div><span id="barprog"></span></div>
  
<div id="toolbar">
    <button class="btn active" id="btn_scribble">Scribble</button>
    <button class="btn" id="btn_sky">Sky 🟦</button>
    <button class="btn" id="btn_terrain">Terrain 🟥</button>
    <span style="color:#555">|</span>
    <span style="font-size:12px; color:#aaa">Brush:</span>
    <input type="range" id="brush_size" min="15" max="70" value="30">
    <span style="color:#555">|</span>
    <br/>
    <span style="font-size:12px; color:#aaa">View:</span>
    <button class="btn view-btn" id="view_original">Original Photo</button>
    <button class="btn view-btn active" id="view_tint">Mask Overlay</button>
    <button class="btn view-btn" id="view_dehazed">Dehazed</button>
    <button class="btn view-btn" id="view_lab">Color Map</button>
    <button class="btn view-btn" id="view_clahe">Edges</button>
    <span style="flex:1"></span>
    <button class="btn" id="btn_undo">Undo (u)</button>
    <button class="btn" id="btn_reset">Reset (c)</button>
    <button class="btn primary" id="btn_save">Save + Next (s)</button>
    <button class="btn" id="btn_skip">Skip (n)</button>
  </div>
  
  <div id="tip">Draw a stroke across any mountain section to snap the skyline.</div>  

  <div id="imgwrap">
    <img id="img">
    <img id="tint">
    <canvas id="live_canvas" width="1080" height="720" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"></canvas>
    <svg id="svg"></svg>
  </div>

<script>
const img = document.getElementById('img'), svg = document.getElementById('svg'), tint = document.getElementById('tint');
let cur = null, tool = 'scribble', viewMode = 'tint', pts = [], strokePts = [], busy = false, dragging = false;

let cursorPos = [540, 360];

function setTool(t) {
  tool = t;
  document.getElementById('btn_scribble').className = 'btn' + (t === 'scribble' ? ' active' : '');
  document.getElementById('btn_sky').className = 'btn' + (t === 'sky_brush' ? ' active' : '');
  document.getElementById('btn_terrain').className = 'btn' + (t === 'terrain_brush' ? ' active' : '');
  render();
}

document.getElementById('btn_scribble').onclick = () => setTool('scribble');
document.getElementById('btn_sky').onclick = () => setTool('sky_brush');
document.getElementById('btn_terrain').onclick = () => setTool('terrain_brush');

function pos(evt) {
  const r = img.getBoundingClientRect();
  return [Math.max(0, Math.min(1080, (evt.clientX - r.left) / r.width * 1080)),
          Math.max(0, Math.min(720, (evt.clientY - r.top) / r.height * 720))];
}

function render() {
  svg.innerHTML = '';

  // 1. Current red skyline path
  if (pts.length > 1) {
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    poly.setAttribute('points', pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' '));
    poly.setAttribute('fill', 'none');
    poly.setAttribute('stroke', '#ff2222');
    poly.setAttribute('stroke-width', '2.5');
    svg.appendChild(poly);
  }

  // 2. Active dashed blue scribble stroke line
  if (tool === 'scribble' && dragging && strokePts.length > 1) {
    const stroke = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    stroke.setAttribute('points', strokePts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' '));
    stroke.setAttribute('fill', 'none');
    stroke.setAttribute('stroke', '#38bdf8');
    stroke.setAttribute('stroke-width', '3.5');
    stroke.setAttribute('stroke-dasharray', '6 3');
    svg.appendChild(stroke);
  }

  // 3. Circular brush cursor for Sky / Terrain brushes
  if (tool === 'sky_brush' || tool === 'terrain_brush') {
    const r = parseInt(document.getElementById('brush_size').value);
    const circ = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circ.setAttribute('cx', cursorPos[0]);
    circ.setAttribute('cy', cursorPos[1]);
    circ.setAttribute('r', r);
    circ.setAttribute('fill', tool === 'sky_brush' ? 'rgba(30, 110, 255, 0.35)' : 'rgba(239, 68, 68, 0.35)');
    circ.setAttribute('stroke', tool === 'sky_brush' ? '#38bdf8' : '#ef4444');
    circ.setAttribute('stroke-width', '1.5');
    svg.appendChild(circ);
  }
}

const liveCanvas = document.getElementById('live_canvas');
const ctx = liveCanvas.getContext('2d');

function drawLiveBrush(p) {
  if (tool !== 'sky_brush' && tool !== 'terrain_brush') return;
  const r = parseInt(document.getElementById('brush_size').value);
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, 2 * Math.PI);
  ctx.fillStyle = tool === 'sky_brush' ? 'rgba(30, 110, 255, 0.45)' : 'rgba(255, 140, 20, 0.45)';
  ctx.fill();
}

svg.addEventListener('pointerdown', evt => {
  svg.setPointerCapture(evt.pointerId);
  dragging = true;
  const p = pos(evt);
  strokePts = [p];
  if (tool === 'sky_brush' || tool === 'terrain_brush') {
    drawLiveBrush(p);
  }
  render();
});

svg.addEventListener('pointermove', evt => {
  cursorPos = pos(evt);
  if (dragging) {
    strokePts.push(cursorPos);
    if (tool === 'sky_brush' || tool === 'terrain_brush') {
      drawLiveBrush(cursorPos);
    }
  }
  render();
});

svg.addEventListener('pointerup', evt => {
  if (dragging) {
    dragging = false;
    ctx.clearRect(0, 0, 1080, 720); // Clear live canvas overlay after sync
    const radius = parseInt(document.getElementById('brush_size').value);
    sendAction('scribble_brush', {points: strokePts, tool: tool, radius: radius});
    strokePts = [];
  }
});

function setViewMode(v) {
  viewMode = v;
  ['tint', 'original', 'dehazed', 'lab', 'clahe'].forEach(m => {
    const btn = document.getElementById('view_' + m);
    if (btn) btn.className = 'btn view-btn' + (v === m ? ' active' : '');
  });
  if (cur) sendAction('view_mode', {view_mode: viewMode});
}

function sendAction(endpoint, body) {
  if (busy || !cur) return;
  busy = true;
  fetch('/api/' + endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({sid: cur.sid, view_mode: viewMode}, body))
  }).then(r => r.json()).then(apply);
}

// (duplicate handlers removed — first set at line 741 handles all tools)

window.addEventListener('keydown', evt => {
  if (evt.key === 's') sendAction('annotate', {});
  else if (evt.key === 'n') sendAction('skip', {});
  else if (evt.key === 'u') sendAction('undo', {});
  else if (evt.key === 'c') sendAction('reset', {});
});

document.getElementById('view_tint').onclick = () => setViewMode('tint');
document.getElementById('view_original').onclick = () => setViewMode('original');
document.getElementById('view_dehazed').onclick = () => setViewMode('dehazed');
document.getElementById('view_lab').onclick = () => setViewMode('lab');
document.getElementById('view_clahe').onclick = () => setViewMode('clahe');
document.getElementById('btn_undo').onclick = () => sendAction('undo', {});
document.getElementById('btn_reset').onclick = () => sendAction('reset', {});
document.getElementById('btn_save').onclick = () => sendAction('annotate', {});
document.getElementById('btn_skip').onclick = () => sendAction('skip', {});

function apply(d) {
  if (d.error) { busy = false; document.getElementById('status').textContent = d.error; return; }
  cur = d;
  pts = d.points || [];
  img.src = d.img;
  tint.src = 'data:image/png;base64,' + d.tint;
  document.getElementById('status').textContent = 'Sample ' + d.idx + ' / ' + d.total;
  document.getElementById('sub').textContent = d.sid;
  document.getElementById('barfill').style.width = ((d.done / d.total) * 100) + '%';
  document.getElementById('barprog').textContent = d.done + '/' + d.total + ' processed';
  render();
  busy = false;
}

fetch('/api/next').then(r => r.json()).then(apply);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, HTML.encode(), "text/html")
        elif u.path.startswith("/api/img/"):
            raw_filename = u.path[len("/api/img/") :]
            sid = os.path.splitext(raw_filename)[0]
            img_p = find_image_path(sid)
            if img_p and os.path.exists(img_p):
                self._send(200, open(img_p, "rb").read(), "image/png")
            else:
                self._send(404, b"not found")
        elif u.path == "/api/next":
            sid = next(
                (
                    s
                    for s in STATE["sids"]
                    if s not in STATE["annotations"] and s not in STATE["skipped"]
                ),
                None,
            )
            if sid is None:
                self._send(200, json.dumps({"error": "All done!"}).encode())
            else:
                with STATE["lock"]:
                    self._send(200, json.dumps(sample_payload(sid)).encode())

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        sid = body.get("sid")
        vm = body.get("view_mode", "tint")

        with STATE["lock"]:
            if u.path == "/api/scribble_brush":
                apply_scribble_brush(
                    sid,
                    body.get("points", []),
                    body.get("tool", "scribble"),
                    int(body.get("radius", 30)),
                )
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/view_mode":
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/undo":
                pop_hist(sid)
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/reset":
                init_line = load_line(sid)
                set_line_and_sync_mask(sid, init_line)
                STATE["hist"].pop(sid, None)
                self._send(200, json.dumps(sample_payload(sid, view_mode=vm)).encode())
            elif u.path == "/api/annotate":
                line = line_for(sid)
                STATE["annotations"][sid] = line_to_points(line, step=1)
                save_state()
                sid_next = next(
                    (
                        s
                        for s in STATE["sids"]
                        if s not in STATE["annotations"]
                        and s not in STATE["skipped"]
                    ),
                    None,
                )
                if sid_next:
                    self._send(
                        200, json.dumps(sample_payload(sid_next, view_mode=vm)).encode()
                    )
                else:
                    self._send(200, json.dumps({"error": "All done!"}).encode())
            elif u.path == "/api/skip":
                STATE["skipped"][sid] = True
                save_state()
                sid_next = next(
                    (
                        s
                        for s in STATE["sids"]
                        if s not in STATE["annotations"]
                        and s not in STATE["skipped"]
                    ),
                    None,
                )
                if sid_next:
                    self._send(
                        200, json.dumps(sample_payload(sid_next, view_mode=vm)).encode()
                    )
                else:
                    self._send(200, json.dumps({"error": "All done!"}).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--multi-only", action="store_true", help="Filter panos with multiple crops")
    args = ap.parse_args()
    load_state(args.limit, multi_only=args.multi_only)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Direct-Scribble AI Server running at http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()

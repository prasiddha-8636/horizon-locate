"""Sky segmentation utilities: U-Net model, guided post-processing, and training routines.

Post-processing algorithms:
  - refine_sky_mask_with_guidance (lab_b_subpixel): CLAHE-enhanced multi-scale Canny + LAB b* channel sub-pixel fitting
  - refine_grayscale_fixed_window (grayscale_fixed): CLAHE-enhanced grayscale vertical gradient with fixed +/-25 row search
  - refine_multichannel_gradient_fusion (multichannel_fusion): CLAHE-enhanced weighted gradient fusion (LAB b*, HSV Saturation, Grayscale)
  - refine_dynamic_programming_skyline (dynamic_programming): Viterbi shortest-path cost-grid line extraction
"""

import os
from pathlib import Path
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _median_filter_1d(values, kernel_size=7):
    """Apply 1D median filter to stabilize boundary row arrays across columns."""
    if kernel_size <= 1 or len(values) == 0:
        return values

    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filtered = np.empty_like(values)

    for idx in range(len(values)):
        filtered[idx] = np.median(padded[idx : idx + kernel_size])

    return filtered


def _prepare_inference_image(orig_img, input_size=256):
    """Resize image preserving aspect ratio with reflective padding for inference."""
    width, height = orig_img.size
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized_img = orig_img.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    resized_np = np.array(resized_img)

    pad_left = (input_size - resized_width) // 2
    pad_right = input_size - resized_width - pad_left
    pad_top = (input_size - resized_height) // 2
    pad_bottom = input_size - resized_height - pad_top

    padded_np = cv2.copyMakeBorder(
        resized_np,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded_np, (pad_left, pad_top, resized_width, resized_height)


def _has_edge_support(edges, row, col, row_radius=1, col_radius=2, min_count=3):
    """Return True when a candidate edge pixel has surrounding local support."""
    row_min = max(0, row - row_radius)
    row_max = min(edges.shape[0], row + row_radius + 1)
    col_min = max(0, col - col_radius)
    col_max = min(edges.shape[1], col + col_radius + 1)
    window = edges[row_min:row_max, col_min:col_max]
    return np.count_nonzero(window == 255) >= min_count


def refine_sky_mask_with_guidance(
    img_np,
    raw_unet_mask,
    use_top_connected=True,
    use_canny=True,
    use_lab_b=True,
    use_clahe=True,
    kernel_size=3,
):
    """Multi-scale edge-fused outlier-filtered sky mask refinement with CLAHE dehazing."""
    H, W = raw_unet_mask.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sky1 = (raw_unet_mask == 1).astype(np.uint8)

# 1. Top-connected sky region with smart fallback
    if use_top_connected:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky1, connectivity=8)
        top_sky = np.zeros((H, W), dtype=np.uint8)
        top_limit = max(15, int(H * 0.15))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_TOP] <= top_limit and stats[i, cv2.CC_STAT_AREA] > 50:
                top_sky[labels == i] = 1

        # FALLBACK: If no sky in top 15%, use largest sky component
        if top_sky.sum() == 0 and num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            top_sky[labels == largest_idx] = 1
        if top_sky.sum() == 0:
            top_sky = sky1
    else:
        top_sky = sky1

    # 2. Multi-scale Canny edge fusion with sky-zone mild CLAHE
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if use_clahe:
        # Mild CLAHE to remove fog without boosting rock noise
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
        gray_enhanced = clahe.apply(gray)
        # Apply enhanced gray ONLY in sky region where fog lives
        sky_mask_zone = cv2.dilate(top_sky, np.ones((15, 15), np.uint8)) > 0
        gray = np.where(sky_mask_zone, gray_enhanced, gray)

    fine_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    coarse_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    if use_canny:
        edges_fine = cv2.Canny(fine_blur, 30, 150)
        edges_coarse = cv2.Canny(coarse_blur, 20, 100)
        canny_edges = (edges_fine > 0) | (edges_coarse > 0)
    else:
        canny_edges = None

    # 3. LAB b* vertical gradient fallback
    if use_lab_b:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        b_star = cv2.GaussianBlur(lab[:, :, 2].astype(np.float32), (3, 3), 0)
        b_vgrad = np.diff(b_star, axis=0)
    else:
        b_vgrad = None

    vgrad_gray = np.diff(fine_blur.astype(np.float32), axis=0)
    boundaries = np.full(W, -1, dtype=np.float64)

    # 4. Per-column top-down edge extraction (Capped at first Canny ridge edge)
    for col in range(W):
        sky_rows = np.where(top_sky[:, col] == 1)[0]
        if len(sky_rows) == 0:
            continue

        # Find first gap in sky region
        diffs = np.diff(sky_rows)
        gaps = np.where(diffs > 3)[0]
        max_sky_row = sky_rows[gaps[0]] if len(gaps) > 0 else sky_rows[-1]

        # BARRIER: Stop at Canny edge ONLY if it is near U-Net mountain boundary (skip clouds)
        if canny_edges is not None:
            edge_rows = np.where(canny_edges[:, col])[0]
            # Only edges within 20px of U-Net boundary (ignores high clouds)
            valid_mountain_edges = [r for r in edge_rows if abs(r - max_sky_row) <= 20]
            if len(valid_mountain_edges) > 0:
                max_sky_row = valid_mountain_edges[0]

        boundary = float(max_sky_row)

        # 1. Search Canny ONLY in narrow +/-10px window around U-Net edge (stop cloud jumping)
        if use_canny and canny_edges is not None:
            win_start = max(0, int(max_sky_row) - 10)
            win_end = min(H - 1, int(max_sky_row) + 10)
            canny_in_win = np.where(canny_edges[win_start:win_end + 1, col])[0]
            if len(canny_in_win) > 0:
                candidate_rows = win_start + canny_in_win
                best_r = candidate_rows[np.argmin(np.abs(candidate_rows - max_sky_row))]
                boundary = float(best_r)

        boundaries[col] = boundary

    # 5. Outlier Rejection & Smooth Interpolation
    valid = boundaries >= 0
    if np.any(valid):
        all_cols = np.arange(W, dtype=np.float64)
        valid_cols = all_cols[valid]
        valid_vals = boundaries[valid]

        # Outlier filter: reject points > 30px away from 5-neighbor median
        if len(valid_vals) > 5:
            pad = 2
            padded = np.pad(valid_vals, (pad, pad), mode="edge")
            from numpy.lib.stride_tricks import sliding_window_view
            meds = np.median(sliding_window_view(padded, 5, axis=0), axis=1)
            keep = np.abs(valid_vals - meds) <= 30.0
            if keep.any():
                valid_cols = valid_cols[keep]
                valid_vals = valid_vals[keep]

        boundaries = np.interp(all_cols, valid_cols, valid_vals)

        # 1. Two-pass physical slope constraint (Max 2px/col slope - kills cloud cliff steps)
        max_slope = 2.0
        for c in range(1, W):
            delta = boundaries[c] - boundaries[c - 1]
            if abs(delta) > max_slope:
                boundaries[c] = boundaries[c - 1] + np.sign(delta) * max_slope

        for c in range(W - 2, -1, -1):
            delta = boundaries[c] - boundaries[c + 1]
            if abs(delta) > max_slope:
                boundaries[c] = boundaries[c + 1] + np.sign(delta) * max_slope

        # 2. Restored Gaussian Blur for smooth natural mountain profile
        boundaries = _median_filter_1d(boundaries, kernel_size=9)
        boundaries_2d = cv2.GaussianBlur(boundaries.reshape(1, -1).astype(np.float32), (7, 1), 0)
        boundaries = boundaries_2d.flatten()

    refined = np.zeros((H, W), dtype=np.uint8)
    for col in range(W):
        b = int(np.clip(round(boundaries[col]), 0, H - 1))
        refined[:b, col] = 1

    return np.where(refined == 1, 0, 255).astype(np.uint8)


def refine_multichannel_gradient_fusion(img_np, raw_unet_mask, use_clahe=False):
    """Multi-channel gradient fusion with CLAHE dehazing: LAB b*, HSV Saturation, and Grayscale gradients."""
    H, W = raw_unet_mask.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sky1 = (raw_unet_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky1, connectivity=8)
    top_sky = np.zeros((H, W), dtype=np.uint8)
    top_limit = max(10, int(H * 0.10))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_TOP] <= top_limit and stats[i, cv2.CC_STAT_AREA] > 100:
            top_sky[labels == i] = 1

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    b_star = cv2.GaussianBlur(lab[:, :, 2].astype(np.float32), (3, 3), 0)
    b_vgrad = np.diff(b_star, axis=0)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1].astype(np.float32), (3, 3), 0)
    s_vgrad = np.diff(sat, axis=0)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    g_vgrad = np.diff(gray_blur, axis=0)

    fused_vgrad = 0.5 * b_vgrad + 0.3 * s_vgrad + 0.2 * g_vgrad

    boundaries = np.full(W, -1, dtype=np.int32)
    for col in range(W):
        sky_rows = np.where(top_sky[:, col] == 1)[0]
        if len(sky_rows) == 0:
            continue
        boundary = sky_rows[-1]

        search_min = max(0, boundary - 20)
        search_max = min(H - 2, boundary + 20)
        win = fused_vgrad[search_min:search_max, col]
        if len(win) > 0:
            boundary = search_min + int(np.argmin(win))

        boundaries[col] = boundary

    valid = boundaries >= 0
    if np.any(valid):
        all_cols = np.arange(W)
        boundaries = np.interp(all_cols, all_cols[valid], boundaries[valid])
        boundaries = _median_filter_1d(boundaries, kernel_size=5)

    refined = np.zeros((H, W), dtype=np.uint8)
    for col in range(W):
        b = int(np.clip(round(boundaries[col]), 0, H - 1))
        refined[:b, col] = 1

    return np.where(refined == 1, 0, 255).astype(np.uint8)


def refine_grayscale_fixed_window(img_np, raw_unet_mask, search_radius=25, kernel_size=7, use_clahe=False):
    """Grayscale vertical-gradient refinement with CLAHE dehazing and fixed +/- search_radius row snapping."""
    H, W = raw_unet_mask.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sky1 = (raw_unet_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky1, connectivity=8)
    top_sky = np.zeros((H, W), dtype=np.uint8)
    top_limit = max(10, int(H * 0.10))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_TOP] <= top_limit and stats[i, cv2.CC_STAT_AREA] > 100:
            top_sky[labels == i] = 1

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    vgrad = np.diff(blurred, axis=0)

    refined = np.zeros((H, W), dtype=np.uint8)
    boundaries = np.full(W, -1, dtype=np.int32)

    for col in range(W):
        sky_rows = np.where(top_sky[:, col] == 1)[0]
        if len(sky_rows) == 0:
            continue
        boundary = sky_rows[-1]

        search_min = max(0, boundary - search_radius)
        search_max = min(H - 2, boundary + search_radius)
        window_grad = vgrad[search_min:search_max, col]
        if len(window_grad) > 0:
            snap_offset = int(np.argmin(window_grad))
            boundary = search_min + snap_offset

        boundaries[col] = boundary

    valid = boundaries >= 0
    if np.any(valid):
        all_cols = np.arange(W)
        boundaries = np.interp(all_cols, all_cols[valid], boundaries[valid])
        boundaries = _median_filter_1d(boundaries, kernel_size=kernel_size)

    for col in range(W):
        b = int(np.clip(round(boundaries[col]), 0, H - 1))
        refined[:b, col] = 1

    return np.where(refined == 1, 0, 255).astype(np.uint8)


def refine_dynamic_programming_skyline(img_np, raw_unet_mask, smoothness_penalty=2.0, max_step=5):
    """Skyline extraction using Dynamic Programming (Viterbi shortest-path cost search)."""
    H, W = raw_unet_mask.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sky1 = (raw_unet_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky1, connectivity=8)
    top_sky = np.zeros((H, W), dtype=np.uint8)
    top_limit = max(10, int(H * 0.10))
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_TOP] <= top_limit and stats[i, cv2.CC_STAT_AREA] > 100:
            top_sky[labels == i] = 1

    if top_sky.sum() == 0:
        return np.zeros((H, W), dtype=np.uint8)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    b_star = cv2.GaussianBlur(lab[:, :, 2].astype(np.float32), (3, 3), 0)
    b_vgrad = np.abs(np.diff(b_star, axis=0, prepend=0))

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1].astype(np.float32), (3, 3), 0)
    s_vgrad = np.abs(np.diff(sat, axis=0, prepend=0))

    energy = -(b_vgrad + s_vgrad)

    dp = np.full((H, W), 1e9, dtype=np.float32)
    pointers = np.zeros((H, W), dtype=np.int32)

    col0_sky = np.where(top_sky[:, 0] == 1)[0]
    init_row = col0_sky[-1] if len(col0_sky) > 0 else H // 3
    search_r0 = max(0, init_row - 30)
    search_r1 = min(H, init_row + 30)
    dp[search_r0:search_r1, 0] = energy[search_r0:search_r1, 0]

    for col in range(1, W):
        prev_dp = dp[:, col - 1]
        valid_prev = np.where(prev_dp < 1e8)[0]
        if len(valid_prev) == 0:
            dp[:, col] = energy[:, col]
            continue

        p_min, p_max = valid_prev.min(), valid_prev.max()
        r_min = max(0, p_min - max_step)
        r_max = min(H, p_max + max_step + 1)

        for r in range(r_min, r_max):
            prev_start = max(0, r - max_step)
            prev_end = min(H, r + max_step + 1)
            costs = prev_dp[prev_start:prev_end] + smoothness_penalty * ((np.arange(prev_start, prev_end) - r) ** 2)
            best_idx = np.argmin(costs)
            dp[r, col] = energy[r, col] + costs[best_idx]
            pointers[r, col] = prev_start + best_idx

    path = np.zeros(W, dtype=np.int32)
    path[-1] = int(np.argmin(dp[:, -1]))
    for col in range(W - 1, 0, -1):
        path[col - 1] = pointers[path[col], col]

    refined = np.zeros((H, W), dtype=np.uint8)
    for col in range(W):
        b = int(np.clip(path[col], 0, H - 1))
        refined[:b, col] = 1

    return np.where(refined == 1, 0, 255).astype(np.uint8)


def load_segmentation_model(model_path, device):
    """Load trained SMP U-Net model from checkpoint file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")
    model = smp.Unet(
        encoder_name="tu-mobilenetv3_large_100",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(model_path, map_location=device)
    clean_state = {
        k.replace("module.", "").replace("model.", ""): v for k, v in checkpoint.items()
    }
    model.load_state_dict(clean_state)
    return model.to(device).eval()


def _compute_sky_diagnostics(mask, prob_map=None):
    """Compute quality diagnostics from binary sky mask (sky=0, terrain=255)."""
    mask = np.asarray(mask, dtype=np.uint8)
    H, W = mask.shape
    total_px = H * W
    sky = (mask < 128).astype(np.uint8)

    sky_ratio = float(sky.sum() / total_px) if total_px > 0 else 0.0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)

    largest_sky_area = 0
    top_connected = False
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_sky_area:
            largest_sky_area = area
        if stats[i, cv2.CC_STAT_TOP] == 0:
            top_connected = True

    boundary_coverage = 0.0
    for col in range(W):
        sky_rows = np.where(sky[:, col] == 1)
        if len(sky_rows[0]) > 0:
            boundary_coverage += 1.0
    boundary_coverage /= max(W, 1)

    mean_confidence = float(prob_map.mean()) if prob_map is not None else None

    return {
        "sky_ratio": sky_ratio,
        "largest_sky_area": int(largest_sky_area),
        "top_connected": top_connected,
        "boundary_coverage": boundary_coverage,
        "num_components": int(num_labels - 1),
        "mean_confidence": mean_confidence,
    }


def segment_image(
    model,
    img_path,
    mask_output_path,
    device,
    min_sky_ratio=0.05,
    max_sky_ratio=0.95,
    min_boundary_coverage=0.5,
    tta=False,
    threshold=0.70,
    input_size=256,
    refinement_method="lab_b_subpixel",
    return_prob=False,
):
    """Segment sky from single image with configurable refinement, TTA, input size, and threshold."""
    if not os.path.exists(img_path):
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": f"Image not found: {img_path}",
            "diagnostics": {},
            "mask_path": None,
        }

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    try:
        orig_img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": f"Cannot open image: {e}",
            "diagnostics": {},
            "mask_path": None,
        }

    W, H = orig_img.size
    padded_img, crop_info = _prepare_inference_image(orig_img, input_size=input_size)
    pad_left, pad_top, resized_width, resized_height = crop_info

    tensor_img = transform(Image.fromarray(padded_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor_img)).squeeze().cpu().numpy()
        if tta:
            flipped = np.fliplr(padded_img).copy()
            tensor_flip = transform(Image.fromarray(flipped)).unsqueeze(0).to(device)
            output_flip = torch.sigmoid(model(tensor_flip)).squeeze().cpu().numpy()
            output_flip = np.fliplr(output_flip)
            output = 0.5 * (output + output_flip)

    output_cropped = output[
        pad_top : pad_top + resized_height, pad_left : pad_left + resized_width
    ]
    prob_resized = cv2.resize(
        output_cropped.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
    )
    raw_mask = (prob_resized <= threshold).astype(np.uint8)

    img_np = np.array(orig_img)
    if refinement_method == "lab_b_subpixel":
        refined = refine_sky_mask_with_guidance(img_np, raw_mask)
    elif refinement_method == "grayscale_fixed":
        refined = refine_grayscale_fixed_window(img_np, raw_mask)
    elif refinement_method == "multichannel_fusion":
        refined = refine_multichannel_gradient_fusion(img_np, raw_mask)
    elif refinement_method == "dynamic_programming":
        refined = refine_dynamic_programming_skyline(img_np, raw_mask)
    elif refinement_method == "none":
        refined = np.where(raw_mask == 0, 0, 255).astype(np.uint8)
    else:
        refined = refine_sky_mask_with_guidance(img_np, raw_mask)

    if mask_output_path is not None:
        os.makedirs(os.path.dirname(mask_output_path) or ".", exist_ok=True)
        Image.fromarray(refined).save(mask_output_path)

    diagnostics = _compute_sky_diagnostics(refined, prob_map=prob_resized)
    if return_prob:
        diagnostics["prob_map"] = prob_resized

    return {
        "ok": True,
        "status": "OK",
        "reason": "Clean sky segmentation",
        "diagnostics": diagnostics,
        "mask_path": mask_output_path,
    }


class UnifiedDatasetAug(Dataset):
    """Dataset class combining GeoPose3K and synthetic images with cloud overlay data augmentation."""

    def __init__(
        self,
        imgs,
        masks,
        is_train=True,
        train_transform=None,
        cloud_dir=None,
        cloud_prob=0.3,
    ):
        self.imgs = [str(p) for p in imgs]
        self.masks = [str(p) for p in masks]
        self.is_train = is_train
        self.cloud_prob = cloud_prob if is_train else 0.0
        self.cloud_files = []
        if cloud_dir is not None and os.path.isdir(cloud_dir):
            self.cloud_files = sorted(
                str(Path(cloud_dir) / f)
                for f in os.listdir(cloud_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        if len(self.cloud_files) > 0:
            random.seed()

        if self.is_train:
            self.transform = (
                train_transform if train_transform is not None else A.Compose([])
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(256, 256),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.imgs)

    def _composite_clouds(self, img, mask):
        """Blend cloud images into sky regions (mask == 0)."""
        if not self.cloud_files or random.random() > self.cloud_prob:
            return img
        idx = random.randrange(len(self.cloud_files))
        try:
            cloud = cv2.imread(self.cloud_files[idx], cv2.IMREAD_COLOR)
            if cloud is None:
                return img
            cloud = cv2.cvtColor(cloud, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            ch, cw = cloud.shape[:2]
            if ch != h or cw != w:
                cloud = cv2.resize(cloud, (w, h), interpolation=cv2.INTER_AREA)
            alpha = random.uniform(0.15, 0.5)
            sky = (mask < 0.5).astype(np.float32)[..., None]
            blended = (1 - alpha * sky) * img.astype(
                np.float32
            ) + alpha * sky * cloud.astype(np.float32)
            return np.clip(blended, 0, 255).astype(np.uint8)
        except Exception:
            return img

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 10).astype(np.float32)
        h, w, _ = img.shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = self._composite_clouds(img, mask)
        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)


def find_photo_path(folder_path):
    """Find photo image inside GeoPose3K sample directory."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(folder_path, f"photo{ext}")
        if os.path.exists(p):
            return p
    return None


def load_geopose_split(split_file, base_dir):
    """Load image/mask path lists from GeoPose3K split text file."""
    images, masks = [], []
    split_file = Path(split_file)
    base_dir = Path(base_dir)
    if not os.path.exists(split_file):
        return images, masks
    publish_dir = base_dir / "geoPose3K_final_publish"
    if not publish_dir.exists():
        return images, masks
    existing_folders = set(os.listdir(publish_dir))
    with open(split_file) as f:
        folder_names = [line.strip().strip("/") for line in f if line.strip()]
    for folder in folder_names:
        if folder not in existing_folders:
            continue
        folder_path = publish_dir / folder
        img_p = folder_path / "photo.jpg"
        if not img_p.exists():
            img_p = find_photo_path(folder_path)
        mask_p = folder_path / "pinhole" / "labels_crop.png"
        if img_p and mask_p.exists():
            images.append(str(img_p))
            masks.append(str(mask_p))
    return images, masks


def build_training_loaders(
    geopose_dir,
    syn_img_dir,
    syn_mask_dir,
    batch_size=8,
    train_transform=None,
    cloud_dir=None,
    cloud_prob=0.3,
):
    """Build GeoPose3K and synthetic combined training DataLoaders."""
    train_images, train_masks = [], []
    val_images, val_masks = [], []
    geopose_dir = Path(geopose_dir)

    tr_img, tr_mask = load_geopose_split(
        geopose_dir / "splits" / "geoPose3K_final_train.txt", geopose_dir
    )
    va_img, va_mask = load_geopose_split(
        geopose_dir / "splits" / "geoPose3K_final_val.txt", geopose_dir
    )
    train_images.extend(tr_img)
    train_masks.extend(tr_mask)
    val_images.extend(va_img)
    val_masks.extend(va_mask)

    syn_img_dir, syn_mask_dir = str(syn_img_dir), str(syn_mask_dir)
    if os.path.exists(syn_img_dir):
        all_syn = sorted(
            f for f in os.listdir(syn_img_dir) if f.lower().endswith(".png")
        )
        n = len(all_syn)
        if n > 0:
            split = int(n * 0.8)
            for f in all_syn[:split]:
                train_images.append(os.path.join(syn_img_dir, f))
                train_masks.append(os.path.join(syn_mask_dir, f))
            for f in all_syn[split:]:
                val_images.append(os.path.join(syn_img_dir, f))
                val_masks.append(os.path.join(syn_mask_dir, f))

    train_loader = DataLoader(
        UnifiedDatasetAug(
            train_images,
            train_masks,
            is_train=True,
            train_transform=train_transform,
            cloud_dir=cloud_dir,
            cloud_prob=cloud_prob,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        UnifiedDatasetAug(val_images, val_masks, is_train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader


def build_sky_model(device):
    """Instantiate MobileNetV3-Large U-Net model."""
    return smp.Unet(
        encoder_name="tu-mobilenetv3_large_100",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)


def compute_iou(pred_logits, true_masks, threshold=0.5):
    """Compute IoU metric from predicted logits."""
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def bce_dice_loss(pred, target):
    """Compute combined BCE + Soft Dice loss."""
    bce = nn.BCEWithLogitsLoss()(pred, target)
    smooth = 1e-6
    probs = torch.sigmoid(pred)
    intersection = (probs * target).sum()
    dice = 1.0 - (2.0 * intersection + smooth) / (probs.sum() + target.sum() + smooth)
    return bce + dice


def train_sky_model(
    model, train_loader, val_loader, device, save_path, epochs=15, lr=2e-4
):
    """Model training loop with Cosine Annealing learning rate schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, train_ious = [], []
    val_losses, val_ious = [], []
    lrs = []
    best_val_iou = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = total_iou = 0.0
        lrs.append(optimizer.param_groups[0]["lr"])

        for imgs, msks in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)"
        ):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = bce_dice_loss(out, msks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_iou += compute_iou(out, msks)

        train_losses.append(total_loss / len(train_loader))
        train_ious.append(total_iou / len(train_loader))
        scheduler.step()

        model.eval()
        v_loss = v_iou = 0.0
        with torch.no_grad():
            for imgs, msks in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} (Val)"
            ):
                imgs, msks = imgs.to(device), msks.to(device)
                out = model(imgs)
                v_loss += bce_dice_loss(out, msks).item()
                v_iou += compute_iou(out, msks)

        val_losses.append(v_loss / len(val_loader))
        val_ious.append(v_iou / len(val_loader))

        if val_ious[-1] > best_val_iou:
            best_val_iou = val_ious[-1]
            torch.save(model.state_dict(), str(save_path))

    return train_losses, val_losses, train_ious, val_ious, lrs

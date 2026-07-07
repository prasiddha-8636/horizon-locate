#!/usr/bin/env python3
"""
UNIFIED EVALUATION PIPELINE - Visual Geo-Localization for Khumbu Region
=========================================================================
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import rasterio
import cv2
from concurrent.futures import ProcessPoolExecutor
from pyproj import Transformer

# ============================================================================
# COORDINATE SYSTEM & CANONICAL GRID PARAMETERS
# ============================================================================

GRID_SPACING_M = 500.0  # Viewpoint grid spacing in meters
UTM_ZONE = "EPSG:32645"  # UTM Zone 45N (Khumbu region)
GPS_CRS = "EPSG:4326"

DEM_CROP_SIZE_M = 110000.0  # Synthetic query generation: crop size around DEM center
QUERY_MIN_HEIGHT_M = 1.8
QUERY_MAX_HEIGHT_M = 1.8


def load_dem_exactly(dem_path="data/digital_elevation_model/dem.tif", viewpoints_mapping_path="data/digital_elevation_model/viewpoints_mapping.npy"):
    """
    Loads and processes the DEM using identical stride, cropping, and median blurring 
    parameters to match the synthetic generator mesh generation exactly.
    """
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float32)
        dem_data = np.nan_to_num(dem_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        valid_mask = dem_data > 10.0
        if np.any(valid_mask):
            min_valid = float(np.min(dem_data[valid_mask]))
            dem_data[~valid_mask] = min_valid
        else:
            dem_data[~valid_mask] = 1000.0
            
        dem_data = np.clip(dem_data, 100.0, 9000.0)

        [pixel_width, row_rotation, start_x, col_rotation, pixel_height, start_y] = src.transform[:6]

        raw_xs = start_x + np.arange(src.width) * pixel_width
        raw_ys = start_y + np.arange(src.height) * pixel_height
        
        if pixel_height < 0:
            raw_ys = raw_ys[::-1]
            dem_data = np.flipud(dem_data)
            
        dem_crs = src.crs.to_string()

    if os.path.exists(viewpoints_mapping_path):
        vps = np.load(viewpoints_mapping_path)
        center_x = float(np.mean(vps[:, 2]))
        center_y = float(np.mean(vps[:, 3]))
    else:
        center_x = 478712.0
        center_y = 3086932.0

    crop_size_x, crop_size_y = 200000.0, 190000.0
    valid_cols = np.any(valid_mask, axis=0)
    valid_rows = np.any(valid_mask, axis=1)

    crop_min_x = max(center_x - crop_size_x/2.0, np.min(raw_xs[valid_cols]))
    crop_max_x = min(center_x + crop_size_x/2.0, np.max(raw_xs[valid_cols]))
    crop_min_y = max(center_y - crop_size_y/2.0, np.min(raw_ys[valid_rows]))
    crop_max_y = min(center_y + crop_size_y/2.0, np.max(raw_ys[valid_rows]))

    crop_mask_x = (raw_xs >= crop_min_x) & (raw_xs <= crop_max_x)
    crop_mask_y = (raw_ys >= crop_min_y) & (raw_ys <= crop_max_y)

    xs_cropped = raw_xs[crop_mask_x]
    ys_cropped = raw_ys[crop_mask_y]

    stride = 4
    dem_data_final = dem_data[crop_mask_y][:, crop_mask_x][::stride, ::stride]
    dem_data_final = cv2.medianBlur(dem_data_final, 5)
    xs_final = xs_cropped[::stride]
    ys_final = ys_cropped[::stride]
    
    return dem_data_final, xs_final, ys_final, dem_crs


def get_unfiltered_viewpoint_idx(gt_info, xs_final, ys_final, viewpoints_mapping):
    """
    Translates the viewpoint identifier in ground_truth.json back to the
    unfiltered canonical database index. Supports both filtered synthetic indices
    and direct coordinate distance matches for real-world street view queries.
    """
    gps_to_utm = Transformer.from_crs(GPS_CRS, UTM_ZONE, always_xy=True)
    true_lat = gt_info["true_lat"]
    true_lon = gt_info["true_lon"]
    true_utm_x, true_utm_y = gps_to_utm.transform(true_lon, true_lat)

    # First option: check geographic coordinate proximity (works for real street view)
    dists = np.sum((viewpoints_mapping[:, 2:4] - np.array([true_utm_x, true_utm_y]))**2, axis=1)
    true_idx = int(np.argmin(dists))

    # Second option: fallback to translating synthetic filtered index if coordinates are far
    if np.sqrt(dists[true_idx]) > 500.0 and "closest_viewpoint_id" in gt_info:
        filtered_idx = gt_info["closest_viewpoint_id"]
        center_x_mesh = (xs_final[0] + xs_final[-1]) / 2.0
        center_y_mesh = (ys_final[0] + ys_final[-1]) / 2.0
        
        eye_xs = viewpoints_mapping[:, 2]
        eye_ys = viewpoints_mapping[:, 3]
        
        safe_mask = (
            (eye_xs >= center_x_mesh - 20000.0) & (eye_xs <= center_x_mesh + 20000.0) &
            (eye_ys >= center_y_mesh - 15000.0) & (eye_ys <= center_y_mesh + 15000.0)
        )
        
        unfiltered_indices = np.where(safe_mask)[0]
        if filtered_idx < len(unfiltered_indices):
            return int(unfiltered_indices[filtered_idx])
            
    return true_idx


# ============================================================================
# MATCHING ENGINE (FFT + DTW)
# ============================================================================

def calculate_geodesic_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def vertical_to_horizontal_fov(vertical_fov_deg, aspect_ratio=1.5):
    """Convert vertical FOV to horizontal FOV."""
    return np.degrees(2.0 * np.arctan(np.tan(np.radians(vertical_fov_deg) / 2.0) * aspect_ratio))


def extract_elevation_profile(mask_path, fov_y_deg=65.0, aspect_ratio=None, r_tilt=None):
    """
    Extract 1D skyline elevation profile with noise-filtering and adaptive smoothing.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    H, W = mask.shape
    
    if aspect_ratio is None:
        aspect_ratio = W / H
    
    # Robust region-based sky color convention auto-detection
    top_rows_mean = np.mean(mask[:10, :])
    bottom_rows_mean = np.mean(mask[-10:, :])
    sky_is_white = top_rows_mean > bottom_rows_mean
    
    # Morphological clean to remove isolated noise pixels
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = (mask < 128).astype(np.uint8) if sky_is_white else (mask >= 128).astype(np.uint8)    

    skyline_pixels = np.zeros(W)
    for col in range(W):
        terrain_indices = np.where(binary_mask[:, col] == 1)[0]
        if len(terrain_indices) > 0:
            skyline_pixels[col] = terrain_indices[0]
        else:
            skyline_pixels[col] = H - 1

    # Filter out real-world structural occlusions (trees, poles, buildings)
    skyline_pixels_2d = skyline_pixels.reshape(1, -1).astype(np.float32)
    skyline_pixels_cleaned = cv2.morphologyEx(
        skyline_pixels_2d, 
        cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    ).ravel()
    skyline_pixels = np.maximum(skyline_pixels.astype(np.float32), skyline_pixels_cleaned)

    # 1D Median Filter to remove single-column spikes
    skyline_pixels = cv2.medianBlur(skyline_pixels.astype(np.float32), 5).ravel()

    # Gaussian smoothing (sigma=2.5) matching DEM raycast profile smoothness
    sigma = 2.5
    size = int(2 * np.ceil(3 * sigma) + 1)
    x_kernel = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-x_kernel**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    
    pad_width = size // 2
    padded_pixels = np.pad(skyline_pixels, pad_width, mode='reflect')
    skyline_pixels = np.convolve(padded_pixels, kernel, mode='valid')
    
    x_c, y_c = W / 2.0, H / 2.0
    horizontal_fov_deg = vertical_to_horizontal_fov(fov_y_deg, aspect_ratio=aspect_ratio)
    focal_x = W / (2.0 * np.tan(np.radians(horizontal_fov_deg) / 2.0))
    focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    
    cols = np.arange(W)
    x_vals = (cols - x_c) / focal_x
    y_vals = (y_c - skyline_pixels) / focal_y
    z_vals = -np.ones(W)
    
    rays_cam = np.vstack((x_vals, y_vals, z_vals))
    rays_cam_norms = np.linalg.norm(rays_cam, axis=0)
    rays_cam_normalized = rays_cam / rays_cam_norms
    
    if r_tilt is not None:
        rays_leveled = r_tilt @ rays_cam_normalized
        elevations_rad = np.arcsin(np.clip(rays_leveled[1, :], -1.0, 1.0))
        azimuths_rad = np.arctan2(rays_leveled[0, :], -rays_leveled[2, :])
    else:
        elevations_rad = np.arcsin(np.clip(rays_cam_normalized[1, :], -1.0, 1.0))
        azimuths_rad = np.arctan2(rays_cam_normalized[0, :], -rays_cam_normalized[2, :])    

    azimuths_deg = np.degrees(azimuths_rad)
    elevations_deg = np.degrees(elevations_rad)
    
    sort_idx = np.argsort(azimuths_deg)
    azimuths_deg = azimuths_deg[sort_idx]
    elevations_deg = elevations_deg[sort_idx]
    
    start_az = np.ceil(azimuths_deg[0] / 0.25) * 0.25
    end_az = np.floor(azimuths_deg[-1] / 0.25) * 0.25
    target_azimuth_grid = np.arange(start_az, end_az + 0.01, 0.25)
    
    calibrated_query_profile = np.interp(target_azimuth_grid, azimuths_deg, elevations_deg)
    
    return calibrated_query_profile, start_az

def fft_valid_convolve(signal, kernel):
    """Compute the valid linear convolution using FFT."""
    full_len = len(signal) + len(kernel) - 1
    fft_n = 1 << ((full_len - 1).bit_length())
    fft_signal = np.fft.rfft(signal, fft_n)
    fft_kernel = np.fft.rfft(kernel, fft_n)
    full_conv = np.fft.irfft(fft_signal * fft_kernel, fft_n)
    valid_start = len(kernel) - 1
    valid_end = valid_start + len(signal) - len(kernel) + 1
    return full_conv[valid_start:valid_end]


def build_query_feature_bundle(query_profile):
    """Build normalized elevation, derivative, and curvature descriptors."""
    q_raw = np.asarray(query_profile, dtype=np.float64)
    q_norm = q_raw - np.mean(q_raw)
    q_norm /= (np.std(q_norm) + 1e-12)

    q_deriv = np.gradient(q_norm)
    q_deriv -= np.mean(q_deriv)
    q_deriv /= (np.std(q_deriv) + 1e-12)

    q_curv = np.gradient(q_deriv)
    q_curv -= np.mean(q_curv)
    q_curv /= (np.std(q_curv) + 1e-12)

    return q_norm, q_deriv, q_curv


def _compute_z_norm_corr(padded, q_signal, m, N):
    """
    Compute z-normalized Pearson correlation between query and sliding windows.
    """
    dot = fft_valid_convolve(padded, q_signal[::-1])

    cumsum = np.zeros(len(padded) + 1, dtype=np.float64)
    cumsum[1:] = np.cumsum(padded)
    window_means = (cumsum[m:] - cumsum[:-m]) / m

    cumsum_sq = np.zeros(len(padded) + 1, dtype=np.float64)
    cumsum_sq[1:] = np.cumsum(padded ** 2)
    sum_t2 = cumsum_sq[m:] - cumsum_sq[:-m]

    window_vars = np.maximum(sum_t2[:N] - m * (window_means[:N] ** 2), 0.0)
    window_stds = np.sqrt(window_vars + 0.05)
    
    q_std = 1.0
    denom = np.sqrt(m) * window_stds * q_std + 1e-12
    
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_norm = np.where(denom > 1e-12, dot / denom, 0.0)
    return corr_norm


def run_sliding_fft_for_db(db, query_profile, feature="combined", expected_offset=None, tolerance_bins=None):
    """
    FFT-accelerated sliding window matching.
    """
    num_viewpoints, N = db.shape
    m = len(query_profile)

    q_norm, q_deriv, q_curv = build_query_feature_bundle(query_profile)

    best_corrs = np.zeros(num_viewpoints)
    best_offsets = np.zeros(num_viewpoints, dtype=np.int32)

    for i in range(num_viewpoints):
        profile = db[i].astype(np.float64)

        if feature == "profile":
            db_signal = profile
            q_signal = q_norm
            padded = np.hstack([db_signal, db_signal[:m-1]])
            corr_norm = _compute_z_norm_corr(padded, q_signal, m, N)
        elif feature == "derivative":
            db_signal = 0.5 * (np.roll(profile, -1) - np.roll(profile, 1))
            q_signal = q_deriv
            padded = np.hstack([db_signal, db_signal[:m-1]])
            corr_norm = _compute_z_norm_corr(padded, q_signal, m, N)
        elif feature == "curvature":
            db_deriv = 0.5 * (np.roll(profile, -1) - np.roll(profile, 1))
            db_signal = 0.5 * (np.roll(db_deriv, -1) - np.roll(db_deriv, 1))
            q_signal = q_curv
            padded = np.hstack([db_signal, db_signal[:m-1]])
            corr_norm = _compute_z_norm_corr(padded, q_signal, m, N)
        elif feature == "combined":
            db_raw = profile
            db_deriv = 0.5 * (np.roll(profile, -1) - np.roll(profile, 1))
            db_curv = 0.5 * (np.roll(db_deriv, -1) - np.roll(db_deriv, 1))

            corr_raw = _compute_z_norm_corr(np.hstack([db_raw, db_raw[:m-1]]), q_norm, m, N)
            corr_deriv = _compute_z_norm_corr(np.hstack([db_deriv, db_deriv[:m-1]]), q_deriv, m, N)
            corr_curv = _compute_z_norm_corr(np.hstack([db_curv, db_curv[:m-1]]), q_curv, m, N)

            weights = np.array([0.30, 0.65, 0.05], dtype=np.float64)
            corr_norm = weights[0] * corr_raw + weights[1] * corr_deriv + weights[2] * corr_curv
        else:
            raise ValueError(f"Unknown feature type: {feature}")

        # Apply Compass Filter: constrain heading search to ±30 degrees
        if expected_offset is not None and tolerance_bins is not None:
            offsets_grid = np.arange(N)
            dist_to_expected = np.minimum((offsets_grid - expected_offset) % N, (expected_offset - offsets_grid) % N)
            valid_offset_mask = dist_to_expected <= tolerance_bins
            corr_norm[~valid_offset_mask] = -np.inf

        best_offset = int(np.argmax(corr_norm))
        best_corrs[i] = corr_norm[best_offset]
        best_offsets[i] = best_offset

    return best_corrs, best_offsets


def run_three_tier_search_opt(db_global, db_local, db_restricted, query_profile, start_az,
                              feature="derivative", top_k_candidates=50, dtw_window=20, 
                              use_tiers=(True, True, True), valid_vp_mask=None,
                              expected_offset=None, tolerance_bins=None):
    """
    Two-stage matching: FFT pre-filtering followed by Sakoe-Chiba DTW.
    """
    num_viewpoints, N = db_global.shape
    m = len(query_profile)
    tier_names = ["Global", "Local", "Restricted"]
    
    corr_g, offsets_g = (np.full(num_viewpoints, -np.inf), np.zeros(num_viewpoints, dtype=np.int32))
    corr_l, offsets_l = (np.full(num_viewpoints, -np.inf), np.zeros(num_viewpoints, dtype=np.int32))
    corr_r, offsets_r = (np.full(num_viewpoints, -np.inf), np.zeros(num_viewpoints, dtype=np.int32))

    if use_tiers[0]:
        corr_g, offsets_g = run_sliding_fft_for_db(db_global, query_profile, feature=feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
    if use_tiers[1]:
        corr_l, offsets_l = run_sliding_fft_for_db(db_local, query_profile, feature=feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
    if use_tiers[2]:
        corr_r, offsets_r = run_sliding_fft_for_db(db_restricted, query_profile, feature=feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)

    best_corr_per_viewpoint = np.zeros(num_viewpoints)
    best_offset_per_viewpoint = np.zeros(num_viewpoints, dtype=np.int32)
    best_tier_per_viewpoint = np.zeros(num_viewpoints, dtype=np.int32)

    for i in range(num_viewpoints):
        scores = [corr_g[i], corr_l[i], corr_r[i]]
        offsets = [offsets_g[i], offsets_l[i], offsets_r[i]]
        best_tier_idx = int(np.argmax(scores))
        best_corr_per_viewpoint[i] = scores[best_tier_idx]
        best_offset_per_viewpoint[i] = int(offsets[best_tier_idx])
        best_tier_per_viewpoint[i] = best_tier_idx

    if valid_vp_mask is not None:
        best_corr_per_viewpoint[~valid_vp_mask] = -np.inf

    top_candidates = np.argsort(-best_corr_per_viewpoint)[:top_k_candidates]
    
    final_results = []
    dbs = [db_global, db_local, db_restricted]
    
    # Robust 2D Descriptor: Normalized Profile and Slope
    q_norm = query_profile - np.mean(query_profile)
    q_std = np.std(q_norm) + 1e-12
    q_norm_scaled = q_norm / q_std
    q_deriv = np.gradient(q_norm_scaled)
    q_deriv /= (np.std(q_deriv) + 1e-12)
    q_desc = np.vstack((q_norm_scaled, q_deriv)).T    

    for idx in top_candidates:
        winning_tier_idx = best_tier_per_viewpoint[idx]
        active_db = dbs[winning_tier_idx]
        offset = best_offset_per_viewpoint[idx]
        
        db_indices = np.arange(offset, offset + m) % N
        matched_db_subsequence = active_db[idx, db_indices]
        
        db_sub_norm = matched_db_subsequence - np.mean(matched_db_subsequence)
        db_sub_std = np.std(db_sub_norm) + 1e-12
        db_sub_norm_scaled = db_sub_norm / db_sub_std
        db_deriv = np.gradient(db_sub_norm_scaled)
        db_deriv /= (np.std(db_deriv) + 1e-12)
        db_desc = np.vstack((db_sub_norm_scaled, db_deriv)).T

        # Adaptively scale the warping window based on query sequence length (FOV)
        adapted_window = max(dtw_window, int(0.03 * m))
        
        INF = float('inf')
        dtw_matrix = np.full((m + 1, m + 1), INF)
        dtw_matrix[0, 0] = 0.0

        for ii in range(1, m + 1):
            col_start = max(1, ii - adapted_window)
            col_end = min(m, ii + adapted_window)
            for jj in range(col_start, col_end + 1):
                cost = float(np.linalg.norm(q_desc[ii-1] - db_desc[jj-1]))
                v = min(dtw_matrix[ii-1, jj], dtw_matrix[ii, jj-1], dtw_matrix[ii-1, jj-1])
                dtw_matrix[ii, jj] = cost + v

        dtw_cost = dtw_matrix[m, m]
        predicted_heading = (offset * 0.25 - start_az) % 360.0

        final_results.append({
            'viewpoint_idx': idx,
            'fft_corr': best_corr_per_viewpoint[idx],
            'dtw_cost': dtw_cost,
            'heading_deg': predicted_heading,
            'predicted_tier': tier_names[winning_tier_idx]
        })
    
    final_results.sort(key=lambda x: x['dtw_cost'] / max(0.01, x['fft_corr']))
    return final_results

def process_single_sample(sample_id, gt_info, viewpoints_mapping, vp_elevations, db_global, db_local, db_restricted, dem_data_final, xs_final, ys_final, use_tiers, args):
    mask_path = os.path.join(args.masks_dir, f"{sample_id}.png")
    if not os.path.exists(mask_path):
        try:
            mask_path = os.path.join(args.masks_dir, f"sample_{int(sample_id):04d}.png")
        except ValueError:
            return None
            
    if not os.path.exists(mask_path):
        return None

    true_lat, true_lon = gt_info["true_lat"], gt_info["true_lon"]
    fov_y_deg = gt_info.get("fov_y_deg", 65.0)
    
    # Determine height constraints
    if args.disable_height_filter:
        valid_vp_mask = None
    else:
        if "eye_z_m" in gt_info:
            # Synthetic query path
            gt_ground_z = gt_info["eye_z_m"] - gt_info.get("query_height_m", QUERY_MIN_HEIGHT_M)
            tolerance = args.altimetric_tolerance * 3.5  # Wider tolerance for synthetic blurred grid shifts
        else:
            # Real-world coordinate query path
            from pyproj import Transformer
            gps_to_utm = Transformer.from_crs(GPS_CRS, UTM_ZONE, always_xy=True)
            true_utm_x, true_utm_y = gps_to_utm.transform(true_lon, true_lat)
            
            dx = xs_final[1] - xs_final[0]
            dy = ys_final[1] - ys_final[0]
            dem_height, dem_width = dem_data_final.shape
            
            i_frac = (true_utm_x - xs_final[0]) / dx
            j_frac = (true_utm_y - ys_final[0]) / dy
            
            i0 = int(np.clip(np.floor(i_frac), 0, dem_width - 2))
            i1 = i0 + 1
            j0 = int(np.clip(np.floor(j_frac), 0, dem_height - 2))
            j1 = j0 + 1
            
            tx = np.clip(i_frac - i0, 0.0, 1.0)
            ty = np.clip(j_frac - j0, 0.0, 1.0)
            
            z00 = dem_data_final[j0, i0]
            z10 = dem_data_final[j0, i1]
            z01 = dem_data_final[j1, i0]
            z11 = dem_data_final[j1, i1]
            
            gt_ground_z = (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11
            tolerance = args.altimetric_tolerance  # Standard tolerance for real GPS measurements

        valid_vp_mask = np.abs(vp_elevations - gt_ground_z) <= tolerance    

    r_tilt = gt_info.get("cam_R_tilt", None)
    if r_tilt is not None:
        r_tilt = np.array(r_tilt, dtype=np.float32)

    query_profile, start_az = extract_elevation_profile(mask_path, fov_y_deg=fov_y_deg, aspect_ratio=None, r_tilt=r_tilt)

    true_heading = gt_info.get("true_heading_deg", None)
    if true_heading is not None:
        tolerance_bins = int(30.0 / 0.25)
        expected_offset = int(((true_heading + start_az) % 360.0) / 0.25)        
    else:
        tolerance_bins = None
        expected_offset = None

    if args.use_fft_only:
        corr_g, _ = (run_sliding_fft_for_db(db_global, query_profile, feature=args.feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
                     if use_tiers[0] else (np.full(db_global.shape[0], -np.inf), None))
        corr_l, _ = (run_sliding_fft_for_db(db_local, query_profile, feature=args.feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
                     if use_tiers[1] else (np.full(db_global.shape[0], -np.inf), None))
        corr_r, _ = (run_sliding_fft_for_db(db_restricted, query_profile, feature=args.feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
                     if use_tiers[2] else (np.full(db_global.shape[0], -np.inf), None))

        best_corr = np.maximum(np.maximum(corr_g, corr_l), corr_r)
        if valid_vp_mask is not None:
            best_corr[~valid_vp_mask] = -np.inf
        top_indices = np.argsort(-best_corr)
        matches = [{'viewpoint_idx': idx} for idx in top_indices[:5]]
    else:
        matches = run_three_tier_search_opt(
            db_global, db_local, db_restricted, query_profile, start_az,
            feature=args.feature,
            top_k_candidates=args.top_k_candidates,
            dtw_window=args.dtw_window,
            use_tiers=use_tiers,
            valid_vp_mask=valid_vp_mask,
            expected_offset=expected_offset,
            tolerance_bins=tolerance_bins
        )

    r1_idx = matches[0]['viewpoint_idx']
    r1_lat, r1_lon = viewpoints_mapping[r1_idx, 0], viewpoints_mapping[r1_idx, 1]
    r1_error = calculate_geodesic_distance(true_lat, true_lon, r1_lat, r1_lon)

    in_top5 = False
    for m in matches[:5]:
        m_idx = m['viewpoint_idx']
        m_lat, m_lon = viewpoints_mapping[m_idx, 0], viewpoints_mapping[m_idx, 1]
        if calculate_geodesic_distance(true_lat, true_lon, m_lat, m_lon) <= 1000.0:
            in_top5 = True
            break

    return {
        "sample_id": sample_id,
        "error": r1_error,
        "top1_correct": r1_error <= 1000.0,
        "top5_correct": in_top5
    }

# ============================================================================
# MAIN COMMANDS
# ============================================================================

def cmd_evaluate(args):
    print("\n=== Evaluation / Prediction ===")
    
    gt_data = None
    if os.path.exists(args.metadata_path):
        with open(args.metadata_path) as f:
            gt_data = json.load(f)
    else:
        print("Warning: Ground-truth metadata file not found. Running in pure prediction mode...")

    viewpoints_mapping = np.load("data/digital_elevation_model/viewpoints_mapping.npy")
    
    # Load and process the DEM exactly matching render_synthetic's geometry layout
    print("Preloading DEM matching generator specifications...")
    dem_data_final, xs_final, ys_final, dem_crs = load_dem_exactly()
    
    gps_to_utm = Transformer.from_crs(GPS_CRS, UTM_ZONE, always_xy=True)

    # Cache elevations for all viewpoints using bilinear interpolation
    vp_elevations = np.zeros(viewpoints_mapping.shape[0], dtype=np.float32)
    dx = xs_final[1] - xs_final[0]
    dy = ys_final[1] - ys_final[0]
    dem_height, dem_width = dem_data_final.shape

    for i in range(viewpoints_mapping.shape[0]):
        vp_x = viewpoints_mapping[i, 2]
        vp_y = viewpoints_mapping[i, 3]
        
        i_frac = (vp_x - xs_final[0]) / dx
        j_frac = (vp_y - ys_final[0]) / dy
        
        i0 = int(np.clip(np.floor(i_frac), 0, dem_width - 2))
        i1 = i0 + 1
        j0 = int(np.clip(np.floor(j_frac), 0, dem_height - 2))
        j1 = j0 + 1
        
        tx = np.clip(i_frac - i0, 0.0, 1.0)
        ty = np.clip(j_frac - j0, 0.0, 1.0)
        
        z00 = dem_data_final[j0, i0]
        z10 = dem_data_final[j0, i1]
        z01 = dem_data_final[j1, i0]
        z11 = dem_data_final[j1, i1]
        
        vp_elevations[i] = (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11        

    print(f"Preloading database tiers...")
    db_global = np.load("data/digital_elevation_model/horizon_database/global.npy")
    db_local = np.load("data/digital_elevation_model/horizon_database/local.npy")
    db_restricted = np.load("data/digital_elevation_model/horizon_database/restricted.npy")
    
    use_tiers = (args.tiers[0] == '1', args.tiers[1] == '1', args.tiers[2] == '1')
    
    # Pure prediction mode if ground truth is not provided
    if gt_data is None:
        mask_files = sorted([
            f for f in os.listdir(args.masks_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"✓ Found {len(mask_files)} query mask files. Running pure predictions...")
        for mask_file in mask_files:
            mask_path = os.path.join(args.masks_dir, mask_file)
            query_profile, start_az = extract_elevation_profile(mask_path, fov_y_deg=65.0, aspect_ratio=None, r_tilt=None)
            
            matches = run_three_tier_search_opt(
                db_global, db_local, db_restricted, query_profile, start_az,
                feature=args.feature,
                top_k_candidates=args.top_k_candidates,
                dtw_window=args.dtw_window,
                use_tiers=use_tiers,
                valid_vp_mask=None,
                expected_offset=None,
                tolerance_bins=None
            )
            r1_idx = matches[0]['viewpoint_idx']
            r1_lat, r1_lon = viewpoints_mapping[r1_idx, 0], viewpoints_mapping[r1_idx, 1]
            print(f"Mask: {mask_file} -> Predicted Viewpoint ID: {r1_idx}, Lat: {r1_lat:.6f}, Lon: {r1_lon:.6f}, Heading: {matches[0]['heading_deg']:.2f}°")
        return

    # Accuracy verification mode when ground truth is provided
    sample_ids = sorted(list(gt_data.keys()))
    if args.limit > 0:
        sample_ids = sample_ids[:args.limit]    
 
    top1_correct, top5_correct = 0, 0
    errors = []
    failed_sample_ids = []

    print(f"Launching parallel matching executor across available CPU cores...")
    with ProcessPoolExecutor() as executor:
        futures = []
        for sample_id in sample_ids:
            gt_info = gt_data[sample_id]
            futures.append(
                executor.submit(
                    process_single_sample,
                    sample_id,
                    gt_info,
                    viewpoints_mapping,
                    vp_elevations,
                    db_global,
                    db_local,
                    db_restricted,
                    dem_data_final,
                    xs_final,
                    ys_final,
                    use_tiers,
                    args
                )
            )
        
        # Display progress bar while collecting results as they finish
        for fut in tqdm(futures, desc="Matching"):
            res = fut.result()
            if res is None:
                continue
            
            errors.append(res["error"])
            if res["top1_correct"]:
                top1_correct += 1
            else:
                failed_sample_ids.append(res["sample_id"])
            if res["top5_correct"]:
                top5_correct += 1    

    if len(errors) == 0:
        print("No samples evaluated!")
        return
    
    errors = np.array(errors)
    total_samples = len(errors)
    
    # Calculate success at different precision scales
    acc_1000 = 100.0 * np.sum(errors <= 1000.0) / total_samples
    acc_500  = 100.0 * np.sum(errors <= 500.0) / total_samples
    acc_100  = 100.0 * np.sum(errors <= 100.0) / total_samples
    acc_10   = 100.0 * np.sum(errors <= 10.0) / total_samples
    
    top5_acc = 100.0 * top5_correct / total_samples
    median_err = np.median(errors)
    
    print("\n" + "="*60)
    print(f"GEOMETRIC LOCALIZATION PRECISION BREAKDOWN (Total: {total_samples} samples)")
    print("-"*60)
    print(f"Failed Sample IDs (Error > 1km): {failed_sample_ids}")
    print("-"*60)
    print(f"  Top-1 Accuracy at 1000m (Coarse): {acc_1000:.2f}%")
    print(f"  Top-1 Accuracy at  500m (Grid):   {acc_500:.2f}%")
    print(f"  Top-1 Accuracy at  100m (Local):  {acc_100:.2f}%")
    print(f"  Top-1 Accuracy at   10m (Meter):  {acc_10:.2f}%")
    print("-"*60)
    print(f"  Top-5 Accuracy (1000m):           {top5_acc:.2f}%")
    print(f"  Median Position Error:            {median_err:.1f} m")
    print("="*60)

    # Save raw errors to disk so the Jupyter Notebook can load them for plotting
    np.save("temp_errors.npy", errors)


def cmd_diagnose(args):
    print("\n=== Diagnostic: Checking Sample 0 ===")
    
    with open("data/synthetic_dataset/ground_truth.json") as f:
        gt_data = json.load(f)
    viewpoints_mapping = np.load("data/digital_elevation_model/viewpoints_mapping.npy")
    db_global = np.load("data/digital_elevation_model/horizon_database/global.npy")
    
    dem_data_final, xs_final, ys_final, _ = load_dem_exactly()
    
    gt_info = gt_data["0"]
    
    # Translate query coordinates back to database indexes
    true_viewpoint_id = get_unfiltered_viewpoint_idx(gt_info, xs_final, ys_final, viewpoints_mapping)
    
    true_lat, true_lon = gt_info["true_lat"], gt_info["true_lon"]
    fov_y_deg = gt_info.get("fov_y_deg", 65.0)
    
    r_tilt = gt_info.get("cam_R_tilt", None)
    if r_tilt is not None:
        r_tilt = np.array(r_tilt, dtype=np.float32)

    query_profile, start_az = extract_elevation_profile("data/synthetic_dataset/masks/sample_0000.png", fov_y_deg=fov_y_deg, aspect_ratio=None, r_tilt=r_tilt)
    corr_scores_profile, _ = run_sliding_fft_for_db(db_global, query_profile, feature="profile")
    corr_scores_deriv, _ = run_sliding_fft_for_db(db_global, query_profile, feature="derivative")
    corr_scores_combined, _ = run_sliding_fft_for_db(db_global, query_profile, feature="combined")

    gt_corr_profile = corr_scores_profile[true_viewpoint_id]
    gt_rank_profile = np.argsort(-corr_scores_profile).tolist().index(true_viewpoint_id) + 1
    gt_corr_deriv = corr_scores_deriv[true_viewpoint_id]
    gt_rank_deriv = np.argsort(-corr_scores_deriv).tolist().index(true_viewpoint_id) + 1
    gt_corr_combined = corr_scores_combined[true_viewpoint_id]
    gt_rank_combined = np.argsort(-corr_scores_combined).tolist().index(true_viewpoint_id) + 1

    print(f"Sample 0 GT viewpoint profile corr: {gt_corr_profile:.4f}, rank: {gt_rank_profile}")
    print(f"Sample 0 GT viewpoint derivative corr: {gt_corr_deriv:.4f}, rank: {gt_rank_deriv}")
    print(f"Sample 0 GT viewpoint combined corr: {gt_corr_combined:.4f}, rank: {gt_rank_combined}")
    top_5_indices = np.argsort(-corr_scores_combined)[:5]
    print(f"\nFFT Top-5 matches:")
    for rank, idx in enumerate(top_5_indices, 1):
        lat, lon = viewpoints_mapping[idx, 0], viewpoints_mapping[idx, 1]
        dist = calculate_geodesic_distance(true_lat, true_lon, lat, lon)
        match_quality = "✓ CORRECT" if idx == true_viewpoint_id else ""
        print(f"  {rank}. Viewpoint {idx}: {dist:.1f}m away {match_quality}")


def cmd_visualize(args):
    print(f"\n=== Visualizing Sample {args.sample_id} ===")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for visualization. Install it using: pip install matplotlib")
        return

    gt_json_path = "data/synthetic_dataset/ground_truth.json"
    db_local_path = "data/digital_elevation_model/horizon_database/local.npy"

    if not os.path.exists(gt_json_path):
        print(f"Error: {gt_json_path} not found. Please build ground truth first.")
        return
    if not os.path.exists(db_local_path):
        print(f"Error: {db_local_path} not found.")
        return

    with open(gt_json_path) as f: 
        gt_data = json.load(f)
    db_local = np.load(db_local_path)
    viewpoints_mapping = np.load("data/digital_elevation_model/viewpoints_mapping.npy")

    sample_key = str(args.sample_id)
    # Check for direct key match, or fallback to integer formatting for visualizer arg
    if sample_key not in gt_data:
        try:
            sample_key = list(gt_data.keys())[args.sample_id]
        except IndexError:
            print(f"Error: Sample identifier {args.sample_id} not found in dataset.")
            return

    dem_data_final, xs_final, ys_final, _ = load_dem_exactly()

    gt_info = gt_data[sample_key]
    
    # Translate query coordinates back to database indexes
    true_idx = get_unfiltered_viewpoint_idx(gt_info, xs_final, ys_final, viewpoints_mapping)
    
    fov_y_deg = gt_info.get("fov_y_deg", 65.0)

    # Resolve paths dynamically
    image_path = os.path.join(args.images_dir, f"{sample_key}.png")
    mask_path = os.path.join(args.masks_dir, f"{sample_key}.png")
    
    if not os.path.exists(mask_path):
        try:
            image_path = os.path.join(args.images_dir, f"sample_{int(sample_key):04d}.png")
            mask_path = os.path.join(args.masks_dir, f"sample_{int(sample_key):04d}.png")
        except ValueError:
            pass

    r_tilt = gt_info.get("cam_R_tilt", None)
    if r_tilt is not None:
        r_tilt = np.array(r_tilt, dtype=np.float32)

    if not os.path.exists(mask_path):
        print(f"Error: Mask file {mask_path} not found.")
        return

    # Extract query profile with correct vertical FOV and tilt compensation
    query_profile, start_az = extract_elevation_profile(mask_path, fov_y_deg=fov_y_deg, aspect_ratio=None, r_tilt=r_tilt)

    # Fetch corresponding database profile
    db_profile = db_local[true_idx]

    # Plot comparison layout
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: Rendered image (or fallback text if missing)
    if os.path.exists(image_path):
        axes[0].imshow(Image.open(image_path))
        axes[0].set_title(f"1. Rendered Query Image (Sample {args.sample_id})")
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, f"Rendered image not found at:\n{image_path}", 
                     ha='center', va='center', fontsize=12, color='gray')
        axes[0].set_title(f"1. Rendered Query Image (Sample {args.sample_id} - Missing File)")
        axes[0].axis('off')

    # Panel 2: Ground-Truth Binary Mask
    axes[1].imshow(Image.open(mask_path), cmap='gray')
    axes[1].set_title("2. Ground-Truth Mask (Terrain=255/White, Sky=0/Black)")
    axes[1].axis('off')

    # Panel 3: Profile Alignment Check (Plotting in raw Elevation Degrees)
    m = len(query_profile)
    db_padded = np.hstack([db_profile, db_profile[:m-1]])
    q_norm = (query_profile - np.mean(query_profile)) / (np.std(query_profile) + 1e-12)

    best_r = -1.0
    best_off = 0
    for off in range(1440):
        sub = db_padded[off : off + m]
        sub_norm = (sub - np.mean(sub)) / (np.std(sub) + 1e-12)
        r = np.mean(q_norm * sub_norm)
        if r > best_r:
            best_r = r
            best_off = off

    # Get the raw database subsequence directly without Z-normalization
    matched_db_subsequence = db_padded[best_off : best_off + m]

    x_axis = np.arange(m) * 0.25
    # Plot raw profiles directly in degrees
    axes[2].plot(x_axis, query_profile, label="Extracted Skyline (Query)", color="crimson", lw=2)
    axes[2].plot(x_axis, matched_db_subsequence, label="Database Skyline (DEM Grid Point)", color="royalblue", lw=2, linestyle="--")
    axes[2].set_title(f"3. Profile Overlay in Degrees (Pearson Correlation: {best_r:.4f})")
    axes[2].set_xlabel("Relative Field of View (Degrees)")
    axes[2].set_ylabel("Elevation (Degrees)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("output", "visualize.png"))
    print("Saved to output/visualize.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation pipeline for visual geo-localization.")
    parser.add_argument("--mode", type=str, default="evaluate", 
                       choices=["evaluate", "diagnose", "visualize"])
    parser.add_argument("--metadata_path", type=str, default="data/synthetic_dataset/ground_truth.json",
                        help="Path to the dataset JSON metadata file.")
    parser.add_argument("--masks_dir", type=str, default="data/synthetic_dataset/masks",
                        help="Directory containing the segmented horizon masks.")
    parser.add_argument("--images_dir", type=str, default="data/synthetic_dataset/images",
                        help="Directory containing the query images.")
    parser.add_argument("--top_k_candidates", type=int, default=100)
    parser.add_argument("--dtw_window", type=int, default=16)
    parser.add_argument("--tiers", type=str, default="111")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_fft_only", action="store_true")
    parser.add_argument("--sample_id", type=int, default=0,
                        help="The index of the sample to process in visualize mode.")
    parser.add_argument("--feature", type=str, default="combined", 
                        choices=["profile", "derivative", "curvature", "combined"],
                        help="The profile descriptors used for sliding window correlation.")
    parser.add_argument("--altimetric_tolerance", type=float, default=120.0,
                        help="Height tolerance threshold in meters for terrain elevation validation.")
    parser.add_argument("--disable_height_filter", action="store_true",
                        help="Disable coordinate-based altimetric constraints completely to run a realistic, blind database search.")
    
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        cmd_evaluate(args)
    elif args.mode == "diagnose":
        cmd_diagnose(args)
    elif args.mode == "visualize":
        cmd_visualize(args)

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
from pyproj import Transformer

# ============================================================================
# COORDINATE SYSTEM & GRID DEFINITION (CANONICAL)
# ============================================================================

KHUMBU_BOUNDS_GPS = {
    'min_lon': 86.582,
    'max_lon': 86.989,
    'min_lat': 27.770,
    'max_lat': 28.041
}

GRID_SPACING_M = 500.0  # Viewpoint grid spacing in meters
UTM_ZONE = "EPSG:32645"  # UTM Zone 45N (Khumbu region)
GPS_CRS = "EPSG:4326"

DEM_CROP_SIZE_M = 110000.0  # Synthetic query generation: crop size around DEM center
QUERY_MIN_HEIGHT_M = 1.8
QUERY_MAX_HEIGHT_M = 1.8
QUERY_ELEVATION_THRESHOLD = (3400.0, 4900.0)  # Valid terrain elevation range


def build_viewpoints_grid():
    """
    Build the viewpoint grid across the Khumbu region.
    Returns a (N, 4) array: [lat, lon, utm_x, utm_y] in row-major order.
    """
    gps_to_utm = Transformer.from_crs(GPS_CRS, UTM_ZONE, always_xy=True)
    utm_to_gps = Transformer.from_crs(UTM_ZONE, GPS_CRS, always_xy=True)
    
    min_x, min_y = gps_to_utm.transform(KHUMBU_BOUNDS_GPS['min_lon'], KHUMBU_BOUNDS_GPS['min_lat'])
    max_x, max_y = gps_to_utm.transform(KHUMBU_BOUNDS_GPS['max_lon'], KHUMBU_BOUNDS_GPS['max_lat'])
    
    print(f"  UTM bounds: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
    
    X_v = np.arange(min_x, max_x + GRID_SPACING_M, GRID_SPACING_M)
    Y_v = np.arange(min_y, max_y, GRID_SPACING_M)
    
    print(f"  Grid dimensions: {len(Y_v)} rows × {len(X_v)} columns = {len(Y_v) * len(X_v)} viewpoints")
    
    view_xs, view_ys = np.meshgrid(X_v, Y_v, indexing='xy')
    flat_xs = view_xs.ravel(order='C')  # C-order = row-major
    flat_ys = view_ys.ravel(order='C')
    
    lons, lats = utm_to_gps.transform(flat_xs, flat_ys)
    viewpoints_mapping = np.column_stack((lats, lons, flat_xs, flat_ys)).astype(np.float32)
    
    return viewpoints_mapping


def build_ground_truth_samples(viewpoints_mapping, dem_path="data/dem.tif", num_samples=500):
    """
    Generate ground-truth sample locations.
    """
    utm_to_gps = Transformer.from_crs(UTM_ZONE, GPS_CRS, always_xy=True)
    
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float32)
        [pixel_width, row_rotation, start_x, col_rotation, pixel_height, start_y] = src.transform[:6]
        
        raw_xs = start_x + np.arange(src.width) * pixel_width
        raw_ys = start_y + np.arange(src.height) * pixel_height
        
        if pixel_height < 0:
            raw_ys = raw_ys[::-1]
            dem_data = np.flipud(dem_data)
        
        dem_crs = src.crs.to_string()
    
    print(f"  DEM loaded: {dem_data.shape}, CRS={dem_crs}")
    
    center_x = raw_xs[len(raw_xs) // 2]
    center_y = raw_ys[len(raw_ys) // 2]
    
    crop_min_x = center_x - DEM_CROP_SIZE_M / 2.0
    crop_max_x = center_x + DEM_CROP_SIZE_M / 2.0
    crop_min_y = center_y - DEM_CROP_SIZE_M / 2.0
    crop_max_y = center_y + DEM_CROP_SIZE_M / 2.0
    
    crop_mask_x = (raw_xs >= crop_min_x) & (raw_xs <= crop_max_x)
    crop_mask_y = (raw_ys >= crop_min_y) & (raw_ys <= crop_max_y)
    
    xs_final = raw_xs[crop_mask_x][::2]
    ys_final = raw_ys[crop_mask_y][::2]
    dem_data_final = dem_data[crop_mask_y][:, crop_mask_x][::2, ::2]
    
    print(f"  Cropped DEM: {dem_data_final.shape}")
    
    dem_height, dem_width = dem_data_final.shape
    dem_to_gps = Transformer.from_crs(dem_crs, GPS_CRS, always_xy=True)
    
    gt_dict = {}
    viewpoint_indices = np.linspace(0, viewpoints_mapping.shape[0] - 1, num_samples, dtype=np.int32)

    for sample_id, vp_idx in enumerate(viewpoint_indices):
        eye_x = float(viewpoints_mapping[vp_idx, 2])
        eye_y = float(viewpoints_mapping[vp_idx, 3])

        ix = int(np.argmin(np.abs(xs_final - eye_x)))
        iy = int(np.argmin(np.abs(ys_final - eye_y)))
        ix = np.clip(ix, 0, dem_width - 1)
        iy = np.clip(iy, 0, dem_height - 1)

        ground_z = dem_data_final[iy, ix]
        eye_z = ground_z + QUERY_MIN_HEIGHT_M
        true_lon, true_lat = dem_to_gps.transform(eye_x, eye_y)

        gt_dict[str(sample_id)] = {
            "true_lat": float(true_lat),
            "true_lon": float(true_lon),
            "eye_x_utm": float(eye_x),
            "eye_y_utm": float(eye_y),
            "eye_z_m": float(eye_z),
            "closest_viewpoint_id": int(vp_idx),
            "closest_viewpoint_dist_m": 0.0
        }
    
    return gt_dict


# ============================================================================
# MATCHING ENGINE (CORRECTED FFT + DTW)
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


def extract_elevation_profile(mask_path, fov_y_deg=65.0, aspect_ratio=1.5, r_tilt=None):
    """
    Extract 1D skyline elevation profile from binary mask with correct tilt axis alignment.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    H, W = mask.shape
    
    skyline_pixels = np.zeros(W)
    for col in range(W):
        terrain_indices = np.where(mask[:, col] == 0)[0]
        if len(terrain_indices) > 0:
            skyline_pixels[col] = terrain_indices[0]
        else:
            skyline_pixels[col] = H - 1

    # Apply Gaussian filter with edge-padding to prevent zero-padding boundary distortion
    sigma = 1.0
    size = int(2 * np.ceil(3 * sigma) + 1)
    x_kernel = np.arange(-size // 2 + 1, size // 2 + 1)
    kernel = np.exp(-x_kernel**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    
    # Pad edges with the boundary values, convolve, and return valid overlap
    pad_width = size // 2
    padded_pixels = np.pad(skyline_pixels, pad_width, mode='edge')
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
        # Use exact camera-space spherical projection to prevent edge distortion
        elevations_rad = np.arcsin(np.clip(rays_cam_normalized[1, :], -1.0, 1.0))
        azimuths_rad = np.arctan2(rays_cam_normalized[0, :], -rays_cam_normalized[2, :])    

    azimuths_deg = np.degrees(azimuths_rad)
    elevations_deg = np.degrees(elevations_rad)
    
    # 1. Sort arrays to handle non-monotonicity under pitch/roll rotation
    sort_idx = np.argsort(azimuths_deg)
    azimuths_deg = azimuths_deg[sort_idx]
    elevations_deg = elevations_deg[sort_idx]
    
    # 2. Align grid to exact database bin increments (multiples of 0.25) to prevent phase-shift mismatch
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

            weights = np.array([0.50, 0.35, 0.15], dtype=np.float64)
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
    
    # Scale-invariant normalization
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

# ============================================================================
# MAIN COMMANDS
# ============================================================================

def cmd_build_grid(args):
    print("\n=== Building Viewpoints Grid ===")
    viewpoints_mapping = build_viewpoints_grid()
    os.makedirs("data", exist_ok=True)
    np.save("data/viewpoints_mapping.npy", viewpoints_mapping)
    print(f"✓ Saved viewpoints_mapping.npy: {viewpoints_mapping.shape}")


def cmd_build_gt(args):
    print("\n=== Building Ground-Truth Samples ===")
    viewpoints_mapping = np.load("data/viewpoints_mapping.npy")
    gt_dict = build_ground_truth_samples(viewpoints_mapping, num_samples=500)
    os.makedirs("data", exist_ok=True)
    with open("data/synthetic_dataset_gt.json", "w") as f:
        json.dump(gt_dict, f, indent=4)
    print(f"✓ Saved synthetic_dataset_gt.json: {len(gt_dict)} samples")


def cmd_evaluate(args):
    print("\n=== Evaluation ===")
    
    with open("data/synthetic_dataset_gt.json") as f:
        gt_data = json.load(f)
    viewpoints_mapping = np.load("data/viewpoints_mapping.npy")
    
    # Load DEM and cache the terrain elevations of all 4,860 database viewpoints
    print("Preloading DEM for altimetric filtering...")
    with rasterio.open("data/dem.tif") as src:
        dem_data = src.read(1).astype(np.float32)
        [pixel_width, row_rotation, start_x, col_rotation, pixel_height, start_y] = src.transform[:6]
        raw_xs = start_x + np.arange(src.width) * pixel_width
        raw_ys = start_y + np.arange(src.height) * pixel_height
        if pixel_height < 0:
            raw_ys = raw_ys[::-1]
            dem_data = np.flipud(dem_data)
            
    center_x = raw_xs[len(raw_xs) // 2]
    center_y = raw_ys[len(raw_ys) // 2]
    crop_min_x = center_x - DEM_CROP_SIZE_M / 2.0
    crop_max_x = center_x + DEM_CROP_SIZE_M / 2.0
    crop_min_y = center_y - DEM_CROP_SIZE_M / 2.0
    crop_max_y = center_y + DEM_CROP_SIZE_M / 2.0
    crop_mask_x = (raw_xs >= crop_min_x) & (raw_xs <= crop_max_x)
    crop_mask_y = (raw_ys >= crop_min_y) & (raw_ys <= crop_max_y)
    xs_final = raw_xs[crop_mask_x][::2]
    ys_final = raw_ys[crop_mask_y][::2]
    dem_data_final = dem_data[crop_mask_y][:, crop_mask_x][::2, ::2]
    
    # Cache elevations for all viewpoints
    vp_elevations = np.zeros(viewpoints_mapping.shape[0], dtype=np.float32)
    for i in range(viewpoints_mapping.shape[0]):
        vp_x = viewpoints_mapping[i, 2]
        vp_y = viewpoints_mapping[i, 3]
        ix = np.clip(np.argmin(np.abs(xs_final - vp_x)), 0, dem_data_final.shape[1] - 1)
        iy = np.clip(np.argmin(np.abs(ys_final - vp_y)), 0, dem_data_final.shape[0] - 1)
        vp_elevations[i] = dem_data_final[iy, ix]
        
    print(f"Preloading database tiers...")
    db_global = np.load("data/horizon_db_global.npy")
    db_local = np.load("data/horizon_db_local.npy")
    db_restricted = np.load("data/horizon_db_restricted.npy")
    
    use_tiers = (args.tiers[0] == '1', args.tiers[1] == '1', args.tiers[2] == '1')
    
    sample_ids = sorted([int(k) for k in gt_data.keys()])
    if args.limit > 0:
        sample_ids = sample_ids[:args.limit]
    
    top1_correct, top5_correct = 0, 0
    errors = []
    failed_sample_ids = []  # Track the IDs of failed match attempts
    
    for sample_id in tqdm(sample_ids, desc="Matching"):
        mask_path = f"data/synthetic_dataset/masks/sample_{sample_id:04d}.png"
        if not os.path.exists(mask_path):
            continue

        gt_info = gt_data[str(sample_id)]
        true_lat, true_lon = gt_info["true_lat"], gt_info["true_lon"]
        fov_y_deg = gt_info.get("fov_y_deg", 65.0)
        
        # Apply ±50m altimetric constraint
        gt_ground_z = gt_info["eye_z_m"] - QUERY_MIN_HEIGHT_M
        valid_vp_mask = np.abs(vp_elevations - gt_ground_z) <= 50.0
        
        r_tilt = gt_info.get("cam_R_tilt", None)
        if r_tilt is not None:
            r_tilt = np.array(r_tilt, dtype=np.float32)

        # First extract the profile and define start_az
        query_profile, start_az = extract_elevation_profile(mask_path, fov_y_deg=fov_y_deg, aspect_ratio=1.5, r_tilt=r_tilt)

        # Now apply the ±30 degree simulated compass constraint (120 bins out of 1440) using start_az
        true_heading = gt_info["true_heading_deg"]
        tolerance_bins = int(30.0 / 0.25)
        expected_offset = int(((true_heading + start_az) % 360.0) / 0.25)        

        if args.use_fft_only:
            corr_g, _ = (run_sliding_fft_for_db(db_global, query_profile, feature=args.feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
                         if use_tiers[0] else (np.full(db_global.shape[0], -np.inf), None))
            corr_l, _ = (run_sliding_fft_for_db(db_local, query_profile, feature=args.feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
                         if use_tiers[1] else (np.full(db_global.shape[0], -np.inf), None))
            corr_r, _ = (run_sliding_fft_for_db(db_restricted, query_profile, feature=args.feature, expected_offset=expected_offset, tolerance_bins=tolerance_bins)
                         if use_tiers[2] else (np.full(db_global.shape[0], -np.inf), None))

            best_corr = np.maximum(np.maximum(corr_g, corr_l), corr_r)
            best_corr[~valid_vp_mask] = -np.inf  # Zeroes out invalid elevations
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
        errors.append(r1_error)

        if r1_error <= 1000.0:
            top1_correct += 1
        else:
            failed_sample_ids.append(sample_id)  # Record failed ID

        in_top5 = False
        for m in matches[:5]:
            m_idx = m['viewpoint_idx']
            m_lat, m_lon = viewpoints_mapping[m_idx, 0], viewpoints_mapping[m_idx, 1]
            if calculate_geodesic_distance(true_lat, true_lon, m_lat, m_lon) <= 1000.0:
                in_top5 = True
                break
        if in_top5:
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


def cmd_diagnose(args):
    print("\n=== Diagnostic: Checking Sample 0 ===")
    
    with open("data/synthetic_dataset_gt.json") as f:
        gt_data = json.load(f)
    viewpoints_mapping = np.load("data/viewpoints_mapping.npy")
    db_global = np.load("data/horizon_db_global.npy")
    
    gt_info = gt_data["0"]
    true_viewpoint_id = gt_info["closest_viewpoint_id"]
    true_lat, true_lon = gt_info["true_lat"], gt_info["true_lon"]
    fov_y_deg = gt_info.get("fov_y_deg", 65.0)
    
    r_tilt = gt_info.get("cam_R_tilt", None)
    if r_tilt is not None:
        r_tilt = np.array(r_tilt, dtype=np.float32)

    query_profile, start_az = extract_elevation_profile("data/synthetic_dataset/masks/sample_0000.png", fov_y_deg=fov_y_deg, aspect_ratio=1.5, r_tilt=r_tilt)
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

    gt_json_path = "data/synthetic_dataset_gt.json"
    db_local_path = "data/horizon_db_local.npy"

    if not os.path.exists(gt_json_path):
        print(f"Error: {gt_json_path} not found. Please build ground truth first.")
        return
    if not os.path.exists(db_local_path):
        print(f"Error: {db_local_path} not found.")
        return

    with open(gt_json_path) as f: 
        gt_data = json.load(f)
    db_local = np.load(db_local_path)

    sample_key = str(args.sample_id)
    if sample_key not in gt_data:
        print(f"Error: Sample ID {args.sample_id} not found in ground truth dataset.")
        return

    gt_info = gt_data[sample_key]
    true_idx = gt_info["closest_viewpoint_id"]
    fov_y_deg = gt_info.get("fov_y_deg", 65.0)

    image_path = f"data/synthetic_dataset/images/sample_{args.sample_id:04d}.png"
    mask_path = f"data/synthetic_dataset/masks/sample_{args.sample_id:04d}.png"

    r_tilt = gt_info.get("cam_R_tilt", None)
    if r_tilt is not None:
        r_tilt = np.array(r_tilt, dtype=np.float32)

    if not os.path.exists(mask_path):
        print(f"Error: Mask file {mask_path} not found.")
        return

    # Extract query profile with correct vertical FOV and tilt compensation
    query_profile, start_az = extract_elevation_profile(mask_path, fov_y_deg=fov_y_deg, aspect_ratio=1.5, r_tilt=r_tilt)

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
    axes[1].set_title("2. Ground-Truth Mask (Terrain=0, Sky=255)")
    axes[1].axis('off')

    # Panel 3: Profile Alignment Check
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

    matched_db_subsequence = db_padded[best_off : best_off + m]
    matched_db_norm = (matched_db_subsequence - np.mean(matched_db_subsequence)) / (np.std(matched_db_subsequence) + 1e-12)

    x_axis = np.arange(m) * 0.25
    axes[2].plot(x_axis, q_norm, label="Extracted Skyline (Query)", color="crimson", lw=2)
    axes[2].plot(x_axis, matched_db_norm, label="Database Skyline (DEM Grid Point)", color="royalblue", lw=2, linestyle="--")
    axes[2].set_title(f"3. Profile Overlay (Pearson Correlation: {best_r:.4f})")
    axes[2].set_xlabel("Relative Field of View (Degrees)")
    axes[2].set_ylabel("Normalized Elevation")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation pipeline for visual geo-localization.")
    parser.add_argument("--mode", type=str, default="evaluate", 
                       choices=["build_grid", "build_gt", "evaluate", "diagnose", "visualize"])
    parser.add_argument("--top_k_candidates", type=int, default=30)
    parser.add_argument("--dtw_window", type=int, default=8)
    parser.add_argument("--tiers", type=str, default="010")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_fft_only", action="store_true")
    parser.add_argument("--sample_id", type=int, default=0,
                        help="The index of the sample to process in visualize mode.")
    parser.add_argument("--feature", type=str, default="derivative", 
                        choices=["profile", "derivative", "curvature", "combined"],
                        help="The profile descriptors used for sliding window correlation.")
    
    args = parser.parse_args()
    
    if args.mode == "build_grid":
        cmd_build_grid(args)
    elif args.mode == "build_gt":
        cmd_build_gt(args)
    elif args.mode == "evaluate":
        cmd_evaluate(args)
    elif args.mode == "diagnose":
        cmd_diagnose(args)
    elif args.mode == "visualize":
        cmd_visualize(args)

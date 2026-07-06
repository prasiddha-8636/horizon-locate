"""Core matching utilizing fastdtw, scipy.stats, and scipy.signal against a single database."""
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.signal import correlate
from scipy.stats import zscore
from fastdtw import fastdtw


def _feature_bundle(profile):
    """Extract and z-score normalize features using standard scipy.stats utilities."""
    profile = np.asarray(profile, dtype=np.float64)
    
    value = zscore(profile)
    d1 = zscore(np.gradient(value))
    d2 = zscore(np.gradient(d1))
    return value, d1, d2


def fft_prefilter(db_matrix, query_profile, bin_deg,
                  weights=(0.33, 0.33, 0.33),
                  expected_offset_deg=None, tolerance_deg=None):
    """
    Scores database horizons against query_profile using cross-correlation.
    """
    n_horizons, horizon_length = db_matrix.shape
    profile_length = len(query_profile)
    
    query_val, query_d1, query_d2 = _feature_bundle(query_profile)
    
    offset_mask = None
    if expected_offset_deg is not None:
        bins = np.arange(horizon_length)
        expected_bin = (expected_offset_deg / bin_deg) % horizon_length
        tolerance_bins = tolerance_deg / bin_deg
        circular_dist = np.minimum(
            (bins - expected_bin) % horizon_length,
            (expected_bin - bins) % horizon_length
        )
        offset_mask = circular_dist <= tolerance_bins

    best_corr = np.zeros(n_horizons)
    best_offset = np.zeros(n_horizons, dtype=np.int32)

    for i in range(n_horizons):
        horizon = db_matrix[i].astype(np.float64)
        h_val, h_d1, h_d2 = _feature_bundle(horizon)

        pad_val = np.hstack([h_val, h_val[:profile_length - 1]])
        pad_d1 = np.hstack([h_d1, h_d1[:profile_length - 1]])
        pad_d2 = np.hstack([h_d2, h_d2[:profile_length - 1]])

        corr_val = correlate(pad_val, query_val, mode="valid")
        corr_d1 = correlate(pad_d1, query_d1, mode="valid")
        corr_d2 = correlate(pad_d2, query_d2, mode="valid")

        corr = weights[0] * corr_val + weights[1] * corr_d1 + weights[2] * corr_d2
        
        if offset_mask is not None:
            corr = np.where(offset_mask[:len(corr)], corr, -np.inf)

        best_bin = int(np.argmax(corr))
        best_offset[i] = best_bin
        best_corr[i] = corr[best_bin]

    return best_corr, best_offset


def match_query(db_matrix, bin_deg, query_profile, top_k=30, dtw_window=15,
                valid_vp_mask=None, expected_offset_deg=None,
                tolerance_deg=20.0, weights=(0.33, 0.33, 0.33),
                spatial_stride=5):
    """
    Two-stage matching directly on a single database matrix.
    Uses spatial_stride to run a fast coarse-spatial search, then refines locally.
    """
    profile_length = len(query_profile)
    n_viewpoints = db_matrix.shape[0]
    
    # 1. Coarse Spatial Search: Evaluate only every Nth viewpoint
    coarse_indices = np.arange(0, n_viewpoints, spatial_stride)
    coarse_db = db_matrix[coarse_indices]
    
    coarse_vp_mask = None
    if valid_vp_mask is not None:
        coarse_vp_mask = valid_vp_mask[coarse_indices]

    corr_scores, offsets = fft_prefilter(
        coarse_db, query_profile, bin_deg,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
        weights=weights
    )

    if coarse_vp_mask is not None:
        corr_scores = np.where(coarse_vp_mask, corr_scores, -np.inf)

    # 2. Local Fine Refinement: Gather neighbors of the top coarse candidates
    top_coarse_indices = np.argsort(-corr_scores)[:5]
    fine_candidate_set = set()
    
    for idx in top_coarse_indices:
        if corr_scores[idx] == -np.inf:
            continue
        global_coarse_idx = coarse_indices[idx]
        
        local_range = np.arange(
            max(0, global_coarse_idx - spatial_stride),
            min(n_viewpoints, global_coarse_idx + spatial_stride + 1)
        )
        fine_candidate_set.update(local_range)
        
    fine_indices = np.array(sorted(fine_candidate_set), dtype=np.int32)
    
    # Run full-resolution matching ONLY on this fine subset
    fine_db = db_matrix[fine_indices]
    fine_vp_mask = None
    if valid_vp_mask is not None:
        fine_vp_mask = valid_vp_mask[fine_indices]
        
    fine_corr, fine_offsets = fft_prefilter(
        fine_db, query_profile, bin_deg,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
        weights=weights
    )
    
    if fine_vp_mask is not None:
        fine_corr = np.where(fine_vp_mask, fine_corr, -np.inf)
        
    query_val, query_d1, _ = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    # 3. DTW Sequence Alignment on top-k fine candidates
    candidates = []
    top_k_fine = np.argsort(-fine_corr)[:top_k]

    for idx in top_k_fine:
        if fine_corr[idx] == -np.inf:
            continue

        global_vp_idx = fine_indices[idx]
        horizon = db_matrix[global_vp_idx]
        offset = fine_offsets[idx]

        windowed = horizon[np.arange(offset, offset + profile_length) % len(horizon)]
        db_val, db_d1, _ = _feature_bundle(windowed)
        db_features = np.vstack([db_val, db_d1]).T

        # DTW alignment via standard fastdtw library
        dtw_cost, _ = fastdtw(query_features, db_features, radius=dtw_window, dist=euclidean)

        candidates.append({
            "viewpoint_idx": int(global_vp_idx),
            "fft_corr": float(fine_corr[idx]),
            "dtw_cost": float(dtw_cost),
            "offset_bin": int(offset),
        })

    candidates.sort(key=lambda c: c["dtw_cost"] / max(0.01, c["fft_corr"]))
    return candidates


def finalize_matches(result_view, query_profile, dtw_window):
    """Refinement helper using fastdtw on streamed candidate row-views."""
    query_val, query_d1, _ = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    candidates = []
    for corr, _, vp_idx, offset, horizon_window in result_view.items():
        db_val, db_d1, _ = _feature_bundle(horizon_window)
        db_features = np.vstack([db_val, db_d1]).T

        dtw_cost, _ = fastdtw(query_features, db_features, radius=dtw_window, dist=euclidean)

        candidates.append({
            "viewpoint_idx": int(vp_idx),
            "fft_corr": float(corr),
            "dtw_cost": float(dtw_cost),
            "offset_bin": int(offset),
        })

    candidates.sort(key=lambda c: c["dtw_cost"] / max(0.01, c["fft_corr"]))
    return candidates
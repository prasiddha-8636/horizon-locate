"""Runs batch evaluation of terrain mask profiles against the spatial database."""
import os
import json
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic
from tqdm.auto import tqdm

from src.matching import match_query, fft_prefilter, finalize_matches


def _stream_horizon_chunks(parquet_path, chunk_rows=4000):
    """Streams database horizon curves in small, memory-safe row-chunks."""
    pf = pq.ParquetFile(parquet_path)
    bin_deg = None
    global_offset = 0
    
    for batch in pf.iter_batches(batch_size=chunk_rows, columns=["raw_horizon_deg"]):
        df_batch = batch.to_pandas()
        
        if bin_deg is None:
            horizon_grid = np.asarray(df_batch["raw_horizon_deg"].iloc[0], dtype=np.float64)
            bin_deg = 360.0 / len(horizon_grid)
        
        chunk_matrix = np.stack(df_batch["raw_horizon_deg"].to_numpy()).astype(np.float32)
        
        yield chunk_matrix, bin_deg, global_offset
        global_offset += len(df_batch)


def _fetch_rows(parquet_path, row_indices):
    """Fetches only the specific requested row indices from disk to minimize RAM."""
    pf = pq.ParquetFile(parquet_path)
    needed_indices = set(int(i) for i in row_indices)
    index_to_location = {}
    cumulative_rows = 0
    
    for row_group_id in range(pf.num_row_groups):
        num_rows_in_group = pf.metadata.row_group(row_group_id).num_rows
        
        for row_index in list(needed_indices):
            if cumulative_rows <= row_index < cumulative_rows + num_rows_in_group:
                position_in_group = row_index - cumulative_rows
                index_to_location[row_index] = (row_group_id, position_in_group)
                needed_indices.discard(row_index)
        
        cumulative_rows += num_rows_in_group
        if not needed_indices:
            break
            
    result = {}
    rows_by_group = {}
    for row_index, (rg_id, position) in index_to_location.items():
        if rg_id not in rows_by_group:
            rows_by_group[rg_id] = []
        rows_by_group[rg_id].append((position, row_index))
        
    for row_group_id, row_positions in rows_by_group.items():
        df_group = pf.read_row_group(row_group_id, columns=["raw_horizon_deg"]).to_pandas()
        horizons = df_group["raw_horizon_deg"].to_numpy()
        
        for position, row_index in row_positions:
            result[row_index] = np.asarray(horizons[position], dtype=np.float32)
            
    return result


class _RowView:
    """Wrapper class to format fetched candidate rows for DTW alignment."""
    def __init__(self, rows):
        self.rows = rows
    def items(self):
        return sorted(self.rows, key=lambda x: -x[0])


def run_evaluation(ground_truth_path, db_path, masks_dir,
                   height_tolerance_m=200.0, top_k=30, dtw_window=15,
                   correct_dist_m=500.0, limit=0,
                   use_altimeter=True, use_compass=True, compass_tolerance_deg=20.0,
                   bbox=None, weights=(0.33, 0.33, 0.33),
                   min_std_deg=1.5, min_max_elev_deg=1.0,
                   chunk_rows=4000, sample_batch_size=8, spatial_stride=5):
    """
    Evaluates matches on a single database using memory-bounded chunk-streaming.
    Keeps RAM usage extremely low by streaming the database on-the-fly.
    """
    with open(ground_truth_path) as f:
        gt_data = json.load(f)
        
    sample_ids = list(gt_data.keys())
    if limit > 0:
        sample_ids = sample_ids[:limit]
        
    # Load ONLY coordinate and elevation columns (takes ~2 MB RAM)
    metadata = pd.read_parquet(db_path, columns=["lon", "lat", "elevation_m"])
    lon = metadata["lon"].to_numpy()
    lat = metadata["lat"].to_numpy()
    elev_m = metadata["elevation_m"].to_numpy()
    n_vp = len(metadata)
    del metadata
    gc.collect()

    # Determine angular resolution dynamically from raw_horizon_deg length
    pf = pq.ParquetFile(db_path)
    first_batch = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    horizon_grid = np.asarray(first_batch.to_pandas()["raw_horizon_deg"].iloc[0], dtype=np.float64)
    bin_deg = 360.0 / len(horizon_grid)

    # Filter to only queries that have generated masks (checking both naming patterns)
    valid_sids = []
    for sid in sample_ids:
        mask_path_fixed = os.path.join(masks_dir, f"sample_{int(sid):04d}.png")
        mask_path_raw = os.path.join(masks_dir, f"{sid}.png")
        if os.path.exists(mask_path_fixed) or os.path.exists(mask_path_raw):
            valid_sids.append(sid)
            
    if not valid_sids:
        print(f"[Warning] No generated sky mask files found in: {masks_dir}")
        summary = {
            "n_samples": 0,
            "skipped_flat": len(sample_ids),
            "top1_acc_500m": 0.0,
            "top1_acc_100m": 0.0,
            "top5_acc_500m": 0.0,
            "median_error_m": float('nan'),
            "mean_error_m": float('nan'),
        }
        return pd.DataFrame(), summary
        
    all_results = []
    total_batches = max(1, (len(valid_sids) + sample_batch_size - 1) // sample_batch_size)
    
    # Process queries in standard Python loop with clear print logs
    for batch_idx in range(0, len(valid_sids), sample_batch_size):
        batch_sids = valid_sids[batch_idx:batch_idx + sample_batch_size]
        batch_queries = {}
        
        current_batch_num = (batch_idx // sample_batch_size) + 1
        print(f"\n[Batch {current_batch_num}/{total_batches}] Processing queries: {', '.join(batch_sids)}")
        
        for sample_id in batch_sids:
            gt_info = gt_data[sample_id]
            mask_path = os.path.join(masks_dir, f"sample_{int(sample_id):04d}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(masks_dir, f"{sample_id}.png")
            
            try:
                fov = gt_info.get("fov_y_deg", 65.0)
                r_tilt = np.array(gt_info["cam_R_tilt"]) if gt_info.get("cam_R_tilt") else None
                profile, azimuth_start = extract_elevation_profile(
                    mask_path, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=bin_deg
                )
                
                is_valid, _ = is_profile_applicable(
                    profile, min_std_deg=min_std_deg, min_max_elev_deg=min_max_elev_deg
                )
                if not is_valid:
                    continue
                    
                expected_offset = None
                if use_compass and "true_heading_deg" in gt_info:
                    expected_offset = (gt_info["true_heading_deg"] + azimuth_start) % 360.0
                    
                gt_elevation = None
                if use_altimeter and "eye_z_m" in gt_info:
                    gt_elevation = gt_info["eye_z_m"] - gt_info.get("query_height_m", 1.8)
                    
                batch_queries[sample_id] = {
                    "gt_info": gt_info,
                    "profile": profile,
                    "expected_offset": expected_offset,
                    "gt_elevation": gt_elevation,
                    "best_corr": np.full(n_vp, -np.inf, dtype=np.float32),
                    "best_offset": np.zeros(n_vp, dtype=np.int32),
                }
            except Exception:
                continue
                
        if not batch_queries:
            continue
            
        # == PASS 1: Stream the database in small chunks, accumulate cross-correlations ==
        chunk_iter = _stream_horizon_chunks(db_path, chunk_rows)
        total_chunks = (n_vp + chunk_rows - 1) // chunk_rows
        
        # High-visibility active crawler showing database scan progress in real-time
        desc_msg = f"  Batch {current_batch_num}/{total_batches}: Scanning DB"
        for chunk_matrix, _, chunk_start in tqdm(chunk_iter, total=total_chunks, desc=desc_msg):
            chunk_end = chunk_start + chunk_matrix.shape[0]
            chunk_slice = slice(chunk_start, chunk_end)
            chunk_elevations = elev_m[chunk_slice]
            
            # Apply spatial_stride decimation directly inside the chunk stream
            stride_indices = np.arange(0, chunk_matrix.shape[0], spatial_stride)
            stride_matrix = chunk_matrix[stride_indices]
            stride_elevations = chunk_elevations[stride_indices]
            
            for sample_id, query_state in batch_queries.items():
                corr_scores, offsets = fft_prefilter(
                    stride_matrix, query_state["profile"], bin_deg,
                    weights=weights,
                    expected_offset_deg=query_state["expected_offset"],
                    tolerance_deg=compass_tolerance_deg
                )
                
                if query_state["gt_elevation"] is not None:
                    elevation_valid = np.abs(stride_elevations - query_state["gt_elevation"]) <= height_tolerance_m
                    corr_scores = np.where(elevation_valid, corr_scores, -np.inf)
                    
                # Map the decimated indices back to global database coordinates
                global_stride_indices = np.arange(chunk_start, chunk_end, spatial_stride)
                current_best = query_state["best_corr"][global_stride_indices]
                is_better = corr_scores > current_best
                
                if np.any(is_better):
                    current_best[is_better] = corr_scores[is_better]
                    query_state["best_offset"][global_stride_indices[is_better]] = offsets[is_better]
                    
            del chunk_matrix
            gc.collect()
            
        # == PASS 2: Local fine refinement and DTW on top candidates ==
        for sample_id, query_state in batch_queries.items():
            valid_vps = np.where(query_state["best_corr"] > -np.inf)[0]
            if len(valid_vps) == 0:
                continue
                
            # Get top coarse candidates
            top_coarse_vps = valid_vps[np.argsort(-query_state["best_corr"][valid_vps])][:5]
            
            # Expand to include local fine neighbors
            fine_candidate_set = set()
            for coarse_idx in top_coarse_vps:
                local_range = np.arange(
                    max(0, coarse_idx - spatial_stride),
                    min(n_vp, coarse_idx + spatial_stride + 1)
                )
                fine_candidate_set.update(local_range)
            fine_indices = np.array(sorted(fine_candidate_set), dtype=np.int32)
            
            # Fetch horizon curves ONLY for this tiny subset
            fetched_horizons = _fetch_rows(db_path, fine_indices)
            
            result_rows = []
            profile_length = len(query_state["profile"])
            
            for vp_idx in fine_indices:
                horizon_curve = fetched_horizons[vp_idx]
                horizon_length = len(horizon_curve)
                offset_bin = int(query_state["best_offset"][vp_idx])
                
                # Dynamic local cross-correlation score check
                windowed = horizon_curve[np.arange(offset_bin, offset_bin + profile_length) % horizon_length]
                
                result_rows.append((
                    float(query_state["best_corr"][vp_idx]),
                    "main",
                    int(vp_idx),
                    offset_bin,
                    windowed
                ))
                
            # Perform DTW alignment
            matches = finalize_matches(_RowView(result_rows), query_state["profile"], dtw_window)
            if not matches:
                continue
                
            gt_info = query_state["gt_info"]
            true_lat, true_lon = gt_info["true_lat"], gt_info["true_lon"]
            
            best_match = matches[0]
            matched_lat = lat[best_match["viewpoint_idx"]]
            matched_lon = lon[best_match["viewpoint_idx"]]
            
            error_m = geodesic((true_lat, true_lon), (matched_lat, matched_lon)).meters
            
            top5_ok = False
            for match in matches[:5]:
                match_lat = lat[match["viewpoint_idx"]]
                match_lon = lon[match["viewpoint_idx"]]
                match_err = geodesic((true_lat, true_lon), (match_lat, match_lon)).meters
                if match_err <= correct_dist_m:
                    top5_ok = True
                    break
                    
            all_results.append({
                "sample_id": sample_id,
                "error_m": error_m,
                "top1_ok": error_m <= correct_dist_m,
                "top5_ok": top5_ok,
            })
            
        del batch_queries
        gc.collect()
        
    df = pd.DataFrame(all_results)
    if df.empty:
        summary = {
            "n_samples": 0,
            "skipped_flat": len(valid_sids),
            "top1_acc_500m": 0.0,
            "top1_acc_100m": 0.0,
            "top5_acc_500m": 0.0,
            "median_error_m": float('nan'),
            "mean_error_m": float('nan'),
        }
        return df, summary
        
    errors = df["error_m"].to_numpy()
    summary = {
        "n_samples": len(df),
        "skipped_flat": len(valid_sids) - len(df),
        "top1_acc_500m": 100.0 * np.mean(errors <= 500.0),
        "top1_acc_100m": 100.0 * np.mean(errors <= 100.0),
        "top5_acc_500m": 100.0 * df["top5_ok"].mean(),
        "median_error_m": float(np.median(errors)),
        "mean_error_m": float(np.mean(errors)),
    }
    return df, summary


# Simple helper import needed by the evaluator
from src.query_profile import extract_elevation_profile, is_profile_applicable
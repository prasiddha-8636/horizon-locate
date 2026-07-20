"""Runs batch evaluation of terrain mask profiles against the spatial database."""

import os
import json
import gc
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic
from tqdm.auto import tqdm

from src.matching import (
    fft_prefilter,
    finalize_matches,
    feature_bundle_matrix,
    ncc_scores,
)
from src.horizon_format import decode_horizon_uint8, decode_horizon_column
from src.query_profile import extract_elevation_profile, is_profile_applicable


def _stream_horizon_chunks(parquet_path, chunk_rows=4000):
    """Streams database horizon curves in small, memory-safe row-chunks."""
    pf = pq.ParquetFile(parquet_path)
    bin_deg = None
    global_offset = 0

    for batch in pf.iter_batches(batch_size=chunk_rows, columns=["raw_horizon_deg"]):
        df_batch = batch.to_pandas()

        if bin_deg is None:
            horizon_grid = np.asarray(
                df_batch["raw_horizon_deg"].iloc[0], dtype=np.float64
            )
            bin_deg = 360.0 / len(horizon_grid)

        chunk_matrix = decode_horizon_column(df_batch["raw_horizon_deg"].to_numpy())

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
        df_group = pf.read_row_group(
            row_group_id, columns=["raw_horizon_deg"]
        ).to_pandas()
        horizons = df_group["raw_horizon_deg"].to_numpy()

        for position, row_index in row_positions:
            result[row_index] = decode_horizon_uint8(horizons[position])

    return result


class _RowView:
    """Wrapper class to format fetched candidate rows for DTW alignment."""

    def __init__(self, rows):
        self.rows = rows

    def items(self):
        return sorted(self.rows, key=lambda x: -x[0])

    def __iter__(self):
        return iter(self.items())


def load_ground_truth(ground_truth_path, limit=0):
    """Load ground-truth metadata and apply optional sample limit."""
    with open(ground_truth_path) as f:
        gt_data = json.load(f)

    sample_ids = list(gt_data.keys())
    if limit > 0:
        sample_ids = sample_ids[:limit]

    return gt_data, sample_ids


def load_db_metadata(db_path):
    """Load only lightweight DB columns needed for geospatial evaluation."""
    metadata = pd.read_parquet(db_path, columns=["lon", "lat", "elevation_m"])
    lon = metadata["lon"].to_numpy()
    lat = metadata["lat"].to_numpy()
    elev_m = metadata["elevation_m"].to_numpy()
    n_vp = len(metadata)
    del metadata
    gc.collect()
    return lon, lat, elev_m, n_vp


def infer_bin_size_deg(db_path):
    """Infer angular bin resolution from DB horizon array length."""
    pf = pq.ParquetFile(db_path)
    first_batch = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    horizon_grid = np.asarray(
        first_batch.to_pandas()["raw_horizon_deg"].iloc[0], dtype=np.float64
    )
    return 360.0 / len(horizon_grid)


def filter_samples_with_masks(sample_ids, masks_dir):
    """Keep only sample IDs that have an existing predicted mask image."""
    valid_sids = []
    for sid in sample_ids:
        mask_path_raw = os.path.join(masks_dir, f"{sid}.png")
        if os.path.exists(mask_path_raw):
            valid_sids.append(sid)
            continue
        try:
            mask_path_fixed = os.path.join(masks_dir, f"sample_{int(sid):04d}.png")
            if os.path.exists(mask_path_fixed):
                valid_sids.append(sid)
        except (ValueError, TypeError):
            pass
    return valid_sids


def _resolve_mask_path(masks_dir, sample_id):
    """Return whichever naming convention exists for the sample mask."""
    mask_path_raw = os.path.join(masks_dir, f"{sample_id}.png")
    if os.path.exists(mask_path_raw):
        return mask_path_raw
    try:
        mask_path_fixed = os.path.join(masks_dir, f"sample_{int(sample_id):04d}.png")
        if os.path.exists(mask_path_fixed):
            return mask_path_fixed
    except (ValueError, TypeError):
        pass
    return mask_path_raw


def build_batch_queries(
    batch_sids,
    gt_data,
    masks_dir,
    bin_deg,
    n_vp,
    min_std_deg=1.5,
    min_max_elev_deg=1.0,
    use_altimeter=True,
    use_compass=True,
):
    """Create per-query states (profile, constraints, coarse-score buffers)."""
    batch_queries = {}
    for sample_id in batch_sids:
        gt_info = gt_data[sample_id]
        mask_path = _resolve_mask_path(masks_dir, sample_id)

        try:
            fov = gt_info.get("fov_y_deg", 65.0)
            r_tilt = (
                np.array(gt_info["cam_R_tilt"]) if gt_info.get("cam_R_tilt") else None
            )
            pr = extract_elevation_profile(
                mask_path, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=bin_deg
            )
            if not pr["ok"]:
                continue
            profile = pr["profile"]
            azimuth_start = pr["start_az"]

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

    return batch_queries


def run_batch_coarse_scan(
    batch_queries,
    db_path,
    elev_m,
    n_vp,
    chunk_rows=4000,
    spatial_stride=12,
    weights=(0.5, 0.5),
    compass_tolerance_deg=20.0,
    height_tolerance_m=200.0,
    progress_desc="Scanning DB",
):
    """Run chunked FFT prefilter over full DB and update per-query best coarse hits."""
    chunk_iter = _stream_horizon_chunks(db_path, chunk_rows)
    total_chunks = (n_vp + chunk_rows - 1) // chunk_rows

    for chunk_matrix, bin_deg, chunk_start in tqdm(
        chunk_iter, total=total_chunks, desc=progress_desc
    ):
        chunk_end = chunk_start + chunk_matrix.shape[0]
        chunk_slice = slice(chunk_start, chunk_end)
        chunk_elevations = elev_m[chunk_slice]

        stride_indices = np.arange(0, chunk_matrix.shape[0], spatial_stride)
        stride_matrix = chunk_matrix[stride_indices]
        stride_elevations = chunk_elevations[stride_indices]

        # DB features are shared across all queries in the batch
        db_val, db_d1 = feature_bundle_matrix(stride_matrix)
        global_stride_indices = np.arange(chunk_start, chunk_end, spatial_stride)

        for query_state in batch_queries.values():
            corr_scores, offsets = ncc_scores(
                db_val,
                db_d1,
                query_state["profile"],
                bin_deg,
                weights=weights,
                expected_offset_deg=query_state["expected_offset"],
                tolerance_deg=compass_tolerance_deg,
            )

            if query_state["gt_elevation"] is not None:
                elevation_valid = (
                    np.abs(stride_elevations - query_state["gt_elevation"])
                    <= height_tolerance_m
                )
                corr_scores = np.where(elevation_valid, corr_scores, -np.inf)

            is_better = corr_scores > query_state["best_corr"][global_stride_indices]

            if np.any(is_better):
                vp_indices = global_stride_indices[is_better]
                query_state["best_corr"][vp_indices] = corr_scores[is_better]
                query_state["best_offset"][vp_indices] = offsets[is_better]

        del chunk_matrix
        gc.collect()


def refine_query_with_dtw(
    query_state,
    db_path,
    spatial_stride,
    n_vp,
    lat,
    lon,
    dtw_window=15,
    correct_dist_m=500.0,
):
    """Refine one query from coarse hits into final DTW-ranked matches."""
    valid_vps = np.where(query_state["best_corr"] > -np.inf)[0]
    if len(valid_vps) == 0:
        return None

    top_coarse_vps = valid_vps[np.argsort(-query_state["best_corr"][valid_vps])][:5]

    fine_candidate_set = set()
    for coarse_idx in top_coarse_vps:
        local_range = np.arange(
            max(0, coarse_idx - spatial_stride),
            min(n_vp, coarse_idx + spatial_stride + 1),
        )
        fine_candidate_set.update(local_range)
    fine_indices = np.array(sorted(fine_candidate_set), dtype=np.int32)

    fetched_horizons = _fetch_rows(db_path, fine_indices)
    result_rows = []
    profile_length = len(query_state["profile"])

    for vp_idx in fine_indices:
        horizon_curve = fetched_horizons[vp_idx]
        horizon_length = len(horizon_curve)
        offset_bin = int(query_state["best_offset"][vp_idx])
        windowed = horizon_curve[
            np.arange(offset_bin, offset_bin + profile_length) % horizon_length
        ]
        result_rows.append(
            (
                float(query_state["best_corr"][vp_idx]),
                "main",
                int(vp_idx),
                offset_bin,
                windowed,
            )
        )

    matches = finalize_matches(
        _RowView(result_rows), query_state["profile"], dtw_window
    )
    if not matches:
        return None

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

    return {
        "sample_id": gt_info.get("sample_id", None),
        "error_m": error_m,
        "top1_ok": error_m <= correct_dist_m,
        "top5_ok": top5_ok,
    }


def summarize_results(df_results, valid_sample_count):
    """Compute aggregate metrics from per-query evaluation rows."""
    return summarize_results_at_thresholds(
        df_results,
        valid_sample_count,
        thresholds_m=[50.0, 100.0, 200.0, 500.0, 1000.0],
    )


def summarize_results_at_thresholds(df_results, valid_sample_count, thresholds_m=None):
    """Compute aggregate metrics at multiple error thresholds."""
    if thresholds_m is None:
        thresholds_m = [50.0, 100.0, 200.0, 500.0, 1000.0]
    if df_results.empty:
        summary = {
            "n_samples": 0,
            "skipped_flat": valid_sample_count,
            "median_error_m": float("nan"),
            "mean_error_m": float("nan"),
        }
        for t in thresholds_m:
            summary[f"top1_acc_{int(t)}m"] = 0.0
        return summary
    errors = df_results["error_m"].to_numpy()
    summary = {
        "n_samples": len(df_results),
        "skipped_flat": valid_sample_count - len(df_results),
        "median_error_m": float(np.median(errors)),
        "mean_error_m": float(np.mean(errors)),
    }
    for t in thresholds_m:
        summary[f"top1_acc_{int(t)}m"] = 100.0 * np.mean(errors <= t)
    if "top5_ok" in df_results.columns:
        summary["top5_acc_500m"] = 100.0 * df_results["top5_ok"].mean()
    return summary


def run_evaluation(
    ground_truth_path,
    db_path,
    masks_dir,
    height_tolerance_m=200.0,
    top_k=30,
    dtw_window=15,
    correct_dist_m=500.0,
    limit=0,
    use_altimeter=True,
    use_compass=True,
    compass_tolerance_deg=20.0,
    bbox=None,
    weights=(0.5, 0.5),
    min_std_deg=1.5,
    min_max_elev_deg=1.0,
    chunk_rows=4000,
    sample_batch_size=8,
    spatial_stride=12,
    checkpoint_dir=None,
):
    """Backward-compatible orchestrator with optional batch-level checkpointing."""
    gt_data, sample_ids = load_ground_truth(ground_truth_path, limit=limit)
    lon, lat, elev_m, n_vp = load_db_metadata(db_path)
    bin_deg = infer_bin_size_deg(db_path)
    valid_sids = filter_samples_with_masks(sample_ids, masks_dir)

    if not valid_sids:
        print(f"[Warning] No generated sky mask files found in: {masks_dir}")
        return pd.DataFrame(), summarize_results(pd.DataFrame(), len(sample_ids))

    all_results = []
    total_batches = max(
        1, (len(valid_sids) + sample_batch_size - 1) // sample_batch_size
    )
    t_start = time.time()

    for batch_idx in range(0, len(valid_sids), sample_batch_size):
        batch_sids = valid_sids[batch_idx : batch_idx + sample_batch_size]
        current_batch_num = (batch_idx // sample_batch_size) + 1
        print(
            f"\n[Batch {current_batch_num}/{total_batches}] Processing queries: {', '.join(batch_sids)}"
        )

        batch_queries = build_batch_queries(
            batch_sids,
            gt_data,
            masks_dir,
            bin_deg,
            n_vp,
            min_std_deg=min_std_deg,
            min_max_elev_deg=min_max_elev_deg,
            use_altimeter=use_altimeter,
            use_compass=use_compass,
        )

        if not batch_queries:
            continue

        run_batch_coarse_scan(
            batch_queries,
            db_path,
            elev_m,
            n_vp,
            chunk_rows=chunk_rows,
            spatial_stride=spatial_stride,
            weights=weights,
            compass_tolerance_deg=compass_tolerance_deg,
            height_tolerance_m=height_tolerance_m,
            progress_desc=f"  Batch {current_batch_num}/{total_batches}: Scanning DB",
        )

        for sample_id, query_state in batch_queries.items():
            result = refine_query_with_dtw(
                query_state,
                db_path,
                spatial_stride,
                n_vp,
                lat,
                lon,
                dtw_window=dtw_window,
                correct_dist_m=correct_dist_m,
            )
            if result is None:
                continue
            result["sample_id"] = sample_id
            all_results.append(result)

        # Batch-level checkpoint: save immediately after processing
        if checkpoint_dir and len(batch_sids) > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"batch_{batch_idx:06d}.csv")
            batch_df = pd.DataFrame(
                [r for r in all_results if r["sample_id"] in batch_sids]
            )
            if not batch_df.empty:
                batch_df.to_csv(ckpt_path, index=False)

        elapsed = time.time() - t_start
        print(
            f"  Batch {current_batch_num}: {len(all_results)} total results, "
            f"elapsed {elapsed:.0f}s ({elapsed / max(1, len(all_results)):.1f}s/sample)",
            flush=True,
        )

        del batch_queries
        gc.collect()

    df = pd.DataFrame(all_results)
    summary = summarize_results(df, len(valid_sids))
    return df, summary




def _merge_topk(
    running_corr,
    running_idx,
    running_offset,
    chunk_corr,
    chunk_offset,
    local_indices,
    k,
):
    """Merge chunk scores into a running top-K (small bounded arrays)."""
    n_running = len(running_corr)
    if n_running == 0:
        take = min(k, len(chunk_corr))
        order = np.argpartition(chunk_corr, -take)[-take:]
        return (
            chunk_corr[order].astype(np.float32),
            local_indices[order],
            chunk_offset[order].astype(np.int32),
        )
    combined_corr = np.concatenate([running_corr, chunk_corr])
    combined_idx = np.concatenate([running_idx, local_indices])
    combined_off = np.concatenate([running_offset, chunk_offset])
    take = min(k, len(combined_corr))
    order = np.argpartition(combined_corr, -take)[-take:]
    return (
        combined_corr[order].astype(np.float32),
        combined_idx[order],
        combined_off[order].astype(np.int32),
    )


def run_parameter_sweep(
    ground_truth_path,
    db_path,
    masks_dir,
    configs,
    limit=0,
    min_std_deg=1.5,
    min_max_elev_deg=1.0,
    chunk_rows=4000,
    spatial_stride=12,
    sample_batch_size=8,
    thresholds_m=None,
    top_k_candidates=200,
):
    """Memory-safe single-DB-pass parameter sweep.

    Streams the DB once per query batch and keeps only a bounded top-K
    candidate list per (config, query), so RAM stays ~chunk-size regardless of
    DB size or number of configs.

    Parameters
    ----------
    configs : dict[str, dict]
        Key = config name; value = dict of overrides. Recognised keys:
        use_altimeter, use_compass, compass_tolerance_deg, height_tolerance_m,
        dtw_window, spatial_stride, weights, top_k, correct_dist_m.
    top_k_candidates : int
        How many coarse candidates to keep per (config, query) before DTW.
    """
    if thresholds_m is None:
        thresholds_m = [50.0, 100.0, 200.0, 500.0, 1000.0]

    gt_data, sample_ids = load_ground_truth(ground_truth_path, limit=limit)
    lon_arr, lat_arr, elev_m, n_vp = load_db_metadata(db_path)
    bin_deg = infer_bin_size_deg(db_path)
    valid_sids = filter_samples_with_masks(sample_ids, masks_dir)

    if not valid_sids:
        return {}, {}

    # --- Phase 1: extract profiles once (cheap, small) ---
    queries = {}  # sid → {profile, gt_info, start_az}
    for sid in valid_sids:
        gt_info = gt_data[sid]
        mask_path = _resolve_mask_path(masks_dir, sid)
        fov = gt_info.get("fov_y_deg", 65.0)
        r_tilt = np.array(gt_info["cam_R_tilt"]) if gt_info.get("cam_R_tilt") else None
        pr = extract_elevation_profile(
            mask_path, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=bin_deg
        )
        if not pr["ok"]:
            continue
        profile = pr["profile"]
        is_valid, _ = is_profile_applicable(
            profile, min_std_deg=min_std_deg, min_max_elev_deg=min_max_elev_deg
        )
        if not is_valid:
            continue
        queries[sid] = {
            "profile": profile,
            "gt_info": gt_info,
            "start_az": pr["start_az"],
        }

    if not queries:
        return {}, {}
    print(f"[Sweep] {len(queries)} valid queries, {len(configs)} configs", flush=True)

    # Precompute per-config metadata
    cfg_meta = {}
    for cfg_name, cfg in configs.items():
        cfg_meta[cfg_name] = {
            "use_altimeter": cfg.get("use_altimeter", True),
            "use_compass": cfg.get("use_compass", True),
            "compass_tolerance_deg": cfg.get("compass_tolerance_deg", 20.0),
            "height_tolerance_m": cfg.get("height_tolerance_m", 200.0),
            "dtw_window": cfg.get("dtw_window", 15),
            "spatial_stride": cfg.get("spatial_stride", spatial_stride),
            "weights": cfg.get("weights", (0.5, 0.5)),
            "correct_dist_m": cfg.get("correct_dist_m", 500.0),
            "top_k": cfg.get("top_k", 30),
        }

    # Phase 2: batch queries; stream DB once per batch; update top-K heaps
    total_batches = max(1, (len(queries) + sample_batch_size - 1) // sample_batch_size)
    topk_state = {}
    for cfg_name in configs:
        topk_state[cfg_name] = {
            sid: (np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros(0, np.int32))
            for sid in queries
        }

    t_start = time.time()
    sid_list = list(queries.keys())

    for b_i in range(0, len(sid_list), sample_batch_size):
        batch_sids = sid_list[b_i : b_i + sample_batch_size]
        batch_meta = {sid: queries[sid] for sid in batch_sids}
        print(
            f"\n[Sweep] Batch {b_i // sample_batch_size + 1}/{total_batches} "
            f"({len(batch_sids)} queries)",
            flush=True,
        )

        pf = pq.ParquetFile(db_path)
        global_offset = 0
        for chunk_id, batch in enumerate(
            pf.iter_batches(batch_size=chunk_rows, columns=["raw_horizon_deg"])
        ):
            df_batch = batch.to_pandas()
            chunk_matrix = np.stack(df_batch["raw_horizon_deg"].to_numpy()).astype(
                np.float32
            )
            chunk_size = len(df_batch)
            # Shared DB features per chunk
            db_val, db_d1 = feature_bundle_matrix(chunk_matrix)
            local_indices = np.arange(
                global_offset, global_offset + chunk_size, dtype=np.int64
            )
            chunk_elev = elev_m[global_offset : global_offset + chunk_size]

            for sid in batch_sids:
                q = batch_meta[sid]
                profile = q["profile"]
                gt_info = q["gt_info"]
                exp_off = None
                if "true_heading_deg" in gt_info:
                    exp_off = (gt_info["true_heading_deg"] + q["start_az"]) % 360.0

                for cfg_name, meta in cfg_meta.items():
                    corr, offsets = ncc_scores(
                        db_val,
                        db_d1,
                        profile,
                        bin_deg,
                        weights=meta["weights"],
                    )
                    gate = np.ones(chunk_size, dtype=bool)
                    if meta["use_compass"] and exp_off is not None:
                        bin_off = offsets.astype(np.float64) * bin_deg
                        diff = np.abs(((bin_off - exp_off + 180.0) % 360.0) - 180.0)
                        gate &= diff <= meta["compass_tolerance_deg"]
                    if meta["use_altimeter"] and "eye_z_m" in gt_info:
                        gt_elev = gt_info["eye_z_m"] - gt_info.get(
                            "query_height_m", 1.8
                        )
                        gate &= (
                            np.abs(chunk_elev - gt_elev) <= meta["height_tolerance_m"]
                        )

                    if np.any(gate):
                        gated_corr = np.where(gate, corr, -np.inf)
                        run_corr, run_idx, run_off = topk_state[cfg_name][sid]
                        topk_state[cfg_name][sid] = _merge_topk(
                            run_corr,
                            run_idx,
                            run_off,
                            gated_corr,
                            offsets,
                            local_indices,
                            k=top_k_candidates,
                        )

            del chunk_matrix
            gc.collect()

            if (chunk_id + 1) % 100 == 0:
                elapsed = time.time() - t_start
                print(
                    f"  DB pass: chunk {chunk_id + 1}, elapsed {elapsed:.0f}s",
                    flush=True,
                )

        del pf
        gc.collect()

    # --- Phase 3: DTW on top-K candidates for each config ---
    all_summaries = {}
    all_dfs = {}
    for cfg_name, cfg in configs.items():
        print(f"\n[Sweep] Refining (DTW): {cfg_name}", flush=True)
        meta = cfg_meta[cfg_name]
        cfg_results = []
        for sid in queries:
            run_corr, run_idx, run_off = topk_state[cfg_name][sid]
            if len(run_idx) == 0:
                continue
            order = np.argsort(-run_corr)
            run_idx = run_idx[order]
            run_off = run_off[order]
            result = _refine_from_raw(
                queries[sid],
                db_path,
                meta["spatial_stride"],
                n_vp,
                lat_arr,
                lon_arr,
                dtw_window=meta["dtw_window"],
                correct_dist_m=meta["correct_dist_m"],
                top_k=run_idx,
                top_offsets=run_off,
            )
            if result is not None:
                result["sample_id"] = sid
                cfg_results.append(result)

        df = pd.DataFrame(cfg_results)
        all_dfs[cfg_name] = df
        all_summaries[cfg_name] = summarize_results_at_thresholds(
            df, len(queries), thresholds_m
        )
        s = all_summaries[cfg_name]
        print(
            f"  {cfg_name}: n={s['n_samples']}, median={s['median_error_m']:.0f}m, "
            f"top1@500m={s.get('top1_acc_500m', 0):.1f}%",
            flush=True,
        )

    return all_dfs, all_summaries


def _refine_from_raw(
    query,
    db_path,
    spatial_stride,
    n_vp,
    lat,
    lon,
    dtw_window=15,
    correct_dist_m=500.0,
    top_k=None,
    top_offsets=None,
):
    """DTW refinement starting from a pre-computed candidate set."""
    if top_k is None or len(top_k) == 0:
        return None
    if top_offsets is None:
        top_offsets = np.zeros(len(top_k), dtype=np.int32)

    fine_candidate_set = set()
    for coarse_idx in top_k:
        local_range = np.arange(
            max(0, int(coarse_idx) - spatial_stride),
            min(n_vp, int(coarse_idx) + spatial_stride + 1),
        )
        fine_candidate_set.update(local_range)
    fine_indices = np.array(sorted(fine_candidate_set), dtype=np.int32)

    fetched_horizons = _fetch_rows(db_path, fine_indices)
    profile = query["profile"]
    result_rows = []
    profile_length = len(profile)
    off_by_idx = {int(i): int(o) for i, o in zip(top_k, top_offsets)}

    for vp_idx in fine_indices:
        horizon_curve = fetched_horizons[vp_idx]
        horizon_length = len(horizon_curve)
        offset_bin = off_by_idx.get(int(vp_idx), 0)
        windowed = horizon_curve[
            np.arange(offset_bin, offset_bin + profile_length) % horizon_length
        ]
        result_rows.append((0.0, "main", int(vp_idx), offset_bin, windowed))

    matches = finalize_matches(_RowView(result_rows), profile, dtw_window)
    if not matches:
        return None

    gt_info = query["gt_info"]
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

    return {
        "sample_id": gt_info.get("sample_id", None),
        "error_m": error_m,
        "top1_ok": error_m <= correct_dist_m,
        "top5_ok": top5_ok,
    }

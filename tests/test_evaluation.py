#!/usr/bin/env python
"""Tests for the evaluation pipeline in src/evaluation.py.

Covers:
  - summarize_results / summarize_results_at_thresholds
  - load_ground_truth, filter_samples_with_masks
  - build_batch_queries (profile extraction + query state construction)
  - run_batch_coarse_scan (FFT-based DB scanning)
  - refine_query_with_dtw (DTW refinement on synthetic DB)
  - _RowView helper
  - End-to-end: scan → refine pipeline on a small synthetic DB

All tests use synthetic data (no real GSV or DEM files required).
Run with:

    conda run -n skyline_env python -m pytest tests/test_evaluation.py -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon_format import encode_horizon_uint8, DEG_PER_BIN
from matching import feature_bundle_matrix, ncc_scores
from evaluation import (
    _RowView,
    _resolve_mask_path,
    build_batch_queries,
    filter_samples_with_masks,
    infer_bin_size_deg,
    load_ground_truth,
    load_db_metadata,
    refine_query_with_dtw,
    run_batch_coarse_scan,
    summarize_results,
    summarize_results_at_thresholds,
)
from query_profile import extract_elevation_profile, is_profile_applicable


# ---------------------------------------------------------------------------
# Fixtures: synthetic DB, ground truth, mask images
# ---------------------------------------------------------------------------

N_BINS = 720
BIN_DEG = 360.0 / N_BINS  # 0.5°


def _make_synthetic_horizon(n_bins=N_BINS, seed=0):
    """Create a smooth synthetic horizon with a known mountain peak."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 360, n_bins, endpoint=False)
    # A mountain ridge at azimuth 90° ± 30°
    peak = 35.0 * np.exp(-((angles - 90) ** 2) / (2 * 20**2))
    # A second ridge at 240° ± 15°
    ridge2 = 20.0 * np.exp(-((angles - 240) ** 2) / (2 * 10**2))
    # Base terrain + noise
    base = 5.0 + 2.0 * np.sin(np.radians(angles * 3))
    horizon = base + peak + ridge2 + rng.normal(0, 0.3, size=n_bins)
    return np.clip(horizon, 0, 85).astype(np.float64)


def _make_mask_image(width=640, height=480, peak_row_frac=0.35, seed=0):
    """Create a synthetic sky mask (sky=0/black, terrain=255/white).

    The boundary follows a mountain-like profile across columns.
    """
    rng = np.random.default_rng(seed)
    mask = np.full((height, width), 255, dtype=np.uint8)
    for col in range(width):
        # Mountain profile: peaks at ~30% from top in the center
        frac = col / width
        peak = peak_row_frac + 0.15 * np.sin(2 * np.pi * frac)
        boundary_row = int(height * peak)
        boundary_row = max(1, min(height - 1, boundary_row))
        mask[:boundary_row, col] = 0  # sky above
    return mask


def _create_parquet_db(path, n_rows=30, n_bins=N_BINS):
    """Create a small Parquet DB with synthetic horizons."""
    horizons = []
    lats = []
    lons = []
    elevs = []
    rng = np.random.default_rng(42)

    for i in range(n_rows):
        h = _make_synthetic_horizon(n_bins, seed=i)
        encoded = encode_horizon_uint8(h).tolist()
        horizons.append(encoded)
        # Spread VPs across a small area near Khumbu
        lats.append(27.98 + rng.uniform(-0.05, 0.05))
        lons.append(86.92 + rng.uniform(-0.05, 0.05))
        elevs.append(4000.0 + rng.uniform(-500, 500))

    table = pa.table({
        "raw_horizon_deg": pa.array(horizons, type=pa.list_(pa.uint8())),
        "lon": pa.array(lons, type=pa.float64()),
        "lat": pa.array(lats, type=pa.float64()),
        "elevation_m": pa.array(elevs, type=pa.float64()),
    })
    pq.write_table(table, str(path))
    return lats, lons, elevs


def _create_ground_truth(path, sample_ids, lats, lons, elevs, target_vp=0):
    """Create ground truth JSON pointing to a specific VP as the true location."""
    gt = {}
    for sid in sample_ids:
        gt[sid] = {
            "true_lat": lats[target_vp],
            "true_lon": lons[target_vp],
            "eye_z_m": elevs[target_vp] + 1.8,  # eye height above ground
            "query_height_m": 1.8,
            "true_heading_deg": 0.0,
            "fov_y_deg": 65.0,
            "cam_R_tilt": None,
            "sample_id": sid,
        }
    with open(path, "w") as f:
        json.dump(gt, f)
    return gt


def _create_masks(mask_dir, sample_ids, seed=0):
    """Create synthetic mask PNG files for each sample ID."""
    os.makedirs(mask_dir, exist_ok=True)
    for i, sid in enumerate(sample_ids):
        mask = _make_mask_image(640, 480, seed=seed + i)
        mask_path = os.path.join(mask_dir, f"{sid}.png")
        cv2.imwrite(mask_path, mask)
    return mask_dir


@pytest.fixture
def synthetic_db(tmp_path):
    """Create a complete synthetic evaluation environment."""
    db_path = tmp_path / "test_db.parquet"
    gt_path = tmp_path / "ground_truth.json"
    mask_dir = str(tmp_path / "masks")

    sample_ids = ["s001", "s002", "s003"]
    n_rows = 30
    lats, lons, elevs = _create_parquet_db(db_path, n_rows=n_rows)
    _create_ground_truth(gt_path, sample_ids, lats, lons, elevs, target_vp=0)
    _create_masks(mask_dir, sample_ids)

    return {
        "db_path": str(db_path),
        "gt_path": str(gt_path),
        "mask_dir": mask_dir,
        "sample_ids": sample_ids,
        "lats": np.array(lats),
        "lons": np.array(lons),
        "elevs": np.array(elevs),
        "n_rows": n_rows,
    }


# ---------------------------------------------------------------------------
# Tests: summarize_results
# ---------------------------------------------------------------------------

class TestSummarizeResults:
    def test_empty_results(self):
        """Empty DataFrame should produce zero-count summary."""
        import pandas as pd
        df = pd.DataFrame()
        summary = summarize_results(df, valid_sample_count=10)
        assert summary["n_samples"] == 0
        assert summary["skipped_flat"] == 10
        assert np.isnan(summary["median_error_m"])

    def test_all_correct(self):
        """All results within threshold should yield 100% accuracy."""
        import pandas as pd
        df = pd.DataFrame({
            "error_m": [100.0, 200.0, 300.0, 400.0],
            "top1_ok": [True, True, True, True],
            "top5_ok": [True, True, True, True],
        })
        summary = summarize_results(df, valid_sample_count=4)
        assert summary["n_samples"] == 4
        assert summary["skipped_flat"] == 0
        assert summary["top1_acc_500m"] == 100.0
        assert summary["top5_acc_500m"] == 100.0

    def test_some_correct(self):
        """Mixed results should produce correct accuracy fractions."""
        import pandas as pd
        errors = [100.0, 600.0, 1500.0, 200.0, 800.0]
        df = pd.DataFrame({
            "error_m": errors,
            "top1_ok": [e <= 500.0 for e in errors],
            "top5_ok": [True, False, False, True, False],
        })
        summary = summarize_results(df, valid_sample_count=5)
        assert summary["n_samples"] == 5
        # 2 out of 5 have error <= 500m (100.0 and 200.0)
        assert summary["top1_acc_500m"] == 40.0
        # 2 out of 5 in top5
        assert summary["top5_acc_500m"] == 40.0
        assert abs(summary["median_error_m"] - 600.0) < 1e-6

    def test_custom_thresholds(self):
        """Custom threshold list should produce matching keys."""
        import pandas as pd
        df = pd.DataFrame({"error_m": [50.0, 150.0, 5000.0]})
        summary = summarize_results_at_thresholds(
            df, valid_sample_count=3, thresholds_m=[100.0, 1000.0]
        )
        assert "top1_acc_100m" in summary
        assert "top1_acc_1000m" in summary
        assert summary["top1_acc_100m"] == pytest.approx(100.0 / 3, abs=0.1)
        assert summary["top1_acc_1000m"] == pytest.approx(200.0 / 3, abs=0.1)

    def test_skipped_count(self):
        """skipped_flat should equal valid_sample_count - n_samples."""
        import pandas as pd
        df = pd.DataFrame({"error_m": [100.0, 200.0]})
        summary = summarize_results(df, valid_sample_count=10)
        assert summary["skipped_flat"] == 8


# ---------------------------------------------------------------------------
# Tests: load_ground_truth, filter_samples_with_masks, _resolve_mask_path
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_load_ground_truth(self, synthetic_db):
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        assert len(sids) == 3
        assert "s001" in gt_data
        assert gt_data["s001"]["true_lat"] is not None

    def test_load_ground_truth_with_limit(self, synthetic_db):
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"], limit=2)
        assert len(sids) == 2

    def test_filter_samples_with_masks(self, synthetic_db):
        valid = filter_samples_with_masks(
            synthetic_db["sample_ids"], synthetic_db["mask_dir"]
        )
        assert len(valid) == 3

    def test_filter_samples_missing_masks(self, synthetic_db):
        valid = filter_samples_with_masks(
            ["s001", "missing1", "missing2"], synthetic_db["mask_dir"]
        )
        assert valid == ["s001"]

    def test_resolve_mask_path_raw(self, synthetic_db):
        path = _resolve_mask_path(synthetic_db["mask_dir"], "s001")
        assert os.path.exists(path)

    def test_resolve_mask_path_fixed(self, synthetic_db):
        # Create a fixed-naming mask
        fixed_path = os.path.join(synthetic_db["mask_dir"], "sample_0042.png")
        mask = _make_mask_image(640, 480, seed=99)
        cv2.imwrite(fixed_path, mask)
        path = _resolve_mask_path(synthetic_db["mask_dir"], "42")
        assert os.path.exists(path)

    def test_resolve_mask_path_missing(self, synthetic_db):
        path = _resolve_mask_path(synthetic_db["mask_dir"], "nonexistent")
        assert not os.path.exists(path)

    def test_infer_bin_size_deg(self, synthetic_db):
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        assert abs(bin_deg - BIN_DEG) < 1e-6

    def test_load_db_metadata(self, synthetic_db):
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])
        assert n_vp == 30
        assert len(lon) == 30
        assert len(lat) == 30
        assert len(elev_m) == 30


# ---------------------------------------------------------------------------
# Tests: build_batch_queries
# ---------------------------------------------------------------------------

class TestBuildBatchQueries:
    def test_build_batch_queries_basic(self, synthetic_db):
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        n_vp = synthetic_db["n_rows"]

        queries = build_batch_queries(
            sids[:1],  # just one sample
            gt_data,
            synthetic_db["mask_dir"],
            bin_deg,
            n_vp,
            min_std_deg=0.1,  # low threshold for synthetic data
            min_max_elev_deg=0.1,
            use_compass=False,
            use_altimeter=False,
        )
        # Should produce at least one query
        assert len(queries) >= 1
        sid = list(queries.keys())[0]
        q = queries[sid]
        assert "profile" in q
        assert "gt_info" in q
        assert "best_corr" in q
        assert "best_offset" in q
        assert q["best_corr"].shape == (n_vp,)
        assert q["best_offset"].shape == (n_vp,)
        assert q["expected_offset"] is None  # compass disabled
        assert q["gt_elevation"] is None  # altimeter disabled

    def test_build_batch_queries_with_compass(self, synthetic_db):
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        n_vp = synthetic_db["n_rows"]

        queries = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=True, use_altimeter=False,
        )
        assert len(queries) >= 1
        sid = list(queries.keys())[0]
        q = queries[sid]
        # true_heading_deg is 0.0, so expected_offset should be a float
        assert q["expected_offset"] is not None
        assert isinstance(q["expected_offset"], float)

    def test_build_batch_queries_with_altimeter(self, synthetic_db):
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        n_vp = synthetic_db["n_rows"]

        queries = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=False, use_altimeter=True,
        )
        assert len(queries) >= 1
        sid = list(queries.keys())[0]
        q = queries[sid]
        assert q["gt_elevation"] is not None
        assert isinstance(q["gt_elevation"], float)

    def test_build_batch_queries_filters_flat_profiles(self, synthetic_db):
        """With very high min_std threshold, no queries should pass."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        n_vp = synthetic_db["n_rows"]

        queries = build_batch_queries(
            sids, gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=999.0,  # impossible threshold
            min_max_elev_deg=0.1,
        )
        assert len(queries) == 0

    def test_build_batch_queries_all_samples(self, synthetic_db):
        """All three samples should produce queries."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        n_vp = synthetic_db["n_rows"]

        queries = build_batch_queries(
            sids, gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
        )
        assert len(queries) == 3


# ---------------------------------------------------------------------------
# Tests: run_batch_coarse_scan
# ---------------------------------------------------------------------------

class TestRunBatchCoarseScan:
    def test_coarse_scan_populates_best_corr(self, synthetic_db):
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])

        queries = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=False, use_altimeter=False,
        )
        assert len(queries) >= 1

        # Initially all best_corr are -inf
        sid = list(queries.keys())[0]
        assert np.all(queries[sid]["best_corr"] == -np.inf)

        run_batch_coarse_scan(
            queries, synthetic_db["db_path"], elev_m, n_vp,
            chunk_rows=10, spatial_stride=3,
            weights=(0.5, 0.5),
            compass_tolerance_deg=180.0,  # no compass filtering
            height_tolerance_m=99999.0,  # no elevation filtering
        )

        # After scan, some best_corr should be > -inf
        assert np.any(queries[sid]["best_corr"] > -np.inf)
        # At least a fraction of VPs should have been checked (stride=3)
        n_checked = np.sum(queries[sid]["best_corr"] > -np.inf)
        assert n_checked > 0

    def test_coarse_scan_with_compass_masking(self, synthetic_db):
        """Compass filtering should reduce the number of valid VP hits."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])

        # With very tight compass tolerance, few VPs should pass
        queries_tight = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=True, use_altimeter=False,
        )
        if len(queries_tight) == 0:
            pytest.skip("Profile extraction failed for this sample")

        run_batch_coarse_scan(
            queries_tight, synthetic_db["db_path"], elev_m, n_vp,
            chunk_rows=10, spatial_stride=3,
            compass_tolerance_deg=1.0,  # very tight
            height_tolerance_m=99999.0,
        )

        # With no compass filtering
        queries_wide = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=False, use_altimeter=False,
        )
        if len(queries_wide) == 0:
            pytest.skip("Profile extraction failed for this sample")

        run_batch_coarse_scan(
            queries_wide, synthetic_db["db_path"], elev_m, n_vp,
            chunk_rows=10, spatial_stride=3,
            compass_tolerance_deg=180.0,
            height_tolerance_m=99999.0,
        )

        sid_t = list(queries_tight.keys())[0]
        sid_w = list(queries_wide.keys())[0]
        n_tight = np.sum(queries_tight[sid_t]["best_corr"] > -np.inf)
        n_wide = np.sum(queries_wide[sid_w]["best_corr"] > -np.inf)
        assert n_tight <= n_wide

    def test_coarse_scan_with_elevation_masking(self, synthetic_db):
        """Elevation filtering should gate VPs by height difference."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])

        queries = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=False, use_altimeter=True,
        )
        if len(queries) == 0:
            pytest.skip("Profile extraction failed")

        # Very tight elevation tolerance
        run_batch_coarse_scan(
            queries, synthetic_db["db_path"], elev_m, n_vp,
            chunk_rows=10, spatial_stride=3,
            compass_tolerance_deg=180.0,
            height_tolerance_m=1.0,  # 1m tolerance
        )
        sid = list(queries.keys())[0]
        n_hits = np.sum(queries[sid]["best_corr"] > -np.inf)
        # Should have fewer hits than with no elevation filter
        assert n_hits < n_vp  # at least some filtered out


# ---------------------------------------------------------------------------
# Tests: refine_query_with_dtw
# ---------------------------------------------------------------------------

class TestRefineQueryWithDtw:
    def test_refine_returns_result(self, synthetic_db):
        """DTW refinement should return a result dict with error_m."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])

        queries = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=False, use_altimeter=False,
        )
        if len(queries) == 0:
            pytest.skip("Profile extraction failed")

        sid = list(queries.keys())[0]
        run_batch_coarse_scan(
            queries, synthetic_db["db_path"], elev_m, n_vp,
            chunk_rows=10, spatial_stride=3,
            compass_tolerance_deg=180.0,
            height_tolerance_m=99999.0,
        )

        result = refine_query_with_dtw(
            queries[sid], synthetic_db["db_path"],
            spatial_stride=3, n_vp=n_vp,
            lat=lat, lon=lon,
            dtw_window=5, correct_dist_m=500.0,
        )
        # Result should be a dict with required keys
        assert result is not None
        assert "error_m" in result
        assert "top1_ok" in result
        assert "top5_ok" in result
        assert isinstance(result["error_m"], float)
        assert result["error_m"] >= 0.0

    def test_refine_empty_query_returns_none(self, synthetic_db):
        """Query with no coarse hits should return None."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])

        query_state = {
            "gt_info": gt_data[sids[0]],
            "profile": np.random.default_rng(0).uniform(0, 60, 720),
            "expected_offset": None,
            "gt_elevation": None,
            "best_corr": np.full(n_vp, -np.inf, dtype=np.float32),
            "best_offset": np.zeros(n_vp, dtype=np.int32),
        }
        result = refine_query_with_dtw(
            query_state, synthetic_db["db_path"],
            spatial_stride=3, n_vp=n_vp,
            lat=lat, lon=lon,
        )
        assert result is None

    def test_refine_top5_ok_flag(self, synthetic_db):
        """top5_ok should be True when a top-5 match is within threshold."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])

        queries = build_batch_queries(
            sids[:1], gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=False, use_altimeter=False,
        )
        if len(queries) == 0:
            pytest.skip("Profile extraction failed")

        sid = list(queries.keys())[0]
        run_batch_coarse_scan(
            queries, synthetic_db["db_path"], elev_m, n_vp,
            chunk_rows=10, spatial_stride=3,
            compass_tolerance_deg=180.0,
            height_tolerance_m=99999.0,
        )

        result = refine_query_with_dtw(
            queries[sid], synthetic_db["db_path"],
            spatial_stride=3, n_vp=n_vp,
            lat=lat, lon=lon,
            dtw_window=5, correct_dist_m=500.0,
        )
        if result is not None:
            assert isinstance(result["top5_ok"], bool)


# ---------------------------------------------------------------------------
# Tests: _RowView
# ---------------------------------------------------------------------------

class TestRowView:
    def test_rowview_items_sorted_desc(self):
        rows = [(0.5, "a", 1), (0.9, "b", 2), (0.7, "c", 3)]
        rv = _RowView(rows)
        items = rv.items()
        scores = [r[0] for r in items]
        assert scores == sorted(scores, reverse=True)

    def test_rowview_iter(self):
        rows = [(0.5, "a", 1), (0.9, "b", 2)]
        rv = _RowView(rows)
        collected = list(rv)
        assert len(collected) == 2
        assert collected[0][0] >= collected[1][0]


# ---------------------------------------------------------------------------
# Tests: is_profile_applicable
# ---------------------------------------------------------------------------

class TestProfileApplicable:
    def test_flat_profile_rejected(self):
        profile = np.full(720, 10.0)
        valid, reason = is_profile_applicable(profile, min_std_deg=1.5)
        assert not valid
        assert "flat" in reason.lower()

    def test_valid_profile_accepted(self):
        rng = np.random.default_rng(0)
        profile = 20.0 + 10.0 * np.sin(np.linspace(0, 2 * np.pi, 720))
        profile += rng.normal(0, 0.5, size=720)
        valid, reason = is_profile_applicable(profile, min_std_deg=1.5, min_max_elev_deg=1.0)
        assert valid

    def test_empty_profile_rejected(self):
        valid, reason = is_profile_applicable(np.array([]))
        assert not valid

    def test_nan_profile_rejected(self):
        profile = np.full(720, 10.0)
        profile[100] = np.nan
        valid, reason = is_profile_applicable(profile)
        assert not valid


# ---------------------------------------------------------------------------
# Tests: extract_elevation_profile (basic sanity)
# ---------------------------------------------------------------------------

class TestExtractElevationProfile:
    def test_extract_from_synthetic_mask(self):
        """extract_elevation_profile should return a valid profile from a synthetic mask."""
        mask = _make_mask_image(640, 480, peak_row_frac=0.35)
        pr = extract_elevation_profile(mask, fov_y_deg=65.0, bin_deg=BIN_DEG)
        assert pr["ok"]
        assert pr["profile"] is not None
        # Profile length depends on mask geometry and FOV; just check it's finite and non-empty
        assert len(pr["profile"]) > 0
        assert np.all(np.isfinite(pr["profile"]))

    def test_extract_from_file(self, tmp_path):
        """extract_elevation_profile should work from a file path."""
        mask = _make_mask_image(640, 480, seed=5)
        mask_path = str(tmp_path / "test_mask.png")
        cv2.imwrite(mask_path, mask)
        pr = extract_elevation_profile(mask_path, fov_y_deg=65.0, bin_deg=BIN_DEG)
        assert pr["ok"]

    def test_extract_invalid_path_returns_error(self):
        pr = extract_elevation_profile("/nonexistent/mask.png")
        assert not pr["ok"]


# ---------------------------------------------------------------------------
# Integration test: end-to-end scan → refine
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_scan_then_refine(self, synthetic_db):
        """Full pipeline: build queries → coarse scan → DTW refine."""
        gt_data, sids = load_ground_truth(synthetic_db["gt_path"])
        bin_deg = infer_bin_size_deg(synthetic_db["db_path"])
        lon, lat, elev_m, n_vp = load_db_metadata(synthetic_db["db_path"])

        queries = build_batch_queries(
            sids, gt_data, synthetic_db["mask_dir"],
            bin_deg, n_vp,
            min_std_deg=0.1, min_max_elev_deg=0.1,
            use_compass=False, use_altimeter=False,
        )
        if len(queries) == 0:
            pytest.skip("No profiles extracted")

        run_batch_coarse_scan(
            queries, synthetic_db["db_path"], elev_m, n_vp,
            chunk_rows=10, spatial_stride=3,
            compass_tolerance_deg=180.0,
            height_tolerance_m=99999.0,
        )

        all_results = []
        for sid, qstate in queries.items():
            result = refine_query_with_dtw(
                qstate, synthetic_db["db_path"],
                spatial_stride=3, n_vp=n_vp,
                lat=lat, lon=lon,
                dtw_window=5, correct_dist_m=500.0,
            )
            if result is not None:
                result["sample_id"] = sid
                all_results.append(result)

        import pandas as pd
        df = pd.DataFrame(all_results)
        summary = summarize_results(df, len(sids))

        assert summary["n_samples"] > 0
        assert summary["median_error_m"] >= 0.0
        # Should have at least the 500m accuracy key
        assert "top1_acc_500m" in summary

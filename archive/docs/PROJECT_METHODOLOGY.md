# SkylineGeolocation — Project Methodology

> Methodology as actually implemented. Not aspirational, not aspirational.
> Last verified against `src/`, `notebooks/`, `data/` state.

## 1. Project Overview

**Goal**: Determine geographic location from a single ground-level photo via skyline matching against a pre-built horizon database.

**Pipeline**:

1. Select geographic region, download elevation data (DEM)
2. Generate horizon database via HORAYZON ray-tracing on the DEM
3. Train a sky segmentation model (U-Net) to extract sky profile from photos
4. Extract elevation-angle profile from the segmented sky region
5. Match query profile against database using FFT cross-correlation + Dynamic Time Warping (DTW)
6. Return top-k candidate locations; measure geodesic error against ground truth

**Target**: top-1 within ~1000 m on synthetic dataset (300 views at viewpoints inside the DEM region). Calibration still uncalibrated (see `AGENTS.md` parameter table).

## 2. Data Pipeline

### 2.1 Region Definition & DEM Download (`notebooks/01_RegionStudy/`)

**Files**:
- `01_define_region.ipynb` — defines a `Region` (lat/lon bbox + EPSG), saves `output/actual_bounds.json`
- `02_download_glo90_dem.ipynb` — GLO90 (~90 m) via dem_stitcher
- `04_download_glo30_dem.ipynb` — GLO30 (~30 m) via dem_stitcher
- `03_analyze_viewshed.ipynb` — GDAL viewshed dashboard, altitude-vs-visibility table

**Inputs**:
- Geographic bounds (lon_min, lon_max, lat_min, lat_max) defined in JSON config
- Earthdata credentials in `.env` (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`)

**Process**:
1. `Region` class (`src/region.py`) holds bbox + computes UTM projection for the region
2. Earthdata query via `earthaccess.search_data(short_name=..., bounding_box=...)`
3. Parallel download via `earthaccess.download(...)` with thread pool (8 workers by default)
4. Merge individual GeoTIFF tiles into a single reprojected raster (`src/dem.py: merge_tiles_to_compressed_tif`)
5. Reproject to UTM EPSG:32645 for the working region

**Outputs**:
- `data/digital_elevation_model/dem_30m.tif` (2.6 GB, primary)
- `data/digital_elevation_model/dem_90m.tif` (coarser fallback)
- `data/digital_elevation_model/dem_30m_compressed.tif` (deflate-compressed variant)
- `notebooks/01_RegionStudy/output/actual_bounds.json`
- `notebooks/01_RegionStudy/output/download_bounds.json`

**Key code**:
- `src/download_utils.py: load_download_bounds`, `login_earthdata`, `search_data_by_bbox`, `download_results_parallel`
- `src/region.py: Region`, `import_region`
- `src/dem.py: save_reprojected_dem`, `merge_tiles_to_compressed_tif`, `create_hybrid_dem_blockwise`

**Dependencies**: `earthaccess`, `rasterio`, `dem_stitcher`, `pyproj`, `osgeo/gdal`

**Known unused**: HMA DEM (`dem_hma_merged_compressed.tif` exists in repo, was downloaded experimentally, is not used by the pipeline). Hybrid source DEM also unused.

### 2.2 Skyline Database Generation (`notebooks/02_SkylineDatabase/`)

**Files**:
- `01_generate_skyline_database.ipynb` — runs `SkylineDatabaseGenerator.generate_database`
- `02_visualize_horizons.ipynb` — interactive explorer (`SkylineHorizonExplorer`)

**Inputs**:
- DEM raster file (`data/digital_elevation_model/dem_30m.tif`)
- Region bounds (`actual_bounds.json`)
- Search radius (default 30 km — chosen as Earth-curvature limit)
- Grid spacing (30 m typical)
- Eye height (1.6 m — human eye)

**Process** (`SkylineDatabaseGenerator.generate_database` in `src/skyline.py`):

1. Convert grid spacing from meters to degrees via `pyproj.Geod.fwd`
2. Build a regular lat/lon grid over the region
3. Load DEM window covering region + buffer (search radius)
4. Rearrange DEM into HORAYZON's expected buffer layout (`horayzon.auxiliary.rearrange_pad_buffer`)
5. For each viewpoint batch (4096 rows default):
   - Convert (lat, lon) to DEM coordinates
   - Ray-cast 360 azimuth directions via `horayzon` (C++ Embree)
   - Each ray returns horizon elevation angle in degrees
   - Result: 360-element float32 array per viewpoint (1° azimuth resolution)
6. Filter out flat profiles (`min_std_deg`, `min_max_elev_deg`)
7. Write to Parquet with columns:
   - `lon`, `lat`, `elevation_m` (point coordinate + ground elevation)
   - `raw_horizon_deg` — variable-length float32 array (length 360)

**Outputs**:
- `notebooks/02_SkylineDatabase/output/skyline_db.parquet` (911 MB, 1,338,650 viewpoints, 327 row groups)
- `notebooks/02_SkylineDatabase/output/terrain_mesh.npy` — compressed trimesh for renderer
- `notebooks/02_SkylineDatabase/output/terrain_meta.json` — DEM dim/extent metadata

**Memory**: 911 MB on disk. ~1.4 KB per viewpoint. **Never load fully into RAM** — use `pq.ParquetFile(...).iter_batches()`.

**Supported kwargs** (for subsetting): `start_idx`, `end_idx`, `batch_size`, `grid_spacing_m`. Quick test: `end_idx=100` processes only first 100 viewpoints.

### 2.3 Synthetic Dataset Generation (`notebooks/03_SyntheticData/`)

**Files**:
- `01_download_clouds_dataset.ipynb` — Kaggle clouds-photos dataset → `data/clouds/`
- `02_download_satellite_texture.ipynb` — Esri Sentinel-2 tiles → `data/satellite_imagery/`
- `03_generate_synthetic_data.ipynb` — `SyntheticSceneGenerator` renders 300 views

**Inputs**:
- DEM
- Satellite texture (Esri/Sentinel-2 tile covering region)
- Cloud backdrop images (~200 files)
- Region bounds

**Process** (`SyntheticSceneGenerator` in `src/synthetic_generator.py`):

1. Load DEM window + satellite texture
2. Crop DEM to a working window (configurable stride)
3. Build trimesh terrain mesh
4. For each random viewpoint:
   - Sample random lat/lon in DEM
   - Render terrain via pyrender (EGL backend)
   - Composite cloud backdrop + sky mask
   - Save RGB image + binary sky mask
5. Save ground truth JSON (lat/lon, eye elevation, camera params)

**Output**:
- `data/synthetic_dataset/images/` (300 PNG, RGB photos)
- `data/synthetic_dataset/masks/` (300 PNG, binary sky masks, ground truth)
- `data/synthetic_dataset/ground_truth.json` — per-sample metadata:
  ```json
  {
    "true_lat": 27.9058,
    "true_lon": 86.7643,
    "eye_x_utm": 476814.03,
    "eye_y_utm": 3086794.5,
    "eye_z_m": 5813.5,
    "closest_viewpoint_id": 2435,
    "closest_viewpoint_dist_m": 0.0,
    "fov_y_deg": 56.3,
    "true_heading_deg": 311.4,
    "cam_R_tilt": [[...], ...]
  }
  ```

**Headless setup**:
- `mountain_engine.py:3` sets `PYOPENGL_PLATFORM=egl`
- `synthetic_generator.py:15` sets `PYRENDER_BACKEND=egl`

## 3. Sky Segmentation Model (`notebooks/04_SkySegmentation/`)

**Files**:
- `01_download_geopose3k_dataset.ipynb` — GeoPose3K (21 GB) → `data/geopose3k/`
- `02_train_sky_segmentation_model.ipynb` — train U-Net on GeoPose3K + synthetic
- `03_segment_test_images.ipynb` — run inference on test set, save to `predicted_masks/`

**Architecture**: U-Net with MobileNetV3-large encoder (`tu-mobilenetv3_large_100` from `segmentation_models_pytorch`).

**Training data**:
- GeoPose3K (`data/geopose3k/`) — real ground-level photos with sky masks
- Synthetic dataset (`data/synthetic_dataset/masks/`) — generated ground truth

**Loss**: BCE + Dice (`bce_dice_loss` in `segmentation.py`).

**Output**: `data/sky_segmentation_unet_model.pth` (26 MB, MobileNetV3-large encoder).

**Inference pipeline** (`segment_image`):
1. Resize image to 256×256 with reflective padding (aspect preserved)
2. U-Net → probability map
3. Threshold at 0.5 → raw binary mask
4. Refine with Canny-edge guidance (`refine_sky_mask_with_guidance`):
   - Connected components: keep only sky regions touching top boundary
   - Snap boundaries to nearest Canny edge within ±8 px
5. Save binary mask

**Returns structured result**:
```python
{
    "ok": bool,
    "status": "OK" | "LOW_CONFIDENCE" | "INVALID_INPUT",
    "reason": str,
    "diagnostics": {
        "sky_ratio": float,
        "boundary_coverage": float,
        "largest_sky_area": int,
        "top_connected": bool,
        "num_components": int,
        "mean_confidence": float | None,
    },
    "mask_path": str,
}
```

Defaults (uncalibrated):
- `min_sky_ratio = 0.05`
- `max_sky_ratio = 0.95`
- `min_boundary_coverage = 0.5`

## 4. Query Profile Extraction (`src/query_profile.py`)

`extract_elevation_profile(mask_path)` translates a binary sky mask into a 1D elevation-angle profile.

**Process**:
1. Detect sky convention (white vs. black via mean of top vs. bottom rows)
2. For each column, find first terrain row (skyline pixel)
3. Apply median filter (kernel=5) to stabilize
4. Project each skyline pixel onto a ray using camera intrinsics (`fov_y_deg`, `aspect_ratio`, `r_tilt`)
5. Convert ray → elevation (asin) + azimuth (arctan2)
6. Sort by azimuth
7. Resample onto uniform azimuth grid at `bin_deg` (default 0.25°)

**Returns structured result**:
```python
{
    "ok": bool,
    "status": "OK" | "LOW_CONFIDENCE" | "INVALID_INPUT" | "NO_SKYLINE",
    "reason": str,
    "profile": np.ndarray,    # elevation in degrees, length ~360 / bin_deg
    "start_az": float,
    "diagnostics": {
        "width": int, "height": int,
        "sky_ratio": float,
        "sky_is_white": bool,
        "boundary_coverage": float,
        "missing_columns": int,
        "hfov_deg": float,
        "fov_y_deg": float,
        "bin_deg": float,
        "profile_std_deg": float,
        "profile_max_deg": float,
        "profile_min_deg": float,
        "profile_length": int,
    },
}
```

## 5. Matching (`src/matching.py`)

### Algorithm

Three stages (in `match_query`):

1. **Coarse spatial search**: `fft_prefilter` on every Nth viewpoint (`spatial_stride=5`)
2. **Local fine refinement**: gather neighbors of top-5 coarse candidates
3. **DTW refinement**: `fastdtw(query_features, db_features, radius=dtw_window)` on top-k

**Feature bundle** per profile: 2 z-scored features stacked:
- `value = safe_zscore(elev)`
- `d1 = safe_zscore(gradient(value))`

Both `_feature_bundle` (single profile) and `_feature_bundle_matrix` (batched DB rows) compute `d1` as the gradient of the **z-scored value** — keep them in sync.

**Safe zscore**: returns zeros if `std(x) < 1e-12` (avoids NaN on flat profiles).

**Confidence**: `score_gap = best_corr - second_corr`. Ambiguous if < `min_score_gap`.

### Returns structured result

```python
{
    "ok": bool,
    "status": "OK" | "LOW_CONFIDENCE" | "NO_MATCH" | "INVALID_QUERY" | "INVALID_INPUT",
    "reason": str,
    "matches": [
        {
            "row_index": int,
            "score": float,
            "fft_corr": float,
            "dtw_distance": float,
            "offset_deg": float,
        }
    ],
    "confidence": {
        "best_score": float,
        "second_score": float,
        "score_gap": float,
        "ambiguous": bool,
    },
    "diagnostics": dict,
}
```

Confidence rules (configurable, all uncalibrated):
- `min_corr = 0.30`
- `min_score_gap = 0.03`

**Failure handling**:
- Flat query → `INVALID_QUERY`
- All-low-correlation matches → `LOW_CONFIDENCE`
- No candidates survived filter → `NO_MATCH`
- Empty DB → `INVALID_INPUT`

### Verified matcher upper bound (2026-08-12)

Exhaustive stride=1 scan over all 1,338,650 VPs with a perfect query profile (the
true VP's own DB horizon, windowed to GT FOV/heading), 2-feature NCC `weights=(0.5,0.5)`:

| SID | TrueVP | Rank/1.3M | True FB | Best FB | Best err |
|-----|--------|-----------|---------|---------|----------|
| 0   | 671249 | **0** | 0.9970 | 0.9970 | 15 m |
| 1   | 806818 | **0** | 0.9910 | 0.9910 | 14 m |
| 2   | 1205848| **0** | 0.9999 | 0.9999 | 9 m |
| 3   | 246088 | **0** | 0.9951 | 0.9951 | 7 m |
| 4   | 560883 | **0** | 0.9999 | 0.9999 | 14 m |

**Interpretation**: with a clean profile the matcher localizes to ≤15 m. The 1°
resolution and 2-feature NCC are sufficient. Mask-level test (2026-08-12) shows
predicted-mask and perfect-mask profiles perform identically on synthetic data —
segmentation is NOT the bottleneck on synthetic. The loss between DB-sliced and
mask-extracted profiles (0.997→0.89 FB) is in render fidelity / profile extraction
geometry. Harness: `scripts/diag_true_rank.py` (correct row-group fetch).

**GSV (RE-EVALUATED 2026-08-13 part 3, correct harness)**: the earlier real-world
baseline was a harness artifact — `gsv_eval.py` selected matches by minimum geodesic
error (ground-truth peek) and the "perfect-mask ceiling" was built from the DB
(circular). With the honest max-corr matcher, GSV U-Net-mask profiles match at median
**~18 km** (0% <1 km, true-VP rank ~12%), barely above the permuted-profile null
(~19.6 km). The projected DB ridge sits on no visible boundary in the photos
(brightness contrast 0.09σ). GeoPose3K is 100% European Alps (zero in-region), so no
non-circular validation existed — the project built **hand-annotated GSV skylines**
(`scripts/annotate_gsv.py` → `scripts/annotated_gsv_eval.py`) as the decisive test.
Synthetic remains fully validated (rank 0, ≤15 m). Harness: `scripts/gsv_eval.py`.

## 6. Evaluation (`src/evaluation.py`)

`run_evaluation(ground_truth_path, db_path, masks_dir, ...)`

**Process**:
1. Load ground truth JSON (sample_id → metadata)
2. Load lightweight DB metadata (`lon`, `lat`, `elevation_m` only, via `pd.read_parquet(columns=[...])`)
3. Infer `bin_deg` from first row of horizon array
4. Filter samples that have predicted masks
5. For each batch (default 8 samples):
   - Build query states (`build_batch_queries`)
   - Stream DB chunks via `iter_batches(batch_size=4000)`
   - Per chunk, run coarse FFT scan on strided DB subset (`run_batch_coarse_scan`)
   - Apply elevation filter (if altimeter used)
   - Apply compass heading filter (if compass used)
   - After streaming: refine top candidates via DTW (`refine_query_with_dtw`)
   - Compute geodesic error for top-1 + top-5
6. Aggregate results into summary

**Memory safety**: never loads full DB. Streams via `pq.ParquetFile(...).iter_batches(batch_size=chunk_rows)`.

**Summary fields** (`summarize_results`):
- `n_samples`: matched samples
- `skipped_flat`: rejected by profile filter
- `top1_acc_500m`: % top-1 within 500 m
- `top1_acc_100m`: % top-1 within 100 m
- `top5_acc_500m`: % any-of-top-5 within 500 m
- `median_error_m`, `mean_error_m`

## 7. Sensor Ablation Study (`notebooks/05_SkylineMatching/02_sensor_study.ipynb`)

Four scenarios run via `run_evaluation`:
- Altimeter + Compass
- Altimeter Only
- Compass Only
- No Sensors

Compares accuracy trade-offs. Currently uses `use_altimeter` and `use_compass` flags.

## 8. End-to-End Reproduction

### Prerequisites
- `skyline_env` conda env created
- `.env` with Earthdata creds (for fresh DEM)
- `data/digital_elevation_model/dem_30m.tif` (2.6 GB)
- `data/synthetic_dataset/` with `ground_truth.json`, `images/`, `masks/`, `predicted_masks/`
- `data/sky_segmentation_unet_model.pth` (26 MB)
- `notebooks/02_SkylineDatabase/output/skyline_db.parquet` (911 MB)

### From scratch (regenerate everything)

1. `01_RegionStudy/01` — define region → save `actual_bounds.json`
2. `01_RegionStudy/02` — download GLO90 DEM
3. `01_RegionStudy/04` — download GLO30 DEM (replace primary)
4. `01_RegionStudy/03` — verify via viewshed dashboard
5. `02_SkylineDatabase/01` — generate `skyline_db.parquet` (hours on full region)
6. `03_SyntheticData/01` — download cloud dataset
7. `03_SyntheticData/02` — download satellite imagery
8. `03_SyntheticData/03` — render synthetic views
9. `04_SkySegmentation/01` — download GeoPose3K
10. `04_SkySegmentation/02` — train U-Net
11. `04_SkySegmentation/03` — segment test/synthetic images → `predicted_masks/`
12. `05_SkylineMatching/01` — evaluate
13. `05_SkylineMatching/02` — sensor ablation
14. `generate_results.ipynb` — final figures

### Skip stages with cached artifacts

Each notebook stage checks if its output already exists. Colab notebooks use `ResumeManifest` (`/tmp/pipeline_state.json`) for resumability.

## 9. Key Design Decisions

### Why FFT + DTW (two-stage)
- FFT cross-correlation is O(N log N), orders of magnitude faster than pairwise DTW
- DTW allows minor profile deformations (DEM noise, segmentation errors)
- Use FFT to find top candidates, DTW to rank them

### Why 30 km search radius
- Earth curvature limits ground-level horizon visibility to ~30 km on standard refracted atmosphere
- See `physics.py:calculate_curvature_drop` — at 30 km, ~140 m hidden, at 100 km ~1500 m

### Why structured-result returns
- Callers depend on knowing whether the pipeline succeeded
- Silently returning `None` or fake coords would propagate false confidence
- All major functions return `{ok, status, reason, diagnostics, data}` dicts

### Why streaming Parquet
- Full DB is 911 MB → OOM risk if loaded
- Streaming via `iter_batches` is the only safe way to process 1.3M viewpoints

### Why safe zscore
- `scipy.stats.zscore` returns NaN when std=0 (flat profile)
- Flat profiles should match flat queries — but NaN propagates everywhere
- `_safe_zscore` returns zeros for std < 1e-12, preserves determinism

## 10. Known Limitations

- **Uncalibrated thresholds**: most numerical defaults (correlation minima, profile std cutoffs, distance thresholds) have not been empirically tuned. See `AGENTS.md` parameter table.
- **Single DEM source**: only GLO30 is used in practice. Multi-source fusion (`create_hybrid_dem_blockwise`) exists but is unused.
- **Single search radius**: only 30 km DB generated. No multi-tier coarse/fine/far variants. Ablation studies would require regenerating DBs at multiple radii.
- **Synthetic-only evaluation**: `scripts/run_eval.py` targets synthetic samples. GSV is
  measured separately by `scripts/gsv_eval.py` (correct row-group fetch, no compass
  gate). Verified 2026-08-13: the GSV accuracy bottleneck is U-Net mask quality on real
  photos (true-VP FB_best 0.65 vs 0.963 perfect-mask ceiling).
- **No compass calibration**: heading filter uses raw `true_heading_deg` from synthetic ground truth. Real photos need magnetic declination correction. GSV panos also have unknown per-pano column-0 rotation, so GSV eval disables compass gating entirely (matching is azimuth-shift-invariant).
- **HORAYZON build hardcoded to conda**: `setup.py:19` reads `os.environ["CONDA_PREFIX"]`. Won't compile in bare Colab without miniconda. Local pre-built artifacts bypass this.

# SkylineGeolocation — Repo Guide

## Setup

```bash
conda env create -f environment.yml
conda activate skyline_env
cd HORAYZON && python -m pip install . && cd ..
```

Earthdata creds in `.env` (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`) for DEM downloads. `.env` is gitignored.

## Environment

- Conda env: `skyline_env` (Python 3.10)
- GPU optional — `scripts/crop_segment.py:84` and `scripts/retrain_dualhead.py:332` auto-detect CUDA via `torch.cuda.is_available()`; `src/segmentation.py` takes `device` as a parameter (no auto-detect inside)
- HORAYZON C++ lib uses Intel Embree + TBB, compiled via `pip install` in `HORAYZON/`
- HORAYZON `setup.py:19` hardcodes `os.environ["CONDA_PREFIX"]` — won't build on bare Colab without conda; DB is already generated locally so eval skips it

## Pipeline (notebook stages)

```
01_RegionStudy       — define region, download DEM (GLO90, GLO30), analyze viewshed
02_SkylineDatabase   — generate horizon DB via HORAYZON, visualize
03_SyntheticData     — download clouds/satellite, render synthetic views + masks
04_SkySegmentation   — download GeoPose3K, train U-Net, segment test images
05_SkylineMatching   — batch evaluation, sensor ablation study
```

Top-level: `notebooks/generate_results.ipynb` — produces final result figures.

## AI Agent Guardrails & Anti-Hallucination Rules

1. **Do NOT Re-Run Conclusively Failed Approaches:**
   - **Saliency Weighting ($\alpha > 0$):** DEAD. Amplifies 30m DEM noise.
   - **Elevation Penalty ($\gamma > 0$):** DEAD. Penalizes true VPs due to camera pitch calibration noise.
   - **2D Chamfer Re-Ranking:** DEAD. Does not rescue Top-1 because true VP is not in Stage-1 candidates.
   - **LAB $b^*$ Segmentation:** DEAD. $b^*$ gradient is too gradual; median boundary error > 240px.
   - **RANSAC Sub-Window Consensus:** DEAD. 0/17 sub-window agreement at True VP.

2. **Parquet Row-Group Indexing Rule:**
   - NEVER fetch rows using `vp // 4096`. Row groups in `skyline_db.parquet` are non-uniform. Always use `np.searchsorted` on cumulative row-group starts (see `scripts/regenerate_gsv.py:37-56`) or `iter_batches()`.

3. **Horizon Quantization Rule:**
   - `skyline_db.parquet` stores `raw_horizon_deg` as `uint8` (0–255 ↔ 0–90°). Every DB reader MUST decode via `src.horizon_format` (`decode_horizon_column` / `decode_horizon_uint8`). Never pass raw `uint8` arrays to `matching.py`.

4. **Production Architecture Constraint:**
   - **Synthetic Queries:** Benchmark deliverable is **100% Top-1 ($\le 15\text{m}$ error)**.
   - **Real Photo Mobile App:** Deployment path is **Avenue 1 (Triple-Sensor Gating: IMU Pitch $\pm 2^\circ$ + Magnetometer Compass $\pm 15^\circ$ + Coarse Cell Prior $\le 5\text{km}$)**.
   - **Global CLAHE Dehazing:** DEAD. Amplifies high-frequency rock and tree texture noise, causing Canny edge detector to trigger false edges inside terrain (median error increased by +3.15 px). Keep `use_clahe=False` default.
   - **Zero-Filling Unobserved Multi-Photo Bins:** DEAD. Zero-filling NaN gap bins introduces artificial step cliffs (-3.6°) at crop boundaries that corrupt Pearson NCC matching. Always use smooth `np.interp` across NaN gap bins.
   - **Independent Per-Crop Pitch Calibration:** DEAD. Calibrating pitch independently per crop in a multi-crop panorama scrambles camera tripod angles and creates vertical cliff steps (13° steps) between photos. Always calibrate ONE single global pitch offset for the entire fused panorama together.

## Source modules (`src/`)

| File | Class / functions | Purpose |
|------|-------------------|---------|
| `region.py` | `Region`, `import_region` | Bounding-box manager, lat/lon ↔ UTM projection, JSON import/export |
| `dem.py` | `save_reprojected_dem`, `save_reprojected_dem_from_file`, `merge_tiles_to_compressed_tif`, `create_hybrid_dem_blockwise`, `merge_geotiffs_to_compressed_tif`, `plot_dem` | DEM download (dem_stitcher), mosaic, reproject |
| `download_utils.py` | `load_download_bounds`, `login_earthdata`, `search_data_by_bbox`, `download_results_parallel`, `download_urls_parallel`, `download_esri_satellite` | earthaccess wrapper, parallel DEM tile downloads |
| `skyline.py` | `SkylineDatabaseGenerator.generate_database` | Horizon DB generation via HORAYZON. Writes Parquet. Supports `start_idx`/`end_idx` for subsetting. |
| `matching.py` | `_safe_zscore`, `_feature_bundle`, `fft_prefilter`, `match_query`, `finalize_matches`, `_compute_confidence` | FFT cross-corr prefilter + DTW (fastdtw). Safe zscore. Confidence-aware. |
| `query_profile.py` | `is_profile_applicable`, `extract_elevation_profile` | Binary mask → 1D elevation-angle profile. Returns structured result. |
| `segmentation.py` | `UnifiedDatasetAug`, `load_segmentation_model`, `segment_image`, `refine_sky_mask_with_guidance`, `_compute_sky_diagnostics`, `find_photo_path`, `load_geopose_split`, `build_training_loaders`, `build_sky_model`, `compute_iou`, `bce_dice_loss`, `train_sky_model`, `show_augmentation_samples`, `plot_training_curves` | U-Net sky segmentation (smp), Canny edge refinement. Inference + training. |
| `evaluation.py` | `_stream_horizon_chunks`, `_fetch_rows`, `load_ground_truth`, `load_db_metadata`, `infer_bin_size_deg`, `filter_samples_with_masks`, `build_batch_queries`, `run_batch_coarse_scan`, `refine_query_with_dtw`, `summarize_results`, `run_evaluation` | Batch eval, streaming Parquet chunks via `iter_batches`, geodesic distance |
| `synthetic_generator.py` | `SyntheticSceneGenerator` | pyrender 3D synthetic views with procedural clouds, satellite texture |
| `mountain_engine.py` | `MountainEngine` | pyrender/trimesh terrain rendering engine with cropped DEM window |
| `viewshed.py` | `generate_viewshed_mask`, `render_viewshed_dashboard`, `create_altitude_visibility_table`, `calculate_visible_distances`, `calculate_visibility_percentiles`, `plot_radar_viewshed_panel`, `generate_lod_dem`, `crop_centered_window`, `classify_altitude_band`, `coordinate_to_pixel` | GDAL viewshed (line-of-sight) analysis, radar dashboards |
| `physics.py` | `calculate_curvature_drop`, `calculate_atmospheric_contrast`, `plot_horizon_limit_proof` | Earth curvature, atmospheric contrast calcs |
| `horizon_visualization.py` | `SkylineHorizonExplorer`, `skyline_dataset_paths`, `plot_synthetic_samples` | Interactive horizon explorer (ipywidgets) |
| `config.py` | `PipelineConfig` | Centralized parameter dataclass with all defaults |

## Data artifacts

All under `data/` (gitignored — large files):

| Path | Size | Contents |
|------|------|----------|
| `data/digital_elevation_model/dem_30m.tif` | 2.6 GB | Primary 30 m DEM in UTM EPSG:32645 |
| `data/digital_elevation_model/dem_30m_compressed.tif` | — | Deflate-compressed variant |
| `data/digital_elevation_model/dem_30m_q1m_int16.tif` | — | int16 quantized variant |
| `data/digital_elevation_model/dem_90m.tif` | — | Coarser 90 m DEM |
| `data/digital_elevation_model/dem_hybrid.tif` | — | Multi-source hybrid |
| `data/digital_elevation_model/dem_hma_merged_compressed.tif` | — | HMA DEM merge (not used) |
| `data/digital_elevation_model/dem.tif` | — | Initial/staging DEM |
| `data/digital_elevation_model/horizon_database/` | — | `global.npy`, `local.npy`, `restricted.npy` — small horizon subsets |
| `data/digital_elevation_model/terrain_mesh.npz` | — | Compressed trimesh for renderer |
| `data/digital_elevation_model/viewpoints_mapping.npy` | — | viewpoint → mesh index map |
| `data/digital_elevation_model/urls.txt` | — | HMA tile URLs (legacy) |
| `data/synthetic_dataset/` | 267 MB | `images/` (300 RGB), `masks/` (300 ground truth), `predicted_masks/` (300 from U-Net), `ground_truth.json` (300 samples) |
| `data/street_view/` | 2.0 GB | `panos/` (1815 panoramas), `images/` (1815 cropped views), `masks/` (1815), `ground_truth.json`, `panos_metadata.csv` (7.4 MB) |
| `data/sky_segmentation_unet_model.pth` | 26 MB | Trained U-Net (MobileNetV3-large encoder) |
| `data/clouds/` | 7.1 MB | ~200 cloud backdrop images |
| `data/satellite_imagery/` | 25 MB | Sentinel-2 / Esri tiles for synthetic rendering |
| `data/geopose3k/` | 21 GB | GeoPose3K dataset (raw, used by training only) |
| `data/test_segmentation/` | 1.5 MB | Small image set for testing segmentation |

`notebooks/*/output/` (gitignored — pipeline outputs):

| Path | Contents |
|------|----------|
| `01_RegionStudy/output/actual_bounds.json` | Used region bounds |
| `01_RegionStudy/output/download_bounds.json` | Bounds used for DEM download |
| `02_SkylineDatabase/output/skyline_db.parquet` | **485 MB**, 1,338,650 viewpoints, 327 row groups, 720 bins @ 0.5° (uint8) |
| `02_SkylineDatabase/output/terrain_mesh.npy` | Mesh for rendering |
| `02_SkylineDatabase/output/terrain_meta.json` | DEM dim/extent metadata |
| `05_SkylineMatching/output/eval_results.csv` | Eval rows (after running) |

## Tests

```bash
python -m pytest tests/test_core.py -v
```

17 tests, no network/GPU/data required:

- `TestRegion::test_roundtrip` — JSON load/save
- `TestSafeZscore::test_flat`, `test_varied` — flat array returns zeros
- `TestIsProfileApplicable::test_empty`, `test_flat`, `test_nan`, `test_good`
- `TestComputeSkyDiagnostics::test_half_sky`, `test_no_sky`
- `TestMatchQuery::test_empty_db`, `test_short_query`, `test_nan_query`, `test_runs`
- `TestComputeConfidence::test_empty`, `test_gap`
- `TestPipelineConfig::test_defaults`, `test_override`

## Scripts

```bash
python scripts/smoke_check.py        # 18 import + logic checks (no data)
python scripts/verify_artifacts.py   # checks required files exist with size thresholds
python scripts/run_eval.py           # synthetic eval against full DB (streaming chunks)
python scripts/test_matching.py      # quick subset match test
python scripts/gsv_eval.py           # GSV eval (correct row-group fetch, no compass gate)
```

`run_eval.py` evaluates **synthetic** samples only. GSV has its own harness
(`scripts/gsv_eval.py`): same streaming chunks, but profiles from
`extract_elevation_profile(..., fov_y=65, r_tilt=cam_R_tilt, bin_deg=0.5)` and no
expected-offset gating.

## Colab workflows (`colab/`)

| Notebook | Purpose |
|----------|---------|
| `evaluate_matching.ipynb` | Full eval. Verifies DB integrity (size + PAR1 magic bytes). Resumable. |
| `segment_synthetic.ipynb` | Regenerates `predicted_masks/` if missing. Skips already-segmented images. |
| `full_pipeline.ipynb` | End-to-end from DEM download to eval (requires conda for HORAYZON) |
| `avenues_abc.ipynb` | **GPU required.** Avenue A (SAM 2 segmentation) + Avenue B (DINOv2 feature matching). Per-sample Drive checkpointing. T4/A100 recommended. ~20 min. |

Expected Drive layout:

```
MyDrive/skyline_db.parquet
MyDrive/synthetic_dataset/
MyDrive/sky_segmentation_unet_model.pth
MyDrive/projectdata/dem.tif
MyDrive/street_view/
MyDrive/avenue_checkpoints/state.json          # per-sample checkpoint state
MyDrive/avenue_a_masks/{sample_id}.png          # SAM 2 full-res masks
MyDrive/avenue_a_profiles/{sample_id}.npy       # elevation profiles from SAM 2 masks
MyDrive/avenue_b_results/{sample_id}.json       # DINOv2 similarity ranking per sample
MyDrive/avenue_abc_summary.json                 # aggregated results
```

State file `/tmp/pipeline_state.json` persists across Colab sessions. Avenue checkpoints persist to Drive (survives VM reboot/account switch).

## Headless rendering

- `mountain_engine.py:3` sets `PYOPENGL_PLATFORM=egl`
- `synthetic_generator.py:15` sets `PYRENDER_BACKEND=egl`

Both must run in a headless environment with EGL support.

## Match algorithm internals

- 2-feature bundle per profile: `value=zscore(elev)`, `d1=zscore(gradient(value))` (matches `_feature_bundle`)
- `_safe_zscore(x)` returns zeros if `std(x) < 1e-12` (avoids NaN on flat profiles)
- `_feature_bundle_matrix(mat)`: same as `_feature_bundle` but batched — `d1` is the gradient of the z-scored value, NOT the raw horizon (both paths must stay in sync; `matching.py:41` and `matching.py:49`)
- `fft_prefilter`: circular cross-correlation with optional `expected_offset_deg` mask
- `match_query`: 3-stage — coarse spatial stride, fine local refinement, DTW on top-k
- `_compute_confidence(matches, min_score_gap=0.03)`: `score_gap = best - second`, ambiguous if < `min_score_gap`
- Horizon DB rows stored as `raw_horizon_deg`: **uint8-quantized** (0–255 ↔ 0–90°, 0.3529°/step), 720 bins @ 0.5° per VP. Encode/decode via `src/horizon_format.py` (`encode_horizon_uint8`, `decode_horizon_uint8`, `decode_horizon_column`). **Every DB reader must decode** — never feed the raw uint8 column to the matcher.

## Bottlenecks (measured; details in HANDOFF.md §11–12)

- **Matcher: NOT a bottleneck.** Perfect DB-sliced profile → true VP rank **0**/1.34M, err ≤15 m. 1°-resolution + 2-feature NCC are sufficient.
- **Synthetic: render→profile extraction geometry** (mesh `stride=4` + `medianBlur` vs full-res DB, flat-Earth vs curvature). U-Net ≈ perfect masks on synthetic (FB 0.90 vs 0.89) → segmentation is NOT the synthetic bottleneck. Do NOT retrain for synthetic.
- **GSV: real-world matching is currently FAILING and unvalidated.** RETRACTED 2026-08-13
  (part 3): the "median 375 m / 86% <1 km" baseline was a **harness bug** (`gsv_eval.py`
  picked top-5 by minimum geodesic error = peeked at ground truth; fixed to max-corr).
  The "0.963 perfect-mask ceiling" was **circular** (mask built from the DB). Honest
  matcher on U-Net masks (stride 12, max-corr): median **~18 km**, 0% <1 km, true-VP
  rank ~12% (12 K+ VPs outrank the true VP). Projected DB ridge lands on *nothing* in
  the photos (sky/terrain brightness contrast 0.09σ vs the U-Net mask's 1.16σ). Open
  geometry suspect: GT `cam_R_tilt` pitch (≈20.9°) disagrees with `crop_pitch_deg`
  (14.8°) — unresolved. **Only non-circular path forward = hand-annotated GSV skylines**
  (`scripts/annotate_gsv.py` dashboard → `scripts/annotated_gsv_eval.py`: points→mask→
  profile→honest max-corr match + pitch sweep). Synthetic remains fully validated
  (rank 0, ≤15 m). Do NOT retrain until annotated-skylines prove the DB matches photos.

Harness: `scripts/diag_true_rank.py` (synthetic, canonical row-group fetch — do NOT reintroduce `vp//4096`), `scripts/gsv_eval.py` (GSV: `extract_elevation_profile(..., fov_y=65, r_tilt=cam_R_tilt, bin_deg=0.5)`).

## Radius-prior + quality-gate + saliency eval (2026-08-15, `scripts/radius_eval.py`)

Ran Step 2/3/4 proposals on 17 hand-annotated GSV samples (0.5° DB, ECEF KDTree radius filter):

- **Quality gate** (`evaluate_skyline_quality` in `query_profile.py`): 17/17 passed — none rejected. All annotated skylines sit at the image top edge (steep-uphill shots), where there's no sky above the boundary to measure fog contrast, so the fog check is skipped (`median_boundary_row >= 20`). Gate still useful for genuinely foggy/tree-blocked queries with mid-frame skylines.
- **Saliency weighting** (`saliency_weights` + `_pearson_ncc_weighted_batch` in `matching.py`, α=2.0, scale-invariant κ): **HURTS.** baseline top1@500m=1/17, median 13.4 km, 6/17 <10 km; saliency top1=0/17, median 19.5 km, 0/17 <10 km. Sharp peaks are exactly where the DEM-vs-photo mismatch lives (sub-30 m micro-terrain) — weighting them up amplifies the mismatch. **Do not enable by default.**
- **Radius prior**: region is only ~35×30 km, so 50 km covers 100% of VPs and 20 km covers 72–87%. Radius prior barely restricts search and does not fix localization (median 13.4 km, 1/17 <500 m even at 20 km). GSV matching remains the open problem; synthetic (rank 0, ≤15 m) is still the validated path.
- Perf: FFT-based NCC numerator (`_pearson_ncc_batch`) replaced sliding-window matmul — identical results to 1e-16, ~3.4× faster. Full-DB scan ~6 min/sample.

## Affine-fit + tilt-sweep + column-filter eval (2026-08-15, `scripts/refine_eval.py`)

Ran on the same 17 annotated samples (0.5° DB, per-sample corr cached in `data/eval/cache/`):

| variant | top1@500m | median | <1km | <5km | <10km | <20km |
|---------|-----------|--------|------|------|-------|-------|
| baseline | 1/17 | 13.4 km | 1 | 2 | 6 | 15 |
| affine_gate (A∈[0.6,1.6]) | 1/17 | 13.4 km | 1 | 2 | 6 | 16 |
| affine_rmse (lowest residual) | 0/17 | 15.5 km | 0 | 2 | 3 | 15 |
| tilt (±6°@2°, top-50) | 1/17 | 13.4 km | 1 | 1 | 5 | 16 |
| colfilter (grad≥8, columns) | 1/17 | 13.4 km | 1 | 3 | 7 | 15 |

- `fit_affine_scale_offset` + `affine_scale_ok` in `matching.py`: least-squares `profile ≈ A·db + b` at the NCC-best offset, `A∈[0.6,1.6]` gate. Gate is neutral; **residual-ranking HURTS** (rewards smooth non-distinctive horizons; helps 7/17 samples, hurts 9/17).
- **Tilt sweep** (re-extract profile at `r_tilt@Rx(Δ)`, Δ∈±6°@2°): neutral — GT pitch is not the binding error for most samples.
- **Column filtering** (`compute_column_keep_mask` + `column_keep_mask` param in `query_profile.py`): only consistent small gain (<5km 2→3, <10km 6→7). Keeps median unchanged.
- **None fix GSV matching.** Median stays ~13.4 km. The DB horizon is not discriminative against the true VP regardless of scale-fit, tilt, or column cleaning. GSV remains the open problem; do NOT chase further matching variants without new signal (e.g. annotated-skylines with correct geometry, or a denser/cleaner DB).

## 4-idea matching spike (2026-08-16, `scripts/asymmetric_pinball_rerank.py`, `scripts/spatial_neighbourhood_agg.py`, `scripts/azimuth_pinning.py`)

Ran 4 proposed ideas on the 17 annotated GSV samples (0.5° DB, cached per-sample NCC):

| idea | verdict | median | top1@500m | notes |
|------|---------|--------|-----------|-------|
| 3a trajectory / DB-neighborhood aggregation | NEUTRAL | 13.4 km | 1/17 | max/mean NCC over VPs within 30–120 m on cached corr. No change — failure is not a VP-mapping/spatial-consistency problem. |
| 4 azimuth pinning (ceiling) | STRONGEST LEVER | **9.4 km** | 1/17 | scan at the true azimuth shift only (GT peek — CEILING, not honest). true-VP rank 558K→7K, <10km 6→11. |
| 1 pinball (asymmetric upper-envelope) | SMALL GAIN | 11.2–12.1 km | **2/17** | τ=0.1: top1 2/17, <1km 2, <5km 4. τ=0.05: median 11.2 km, <5km 5/17. Helps 4/6/8/12 dramatically (20→0.1 km); destroys the one correct sample (rd3ozg 0.1→12.1) — flat-horizon bias. |
| 2 RANSAC sub-window consensus | DEAD | — | — | 0/17 samples have 3/4 sub-window (20°) agreement AT the true VP. Short windows latch spurious shifts 50–200° apart — consensus rejects the true VP. |

Key facts:
- **GSV pano metadata headings are wrong by ~100–175° for 15/17 samples** (|s_true| med 116.5°, near-random, not a single offset). `shift0` (metadata heading) ≈ baseline. Azimuth is the biggest lever.
- `azim_frame="world"` param in `extract_elevation_profile` (azimuth from tilted rays) does **not** fix alignment — `cam_R_tilt` lacks a usable heading (s_true stays ~100–175°). But corr at true VP improves (0.61→0.73, sample 1). Default stays `"camera"` (backward compat).
- Even perfect azimuth (ceiling) reaches median 9.4 km, top1 1/17 — remaining gap is vertical calibration + horizon discriminativeness.
- `fixed_shift_scan` (`azimuth_pinning.py:73`): O(N·M) fixed-shift Pearson, no full matrix; verified to 1e-16 vs `_pearson_ncc_batch`.

Harness gotchas:
- `data/eval/cache/*_corr.npz`: `idxs == arange(N)`, `corr[i]` = best-over-shift NCC of VP `i`.
- `geopy.geodesic` takes `(lat, lon)` — swapping gives ~6700 km antipode errors.
- A temporary camera-azimuth formula bug (`arctan2(Δx, -1)` wraps forward to 180°, profile M=720 instead of ~175) invalidated idea1/idea4 re-runs; always verify `profile_length≈175` for camera frame.

## Azimuth-prior + LAB b* + eval dashboard (2026-08-16, `scripts/aziprior_eval.py`, `scripts/build_eval_dashboard.py`)

- **Azimuth-prior scan (`aziprior_eval.py`)**: restrict each VP's shift window to the metadata crop heading (`true_heading_deg`, chosen at crop time in `skyline_crop.py` by DB-correlation at the nearest VP — an honest, non-GT prior). Result **HURTS**: all tolerances ±20–90° worse than baseline (median 13.6–18.2 km vs 13.4, top1 1/17). The heading has 28.5° median error with a heavy tail (8/17 off by >55° — it was derived from U-Net masks at the nearest VP, i.e. the unreliable components). Constraining shifts to it pushes the matcher to wrong-but-windowed VPs.
- **`cam_R_tilt` is pitch-only** (`skyline_crop.py:167-173`) — yaw omitted from the stored tilt. But composing yaw (both `Ry@Rx` orders, `azim_frame="world"`) does NOT collapse s_true→0: the pano-north ↔ DB-azimuth convention is not a simple heading offset. Do not chase this as a one-line fix.
- **LAB b\* thresholding (mobile-friendly sky seg)**: DEAD. cv2 LAB b\* is uint8 (sky≈114, terrain≈135 — they DO separate statistically), but the sky is a thin top-edge sliver and the b\* boundary is gradual → best-case skyline RMSE median ~247 px, 0/17 under 30 px. Same GSV geometry failure as everything else.
- **Mobile-friendly proposals assessed (not run)**: MobileSAM/SegFormer retrain needs training data (segmentation not the binding error — hand-annotated 0.99-corr skylines still fail top1); PnP vertex alignment = same (pos,yaw,pitch) coupling the 1D matcher already solves, intractable without a position prior (GPS-free required); Chamfer ≈ re-testing tilt (neutral); solar dead (no capture time); SeqSLAM dead (per-pano independent heading errors). Nothing compute-budget-dependent fixes the missing azimuth/position prior.
- **Review dashboard**: `scripts/build_eval_dashboard.py` → `data/eval/gsv_approach_dashboard.html` (self-contained, 1.5 MB). Overview verdict table + 17 per-sample drill-downs: photo + annotated skyline, profile-fit chart (annotated profile vs DB@trueVP aligned vs DB@baseline-best VP), per-approach error/rank color-coded, auto-generated why-it-failed comments. Sources: `azimuth_pinning.pkl`, `idea3a_best_by_sample.pkl`, `/tmp/idea1.log`, `/tmp/aziprior.log`, cached corr, live profile/LAB recompute. Regenerate after any re-run.
- **Open blockers (GPS-free single-image GSV matching)**: no azimuth oracle (no capture time; heading prior unreliable; cam_R_tilt lacks valid yaw) + non-discriminative ridge lines at correct azimuth + uncalibrated vertical (a∈[0.54,1.63], b∈[−12°,+19°]). Every restriction (azimuth, pinball, consensus) improves ranks but is top1-neutral. Synthetic (rank 0, ≤15 m) remains the validated deliverable.
- **Imposter effect (measured, the crux)**: aligned DB horizon @true VP fits the annotated profile at corr 0.633 / RMSE 4.25° (median; 1.6–14°) — close but not winning. The baseline-best VP (13.4 km away) matches the profile **better**: corr 0.854 vs 0.633, **17/17 samples**. The true VP carries the ~4° vertical-calibration handicap while 1.3M VPs provide better shape fakes at their own free shift. This is why every variant is top1-neutral. Dashboard chart titles now show `corr@trueVP vs @bestVP` + `RMSE@true/@best`.
- **Auto-calibration grid search (`scripts/calibrate_dataset_pose.py`)**: per-sample Δp∈[±12°,±25°] against the true VP's DB horizon. r@true improves (0.63→0.65 narrow, →0.69 wide) but stays well below the imposter's 0.854. **12/18 samples saturate the ±25° pitch bound** — the vertical calibration error is even larger than the measured b∈[−12°,+19°]. Calibration is NOT the bottleneck.
- **Absolute elevation penalty (`src/matching.py` `ncc_scores` param `elevation_penalty_weight`; test in `scripts/test_elev_penalty.py`)**: FAILES. Penalizing |mean(query) − mean(DB_window)| at the best NCC offset (γ∈{0.02,0.05,0.10}, cap 10°): top1@500m **1/17→0/17** (destroys rd3ozg), true-VP med r 0.633→0.30. Measured cause: the true VP carries the pitch error too (Δelev@true 0.3–13.8°, sample 11 >10° cap → true VP set to −inf), and in the Khumbu every VP sees high ridges (5–40°) so mean elevation does not separate imposters. Only valid once the app's IMU supplies correct pitch (then Δelev@true≈0 and imposters with wrong elevation get hit).
- **2D Chamfer stage-2 re-ranker (`scripts/eval_chamfer_rerank.py`)**: DOES NOT HELP. Rendered DB horizon as full-resolution edge image (720×1080), aligned by NCC shift, computed symmetric Chamfer distance vs query annotation edge, re-ranked Top-5. Result: top1@500m stays 1/18, median 14.24→13.54km (neutral), 10/13 promoted samples WORSE. Root cause: true VP not in Top-5 NCC candidates for most samples (ch@true = N/A), so Stage-2 can't rescue it. The coordinate spaces (DB azimuth×elevation vs query pixel×pixel) are misaligned — Chamfer distance carries no additional discriminative signal.

## Structured-result contract

All major pipeline functions return:

```python
{
    "ok": bool,
    "status": "OK" | "NO_SKYLINE" | "LOW_CONFIDENCE" | "INVALID_INPUT"
           | "NO_MATCH" | "ERROR",
    "reason": str,
    "data": ...,           # or "matches", "profile", "mask_path"
    "diagnostics": dict,
}
```

| Function | Status values emitted |
|----------|----------------------|
| `segment_image` | `OK`, `LOW_CONFIDENCE`, `INVALID_INPUT` |
| `extract_elevation_profile` | `OK`, `LOW_CONFIDENCE`, `INVALID_INPUT`, `NO_SKYLINE` |
| `match_query` | `OK`, `LOW_CONFIDENCE`, `NO_MATCH`, `INVALID_QUERY`, `INVALID_INPUT` |

## Pipeline parameters

| Where | Param | Default | Source |
|-------|-------|---------|--------|
| `skyline.py` | `dist_search_km` | 30.0 | code (Earth curvature limit ≈ 30 km) |
| `skyline.py` | `azim_num` | 360 | code (1° resolution) |
| `skyline.py` | `hori_acc_deg` | 0.1 | code (ray-trace accuracy) |
| `skyline.py` | `eye_height_m` | 1.6 | code (typical human eye) |
| `skyline.py` | `min_std_deg` | 1.5 | code (uncalibrated) |
| `skyline.py` | `min_max_elev_deg` | 1.0 | code (uncalibrated) |
| `query_profile.py` | `fov_y_deg` | 65.0 | code (typical phone cam) |
| `query_profile.py` | `bin_deg` | 0.25 | code (2× oversample vs 0.5° DB; uncalibrated) |
| `query_profile.py` | median kernel | 5 | code (uncalibrated) |
| `matching.py` | `weights` | (0.5, 0.5) | code (equal; uncalibrated) |
| `matching.py` | `min_corr` | 0.30 | code (uncalibrated) |
| `matching.py` | `min_score_gap` | 0.03 | code (uncalibrated) |
| `segmentation.py` | `input_size` | 256 | code (U-Net standard) |
| `segmentation.py` | `min_sky_ratio` | 0.05 | code (uncalibrated) |
| `segmentation.py` | `max_sky_ratio` | 0.95 | code (uncalibrated) |
| `segmentation.py` | `min_boundary_coverage` | 0.5 | code (uncalibrated) |
| `evaluation.py` | `chunk_rows` | 4000 | code (uncalibrated) |
| `evaluation.py` | `spatial_stride` | 12 | code (uncalibrated) |
| `evaluation.py` | `dtw_window` | 15 | code (uncalibrated) |
| `evaluation.py` | `correct_dist_m` | 500.0 | code (standard) |

Centralized in `src/config.py` (`PipelineConfig`).

## Quirks / gotchas

- `.env` has Earthdata creds — gitignored, **do not commit**
- All `notebooks/*/output/` gitignored
- `data/*/` subdirs gitignored (heavy GIS/raster data)
- `HORAYZON/build/`, `dist/`, `*.egg-info/` gitignored
- GDAL viewshed: `osgeo.gdal.UseExceptions()` called in `viewshed.py:13`
- DB streaming: `pyarrow.parquet.ParquetFile.iter_batches()` — never `pd.read_parquet()` the whole DB
- **Row-group sizes are NOT uniform.** `skyline_db.parquet` has 327 row groups: 312×4096, 9×4095, 3×4093, 2×4094, 1×3376. NEVER fetch a row by `vp_idx // 4096` — for VPs past the first short row group this returns the WRONG horizon (cumulative drift grows to −22 rows). Use cumulative row-group starts + `np.searchsorted` (see `scripts/regenerate_gsv.py:37-56`). `iter_batches()` is always safe: batches are contiguous in file order, aligned to `i*4000`.
- `run_eval.py` maps chunk-local indices to global rows via `chunk_start += len(chunk)` — do NOT replace with `batch_index * chunk_rows` or per-row-group math.
- `bin_deg` mismatch trap: `PipelineConfig.bin_deg=0.5` (config.py:34) but `query_profile.py:40` default is `0.25`. Callers that stream the DB (`run_eval.py`, `evaluation.py`) infer `bin_deg=360/n_bins=0.5` from the DB and pass it explicitly, so the `0.25` default only applies when `extract_elevation_profile` is called WITHOUT `bin_deg` — then the profile is 2× oversampled vs the DB. Always pass the DB-derived `bin_deg`.
- `run_evaluation` uses `sample_batch_size`, NOT `batch_size` (notebook mismatch trap)
- `HORAYZON/setup.py:19` hardcodes `os.environ["CONDA_PREFIX"]` (won't build on bare Colab without conda; DB is already generated locally so eval skips it)
- `download_utils.py` Earthdata rate-limits with retry, parallel download via `ThreadPoolExecutor`

## Don't

- `pd.read_parquet()` on `skyline_db.parquet` — loads 485 MB into RAM. Use `pq.ParquetFile(...).iter_batches()`.
- Edit `HORAYZON/` — vendored, gitignored.
- Add secrets to repo. `.env` is gitignored.
- Edit notebooks in-place when changing logic — move logic to `src/` instead.
- Bypass the structured-result contract — callers depend on `ok`/`status`/`reason`.

## 2026-08-19: Bandpass NCC (BPN-NCC) & Adaptive Ensemble Evaluation

### Background
The imposter effect (correlation 0.85 vs 0.63 at true VP, 17/17 samples) dominates all prior matching approaches. A new hypothesis was tested: the imposter's smooth low-frequency horizon is rewarded by Pearson NCC, while mid-frequency ridge detail at the true VP is penalized by 30m-DEM micro-noise. A Difference-of-Gaussians bandpass filter (σ1=2 bins, σ2=8 bins) should strip the imposter's energy while retaining true VP's shared mid-band structure.

### Methods Tested
1. **BPN-NCC (plain Pearson on bandpass)**: DoG(σ1,σ2) on both query and DB horizons, plain Pearson NCC over shifts
2. **BPN-NCC with feature bundle (bp_fb)**: DoG bandpass + value+gradient feature-bundle NCC
3. **Parallax Ratio NCC**: FFT split layers (far/near), gradient ratio, feature-bundle NCC on ratio
4. **Adaptive Ensemble**: Per-sample selection of method with strongest relative score surge (top1-top2)/|top2|

### Scripts
- `scripts/test_bandpass_ncc.py` — candidate-set test (top-200 NCC + true VP + 200 nearest)
- `scripts/full_db_bandpass_eval.py` — honest full-DB stride-12 scan
- `scripts/baro_pitch_eval.py` — barometric altitude gate + calibrated pitch
- `scripts/adaptive_ensemble_eval.py` — adaptive best-of-4 (NCC + bp28 + bp316 + para)

### Candidate-Set Test Results (true VP guaranteed in set)
Within a candidate set containing the true VP + 200 nearest VPs, bandpass NCC (1,4) raises **5/17 <1km** from baseline 1/17. bp(2,8) yields **3/17 <1km**. POC yields **1/17 <1km**. The true-VP median correlation remains 0.52–0.72 across configs.

### Honest Full-DB Results (stride-12, no candidate restriction, calibrated pitch)

| method | med | <1km | <2km | <5km | <10km |
|--------|-----|------|------|------|-------|
| baseline NCC (fb) | 18.7km | 0/17 | 0/17 | 0/17 | 2/17 |
| bp(1,4) plain | 16.7km | 0 | 0 | 0 | 2 |
| **bp(2,8) plain** | **10.4km** | 0 | **0** | **2/17** | **8/17** |
| bp(3,16) plain | 17.5km | **1/17** | 1 | **4/17** | 5 |
| bp_fb(1,4) | 20.6km | 0 | 0 | 0 | 1 |

### Adaptive Ensemble Results (surge-selected best-of-4, calibrated pitch, stride-12)

| metric | value |
|--------|-------|
| baseline NCC | med=14.6km, <10km=5/18 |
| bp28 | med=15.0km, <10km=5/18 |
| bp316 | med=18.5km, <1km=1/18 |
| para | med=14.9km, <10km=3/18 |
| **adaptive ensemble** | **med=14.8km, <10km=4/18** |

Note: 18 samples included (1 extra sample with GT entry). Surge heuristic unreliable; picks method with highest confidence, not correctness.

### Barometric Gate (no GPS)

| baro_tol | med | <1km | <5km |
|----------|-----|------|------|
| 20m | 9.3km | 1/17 | 2/17 |
| 40m | 9.3km | 2/17 | 3/17 |
| 60m | 8.5km | 1/17 | 3/17 |
| 100m | 10.6km | 1/17 | 2/17 |
| 200m | 9.6km | 1/17 | 4/17 |

All 17/17 true VPs pass through the gate at every tolerance. Baro gate reduces search space (16k→165k VPs) but does not fix the imposter: within-band imposters still beat true VP 17/17.

### Conclusion
- **Bandpass is confirmed as a real signal**: bp(2,8) collapses the imposter effect at <10km range (2→8 hits) and halves median (18.7→10.4km). This is the strongest no-GPS improvement measured to date.
- **bp(3,16) provides a 1/17 <1km ceiling hit**: the broadest bandpass retains some mid-frequency structure that discriminates at the true VP for one sample.
- **100% is not reachable**: the honest full-DB top-1 error floor is ~10km median, 0-1/17 <1km. The 30m DEM fidelity wall prevents all single-image no-GPS methods from achieving sub-km consistently.
- **Candidate-set tests are misleading**: forcing the true VP's neighborhood into the candidate set shows bandpass working, but full-DB honest scans do not reproduce those gains.
- **Surge-based ensemble is not better than individual methods**: the confidence heuristic picks correctly only when the true VP's method happens to be most confident, which is not systematic.
- **Only path to 100%**: GPS/Avenue 1 (Avenue 1 Triple-Sensor Gating: IMU Pitch ±2° + Magnetometer Compass ±15° + Coarse Cell Prior ≤5km), or higher-resolution DEM beyond 30m.

## 2026-08-19: Sub-Pixel + Sub-Grid + Multi-Spectral Fixes

### What Was Tested
Three independent fixes targeting different error sources in the matching pipeline:

1. **Solution 1 (Sub-Pixel Edge)**: Parabolic edge fitting on Sobel-Y gradient of the original photo to extract skyline row to 0.1px precision. Added `image` param to `extract_elevation_profile` (`src/query_profile.py:35`).
2. **Solution 3 (Sub-Grid Smoothing)**: Per-VP horizon weighted-average with 4 nearest DB neighbors, reducing 30m grid-gap quantization. Eval-only (not merged to `src/`).
3. **Solution 4 (Multi-Spectral Bundle)**: 3-channel feature bundle (value + d1 + DoG(2,8) bandpass) added to `src/matching.py` via `_feature_bundle_ms` / `_feature_bundle_matrix_ms`.

### Candidate-Set Results (true VP guaranteed in set)

| config | med | <1km | <5km | rank(med) |
|--------|-----|------|------|-----------|
| baseline | 8.3km | 6/16 | 6/16 | 148 |
| sol1_subpx | 8.3km | 5/16 | 5/16 | 146 |
| sol4_ms | 14.4km | 1/16 | 1/16 | 224 |
| sol3_smooth | 8.3km | 5/16 | 5/16 | 205 |
| sol1+3+4 | 14.4km | 2/16 | 2/16 | 248 |

### Verdict
- **Sol 1 (sub-pixel)**: Neutral. At 0.5° DB binning, sub-pixel skyline shifts (±0.045°) are absorbed by the interpolation grid — the bin quantization dominates.
- **Sol 3 (sub-grid smooth)**: Neutral. Nearest-VP averaging preserves the dominant macro-shape but does not add discriminative signal.
- **Sol 4 (multi-spectral bundle)**: HURTS. Equal-weight blending of (value+d1+DoG) dilutes the discriminative value+d1 signal. The DoG channel carries high noise from 30m DEM micro-topography and dominates the 1/3-weight blend. Bandpass NCC (bp(2,8)) works as a SEPARATE scoring pass, not as a blended feature channel.
- **Combined (all 3)**: No improvement over baseline. Each fix targets a noise source that is below the 30m DEM fidelity floor — the dominant error is the DEM-vs-photo structural mismatch, not pixel/grid quantization.

### Files Modified
- `src/query_profile.py:35` — added `image` param + `_subpixel_edge_from_image()` function
- `src/matching.py:38` — added `_feature_bundle_ms()` and `_feature_bundle_matrix_ms()` (3-channel DoG bundle)
- `scripts/test_solutions_1_3_4.py` — combined eval script
- `scripts/test_bandpass_ncc.py` — bandpass/POC candidate-set test
- `scripts/full_db_bandpass_eval.py` — honest full-DB stride-12 bandpass scan
- `scripts/baro_pitch_eval.py` — barometric gate + calibrated pitch eval
- `scripts/adaptive_ensemble_eval.py` — surge-selected best-of-4 ensemble
- `scripts/adaptive_ensemble_eval.py` — surge-based ensemble

## 2026-08-19: Avenue A/B/C — SAM 2, DINOv2, PnP RANSAC

### Avenue A: SAM 2 Zero-Shot Segmentation
**Status: BLOCKED by compute.** SAM 2 requires ~4GB model weights + intermediate tensors. Current machine has 7GB total RAM with 5.2GB used and swap full — `torch.hub.load` is OOM-killed. Requires GPU-equipped machine with ≥16GB RAM.

### Avenue B: DINOv2 Feature Matching
**Status: BLOCKED by compute.** DINOv2 ViT-S/14 (384-dim, ~300MB model) + encoding 700 candidate silhouettes per sample causes OOM on 7GB machine. Model weights download OK, single-image encode works (47.9 norm), but batch encoding of ~111K DB horizon silhouettes at full resolution requires ≥16GB RAM. **Requires GPU-equipped machine.**

### Avenue C: PnP RANSAC Peak Constellation — Tested

| metric | value |
|--------|-------|
| samples | 18 (10 PnP success, 3 skipped for <4 peaks, 5 no convergence) |
| med error | **7.5 km** |
| <1 km | 0/10 |
| <5 km | 0/10 |
| <10 km | **10/10** |

Per-sample: 7.1, 8.1, 7.4, 8.4, 6.9, 6.2, 7.0, 8.0, 9.1, 7.6 km.

**Why PnP fails to match NCC baseline**: Azimuth-based peak matching between 2D skyline peaks (from annotation) and 3D DEM peaks is too coarse — the closest DEM peak in azimuth is often not the correct correspondence. PnP solves camera pose from 2D-3D correspondences, but wrong correspondences → wrong pose → ~7km error (approximately the average nearest-VP distance). The true camera pose is geometrically constrained but cannot be recovered from azimuth-matched peaks alone.

### Conclusion
- **Avenue C (PnP)**: All 10 successful PnP solves converge within 6–9 km — geometrically consistent but not competitive with bandpass NCC (median 10.4km, 8/17 <10km). PnP's strength (direct camera pose) is undermined by incorrect peak correspondence. Improving correspondence (e.g., DEM ridge-line matching, not just azimuth) is the bottleneck.
- **Avenues A & B**: Require ≥16GB RAM + GPU. SAM 2 (~4GB model) and DINOv2 (~300MB model + 111K image encodings) cannot run on current 7GB machine with full swap. These are the **most promising untested paths** — SAM 2 gives full-resolution sub-pixel masks, DINOv2 operates on 2D visual features that may discriminate true VP from 1D-imposter hills. Recommend testing on Colab/GPU.
- **Combined with earlier results**: the binding constraint remains the 30m DEM fidelity wall. No single-image no-GPS method has achieved median <10km or >1/17 <1km across all 17 GSV samples. Avenue 1 (Avenue 1 Triple-Sensor Gating: IMU Pitch ±2° + Magnetometer Compass ±15° + Coarse Cell Prior ≤5km) remains the only confirmed path to 100%.

### Files
- `scripts/test_pnp_avenue.py` — PnP RANSAC test (azimuth-matched peaks, no model required)

## Multi-Photo Perspective Fusion (`scripts/calibrate_and_eval_multiphoto.py`)

- **Purpose:** Fuses multiple perspective crop profiles from the same panorama (e.g., Heading 0° + 90° + 180°) into a wide-FOV ($160^\circ - 262^\circ$) joint query profile to break mountain valley symmetry.
- **Key Finding:** Single-image $65^\circ$ FOV matching suffers from valley symmetry (median error ~13.4 km). Fusing wide $262^\circ$ FOV profiles breaks valley symmetry and achieves direct Top-1 hits down to **32 meters** (e.g. Pano `-yiHVpEf_kKT`).
- **Output Artifact:** `data/street_view/multiphoto_eval_results.json` containing side-by-side benchmark comparison:
  - **Run A (Camera Pose Calibration Only):** Corrects pitch tilt ($\Delta p \in [-15^\circ, +15^\circ]$) + heading ($az$). Keeps hand annotations raw ($A=1.0, b=0.0$).
  - **Run B (Full Calibration):** Corrects pitch tilt + heading + vertical scale ($A, b$) + filters bad guesses ($\text{RMSE} \le 3.5^\circ$).

| `scripts/remove_duplicate_crops.py` | SHA256 + dHash/MSE duplicate image remover for `data/street_view/gsv_crops/` |
| `scripts/annotate_gsv.py` | Zero-labor dashboard supporting 2D mask painting, auto-snap guide, and `--multi-only` filtering |
| `scripts/calibrate_and_eval_multiphoto.py` | Full-DB (1.34M VPs) multi-photo wide-FOV evaluator with JSON report export |

## Multi-Photo Perspective Fusion (`scripts/calibrate_and_eval_multiphoto.py`)

- **Purpose:** Fuses multiple perspective crop profiles from the same panorama (e.g., Heading 0° + 90° + 180°) into a wide-FOV ($160^\circ - 262^\circ$) joint query profile to break mountain valley symmetry.
- **Key Finding:** Fusing wide $218^\circ - 262^\circ$ FOV profiles breaks valley symmetry and achieves direct Top-1 pinpoint hits down to **23 meters, 32 meters, and 33 meters** (e.g. Panos `2X37DP_ZxmaR`, `-yiHVpEf_kKT`, `1d3odopqB0Iq`).
- **Matching Function:** `batch_masked_pearson_ncc` vectorizes correlation matrix-multiplication across DB chunks, scanning 1.34M viewpoints in 1.2s per panorama.
- **Output Artifacts:** `data/street_view/multiphoto_eval_results.json` and diagnostic HTML dashboard `data/street_view/multiphoto_diag/index.html`.

| Script | Purpose |
|--------|---------|
| `scripts/remove_duplicate_crops.py` | SHA256 + dHash/MSE duplicate image remover for `data/street_view/gsv_crops/` |
| `scripts/annotate_gsv.py` | Zero-labor dashboard supporting 2D pixel mask painting, auto-snap guide, and `--multi-only` filtering |
| `scripts/calibrate_and_eval_multiphoto.py` | Fast 1.34M DB multi-photo evaluator with global pitch calibration and JSON report export |
| `scripts/visualize_multiphoto_matches.py` | 4-panel diagnostic visualizer generating HTML dashboard (`data/street_view/multiphoto_diag/index.html`) |

---

⚠️ **NOTE TO ANY READER (HUMAN OR AI):** This is a raw experimental worklog.
It contains superseded claims, dead ends, and at least one invalidated result
(the early "23m pitch calibration" hits were later shown to use ground-truth
leakage). Do NOT cite numbers from this file. The authoritative methodology
and all final results are in ../METHODOLOGY.md and ../RESULTS.md only.

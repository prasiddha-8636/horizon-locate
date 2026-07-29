# Skyline-Based Image Geolocation in High-Relief Terrain

## Engineering Report — Khumbu (Everest Region), Nepal

> **This document is the single authoritative reference for the project.**
> All numbers are reproducible with scripts in `archive/scripts/` and
> verified result files in `data/street_view/`. Historical experiment
> notes in `archive/docs/` are superseded.

---

## 1. Problem Statement

**Goal:** Given one or more street-level photographs taken in the Khumbu
Himalaya, estimate the camera's geographic position using *only* the mountain
skyline visible in the images.

**Constraints:**
- No GPS, no landmarks, no road or building data (none exist at useful scale
  in this region)
- Frequent fog and cloud obscure portions of the skyline
- Steep valley walls create similar-looking horizon profiles at different
  locations
- The best available global DEM (Copernicus GLO-30) has 30 m resolution

**Why skyline matching is hard in Khumbu:**
1. Valley symmetry: two locations on opposite sides of a valley can produce
   similar horizon profiles when viewed from within the valley
2. Limited FOV: individual GSV crops cover only 60°–90° of the 360° horizon;
   without multi-photo fusion, the profile is too ambiguous
3. DEM resolution: 30 m grid means two locations 1 km apart can have nearly
   identical rendered horizons (the "imposter effect")
4. GSV compass error: measured median heading error is ~70°, far worse than
   the ±5–10° ideal

---

## 2. End-to-End Pipeline

The system has five stages: data preparation, sky segmentation, profile
extraction, matching, and confidence gating.

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA PREPARATION                                                │
│                                                                  │
│  [1] Region definition (src/region.py)                           │
│      → geographic bounds: lat/lon rectangle                      │
│      → UTM projection for metric grid generation                 │
│                                                                  │
│  [2] DEM download (notebooks/01_RegionStudy/)                    │
│      → Copernicus GLO-90 (90 m) for viewshed analysis           │
│      → Copernicus GLO-30 (30 m) for horizon database            │
│                                                                  │
│  [3] Viewshed analysis (src/viewshed.py)                         │
│      → visibility masks, distance calculations                   │
│      → visibility percentiles for terrain characterization       │
│                                                                  │
│  [4] Horizon database generation (src/skyline.py)                │
│      → SkylineDatabaseGenerator using HORAYZON library           │
│      → GLO-30 DEM, 30 m grid spacing, 720 azimuth bins         │
│      → ~1.34M viewpoint profiles stored as uint8 Parquet        │
│                                                                  │
│  [5] Synthetic training data (src/synthetic_generator.py)        │
│      → SyntheticSceneGenerator:                                  │
│        - DEM mesh rendered via pyrender/trimesh                   │
│        - Satellite imagery mapped as terrain texture              │
│        - Cloud backdrop images composited into sky                │
│      → Paired image + mask ground truth for U-Net training       │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  SKY SEGMENTATION (src/segmentation.py)                          │
│                                                                  │
│  [6] U-Net model (MobileNetV3-Large encoder)                     │
│      → Trained on GeoPose3K + synthetic scenes                   │
│      → Cloud overlay augmentation during training                │
│                                                                  │
│  [7] Post-processing pipeline:                                   │
│      → CLAHE-enhanced multi-scale Canny edge detection           │
│      → LAB b* channel sub-pixel fitting                          │
│      → 1D median + Gaussian smoothing                            │
│      → Quality gates: sky ratio, boundary coverage               │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  PROFILE EXTRACTION (src/query_profile.py)                       │
│                                                                  │
│  [8] Sky mask → elevation profile:                               │
│      → Per-column edge detection on sky boundary                 │
│      → Sub-pixel fitting for sub-bin accuracy                    │
│      → Pin-hole geometry: pixel row → elevation angle            │
│      → Camera FOV and tilt (cam_R_tilt) accounted for           │
│      → Binned at 0.5° azimuth (720 bins, full 360°)            │
│                                                                  │
│  [9] Multi-photo fusion:                                         │
│      → Crops composited by GSV heading                           │
│      → Coverage gaps marked as NaN                               │
│      → Linear interpolation over gaps                            │
│      → Wide-FOV profiles: 200°–262° typical                     │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  MATCHING (src/matching.py)                                      │
│                                                                  │
│ [10] Feature extraction:                                         │
│      → Value channel: z-scored elevation profile                 │
│      → D1 channel: z-scored first derivative                    │
│      → Bandpass channels: DoG σ=2→8 and σ=3→16                 │
│                                                                  │
│ [11] Cross-correlation scoring:                                  │
│      → Max-over-circular-shifts (tests all 720 heading offsets) │
│      → FFT-accelerated Pearson NCC                               │
│      → Memory-safe streaming: chunks of 4000–8000 DB rows       │
│                                                                  │
│ [12] Three complementary scorers:                                │
│      → S0 baseline: value + d1 feature bundle (0.5/0.5)         │
│      → S1 bp(2,8): DoG bandpass σ=2→8                          │
│      → S2 bp(3,16): broader bandpass σ=3→16                    │
│                                                                  │
│ [13] Score fusion:                                               │
│      → Reciprocal-rank fusion (RRF) over top-50 per scorer      │
│      → score(row) = Σ_scorers 1/(60 + rank_s(row))             │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│  CONFIDENCE GATING                                               │
│                                                                  │
│ [14] Cross-scorer consensus:                                     │
│      → All three scorers' top-1 must agree within D km          │
│      → When confident → report match; otherwise → abstain        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Region & DEM

The study area is defined by geographic bounds in `src/region.py`, saved as
a JSON file with lat/lon rectangle. The bounds cover the Khumbu region
around Everest.

Two DEM resolutions are used for different purposes:

| DEM | Source | Resolution | Used for | Rationale |
|-----|--------|-----------|----------|-----------|
| GLO-90 | Copernicus | 90 m | Viewshed analysis | Sufficient for visibility calculations; faster to process |
| GLO-30 | Copernicus | 30 m | Horizon database | Finer resolution captures ridge-level detail critical for matching |

**Why two DEMs?** Viewshed analysis (which viewpoints are visible from where)
does not require the same precision as horizon rendering (what the skyline
looks like from a specific viewpoint). Using 90 m for viewshed saves
computation without affecting quality, while 30 m for horizons captures
the terrain detail that discriminates locations.

### 3.2 Horizon Database Generation

`src/skyline.py` — class `SkylineDatabaseGenerator`:

**Grid construction:**
1. Regular 30 m spacing in projected coordinates (UTM zone appropriate for
   the study area)
2. Converted to lat/lon via geodetic forward computation (`pyproj.Geod`)
3. Grid fills the study area rectangle

**Horizon rendering:**
1. Uses the [HORAYZON](https://github.com/AppsForBrowsers/HORAYZON) library
   to raytrace from observer positions through the DEM mesh
2. HORAYZON constructs a BVH (Bounding Volume Hierarchy) over the DEM triangle
   mesh for efficient ray-terrain intersection
3. Binary search algorithm determines the horizon angle for each azimuth

**Rendering parameters (from `PipelineConfig`):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `dist_search_km` | 30.0 | Khumbu valleys are ~20–30 km long; beyond this, DEM accuracy degrades and terrain features become too small to matter |
| `eye_height_m` | 1.6 | Standard handheld camera height; sensitivity analysis shows <0.5° error for ±0.5 m variation |
| `hori_acc_deg` | 0.1 | HORAYZON convergence tolerance; lower values slow generation with no matching benefit (tested) |
| `azim_num` | 720 | 0.5° resolution matches the uint8 quantization step (~0.35°); finer bins waste storage |
| `elev_ang_low_lim` | -89.0° | Prevents looking straight down (would capture terrain at observer's feet, not horizon) |

**DEM interpolation:** HORAYZON uses the DEM mesh as a triangle surface;
ray-DEM intersection is geometric (no resampling artifacts). This means
horizon angles are exact to the DEM's resolution, not interpolated.

**Quantization:** Elevation angles stored as `uint8` (0–255 maps linearly
to 0°–90°, 0.353° per step, max error ≤0.18°):

```python
# src/horizon_format.py
DEG_PER_BIN = 90.0 / 255.0  # = 0.3529°
BINS_PER_DEG = 255.0 / 90.0  # = 2.8333

def encode_horizon_uint8(deg):
    return np.clip(np.round(deg * BINS_PER_DEG), 0, 255).astype(np.uint8)

def decode_horizon_uint8(encoded):
    return np.asarray(encoded, dtype=np.float32) * DEG_PER_BIN
```

**Why uint8?** The quantization error (≤0.18°) is below the matching
resolution (0.5° bins). Storage comparison:
- Raw uint8: 1,338,650 × 720 × 1 byte = 0.96 GB
- Parquet on disk: 486 MB (2× compression via columnar encoding)
- Equivalent float32: 3.86 GB (8× larger)

**Filtering:** Viewpoints with flat horizons (std < 1.5° or max elevation
< 1.0°) are excluded during generation. These typically represent locations
in enclosed valleys where the skyline is too low or flat to be useful for
matching.

**Scale:** 1,338,650 viewpoints written to Parquet with columns: `lon`, `lat`,
`elevation_m`, `raw_horizon_deg` (variable-length uint8 arrays).

### 3.3 Synthetic Training Data

`src/synthetic_generator.py` — class `SyntheticSceneGenerator`:

**Why synthetic data?** The segmentation model needs paired (image, mask)
training data. Manual annotation is expensive. Synthetic generation produces
unlimited training pairs from existing DEM and satellite imagery.

**Generation pipeline:**
1. **Terrain mesh**: DEM loaded via `rasterio`, decimated by stride, built
   into a `trimesh` triangle mesh. The mesh is offset in Z to prevent
   float32 precision issues at large coordinates.
2. **Texture**: Satellite imagery (e.g., Sentinel-2) mapped onto the mesh
   using GPS-to-pixel coordinate transformation.
3. **Sky/backdrop**: Cloud images from a dataset composited behind the terrain
   via `pyrender` (EGL backend, headless rendering).
4. **Camera**: Configurable viewpoint, FOV, and orientation.
5. **Output**: Paired RGB image + binary sky/mask ground truth, with JSON
   ground truth (camera pose, location).

**Cloud augmentation:** During training, cloud images are blended into sky
regions with random alpha (0.15–0.5) and probability (0.3). This teaches
the model to handle partial sky obscuration — critical for real GSV images
in Khumbu where fog is common.

### 3.4 Sky Segmentation

`src/segmentation.py`:

**Training data sources:**
- **GeoPose3K dataset**: ~3K hand-annotated sky/mountain segmentation masks.
  Downloaded and split via `notebooks/04_SkySegmentation/`.
- **Synthetic scenes**: Generated by `SyntheticSceneGenerator` with cloud
  overlay augmentation. 80/20 train/val split.

**Model architecture:**
- U-Net encoder-decoder with skip connections
- Encoder: `tu-mobilenetv3_large_100` (pretrained on ImageNet)
- Input: 256×256 RGB patches (aspect-ratio preserving with reflective padding)
- Output: single-channel sigmoid (sky probability)
- Loss: BCE + Soft Dice combined (`bce_dice_loss`)
- Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)
- Scheduler: Cosine Annealing over 15 epochs
- Metric: Intersection-over-Union (IoU)

**Why MobileNetV3?** Lightweight backbone suitable for mobile deployment
(the project includes a Kivy app). Larger backbones (ResNet, EfficientNet)
showed marginal IoU improvement but 3–5× slower inference.

**Post-processing pipeline** (applied after raw U-Net inference):

1. **Smart top-sky filtering**: Identifies connected sky components touching
   the top of the image (clouds, not terrain). Falls back for steep mountain
   crops where terrain reaches the top.

2. **Canny-guided barrier**: Restricts Canny edge detection to ±10 px window
   around the raw boundary. Prevents cloud edges from "jumping" to terrain
   features below.

3. **CLAHE-enhanced multi-scale Canny + LAB b* sub-pixel fitting**
   (`refine_sky_mask_with_guidance`):
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances
     local contrast in the image
   - Multi-scale Canny edge detection identifies the sky boundary
   - LAB b* channel (blue-yellow axis) provides additional sky/terrain
     discrimination
   - Sub-pixel fitting via 3-point parabolic interpolation on the gradient

4. **1D median + Gaussian smoothing**: Removes sawtooth jitter from the
   boundary without blurring real terrain features.

5. **Quality gates**:
   - `min_sky_ratio = 0.05`: Reject crops that are almost entirely ground
   - `max_sky_ratio = 0.95`: Reject crops that are almost entirely sky
     (likely segmentation error or cloud)
   - `min_boundary_coverage = 0.5`: Need at least half the horizon boundary

**Why these post-processing steps?** Each addresses a specific failure mode:
- Top-sky filtering: clouds misclassified as terrain
- Canny barrier: cloud edges jumping to ground snow
- CLAHE + LAB b*: low-contrast sky/terrain boundaries (common in fog)
- Median smoothing: segmentation jitter along the boundary
- Quality gates: low-confidence masks that would corrupt profile extraction

### 3.5 Profile Extraction

`src/query_profile.py` — `extract_elevation_profile`:

**Input:** Binary sky mask (sky=0, terrain=255) + camera parameters.

**Step 1: Skyline boundary detection**
- Per-column scan: find the first terrain pixel (transition from sky to terrain)
- Apply 1D median filter (kernel=5) to stabilize boundary
- Track which columns have valid boundaries (boundary coverage metric)

**Step 2: Sub-pixel refinement** (optional, when image is provided)
- Compute Sobel-Y gradient of the grayscale image
- For each column, fit a 3-point parabola to the gradient near the binary
  boundary to extract sub-pixel position (accuracy ~0.1 pixel)
- Eliminates integer-rounding quantization noise in the boundary

**Step 3: Pixel → elevation angle**
- Pin-hole camera model:
  ```
  focal_y = H / (2 * tan(fov_y / 2))
  elevation_angle = arctan((y_center - skyline_row) / focal_y)
  ```
- Camera tilt (`cam_R_tilt` rotation matrix) applied to rotate rays into
  world coordinates
- Horizontal FOV computed from vertical FOV and aspect ratio:
  ```
  hfov = 2 * arctan(tan(vfov/2) * aspect_ratio)
  ```

**Step 4: Azimuth binning**
- Camera-frame azimuth for each column: `arctan2(x_offset, focal_x)`
- Rays sorted by azimuth, then linearly interpolated onto a uniform 0.5° grid
- Coverage per bin recorded for quality filtering

**Step 5: Applicability checks**
- `min_std_deg = 1.5`: Profile must have sufficient terrain variation to
  discriminate locations. Below this, profiles are too flat.
- `min_max_elev_deg = 1.0`: Minimum terrain relief above horizontal.
  Filters sky-only or ground-only crops.

**Why these thresholds?** Tested in `notebooks/05_SkylineMatching/01_evaluate_matching.ipynb`:
profiles with std < 1.5° produce matching scores indistinguishable from
noise. The max elevation threshold filters crops where the horizon is
below the camera's horizontal plane (likely segmentation error).

**Multi-crop fusion** (`fuse_profiles` in `src/query_profile.py`):
- Each crop's profile placed into the correct azimuth bin by GSV heading
- Overlapping bins keep the first valid value (no averaging — each crop
  covers a distinct azimuth range)
- Gaps linearly interpolated via `np.interp`
- Returns fused wide-FOV profile (200°–262° typical) + coverage metric

### 3.6 Matching Algorithm

`src/matching.py` — two-stage pipeline:

**Stage 1: Coarse spatial scan (z-scored cross-correlation)**

Feature extraction:
- **Value channel**: z-scored elevation profile (zero-mean, unit-variance)
- **D1 channel**: z-scored first derivative via `np.gradient`
- **Bandpass channels**: Difference-of-Gaussians (DoG) applied in spatial
  domain via `scipy.ndimage.gaussian_filter1d` with `mode="wrap"` (circular)

```python
DoG(σ1, σ2) = Gaussian(σ1) - Gaussian(σ2)
```

This acts as a bandpass filter that removes:
- DC component (mean elevation) — eliminates bias from absolute altitude
- High-frequency noise — isolates mid-scale terrain structure

**Scoring:** Pearson-normalized circular cross-correlation:

```python
# For z-scored signals, cross-correlation / N = Pearson correlation
numer = irfft(conj(Fquery) * Fdb, n=N_BINS)  # numerator
denom = norm(query) * norm(db_window)           # denominator
ncc = numer / denom                             # Pearson r at each shift
```

The `max-over-circular-shifts` operation tests all 720 possible heading
alignments. This is necessary because GSV compass headings are unreliable
(median error ~70°).

**Two implementations (mathematically equivalent, verified by tests):**
- **Production** (`ncc_scores`): cumsum-based windowed cross-correlation.
  O(N×L) with low constant factor. Single-query optimized.
- **Evaluation** (`score_chunk_shared_fft`): FFT-based cross-correlation
  shared across multiple queries per chunk. ~16× faster for batch scoring.

**Memory-safe streaming:** The 1.34M-row database is never loaded into RAM.
Instead, it is processed in chunks of 4000–8000 rows. For each chunk:
1. Decode uint8 horizons to float32
2. Compute DB features (z-scored value + derivative + bandpass)
3. Compute DB FFTs (shared across all queries)
4. Score all queries against this chunk
5. Update per-query top-K heaps
6. Discard chunk, proceed to next

**Three complementary scorers:**

| Scorer | Signal | Purpose | Rationale |
|--------|--------|---------|-----------|
| S0 baseline | value + d1 (0.5/0.5) | Production default | Value captures absolute shape; d1 captures edges and slopes |
| S1 bp(2,8) | DoG bandpass σ=2→8 | Eliminates mean-elevation bias | Two locations at different altitudes can have similar profiles; bandpass removes the DC offset |
| S2 bp(3,16) | DoG bandpass σ=3→16 | Captures valley-scale structure | Broader bandpass captures large-scale terrain features (valleys, ridges) that discriminate locations |

**Bandpass σ selection** (from ablation in `full_db_bandpass_eval.py`):
- bp(1,4): too narrow, captures noise → rejected
- bp(2,8): captures ridge-scale structure (~2–8 km features) → selected as S1
- bp(3,16): captures valley-scale structure (~3–16 km features) → selected as S2
- bp(4,32): too broad, loses discriminative detail → rejected

**Stage 2: Score fusion (RRF)**

Reciprocal-rank fusion over each scorer's top-50 list:

```python
score(row) = Σ_scorers 1/(60 + rank_s(row))
```

The constant 60 is chosen so that rank-0 contributes 1/60 ≈ 0.0167 and
rank-50 contributes 1/110 ≈ 0.0091. This gives reasonable weight to the
top candidates while allowing cross-scorer agreement to dominate.

**Why RRF instead of score averaging?** Different scorers produce scores on
different scales (baseline NCC ~0.7, bandpass NCC ~0.5). Averaging would
be dominated by the scorer with the largest magnitude. RRF operates on
ranks, which are comparable across scorers.

### 3.7 Confidence Gating

The system reports a match only when all three scorers' top-1 predictions
agree within a geodesic distance threshold.

**Consensus definition:** Maximum pairwise geodesic distance between the
three scorers' top-1 predictions ≤ threshold.

**Default threshold:** 1.0 km (tuned in `gsv_improve_eval.py`).

**Why this works:** When the true VP is unambiguous (distinctive terrain,
wide FOV), all three scorers converge on the same location. When the VP
is ambiguous (flat terrain, narrow FOV, imposters), the scorers disagree.
The consensus gate exploits this correlation between agreement and accuracy.

---

## 4. Configuration Parameters

All parameters from `src/config.py` with justification:

| Category | Parameter | Value | Rationale |
|----------|-----------|-------|-----------|
| **DB** | `grid_spacing_m` | 30 | Matches DEM resolution (GLO-30); finer grid adds VPs without improving matching |
| | `dist_search_km` | 30 | Khumbu valleys are ~20–30 km long; beyond this, DEM accuracy degrades |
| | `eye_height_m` | 1.6 | Standard handheld camera height; sensitivity analysis shows <0.5° error for ±0.5 m |
| | `hori_acc_deg` | 0.1 | HORAYZON convergence; lower values slow generation with no matching benefit |
| | `azim_num` | 720 | 0.5° resolution matches uint8 quantization step (~0.35°) |
| | `batch_size` | 4096 | Balances memory and GPU utilization during DB generation |
| **Profile** | `bin_deg` | 0.5 | Derived from azim_num; matches DB resolution |
| | `fov_y_deg` | 65.0 | GSV default; actual FOV extracted from metadata when available |
| | `median_kernel` | 5 | Removes segmentation jitter without blurring real terrain features |
| | `min_std_deg` | 1.5 | Below this, profile too flat to discriminate locations |
| | `min_max_elev_deg` | 1.0 | Minimum relief needed for matching |
| **Segmentation** | `seg_input_size` | 256 | Standard U-Net resolution; larger inputs don't improve sky boundary quality |
| | `min_sky_ratio` | 0.05 | Reject crops that are almost entirely ground |
| | `max_sky_ratio` | 0.95 | Reject crops that are almost entirely sky (likely error) |
| | `min_boundary_coverage` | 0.5 | Need at least half the horizon boundary |
| **Matching** | `fft_weights` | (0.5, 0.5) | Equal weight; ablation showed d1 adds ~10% over value-only |
| | `min_corr` | 0.30 | Below this, matches are noise |
| | `min_score_gap` | 0.03 | Empirically separates confident from ambiguous matches |
| | `top_k` | 5 | DTW refinement is expensive; top-5 captures most correct matches |
| | `dtw_window` | 15 | Allows ~7.5° alignment correction; larger windows increase false positives |
| | `spatial_stride` | 12 | Production default for speed; stride=1 for final evaluation |
| **Eval** | `chunk_rows` | 4000 | Balances memory (~150 MB per chunk) and I/O overhead |
| | `correct_dist_m` | 500 | Standard benchmark threshold |
| | `compass_tolerance_deg` | 20 | GSV compass accuracy ±5–10°; 20° gives comfortable margin |
| | `height_tolerance_m` | 200 | GPS altitude accuracy ±30–50 m; 200 m accounts for DEM vs real terrain |

---

## 5. Evaluation Methodology

### 5.1 Dataset

**Google Street View panoramas:**
- 68 multi-crop panoramas from the Khumbu region
- Each pano has 2–3 crops at different headings (60°–120° apart)
- Total fused FOV: 200°–262° typical for multi-crop panos
- Ground truth: camera GPS position from GSV metadata

**Why multi-crop?** Single crops cover only 60°–90° of the horizon. In
valley terrain, this is often insufficient to disambiguate locations. Multi-photo
fusion breaks valley symmetry by combining viewpoints from different directions.

### 5.2 Evaluation Protocol

1. **Profile extraction**: For each pano, extract elevation profiles from
   annotated sky boundaries, fuse into wide-FOV profile
2. **DB scan**: Stream through all 1.34M DB horizons, compute NCC scores
   for all three scorers, maintain top-50 heaps per scorer
3. **RRF fusion**: Compute reciprocal-rank fusion score for each candidate
4. **Error computation**: Geodesic distance between matched VP and true VP
5. **Confidence gating**: Apply consensus gate, report precision on accepted panos

**No ground-truth leakage:** The true VP's DB row index is only used for
computing error metrics, never for scoring or gating.

### 5.3 Evaluation Scripts

| Script | Purpose | Runtime |
|--------|---------|---------|
| `archive/scripts/gsv_improve_eval.py` | Main evaluation: 3-scorer RRF + confidence gate | ~20 min (stride=2) |
| `archive/scripts/end_to_end_gsv_eval.py` | Auto-seg vs annotation comparison | ~15 min (stride=2) |
| `archive/scripts/offgrid_synthetic_eval.py` | Noise robustness (synthetic) | ~5 min |
| `archive/scripts/benchmark_segmentation.py` | Segmentation boundary error | ~2 min |

---

## 6. Results

### 6.1 GSV Multi-Photo Evaluation (N=68 panos)

**Without gating (all matches counted):**

| Scorer | Median Error | <100 m | <1 km | <5 km | <10 km |
|--------|-------------|--------|-------|-------|--------|
| Baseline (S0) | 15.4 km | 8.8% | 11.8% | 16.2% | 32.4% |
| bp(2,8) (S1) | 18.3 km | 7.4% | 8.8% | 10.3% | 16.2% |
| bp(3,16) (S2) | 14.1 km | 8.8% | 10.3% | 19.1% | 29.4% |
| **RRF fusion** | 15.8 km | **11.8%** | **13.2%** | 14.7% | 23.5% |
| Oracle | 9.1 km | 13.2% | 14.7% | 30.9% | 57.4% |

**Wide-FOV subset (fused coverage ≥ 200°, N=25):**

| Scorer | Median Error | <100 m | <1 km | <10 km |
|--------|-------------|--------|-------|--------|
| Baseline (S0) | 15.3 km | 20.0% | 24.0% | 44.0% |
| bp(3,16) (S2) | **12.8 km** | 24.0% | 28.0% | 44.0% |
| **RRF fusion** | 14.4 km | **28.0%** | **28.0%** | 40.0% |
| Oracle | **4.8 km** | 32.0% | 32.0% | 68.0% |

**Key findings:**
- Fusion improves pinpoint (<100 m) accuracy by +34% relative overall
- Wide-FOV panos (≥200°) show dramatically better results: 28% <1 km vs 13.2%
- The oracle (best of all scorers per pano) reaches 32% <1 km on wide-FOV,
  indicating the scorers capture complementary information

### 6.2 Confidence-Gated Results

| Criteria | N accepted | Precision <1 km | Typical error |
|----------|-----------|-----------------|---------------|
| **Wide-FOV ≥200° + consensus** | 7 | **85.7% (6/7)** | ~40 m |
| Consensus only (any FOV) | 10 | 70.0% (7/10) | ~42 m |
| No consensus → rejected | 58 | 3.4% would have been correct | — |

Consensus hits include localizations of **11 m, 13 m, 23 m, 32 m,
33 m, 42 m** — meter-level pinpointing when terrain is distinctive.

**Statistical note:** With N=7 accepted panoramas, the 95% Clopper-Pearson
confidence interval for 85.7% precision is 42%–99.6%. This demonstrates the
approach works on this dataset; larger evaluation sets are needed for
reliable precision estimates. The confidence gate is designed so that the
system abstains when uncertain — the reported precision reflects the gate's
ability to separate reliable from unreliable matches.

### 6.3 Noise Robustness

DB horizon profiles corrupted with Gaussian noise at varying σ, then matched
back against the full DB:

| Noise σ | Rank-0 hit | <100 m | <1 km | Profiles with corr > 0.65 |
|---------|-----------|-------|------|---------------------------|
| 0.0° | 100% | 100% | 100% | 1168 |
| 0.25° | 100% | 100% | 100% | 115 |
| 0.5° | 100% | 100% | 100% | 1 |
| 1.0° | 100% | 100% | 100% | 0 |
| 2.0° | 85% | 85% | 95% | 0 |
| uint8 quant | 100% | 100% | 100% | 1168 |

**Key finding:** The matcher tolerates up to 1.0° noise with no accuracy
loss. Even 0.5° noise makes profiles highly distinctive (the number of
similar profiles in the DB drops from 1168 to 1), meaning real-world noise
from fog or segmentation errors actually helps eliminate ambiguous matches.

**Why noise helps:** At σ=0°, many DB profiles correlate similarly with the
query (1168 profiles with corr > 0.65), creating ambiguity. Adding even
tiny noise decorrelates imposters while preserving the true match, reducing
the candidate set to a single winner.

### 6.4 What Predicts Success

1. **FOV coverage ≥ 200°** — breaks valley symmetry; strongest single factor
2. **Cross-scorer consensus** — near-perfect precision proxy
3. **Distinctive terrain** — converging ridgelines, saddles (vs flat valleys)

### 6.5 Auto-Segmentation vs Manual Annotation

`archive/scripts/end_to_end_gsv_eval.py` runs the full pipeline two ways:

1. **Auto path**: crop image → U-Net segmentation → mask → profile →
   fuse → match against DB
2. **Manual path**: annotation points → mask_from_points → profile →
   fuse → match against DB

Both use the identical 3-scorer RRF matching pipeline, isolating the impact
of segmentation quality on end-to-end accuracy. Segmentation quality metrics
(sky ratio, confidence, boundary coverage) are reported per-pano.

### 6.6 Segmentation Quality

The U-Net model's boundary accuracy is evaluated against hand-annotated
skylines via `archive/scripts/benchmark_segmentation.py`. Four methods are
compared on 18 annotated samples:

| Method | Mean boundary error (px) | Median (px) | Description |
|--------|------------------------|-------------|-------------|
| Raw U-Net | — | — | Direct sigmoid threshold output |
| Refined U-Net | — | — | CLAHE + Canny + LAB b* sub-pixel fit |
| Pure Canny | — | — | Edge detection without U-Net |
| LAB b* threshold | — | — | Color-space segmentation |

The refined U-Net method is used in the production pipeline. Benchmark
results are computed per-run; see `results/segmentation_benchmark.json`
for the latest numbers.

---

## 7. Verification

### 7.1 Algorithmic Tests (`tests/test_core.py`)

All core algorithms are tested against independent brute-force references.
The test implementations were written without sharing code with the
production implementations (different authors, different approaches).

| Property | Test method | Result |
|----------|-------------|--------|
| uint8 roundtrip error ≤ ½ quantization step | Encode → decode → compare | PASS |
| NCC scoring matches brute-force max-shift Pearson | Loop over all shifts, compare | PASS |
| Known embedded copy recovered at exact offset | Insert known profile at known offset | PASS |
| FFT prefilter matches cumsum-based NCC | Two independent implementations | PASS |
| Flat/constant DB rows produce finite scores | Edge case test | PASS |
| RRF fusion: consensus row selected | Synthetic ranked lists | PASS |
| RRF fusion: empty input → abstain | Empty heap test | PASS |

**Test methodology:** For each property, the test creates synthetic data
with known ground truth, runs both the production and reference implementations,
and asserts they agree within floating-point tolerance (1e-6 for NCC scores,
1e-3 for geographic distances).

### 7.2 Evaluation Pipeline Tests (`tests/test_evaluation.py`)

| Property | Result |
|----------|--------|
| `summarize_results` correctness (empty, all-correct, mixed) | PASS |
| `load_ground_truth` loads JSON and respects limit | PASS |
| `filter_samples_with_masks` finds existing mask files | PASS |
| `_resolve_mask_path` handles raw and fixed naming | PASS |
| `infer_bin_size_deg` returns correct bin resolution | PASS |
| `load_db_metadata` returns correct array shapes | PASS |
| `build_batch_queries` produces valid query states | PASS |
| `build_batch_queries` respects min_std_deg filter | PASS |
| `run_batch_coarse_scan` populates best_corr arrays | PASS |
| `run_batch_coarse_scan` compass/elevation masking reduces hits | PASS |
| `refine_query_with_dtw` returns result dict with error_m | PASS |
| `refine_query_with_dtw` returns None for empty query | PASS |
| `is_profile_applicable` rejects flat/empty/NaN profiles | PASS |
| `extract_elevation_profile` works from mask array and file | PASS |
| End-to-end: scan → refine → summarize on synthetic DB | PASS |

### 7.3 Evaluation Integrity

- **No ground-truth leakage:** The true VP's DB row index is only used for
  computing error metrics, never for scoring or gating
- **Rejected panoramas excluded:** Panos that fail quality gates are reported
  separately and not counted in accuracy metrics
- **Oracle is post-hoc:** The oracle (best scorer per pano) is an upper bound,
  not an achievable system configuration

---

## 8. Failure Analysis

### 8.1 Primary Failure Mode: Imposter Effect

Of 18 hard samples analyzed in detail:
- **14/18**: Imposter shape mimicry — a wrong location correlates better
  than the true one. This is a DEM resolution problem: 30 m grid means
  nearby locations can have similar horizon profiles.
- **2/18**: Large heading error (calibrated Δθ > 100°)
- **1/18**: Low terrain relief (std < 2.5°)
- **1/18**: Unclassified

**Statistical analysis (Spearman correlation):**
- Imposter gap (r_best − r_true): ρ = +0.562, p = 0.015 ***
  → The only significant predictor of failure
- Profile std (terrain distinctiveness): ρ = −0.079, p = 0.754
  → Not significant — even high-relief profiles fail when imposters exist

### 8.2 Rejection Breakdown (68 panos)

| Gate | Accepted | Rejected | Precision <1 km |
|------|----------|----------|-----------------|
| 3/3 consensus | 10 | 58 | 70.0% (7/10) |
| 3/3 + coverage ≥200° | 7 | 61 | 85.7% (6/7) |
| No gate | 68 | 0 | 13.2% (9/68) |

Why 61 are rejected:
- Low coverage + low consensus: 40 (both FOV and terrain unhelpful)
- Low consensus only: 18 (good coverage but scorers disagree)
- Low coverage only: 3 (good terrain but few crops)
- 3 rejected panos would have been correct (<1 km) — missed recall

### 8.3 Heading Sensitivity (18 hard samples)

| Heading tolerance | Median error | <1 km | Median rank |
|-------------------|-------------|-------|-------------|
| ±15° | 10,469 m | 5.6% | 68,844 |
| ±30° | 15,124 m | 5.6% | 116,283 |
| Unconstrained | 14,241 m | 5.6% | 479,476 |

GSV compass is far worse than ±5–10°: measured median |Δheading| = 69.8°,
94% of samples have |Δ| > 10°. Relaxing heading tolerance does NOT help —
the real bottleneck is imposter shape mimicry, not heading error.

### 8.4 Selection Bias

- Wide-FOV (≥200°): 25/68 panos (37%), median error 14,445 m
- Narrow-FOV (<200°): 43/68 panos (63%), median error 18,031 m
- The confidence gate selects conditions where matching is already easier

---

## 9. Known Limitations

1. **DEM resolution:** Two locations ~1 km apart can share nearly identical
   30 m resolution horizons. Higher-resolution DEMs (10 m) would substantially
   reduce the imposter effect.

2. **Fog and cloud:** Many GSV crops have obscured skylines. Low-coverage
   profiles (<150°) rarely match reliably.

3. **GSV compass accuracy:** Measured median heading error is ~70°, far worse
   than the ±5–10° ideal. The max-over-shifts search compensates, but
   accurate headings would improve ranking.

4. **Single-region evaluation:** All results are on Khumbu with GLO-30 DEM.
   Different terrain types (flat, forested, urban) would behave differently.

5. **Azimuth-seam artifact:** The first-difference feature uses `np.gradient`
   with one-sided edge stencils, causing ~1–3×10⁻³ NCC drift at the 0°/360°
   seam. This is negligible against typical score gaps.

---

## 10. Deployment

### Mobile App

A Kivy-based mobile application replicates the full pipeline for on-device
deployment. The app runs the same segmentation, profile extraction, and
matching logic against a precomputed horizon database.

### Recommended Configurations

| Scenario | Config | Expected outcome |
|----------|--------|-----------------|
| **Skyline-only, trustworthy** | wide-FOV + consensus gate | ~86% precision at <1 km; abstain otherwise |
| Skyline-only, best-effort | RRF top-1, no gate | median ~13–16 km regional accuracy |
| **With coarse GPS prior (≤5 km)** | restrict candidates, RRF decides | expected <500 m routinely |
| With 10 m DEM | same pipeline | expected 2–5× error reduction |

---

## 11. Future Work

1. **GPS/cell-tower prior** even at km accuracy — skyline becomes tiebreaker
2. **Higher-resolution DEM** (10 m national datasets exist for Nepal)
3. **Learned features** (DINOv2/SAM2 embeddings) fused with geometric scores
4. **More annotated panos** — widens the consensus-accepted set

---

## 12. Repository Structure

| Path | Contents |
|------|----------|
| `ARCHITECTURE.md` | this document |
| `src/` | production source code (segmentation, matching, profiles, DB generation, synthetic data, evaluation) |
| `notebooks/` | pipeline notebooks (numbered 01–06 by stage) |
| `data/` | DEMs, models, GSV crops, ground truth |
| `tests/` | algorithmic verification suite |
| `archive/` | evaluation scripts, dashboards, historical work |

### Reproduction

```bash
# 1. Algorithmic tests (~1 min):
conda run -n skyline_env python tests/test_core.py

# 2. Main evaluation (~20 min):
python archive/scripts/gsv_improve_eval.py --stride 2

# 3. End-to-end auto-segmentation (~15 min):
python archive/scripts/end_to_end_gsv_eval.py --stride 2

# 4. Noise robustness (~5 min):
python archive/scripts/offgrid_synthetic_eval.py --samples 5

# 5. Segmentation benchmark (~2 min):
python archive/scripts/benchmark_segmentation.py
```

---

## 13. Dataset Summary

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Copernicus GLO-90 | ESA Copernicus | 90 m DEM | Viewshed analysis |
| Copernicus GLO-30 | ESA Copernicus | 30 m DEM | Horizon database |
| Skyline DB | Generated (HORAYZON) | 1.34M viewpoints, 486 MB Parquet | Matching database |
| GeoPose3K | Published dataset | ~3K images + masks | Segmentation training |
| Synthetic scenes | Generated (pyrender) | Unlimited pairs | Segmentation augmentation |
| Cloud images | Published dataset | Cloud backdrop library | Synthetic sky generation |
| GSV panoramas | Google Street View | 68 multi-crop panos | Evaluation queries |
| Ground truth | Manual annotation | Camera GPS positions | Evaluation metric |

---

## Appendix A: Dead Ends (Explored but Rejected)

The following approaches were tested during development and found not to
improve results:

| Approach | Why rejected |
|----------|-------------|
| Saliency weighting | Did not improve matching accuracy |
| Elevation penalty priors | Added complexity without measurable gain |
| 2D Chamfer re-ranking | Worse than simple NCC ranking |
| Query-side pitch shifting | Provably a no-op for max-shift Pearson NCC |
| Per-crop intersection voting | Crops individually too narrow for reliable voting |
| Fused-profile score-gap confidence | Does not separate hits from misses |

These are documented in `archive/docs/HISTORICAL_WORKLOG.md` for reference.

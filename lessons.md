# Handover: Skyline-Based Visual Geo-Localization Pipeline (Khumbu Region)

This document is the consolidated, exhaustive technical guide and codebase archive for the visual geo-localization system designed for the GNSS-denied Khumbu (Everest) region of Nepal. It serves as a comprehensive system summary, cataloging all architectural designs, historical failures, physical models, and corrected mathematical implementations.

---

## 1. System Overview & Pipeline Architecture

The system estimates a camera's WGS84 coordinates by matching the 1D silhouette of a mountain range (skyline) extracted from a photograph against a pre-computed 3D database of synthetic horizons.

```
                  [Smartphone Photograph]
                             │
                             ▼ (1. Sky Segmentation U-Net)
                     [Binary Sky Mask]
                             │
                             ▼ (2. Spherical Projection via R_tilt)
                 [Leveled Elevation Profile]
                             │
                             ▼ (3. Gaussian Smoothing, sigma=2.0)
               [Noise-Free Combined Descriptor]
                             │
             ┌───────────────┴───────────────┐
             ▼ (4. Coarse Filter: Stage 1)    ▼ (5. Fine Refine: Stage 2)
       [FFT Sliding Window MASS]       [Symmetric 2D Normalized DTW]
             │                               │
             └───────────────────────────────┘
                             │
                             ▼ (6. Coordinate Resolution)
                    [WGS84 Lat/Lon Coordinates]
```

### The Production Pipeline:
1. **Skyline Segmentation**: A convolutional U-Net with a MobileNet-V3-Large backbone segments the image into a binary sky-terrain mask (Terrain=0, Sky=255) with **$99.47\%$ validation IoU**.
2. **Cylindrical Elevation Profile Extraction**: The sky-terrain boundary is extracted column-by-column. Rather than assuming a flat-grid camera projection, the columns are projected into true geodetic elevation angles using a planar-to-cylindrical ray transformation.
3. **Physical Tilt & Gaze Correction**: The 3D rays are rotated to a level, gravity-aligned coordinate frame using the camera tilt rotation matrix $R_{\text{tilt}}$. This isolates the camera's physical pitch and roll from the relative azimuth.
4. **Gaussian Pre-Filtering**: The 1D profile is filtered with a 1D Gaussian kernel ($\sigma=2.0$) to smooth out discrete pixel-staircasing transitions.
5. **Two-Stage Matching Engine**:
   *   **Stage 1 (Coarse Filter - MASS)**: Runs a Z-normalized sliding-window cross-correlation via FFT against all 4,860 viewpoints across three database tiers, returning the top $K$ candidates.
   *   **Stage 2 (Fine Re-ranking - DTW)**: Performs a Sakoe-Chiba constrained Dynamic Time Warping alignment on a 2-channel descriptor (normalized profile + normalized pre-computed derivative) to resolve any minor spatial or scale skews.

---

## 2. Comprehensive File Inventory

### 🔴 CORE SYSTEM FILES (Keep)

*   **`unified_evaluation_pipeline.py`**: The central runtime. Contains the viewpoint grid builder, ground-truth generator, ray projection math, MASS sliding-window FFT search, and the 2-channel DTW re-ranking engine. Supports CLI commands for `--mode evaluate`, `diagnose`, `build_grid`, `build_gt`, and `full_ablation`.
*   **`render_synthetic.py`**: The headless procedural renderer. Uses EGL and Pyrender to place cameras across the DEM, direct them toward distant high-altitude peaks, apply varied weather/atmospheric presets, and output photorealistic images, pixel-perfect binary masks, and synchronized metadata (`fov_y_deg`, `true_heading_deg`, and `cam_R_tilt`).
*   **`generate_horizons_from_dem.py`**: Computes 360-degree horizon profiles at 0.25° resolution (1440 columns) using HORAYZON raycasting over the DEM across all 4,860 viewpoints.
*   **`environment.yml`**: Conda virtual environment package specification.

### 🟡 DEEP LEARNING & REAL PHOTO TOOLS (Keep if expanding to real images)

*   **`train_sky_segmentation_aug.py`**: Training script for the SMP U-Net model using mixed synthetic and GeoPose3K datasets with color, haze, and blur augmentations.
*   **`real_photo_preprocess.py`**: Image preprocessing pipeline for actual camera captures. Implements aspect-ratio-preserving cropping and scales predicted masks back to native sensor resolutions.

### ❌ DEPRECATED FILES (Safely Delete)

These legacy scripts have been fully consolidated into the core files and are redundant:
*   `evaluate_pipeline.py` (Superseded by `unified_evaluation_pipeline.py`)
*   `sync_and_rebuild_eval.py` (Superseded by grid-synchronization steps)
*   `run_ablation_study.py` (Superseded by `unified_evaluation_pipeline.py --mode full_ablation`)
*   `diagnose_mismatch.py` (Superseded by `unified_evaluation_pipeline.py --mode diagnose`)
*   `get_dataset_gt.py` (Superseded by `unified_evaluation_pipeline.py --mode build_gt`)
*   `/tmp/` (Safe to remove the entire directory)

---

## 3. Chronological Failure Analysis & Root Cause Resolutions

To prevent regressions, the mathematical and systemic failures solved during development are archived here.

### A. The 81-Column Meshgrid Index Drift
* **Symptoms**: Top-1 evaluation accuracy was stuck at 0.0%, with median positioning errors of 11 km to 20 km.
* **Root Cause**: The database generator and evaluation scripts used mismatched boundaries. The evaluation script used `max_x + GRID_SPACING_M` in `np.arange`, appending an extra column (81 columns instead of 80). When flattened via `ravel()`, the off-by-one column boundary created a cumulative index drift, scrambling coordinates globally.
* **Resolution**: Standardized grid boundaries in all scripts to an exact 80-column matching grid:
  ```python
  X_v = np.arange(min_x, max_x, GRID_SPACING_M)  # Strictly 80 steps
  Y_v = np.arange(min_y, max_y, GRID_SPACING_M)  # Strictly 60 steps
  ```

### B. Vertical Height Inversion (DEM Index-Flipping)
* **Symptoms**: Viewpoints were rendered from underground or at incorrect altitudes.
* **Root Cause**: The DEM GeoTIFF starts from the top-left (North, negative height step). Flipping the DEM vertically using `np.flipud` to align coordinates without updating the origin `start_y` caused coordinate index wrapping, forcing height lookups to index the array backward.
* **Resolution**: Replaced manual index arithmetic with `rasterio`'s native coordinate lookup:
  ```python
  row, col = src.index(eye_x, eye_y)
  ground_z = dem_data[row, col]
  ```

### C. Perspective Flat-Sensor Edge Compression
* **Symptoms**: Squeezed query profiles failing to align with database silhouettes.
* **Root Cause**: The database profiles store true geodetic elevation angles relative to the horizontal plane. Standard camera projection assumes a flat plane where the distance to pixels increases toward the edges.
* **Resolution**: Corrected the planar projection math by computing the radial distance to each column pixel:
  ```python
  dists = np.sqrt(f_px**2 + (cols - x_c)**2)
  elevations_rad = np.arctan((y_c - skyline_pixels) / dists)
  ```

### D. Pixel Staircasing in Derivatives
* **Symptoms**: First-derivative stage-1 matching returned near-zero correlation ($0.0429$).
* **Root Cause**: Extracted masks are discrete. Taking `np.gradient` directly on raw pixel steps amplified high-frequency staircasing noise, introducing massive spikes that ruined the correlation.
* **Resolution**: Applied a 1-D Gaussian filter ($\sigma=2.0$) to smooth out pixel transitions before computing derivatives.

### E. DTW Slicing Boundary Artifacts
* **Symptoms**: Stage-2 DTW selected incorrect, distant viewpoints over true candidates.
* **Root Cause**: Slicing the database profile first and then taking `np.gradient` of that short subsequence introduced severe edge artifacts at the slice boundaries.
* **Resolution**: Always compute the continuous 360-degree derivative of the database profile *first*, and then slice the pre-computed derivative array.

### F. DTW Feature Scale Mismatches
* **Symptoms**: DTW accuracy dropped to 10.0%.
* **Root Cause**: The database descriptor was z-normalized (profile std=1.0, derivative std=1.0), but the query descriptor was un-normalized (profile std $\approx 8.0$, derivative std $\approx 0.3$), causing the L2 distance calculation to be heavily skewed.
* **Resolution**: Symmetrically z-normalized both the profile and derivative channels independently for both the query and database descriptors.

---

## 4. Final Mathematical Implementations

### A. Tilt Correction Matrix ($R_{\text{tilt}}$)
To extract a pitch-invariant, leveled skyline from a tilted camera, the camera coordinates are rotated around gravity to align the forward vector with the horizontal plane:
Let $R$ be the $3 \times 3$ camera rotation matrix. Gaze is along the negative Z-axis, so the world-space camera forward vector is:
$$\mathbf{f} = -R_{:, 2}$$

We project $\mathbf{f}$ onto the horizontal plane:
$$\mathbf{f}_{\text{horiz}} = \begin{bmatrix} f_x \\ f_y \\ 0 \end{bmatrix}, \quad \hat{\mathbf{f}}_{\text{horiz}} = \frac{\mathbf{f}_{\text{horiz}}}{\|\mathbf{f}_{\text{horiz}}\|}$$

The leveled coordinate frame axes in world space are:
$$\mathbf{z}_{\text{level}} = -\hat{\mathbf{f}}_{\text{horiz}}, \quad \mathbf{y}_{\text{level}} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}, \quad \mathbf{x}_{\text{level}} = \mathbf{y}_{\text{level}} \times \mathbf{z}_{\text{level}}$$

Construct the leveled frame rotation matrix $R_{\text{level\_frame}} = [\mathbf{x}_{\text{level}}, \mathbf{y}_{\text{level}}, \mathbf{z}_{\text{level}}]$. The tilt rotation matrix $R_{\text{tilt}}$ is:
$$R_{\text{tilt}} = R_{\text{level\_frame}}^T \cdot R$$

### B. Planar-to-Cylindrical Ray Projection
For each image column, the leveled camera ray is computed and projected into true geodetic elevation $\phi$ and relative azimuth $\alpha$:
$$\mathbf{v}_{\text{camera}} = \begin{bmatrix} (col - x_c) / f_x \\ (y_c - row) / f_y \\ -1 \end{bmatrix}, \quad \hat{\mathbf{v}}_{\text{camera}} = \frac{\mathbf{v}_{\text{camera}}}{\|\mathbf{v}_{\text{camera}}\|}$$

The ray in the leveled frame is:
$$\mathbf{v}_{\text{leveled}} = R_{\text{tilt}} \cdot \hat{\mathbf{v}}_{\text{camera}}$$

The elevation and relative azimuth angles are:
$$\phi = \arcsin\left(\mathbf{v}_{\text{leveled}, y}\right), \quad \alpha = \arctan2\left(\mathbf{v}_{\text{leveled}, x}, -\mathbf{v}_{\text{leveled}, z}\right)$$

### C. Z-Normalized MASS Sliding Window FFT
For a query profile $Q$ of length $m$ and a database profile $T$ of length $N$, MASS computes the Z-normalized Euclidean distance by expanding the term:
$$\text{dist}^2 = 2 \cdot m \cdot (1 - \rho)$$
where the Pearson correlation $\rho$ is calculated using FFT convolution:
$$\rho = \frac{\mathbf{fftconvolve}(T_{\text{padded}}, Q_{\text{flipped}}) - m \cdot \mu_T \cdot \mu_Q}{m \cdot \sigma_T \cdot \sigma_Q}$$

---

## 5. Operations & Troubleshooting Workflow

To reset, verify, or execute the pipeline:

```bash
# 1. Regenerate the reference database with correct camera heights
python generate_horizons_from_dem.py

# 2. Build the synchronized 80x60 grid map (viewpoints_mapping.npy)
python unified_evaluation_pipeline.py --mode build_grid

# 3. Generate ground-truth samples mapped to the aligned grid
python unified_evaluation_pipeline.py --mode build_gt

# 4. Run the visual shape diagnostic on Sample 0
python unified_evaluation_pipeline.py --mode diagnose

# 5. Run the batch evaluation pipeline
python unified_evaluation_pipeline.py --mode evaluate
```

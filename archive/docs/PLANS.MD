# PLANS.MD — SkylineGeolocation Stabilization Plan

## Goal

Make SkylineGeolocation runnable, debuggable, memory-safe, and honest about failure cases. Any AI agent can follow this plan to improve the repo without guessing.

Pipeline: define region → download/reproject DEM → generate skyline DB → train/run segmentation → extract profile → match → evaluate

## Hard Rules

- No committing secrets. No printing `.env`.
- No committing `data/`, `tmp/`, notebook `output/`.
- Prefer `src/` changes over notebook edits.
- Notebooks are orchestration/demo; core logic lives in `src/`.
- No full-DB RAM loads without explicit small-DB call.
- Every failure-prone stage returns `ok: bool, status: str, reason: str, diagnostics: dict`.
- If no confident match: return `NO_MATCH`, not fake coords.

## Environment

```bash
conda activate skyline_env
```

If setting up fresh:

```bash
conda env create -f environment.yml
conda activate skyline_env
cd HORAYZON && python -m pip install . && cd ..
```

Audit missing deps before changes: `fastdtw`, `geopy`, `pyarrow`, `pyprojroot`, `earthaccess`, `osgeo`, `ipywidgets`. Add to `environment.yml` if missing.

## Compute Strategy

| Stage | Where |
|-------|-------|
| Code fixes, smoke tests, profile extraction, matching on tiny synthetic DB, notebook cleanup, plot generation | Local |
| Train segmentation model (reduced), small synthetic dataset, limited evaluation (50–300 samples), final artifacts | Colab free tier |
| Full 30m DEM DB over large region, 1.3M+ viewpoint horizon DB, full U-Net training | Likely not free-tier |

Colab notebooks under `colab/` should:
1. Mount Drive
2. Install deps
3. Restore needed artifacts
4. Run one stage
5. Save outputs to Drive
6. Print exact paths

Required artifacts:

```
data/digital_elevation_model/dem_30m.tif
notebooks/01_RegionStudy/output/actual_bounds.json
notebooks/02_SkylineDatabase/output/*.parquet
data/sky_segmentation_unet_model.pth
notebooks/05_SkylineMatching/output/*.npy
```

## Phase 1 — Repo Health Audit

### 1.1 Read high-value files

`AGENTS.md`, `PROJECT_METHODOLOGY.md`, `environment.yml`, `setup.sh`, `.gitignore`, `src/*.py`, `notebooks/**/*.ipynb`

### 1.2 Verify importability

```bash
python -c "import src.region; import src.dem; import src.skyline; import src.matching; import src.query_profile; import src.evaluation; import src.segmentation; print('ok')"
```

Fix any import failures.

### 1.3 `scripts/smoke_check.py`

Lightweight validation script. No large data required.

Tests:
- Region JSON load/save roundtrip
- Profile extraction from synthetic tiny mask
- Matching against tiny fake DB
- Evaluation returns structured result

## Phase 2 — Failure Transparency

### 2.1 Shared result shape

Return shape for all major stages:

```python
{
    "ok": bool,
    "status": "OK" | "NO_SKYLINE" | "LOW_CONFIDENCE" | "INVALID_INPUT" | "NO_MATCH" | "ERROR",
    "reason": str,
    "data": ...,
    "diagnostics": dict,
}
```

### 2.2 `segmentation.py`

Validate: input path exists, model path exists, report device.

Add diagnostics:
- mean sky probability
- sky pixel ratio
- largest sky component size
- skyline continuity score

Detect bad masks:
- `min_sky_ratio = 0.05`, `max_sky_ratio = 0.95`
- `min_boundary_coverage = 0.50`
- `min_confidence = 0.60`

Bad/blank image → invalid status. Missing model → clear error.

### 2.3 `query_profile.py`

Validate mask dimensions. Detect all-white/all-black. Detect columns with no sky/terrain boundary. Track boundary coverage.

Diagnostics:
- width, height, sky_is_white, boundary_coverage
- missing_columns count
- profile_std_deg, profile_max_deg, profile_min_deg
- hfov_deg, fov_y_deg, bin_deg

Empty/solid masks → fail. Valid mask → profile + diagnostics.

### 2.4 `matching.py`

Key issues:
- `_feature_bundle()` uses `scipy.stats.zscore` which produces NaN on flat arrays. Fix with safe normalization.
- No confidence scoring. Best match returned even if all matches are bad.
- No `NO_MATCH` path.

Add safe zscore wrapper:

```python
def _safe_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    if x.std() < 1e-12:
        return np.zeros_like(x)
    return zscore(x)
```

Matching result fields:

```python
{
    "ok": bool,
    "status": "OK" | "LOW_CONFIDENCE" | "NO_MATCH" | "INVALID_QUERY",
    "matches": [
        {
            "row_index": int, "lat": float, "lon": float,
            "score": float, "fft_corr": float,
            "dtw_distance": float, "offset_deg": float,
        }
    ],
    "confidence": {
        "best_score": float, "second_score": float,
        "score_gap": float, "ambiguous": bool,
    },
    "diagnostics": dict,
}
```

Confidence rules (configurable): `min_corr = 0.15`, `min_score_gap = 0.03`.

Flat query → `INVALID_QUERY`. Random query → `LOW_CONFIDENCE`. Clear match → `OK`.

## Phase 3 — Memory Safety

### 3.1 `evaluation.py`

Already uses `ParquetFile.iter_batches()` — good.

Verify:
- No accidental `pd.read_parquet()` for large DB
- Chunk size configurable (default 4000)
- Top-k across chunks without storing all scores
- NaN in scores doesn't pollute ranking
- `raw_horizon_deg` column name checked before access

### 3.2 `skyline.py`

Add:
- `dry_run` param (process subset for testing)
- `max_viewpoints` limiter
- Logging for per-row time, total projected rows

### 3.3 `synthetic_generator.py` / `mountain_engine.py`

- Verify crop window loads reduced DEM
- Verify stride is configurable
- GC after each synthetic scene

## Phase 4 — Evaluation & Parameter Justification

### 4.1 Threshold audit

| Location | Parameter | Value | Status |
|----------|-----------|-------|--------|
| `skyline.py` | `dist_search_km=30.0` | Default | Should test 10/15/30/60 |
| `skyline.py` | `azim_num=360` | Standard | Keep |
| `skyline.py` | `hori_acc_deg=0.1` | Standard | Keep |
| `skyline.py` | `eye_height_m=1.6` | Standard | Keep |
| `skyline.py` | `min_std_deg=1.5` | Guess | Flag as uncalibrated |
| `skyline.py` | `min_max_elev_deg=1.0` | Guess | Flag as uncalibrated |
| `query_profile.py` | `fov_y_deg=65.0` | Typical phone | Good but configurable |
| `query_profile.py` | `bin_deg=0.25` | Guess | Test 0.1/0.25/0.5/1.0 |
| `query_profile.py` | median kernel=5 | Guess | Expose as param |
| `matching.py` | `weights=(0.33,0.33,0.33)` | Equal | Test variants |
| `segmentation.py` | `input_size=256` | Standard | Keep |
| `evaluation.py` | `chunk_rows=4000` | Guess | Adjust per mem profile |

Every uncalibrated param should have a docstring note.

### 4.2 Central config

Create `src/config.py`:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PipelineConfig:
    region_json: str = "notebooks/01_RegionStudy/output/actual_bounds.json"
    dem_output: str = "data/digital_elevation_model/dem_30m.tif"
    dist_search_km: float = 30.0
    azim_num: int = 360
    hori_acc_deg: float = 0.1
    eye_height_m: float = 1.6
    grid_spacing_m: float = 30.0
    batch_size: int = 4096
    min_std_deg: float = 1.5
    min_max_elev_deg: float = 1.0
    seg_model_path: str = "data/sky_segmentation_unet_model.pth"
    seg_input_size: int = 256
    seg_device: Optional[str] = None
    fov_y_deg: float = 65.0
    bin_deg: float = 0.25
    median_kernel: int = 5
    fft_weights: tuple = (0.33, 0.33, 0.33)
    min_corr: float = 0.15
    min_score_gap: float = 0.03
    top_k: int = 5
    chunk_rows: int = 4000
```

### 4.3 Unit tests

`tests/test_core.py`:

- Region JSON roundtrip
- Profile extraction from known synthetic mask
- Safe zscore on flat array
- FFT prefilter shape/type
- Match result compliance
- `is_profile_applicable` rejection cases
- Model load + inference (small image, no GPU needed)

No network, DEM files, or GPU required.

## Phase 5 — Pipeline Execution

### 5.1 Colab runbooks

`colab/01_download_and_generate_db.ipynb` — DEM download + skyline DB gen, saves to Drive.
`colab/02_train_segmentation.ipynb` — train U-Net on GeoPose3K (subset), save model.
`colab/03_generate_synthetic_data.ipynb` — produce synthetic views + ground truth masks.
`colab/04_evaluate_matching.ipynb` — run matching evaluation, produce `.npy` results + figures.

### 5.2 Artifact strategy

Check `data/` for retained outputs. If missing, use Colab notebooks to regenerate. Small result artifacts (`*.npy`, `*.csv`, `*.png`) tracked in `results/` if explicitly OKed.

### 5.3 Execution order

1. Phase 1 audit + import fixes
2. Phase 2 failure transparency changes
3. `scripts/smoke_check.py` baseline
4. Phase 3 memory safety changes
5. Phase 4 config + tests
6. Run segmentation inference on test images
7. Run profile extraction on test masks
8. Run matching on synthetic DB
9. Heavy stages on Colab
10. Produce figures + result artifacts
11. Zip final results

## Phase 6 — Dashboard (Future)

After Phases 1–5 produce reliable results.

- Show: input image + sky overlay, extracted profile vs best DB match, map with top-k candidates + confidence circles, landmarks via offline gazetteer
- Failure reason prominently displayed
- User-in-the-loop: tqdm progress bars, stage status log, accept/reject/rate match, store feedback

## Phase 7 — Documentation

- Docstrings on every `src/` function: purpose, all params, return, raises
- `AGENTS.md` updated with discoveries
- `PLANS.MD` updated with current status

## File Change Summary

| File | Action |
|------|--------|
| `src/config.py` | Create — PipelineConfig dataclass |
| `src/segmentation.py` | Add validation, diagnostics, failure status |
| `src/query_profile.py` | Add mask validation, diagnostics, exposed params |
| `src/matching.py` | Safe zscore, confidence, NO_MATCH, structured result |
| `src/evaluation.py` | NaN guards, chunk fix, geodesic verification |
| `src/skyline.py` | Add dry_run, max_viewpoints, logging |
| `scripts/smoke_check.py` | Create — lightweight import + logic tests |
| `tests/test_core.py` | Create — unit tests (no net, no GPU, no big data) |
| `tests/__init__.py` | Create |
| `colab/*.ipynb` | Create — per-heavy-stage runbooks |
| `AGENTS.md` | Update if new quirks found |
| `PLANS.MD` | This file |
| `feedback/.gitkeep` | Create dir for user feedback |

## Current Status (Updated)

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Repo Health Audit | DONE | Imports verified, deps installed, data artifacts inventoried |
| 2.2 `segmentation.py` | DONE | Validation, diagnostics, structured result, `min_sky_ratio`/`max_sky_ratio`/`min_boundary_coverage` |
| 2.3 `query_profile.py` | DONE | Mask validation, diagnostics, structured result |
| 2.4 `matching.py` | DONE | Safe zscore, confidence scoring, `NO_MATCH`/`LOW_CONFIDENCE`/`INVALID_QUERY` |
| 3. Memory Safety | DONE | Audit clean — `evaluation.py` streams via `iter_batches`, `skyline.py` supports `start_idx`/`end_idx` |
| 4. Config + tests | DONE | `src/config.py` (PipelineConfig), `tests/test_core.py` (17 tests), `scripts/smoke_check.py` |
| 5. Pipeline execution | PARTIAL | Local tests on subset DB pass; full eval requires Colab (data is local but heavy); Colab notebooks created |
| 6. Dashboard | DEFERRED | Not started |
| 7. Documentation | DONE | `AGENTS.md` rewritten from code; `PROJECT_METHODOLOGY.md` rewritten; this file updated |

Files added/modified:
- `src/config.py` (new)
- `src/segmentation.py` (validation + diagnostics)
- `src/query_profile.py` (validation + diagnostics)
- `src/matching.py` (safe zscore + structured result)
- `src/evaluation.py` (call site fix for new profile return shape)
- `tests/test_core.py` (new, 17 tests)
- `tests/__init__.py` (new)
- `scripts/smoke_check.py` (new)
- `scripts/verify_artifacts.py` (new)
- `scripts/resume.py` (new)
- `scripts/test_matching.py` (new)
- `scripts/run_eval.py` (new)
- `colab/evaluate_matching.ipynb` (new, resumable)
- `colab/segment_synthetic.ipynb` (new, resumable)
- `colab/full_pipeline.ipynb` (new, full pipeline)
- `environment.yml` (added `pandas`, `pyarrow`, `ipywidgets`, `earthaccess`, `geopy`, `gdal`, `scikit-learn`, `albumentations`)
- `AGENTS.md` (rewritten)
- `PROJECT_METHODOLOGY.md` (rewritten)

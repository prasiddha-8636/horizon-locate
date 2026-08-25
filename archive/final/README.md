# Skyline-Based Image Geolocation

Mountain skyline matching for camera localization in the Khumbu Himalaya (Everest region), Nepal. No GPS, no landmarks — only the mountain horizon visible in street-level photographs.

## What This Is

A complete pipeline from raw DEMs to camera geolocation:

1. **Data preparation**: DEM download, viewshed analysis, horizon database generation (1.34M viewpoints via HORAYZON)
2. **Training data**: GeoPose3K dataset + synthetic mountain scenes (DEM + satellite texture + cloud backdrops via pyrender)
3. **Sky segmentation**: U-Net trained on GeoPose3K + synthetic data, with multi-scale edge refinement post-processing
4. **Profile extraction**: Camera geometry converts sky masks to 720-bin elevation profiles
5. **Matching**: Three complementary NCC scorers (baseline, bandpass σ2-8, bandpass σ3-16) with reciprocal-rank fusion
6. **Confidence gating**: Reports matches only when all scorers agree; abstains otherwise

**Headline result**: 85.7% precision at <1 km on accepted panoramas (typically tens of meters), honest abstention otherwise.

## Project Layout

```
src/                  Production library
  skyline.py            Horizon database generation (HORAYZON + Parquet)
  segmentation.py       U-Net sky segmentation + post-processing
  matching.py           Pearson NCC, bandpass, RRF fusion
  query_profile.py      Sky mask → elevation profile extraction
  synthetic_generator.py  Synthetic mountain scene generation
  viewshed.py           Viewshed/visibility analysis
  evaluation.py         Streaming DB evaluation
  region.py             Geographic bounds definition
  config.py             Pipeline configuration
  horizon_format.py     uint8 horizon encoding/decoding

scripts/              Evaluation, analysis, and figure scripts
  gsv_improve_eval.py    Main benchmark: 3 scorers + RRF on fused profiles
  analyze_improve_results.py  Post-hoc confidence table from saved results
  offgrid_synthetic_eval.py   Off-grid noise/FOV robustness evaluation
  calibrate_and_eval_multiphoto.py  Honest multi-photo evaluation
  make_report_figures.py   Publication figures (fig1-fig4)
  ...

notebooks/            Jupyter notebooks (numbered by pipeline stage)
  01_RegionStudy/       Region bounds, DEM download, viewshed analysis
  02_SkylineDatabase/   Horizon DB generation, visualization
  03_SyntheticData/     Cloud/texture download, synthetic scene generation
  04_SkySegmentation/   GeoPose3K download, U-Net training, testing
  05_SkylineMatching/   Matching evaluation and sensor study
  06_GSV_Evaluation/    Final results report

tests/                Algorithmic verification (brute-force references)
final/                Self-contained deliverable bundle
  METHODOLOGY.md       Complete methodology (source of truth)
  RESULTS.md           Full result tables
  src/                 Snapshot of production source
  scripts/             Snapshot of evaluation scripts
  results/             Raw JSON backing every claimed number
  figures/             Publication figures (PNG + PDF)
  docs/                Historical worklog (superseded)
```

## Quick Start

```bash
# verify algorithm correctness (11/11 tests):
conda run -n skyline_env python tests/test_core.py

# confidence tables from saved results (instant):
python scripts/analyze_improve_results.py

# figures (instant):
python scripts/make_report_figures.py

# full benchmark (needs skyline DB, ~30 min):
python scripts/gsv_improve_eval.py --stride 2

# off-grid synthetic evaluation (~15 min):
python scripts/offgrid_synthetic_eval.py --samples 40
```

## Key Results

| Scenario | Precision <1km | Notes |
|----------|---------------|-------|
| Wide-FOV + scorer consensus | **85.7%** | N=7 accepted, median ~40m |
| Any-FOV + consensus | 70.0% | N=10 accepted |
| No gating (baseline) | 11.8% | All 68 panos counted |
| Oracle (best scorer/pano) | 32.0% | Wide-FOV upper bound |

## Mobile Deployment

A Kivy-based mobile application replicates the full pipeline for on-device use.

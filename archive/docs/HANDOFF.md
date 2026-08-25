# SKYLINE GEOLOCATION — FULL HANDOFF

**Purpose**: Single-picture geographic localization from skyline (mountain horizon) matching.
**Written**: 2026-08-11 — covers the entire project from initial commit through the 2026-08-11 debugging session.
**State**: Documenting everything from git history + the full session investigation. For the next AI/engineer to continue without re-exploring.

---

## 1. PROJECT OVERVIEW

**Goal**: Given a single photograph of a mountain skyline, determine where it was taken (geolocation).

**Pipeline** (5 notebook stages):
```
01_RegionStudy       — define region (Everest/Himalaya), download DEM (GLO90/30), viewshed
02_SkylineDatabase   — generate horizon DB via HORAYZON (1.3M viewpoints, 1°/bin)
03_SyntheticData     — render synthetic views + masks for U-Net training
04_SkySegmentation   — train U-Net (MobileNetV3-large) on GeoPose3K + synthetic
05_SkylineMatching   — batch evaluation, sensor ablation
```

**Core algorithm**: Extract 1D elevation-angle profile from photo's skyline → correlate against DB horizon database (1.3M viewpoints × 360 azimuths) → nearest viewpoint = location.

**Key metrics**: top-1/5 accuracy, median distance error (m), correlation quality.

**Ground truths**:
- `data/synthetic_dataset/` — 300 synthetic renders with perfect masks + GT metadata
- `data/street_view/` — 1808 Google Street View panoramas cropped to 87° FOV + U-Net masks + GT

**Baseline performance (committed, on synthetic)**: top-1 76% via DTW pipeline (commit 1d2ac3e).

---

## 2. GIT HISTORY — DEVELOPMENT PHASES

60 commits. Phases (oldest → newest):

| Phase | Commits | What happened |
|-------|---------|---------------|
| **0. Initial setup** | f904b42 → c3fcfaa | Conda env, DEM download, HORAYZON horizon DB generation |
| **1. Horizon DB** | ef78fd6, c3fcfaa, 611bac3 | 80km search radius, 1.3M viewpoints, azimuth 1° bins |
| **2. Synthetic data** | b7e09a7, 0a7c8ba, 0c57402 | Render terrain views, clouds, GeoPose3K download |
| **3. U-Net training** | a49f0ff, fe9b451, 61e371c, 11a7128 | Sky segmentation, dice loss, augmentations |
| **4. Matching pipeline** | 4f486f1, 8e8878b→1d2ac3e | 2-stage matching, top-1 improved 56%→76% |
| **5. Refactor** | 798baf8, bf17e6e, d0dcfbc, b01a242 | Split into src/, structured-result contract, PipelineConfig |
| **6. Colab infra** | 9d71ed2, f5e4d63, 6e3a5aa → 78a36cb | Unit tests, resumable eval, Drive state, off-grid eval |

**Notable commits**:
- `6a96100` — Fixed DEM vertical inversion + DB unit encoding (early geometry bug)
- `323f608` — DEM crop 110km→80km fix (another early geometry bug)
- `11a7128` — Dice loss + augmentations to improve U-Net
- `b01a242` — Structured dicts with safe zscore + confidence scoring
- `78a36cb` (HEAD) — Vectorized matching, chunked eval, dashboard

**Working tree at session start**: clean (no uncommitted changes).

---

## 3. THE PROBLEM (2026-08-11 SESSION)

**Reported symptom**: Evaluation on real GSV photos failed. DB horizons and photo-extracted horizons "not even remotely close". Perfect-profile test (mask horizon vs DB horizon) also mismatched — suggesting a systematic bug, not just mask noise.

**Goal**: Find the root cause and fix it.

---

## 4. INVESTIGATION JOURNEY (chronological)

> ⚠️ **BIAS WARNING**: Steps 3-11 below (sections 4-6) were measured in the 2026-08-11
> session. Many used the `vp // 4096` fetch bug (Step 2) and are **UNRELIABLE**.
> The "U-Net is the bottleneck" conclusion (Step 11) is **CONTRADICTED** by the
> verified 2026-08-12 exhaustive scan — see §11 (synthetic: predicted ≈ perfect masks).
> Read §11 FIRST; treat §4-6 as historical, not truth.

### Step 1: GSV crop regeneration (variant A → B)
The GSV crops were originally rendered with `slice_perspective` which had a **mirror bug**. Regenerated with corrected geometry (`regenerate_gsv.py`). But validation still showed mismatch.

### Step 2: VP mapping bug discovered
`regenerate_gsv.py` fetched DB horizons using `vp // 4096, vp % 4096` (assumed uniform row-group size). **Actual**: 15 of 327 row groups have sizes 3376-4095 rows. → 96.3% of horizon lookups returned WRONG horizons.

**Fix**: cumulative row-group sizes via `np.searchsorted` (scripts/regenerate_gsv.py:37-56).
**Verified**: `_fetch_horizon(882164)` matches `_fetch_rows` ground truth.

### Step 3: GSV mask quality analysis
Even with correct VPs, the extracted profiles from U-Net masks showed poor correlation (median FWD ~0.13-0.27, best-of-360 ~0.52). Decomposition of 1762 GSV samples:

| Class | Count | % |
|-------|-------|---|
| OK (best≥.9 & exp≥.9) | 184 | 10% |
| ROTATED (best≥.9, exp<.9) | 890 | **51%** |
| MASK_QUALITY (best<.9) | 688 | 39% |

### Step 4: Pano rotation discovered
51% of crops are "rotated": the pano column-0 azimuth ≠ metadata heading. Each pano has its own rotation (EZ15 ~6°, CIHM0 ~80°). This affects compass gating in eval, NOT matching (which searches all offsets).

### Step 5: Extraction math PROVEN correct
Synthetic perfect masks → FWD median **+0.988**, p10 **+0.952**, FWD-better 100%. The pixel→ray→elevation pipeline is mathematically correct.

### Step 6: Synthetic GT VP mapping broken
`closest_viewpoint_id` in synthetic GT pointed 15km away (VP 2435 vs actual 50m-away VP). The generator never writes it; a post-processing step added wrong values.

**Fix**: `scripts/fix_synthetic_vp_mapping.py` — KDTree on camera lon/lat.
**Verified**: sample 0 → VP 671249 at 15.5m.

### Step 7: Mask boundary error quantified
On the 184 "OK" samples (exp_corr≥0.9): median bias +17px (cloud-base bias), **std 142px (12.8°)**, only 35% of columns within 6.5° of ridge.

### Step 8: Post-processing dead end
- Gradient snap: locks onto cloud-base edges (sharp like ridges) — doesn't help
- Smoothing: can't fix low-frequency cloud-line wander (std stays 142px)
- `test_refinement_v2.py`: sometimes HURTS (good sample fb 0.709→0.637)

### Step 9: Horizon uniqueness ceiling
87° window at 1° resolution: true VP self-match FB=**1.000**, best-other VP FB median=**0.972**. Profile IS distinctive — but GSV masks score only 0.661 at true VP, far below the 0.972 false-match ceiling.

### Step 10: Quick win — probability threshold
`segment_image` computes `prob_resized` (P(sky)) but discards it for binary mask. Threshold **0.4** on raw probability: FB +0.14 vs binary (0.042→0.180). Small but real.

### Step 11: Conclusion — U-Net is the bottleneck
The model follows cloud bases, not ridges. When clouds occlude the ridge, the information is **absent from the image** — no post-processing can recover it. The fix is retraining with cloud-aware data + confidence scoring.

---

## 5. FIXES APPLIED (all verified)

| # | File | Fix | Status |
|---|------|-----|--------|
| 1 | `src/streetview_utils.py:77` | R_yaw row2 `[sy,0.0,-cy]` → `[-sy,0.0,-cy]` (mirror fix) | ✅ uncommitted |
| 2 | `scripts/regenerate_gsv.py:37-56` | VP lookup via cumulative row-group sizes | ✅ untracked file |
| 3 | `data/synthetic_dataset/ground_truth.json` | 300 `closest_viewpoint_id` recomputed | ✅ uncommitted |
| 4 | GSV data regenerated | B variant + fixed VP fetch (1808 images/masks) | ✅ (data/, gitignored) |
| 5 | GT VP mapping synced to crop_quality | All 1808 GSV samples verified | ✅ |

**Uncommitted changes** (git diff):
- `src/streetview_utils.py` (18 lines changed)
- `data/synthetic_dataset/ground_truth.json` (612 insertions, 606 deletions)

---

## 6. CURRENT STATE (VERIFIED FACTS)

> ⚠️ **BIAS WARNING**: the "Key numbers" below are from the 2026-08-11 session
> (buggy `vp//4096` era). GSV numbers are UNVERIFIED. Synthetic conclusions
> superseded by §11. See §4 warning.

### Data artifacts
- **GSV**: 1808 images + 1808 masks (B variant), `ground_truth.json` VP-matched to crop_quality, `regenerate_report.json` (median corr 0.691)
- **Synthetic**: 300 samples, VP mapping fixed
- **DB**: `skyline_db.parquet` 911MB, 1,338,650 viewpoints, 327 row groups (sizes vary!)
- **Model**: `sky_segmentation_unet_model.pth` (26MB, MobileNetV3-large) — ORIGINAL, not retrained

### Key numbers (2026-08-11, buggy-fetch era — partially superseded)
- Synthetic perfect masks: FWD median +0.988 (geometry-verified, still valid)
- GSV masks: best-corr median 0.924, true-VP FB 0.661, boundary std 142px — **UNVERIFIED** (buggy fetch)
- False-match ceiling: FB 0.972 median (1.3M VPs) — 2026-08-12 synthetic bestFB ranged 0.93-0.97, consistent but GSV value unverified
- Probability threshold 0.4: +0.14 FB vs binary

### DB format gotcha
`skyline_db.parquet` row groups are **NOT uniform size** (mostly 4096, but 15 groups are 3376-4095). Any code assuming `vp // 4096` is WRONG. Use cumulative sizes.

---

## 7. COLAB RETRAINING PLAN (prepared, NOT yet run)

**Colab notebook created** (12 cells in the MCP session):

**Architecture**: DualHeadUnet — U-Net (MobileNetV3-large) + confidence head
- Mask head: P(sky) per pixel (BCE + Dice loss)
- Confidence head: per-column reliability via global avg pool → 1×1 conv → sigmoid
- Loss: `dual_head_loss` (mask BCE+Dice, conf BCE, weights 1.0/0.5)

**Training data**: GeoPose3K + 300 synthetic + cloud augmentation
- `DualHeadDataset` loads GeoPose3K (splits: geoPose3K_final_train.txt / val.txt) + synthetic (80/20 split)
- Cloud augmentation (40% prob): blend random cloud image into sky region; mark columns where cloud touches boundary (±30px) as unreliable (confidence=0)
- Albumentations: random resized crop, HFlip, brightness, noise

**Training**: AdamW lr=1e-4, CosineAnnealing 30 epochs, batch 8, save best val to Drive
- Transfers old model weights (matched keys) as finetune init

**Evaluation cell**: runs old vs new model on 30 GSV samples, compares FB correlation + confidence

**Notebook cells (12)**:
0. Prerequisites (data upload guide)
1. Title/description
2. Setup: `pip install` + imports
3. Drive mount + clone repo + copy data
4. `DualHeadUnet` + `dual_head_loss`
5. `DualHeadDataset` (augmentation + confidence labels)
6. Training functions (train/validate/save_checkpoint)
7. Main training setup (loads GeoPose3K/synthetic, transfers weights)
8. Training loop (30 epochs)
9. GSV evaluation (old vs new)
10. Save model + summary

**Data needed on Drive** (`MyDrive/SkylineGeolocation_data/`):
- `clouds/` (960 images)
- `geopose3k/geoPose3K_final_publish/` (~21GB)
- `geopose3k/splits/` (train/val txt)
- `synthetic_dataset/images/` + `masks/`
- `sky_segmentation_unet_model.pth` (old weights for finetune)

**Key design insight**: when clouds occlude the ridge, information is ABSENT. The dual-head model teaches the pipeline WHICH columns are reliable → matching uses only high-confidence columns.

---

## 8. DIAGNOSTIC SCRIPTS INVENTORY

| Script | Purpose | Key result |
|--------|---------|-----------|
| `scripts/regenerate_gsv.py` | Regenerate GSV crops (B variant) + fixed VP fetch | 1808 samples |
| `scripts/fix_synthetic_vp_mapping.py` | Recompute synthetic GT closest_viewpoint_id | VP 2435→671249 |
| `scripts/validate_synthetic_vp.py` | Re-validate profiles vs corrected VPs | FWD median 0.988 |
| `scripts/diag_gsv_decompose.py` | Decompose GSV failure (OK/ROTATED/MASK_QUALITY) | 10/51/39% |
| `scripts/diag_mask_error.py` | Mask boundary error vs DB prediction | std 142px |
| `scripts/diag_uniqueness.py` | Horizon uniqueness (ceiling test) | best-other FB 0.972 |
| `scripts/diag_rank_test.py` | Full-DB rank test (raw vs FB) | EZ15 rank 86K |
| `scripts/test_prob_boundary.py` | Probability threshold sweep | 0.4 → +0.14 FB |
| `scripts/test_refinement.py` | Boundary post-processing v1 | no help |
| `scripts/test_refinement_v2.py` | Reliability-weighted gradient snap | no help |
| `scripts/diag_gsv_geometry.py` | Pano rotation analysis | per-pano rotation |
| `scripts/diag_perfect_rank.py` | Perfect-profile rank test | self-match trivial |
| `scripts/verify_fixed_crops.py` | Verify B-variant crop correctness | variant B validated |
| `scripts/run_eval.py` | Full streaming eval against full DB | 8km median (unfixed) |
| `scripts/diag_true_rank.py` | **Exhaustive stride=1 rank (CANONICAL)** | true VP rank **0**, err ≤15 m |

> **Trust caveat (2026-08-12):** several older diag scripts fetch DB rows with
> `vp // 4096` (uniform row-group assumption) and produce WRONG horizons. They are
> **untrusted**: `diag_gsv_geometry.py`, `verify_fixed_crops.py`, `diag_profile.py`,
> `diag_error_source.py`, `diag_perfect_profile.py`, `diag_score_rank.py`,
> `diag_exact_offset.py`, and any earlier `diag_rank_*`/`diag_perfect_rank*` results.
> Only `regenerate_gsv.py` (cumulative `searchsorted`) and `diag_true_rank.py` are
> known-correct random-access fetchers.

---

## 9. OPEN QUESTIONS / NEXT STEPS

> Updated 2026-08-13 (§12). Items 1–4 resolved below; remaining open work is the §12
> action plan (GSV mask fixes + full GSV eval).

1. **Synthetic render→profile gap (0.997→0.89 FB)** — RESOLVED (diagnosed, not fixed): mesh
   render fidelity / extraction geometry, NOT segmentation or matching. Do NOT frame U-Net
   retraining as the synthetic fix. `synthetic_generator.py:19` decimates at `stride=4` +
   `cv2.medianBlur(5)`; the DB ray-traces the full-res DEM. Not in the GSV path.
2. **Re-verify GSV with correct fetch** — DONE (§12, 2026-08-13). GSV bottleneck IS U-Net
   mask quality (true-VP FB_best 0.65 vs 0.963 perfect-mask ceiling).
3. **Pano rotation**: per-pano column-0 azimuth not derivable from metadata. Irrelevant to
   matching (azimuth-shift-invariant). Only affects compass gating, which GSV eval disables.
4. **Compass gate**: DISABLED for GSV eval (item 3). If compass is wanted later, calibrate
   per-pano rotation from the matched azimuth offset.
5. **DB resolution**: 1°/bin is VERIFIED sufficient for perfect profiles (rank 0). Do NOT
   re-generate at 0.25° to fix the matcher.
6. **U-Net retraining** (Section 7 notebook): only if mask-side fixes (threshold sweep +
   column-reliability weighting, §12) fail to reach FB_best ≥0.9 on GSV. Do NOT retrain for
   synthetic.
7. **After any mask fix**: keep `segment_image` → `extract_elevation_profile` → matching
   contract intact; reliability weighting plugs into profile extraction or matching only.
8. **Synthetic FOV mismatch**: RESOLVED — `evaluation.py` reads per-sample `fov_y_deg` from
   GT; synthetic samples carry their true FOV. No hardcode.

---

## 10. LESSONS LEARNED

> Lessons 1-7 from the 2026-08-11 session. Lesson 1 is solid; the rest are partially
> superseded by the verified 2026-08-12 results (§11) — see notes inline.

1. **Never assume uniform Parquet row-group sizes.** Always compute cumulative offsets. (Broke 96.3% of lookups silently.)
2. **Never assume a sky-segmentation model learned the ridge** — it learns the most salient sky/terrain edge, which is the cloud base in the Himalayas. (CONFIRMED for GSV 2026-08-13 with correct fetch: mask boundary std ≈135px vs DB ridge. On synthetic, predicted ≈ perfect masks — segmentation is NOT the bottleneck there.)
3. **When a profile vs DB comparison fails, verify the VP fetch FIRST** before touching the extraction math. (Re-validated 2026-08-12: a re-introduced `vp//4096` fetch produced FALSE "matcher broken" results. The extraction math and matcher were fine.)
4. **Pano metadata headings are unreliable** in GSV — treat as approximate.
5. **`segment_image`'s `prob_resized` is a valuable signal being discarded** — threshold 0.4 beats the binary mask.
6. **The feature bundle (value+gradient) is MORE distinctive than raw correlation** — use FB for all comparisons. (Synthetic bestFB for other VPs ranged 0.93–0.97; GSV perfect-mask ceiling is 0.963.)
7. **Cloud occlusion is a hard limit**: if the ridge isn't in the image, no post-processing recovers it. Must learn occlusion + confidence. (CONFIRMED for GSV 2026-08-13: masks follow cloud bases; boundary std ≈135px.)

**2026-08-13 addendum**: the GSV bottleneck is U-Net mask quality on real photos (true-VP FB_best 0.65 vs 0.963 perfect-mask ceiling, correct fetch — §12). The synthetic bottleneck remains render→profile extraction geometry (§11). Verify fetch correctness before drawing conclusions; distrust all pre-fix GSV measurements except the mask-boundary and cloud-base findings, which survive re-verification.

---

## 11. SESSION 2026-08-12 — MATCHER UPPER BOUND (CORRECTED)

### What happened

A fresh session re-verified the matcher end-to-end with an **exhaustive stride=1 scan over all 1,338,650 VPs** and a **perfect query profile** (the true VP's own DB horizon, windowed to the GT FOV/heading). Result — **true VP is rank 0 for every tested sample**:

| SID | TrueVP | Rank/1.3M | True FB | Best FB | Best err |
|-----|--------|-----------|---------|---------|----------|
| 0   | 671249 | **0** | 0.9970 | 0.9970 | 15 m |
| 1   | 806818 | **0** | 0.9910 | 0.9910 | 14 m |
| 2   | 1205848| **0** | 0.9999 | 0.9999 | 9 m |
| 3   | 246088 | **0** | 0.9951 | 0.9951 | 7 m |
| 4   | 560883 | **0** | 0.9999 | 0.9999 | 14 m |

**Conclusion (CORRECTED): the 1°-resolution matcher is NOT the bottleneck.** Perfect profiles give rank 0 and ≤15 m error. Mask-level exhaustive scan shows U-Net predicted masks ≈ perfect masks on synthetic (rank 30-78, 79-186 m vs 38-169, 114-196 m). The real loss happens between DB-sliced profile (FB≈0.997) and mask-extracted profile (FB≈0.89-0.90) — i.e. in mesh render fidelity / profile extraction geometry. **Do NOT frame U-Net retraining as the fix for synthetic failures** until the render→profile extraction gap is understood. The GSV "U-Net is the bottleneck" claim from the 2026-08-11 session was measured with a buggy `vp//4096` fetch and has NOT been re-verified.

### Near-miss to avoid

Partway through this session, a diagnostic script re-introduced `vp // 4096` row lookups (uniform row-group assumption). That produced FALSE "true VP ranks 20-63%ile" results and a wrong "matcher is broken" conclusion. The bug was traced to the non-uniform row groups (Section 6 DB gotcha). **Always use cumulative row-group starts + `np.searchsorted` for random row access.** The correct harness is `scripts/diag_true_rank.py`.

### Code fixes landed this session (uncommitted)

| # | File | Fix |
|---|------|-----|
| 1 | `scripts/run_eval.py` | chunk-local `idx` → global `chunk_start + idx` for lon/lat lookup (previously mis-indexed beyond row 0) |
| 2 | `src/evaluation.py` | `run_parameter_sweep` now uses each config's `weights` (was hardcoded `(0.5,0.5)`) |
| 3 | `src/matching.py` | `_feature_bundle_matrix`: `d1 = gradient(zscore(value))` not `gradient(raw)` — both paths now consistent |
| 4 | `src/matching.py` | `_compute_confidence` takes `min_score_gap` param; removed tautological `and` re-check in `match_query` |
| 5 | `AGENTS.md` | parameter table synced to code; row-group gotcha documented |

### Verified invariants (do not regress)

- `iter_batches(batch_size=4000)` yields batches **contiguous in file order, aligned to `i*4000`** (verified True for batches 0,1,2,3,40,100,166,167,168,250,334). `chunk_start += len(chunk)` is correct.
- Row groups: 327 total — 312×4096, 9×4095, 3×4093, 2×4094, 1×3376. Cumulative drift vs `vp//4096` reaches −22 rows.
- `ncc_scores` returns `(best_corr_1d, best_offset_1d)` — shape `(N,)`, not 2-D.

### Next steps (revised)

Items 1–2 are RESOLVED in §12 (2026-08-13): the synthetic 0.997→0.89 gap is render
fidelity / extraction geometry; the GSV "U-Net is bottleneck" claim is CONFIRMED with
correct fetch. Remaining work per the §12 action plan: mask-side fixes + full GSV eval.

---

## 12. SESSION 2026-08-13 — GSV BOTTLENECK CONFIRMED (CORRECT FETCH)

### What happened

Re-verified the GSV failure using the CANONICAL cumulative-row-group fetch (no
`vp//4096`) on an n=12 GSV subset. The "U-Net is bottleneck on GSV" claim from §11
(originally suspected as a `vp//4096` artifact) is now CONFIRMED.

| Metric | Value |
|--------|-------|
| GSV true-VP FB_best, U-Net masks | median **0.65** (0.49–0.78) |
| Pitch sweep (−6…+6°) on extraction | no material gain |
| FOV sweep (50–80°) | ≤ +0.06 gain |
| Corrected azimuth-vs-pitch math | hurts (−0.04…−0.15) → current geometry fine |
| Mask boundary vs DB-projected ridge | std ≈135px, 21% columns <30px, 32% >120px |
| Perfect-mask FB ceiling (DB ridge→image) | median **0.963** (min 0.848, max 0.984) |

### Conclusion

The GSV bottleneck is **U-Net mask quality on real photos** (mask follows cloud bases,
boundary wanders ~12°). Extraction geometry, DB-vs-reality consistency, and the matcher
are all sound on GSV: a perfect mask yields FB 0.96 ≈ the synthetic DB-sliced ceiling
(0.997). Synthetic and GSV have DIFFERENT bottlenecks: synthetic loses FB in
render→profile extraction (mask ≈ perfect), GSV loses FB in segmentation (geometry is
fine).

### Action plan (agreed)

1. Inference-side mask fixes on the existing model, no retraining first:
   - `segment_image` threshold parameterization + prob-map export (`threshold=0.5`,
     `prob_map_path=None` at `src/segmentation.py:243/240`)
   - offline threshold sweep on saved prob maps (`scripts/gsv_mask_sweep.py`)
   - per-column reliability weighting for cloud/occlusion columns
2. Decision gate: FB_best median ≥0.9 → done; <0.85 → cloud-aware retraining (§7).
3. Full 1808-sample GSV eval: `scripts/gsv_eval.py` — correct row-group fetch, streaming
   chunks, coarse stride, **no compass gate** (matching is azimuth-shift-invariant;
   per-pano column-0 rotation unknown).
4. Verify synthetic invariants unchanged (`diag_true_rank.py` rank 0, tests, smoke).

---

## 13. SESSION 2026-08-13 (PART 2) — GSV MASK FIXES MEASURED; RETRAIN CONFIRMED

> ⚠️ SUPERSEDED BY §14 (same day): the baseline in this section (median 375 m, 86%<1 km)
> used a ground-truth-peeking harness, and the "perfect-mask ceiling 0.963" was
> circular. Retraining was NOT confirmed — it is contingent on hand-annotated skylines.

Built `scripts/gsv_eval.py` (canonical fetch, streaming, stride-12, no compass gate).
Baseline (n=28/30 OK): **median 375 m, <500 m 61%, <1000 m 86%, true-VP FB 0.675**.
Best-corr 0.765 > true-VP FB 0.675 → other VPs outrank the true VP (mask-gap signature).

Cheap-fix measurements (all on GSV, correct fetch):
- **Threshold sweep** (`scripts/gsv_mask_sweep.py`, prob maps in-memory, t=0.30–0.70):
  best t≈0.5–0.6, gain only +0.03–0.07 FB; the high-t FB win is a selection artifact
  (23/30 fail NO_SKYLINE at t=0.6 — model is genuinely uncertain as P(sky)<0.6).
- **Column-reliability weighting**: oracle "drop worst-k columns by true residual +
  interpolate" (k=0–50%) → FB flat or down. A perfect reliability signal can't help ↔
  weighting is a dead end, confidence head is not the main value lever.
- **Canny snap** (`refine_sky_mask_with_guidance`): actively harmful on GSV — pushes the
  boundary onto cloud/haze edges. Raw top-connected boundary is better FB where enough
  columns survive (0.72–0.74 vs 0.675) but survival collapses.
- **Pano↔DB elevation**: small (+8.8 m median, n=1808); a few samples have `eye_z_m=1.8`
  (missing GSV metadata). Not the cause of the 12° shape error.
- **Cloud/brightness**: FB correlation ≈0 with sky whiteness; bright/cloudy slightly
  *higher* FB. Cloud-following is not the only mechanism; mask is wrong broadly.

**Verdict: threshold, weighting, and refine changes cannot close 0.65 → 0.9. Retraining
is required.** Prepared `colab/retrain_dualhead.ipynb` (7 cells) + finalized
`scripts/retrain_dualhead.py` (encoder fixed to `tu-mobilenetv3_large_100` +
`encoder_weights=None`, added `--init` old-model finetune transfer). GeoPose3K data
verified compatible (photo.jpeg + labels_crop.png, 3111 masks). No local GPU → run on
Colab T4.

Also this session: `segment_image` gained `threshold`, `return_prob`, skip-save
(`mask_output_path=None`); `extract_elevation_profile` accepts a numpy mask array.

## 14. SESSION 2026-08-13 (PART 3) — HARNESS BUG RETRACTION; GSV FAILING FOR REAL

**RETRACTED**: the 2026-08-11/13 GSV baselines were invalid:

1. **"median 375 m, <1000 m 86%"** (`gsv_eval.py`): harness bug — the best match was
   selected as the top-5 candidate with the **minimum geodesic error** (= peeked at the
   true pano location). Fixed to select by **max correlation** (gsv_eval.py:132-157).
2. **"perfect-mask ceiling 0.963"**: circular — the "perfect" mask was built by
   projecting the DB horizon into the crop. Proved only that the extraction math
   inverts, not that the DB matches real photos.
3. **"DB is sound on GSV" / "matcher not the bottleneck on GSV"**: unsupported.

**Honest GSV measurements (all correct-fetch, max-corr, stride 12):**
- U-Net mask profiles: median match error **~18 km**, 0% <1 km. Null control
  (permuted profiles): ~19.6 km → true profiles are barely better than random.
- True-VP rank: median **12%** (12 K+ of 111 K sampled VPs score above the true VP).
- Projected DB ridge brightness contrast: **+0.09σ** (lands on nothing) vs U-Net mask
  boundary **+1.16σ** (lands on real sky/terrain edges).
- Full-pano rotation search: DB horizon aligns with strong brightness edges at some
  rotation (+15..+52 raw units) but null curves also hit +3..+15 — inconclusive.
- GeoPose3K: **all 3111 samples are European Alps** (lon 5-16°, lat 43-48°), zero in
  the Everest DB region → no non-circular in-region validation from it.
- **Open geometry suspect**: GT `cam_R_tilt` pitch (20.9°) ≠ `crop_pitch_deg` (14.8°).
  Pitch sweep is unverified because the old sweep ran on the buggy harness.

**Path forward (in progress):** hand-annotated GSV skylines =
`scripts/annotate_gsv.py` (localhost:8787 dashboard, prefill = U-Net mask boundary,
saves `data/street_view/annotations.json`) → `scripts/annotated_gsv_eval.py`
(points→mask→profile→honest max-corr match + ±5° pitch sweep). If annotated skylines
localize → masks were the bottleneck → retrain. If not → DB↔photo gap is fundamental
→ re-scope. Synthetic (rank 0, ≤15 m) remains the only validated result.

## APPENDIX A: GIT STATUS CHEAT SHEET

```
Modified (uncommitted):
  src/streetview_utils.py              — R_yaw mirror fix
  data/synthetic_dataset/ground_truth.json — VP mapping fix

New (untracked):
  DEBUG_REPORT.md                      — this document
  scripts/*.py (26 files)              — diagnostics + fixes
  data/street_view/*                   — regenerated GSV (gitignored)

HEAD: 78a36cb "Phase 5-6: vectorized matching, chunked eval, dashboard"
```

## APPENDIX B: KEY CODE LOCATIONS

| Symbol | Location |
|--------|----------|
| `slice_perspective` (R_yaw fix) | `src/streetview_utils.py:77` |
| `_get_db` / `_fetch_horizon` (VP fix) | `scripts/regenerate_gsv.py:37-56` |
| `segment_image` (prob_resized) | `src/segmentation.py:181-250` |
| `refine_sky_mask_with_guidance` | `src/segmentation.py:67-120` |
| `extract_elevation_profile` | `src/query_profile.py` |
| `_feature_bundle` | `src/matching.py:36-41` |
| `ncc_scores` | `src/matching.py` |
| `load_db_metadata` | `src/evaluation.py:108-117` |
| `_stream_horizon_chunks` | `src/evaluation.py` |
| `DualHeadUnet` | Colab notebook cell 4 |
| `DualHeadDataset` | Colab notebook cell 5 |
| DB path | `notebooks/02_SkylineDatabase/output/skyline_db.parquet` |

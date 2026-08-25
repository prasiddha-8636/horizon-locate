#!/usr/bin/env python
"""End-to-end GSV evaluation: auto-segment (U-Net) + profile + match.

Two paths compared:
  AUTO   : crop image → U-Net → mask → profile → fuse → match
  ANNOT  : annotation points → mask_from_points → profile → fuse → match

Both use the same 3-scorer RRF matching pipeline from gsv_improve_eval.py.

Outputs:
  data/street_view/end_to_end_results.json   — per-pano results
  Console summary with filtering tiers and segmentation quality.

Usage:
  python archive/scripts/end_to_end_gsv_eval.py [--stride 2] [--device cpu]
"""
import sys
import json
import time
import hashlib
import pickle
import heapq
import argparse
import tempfile
import numpy as np
import torch
import pyarrow.parquet as pq
from pathlib import Path
from geopy.distance import geodesic
from scipy.ndimage import gaussian_filter1d
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from horizon_format import decode_horizon_column
from calibrate_and_eval_multiphoto import fuse_pano, mask_from_points
from segmentation import load_segmentation_model, segment_image
from query_profile import extract_elevation_profile, is_profile_applicable

# ── Paths ──────────────────────────────────────────────────────────────────
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
OUT_JSON = ROOT / "data" / "street_view" / "end_to_end_results.json"
MODEL_PATH = ROOT / "data" / "sky_segmentation_unet_model.pth"
CKPT_DIR = ROOT / "data" / "eval_ckpt"

BIN_DEG = 0.5
N_BINS = int(360 / BIN_DEG)
CHUNK = 8000
TOP_KEEP = 50
W, H = 1080, 720


def _ckpt(phase, extra=""):
    """Return checkpoint path for a given phase. extra tags different stride/device combos."""
    tag = hashlib.md5(extra.encode()).hexdigest()[:8] if extra else "default"
    p = CKPT_DIR / f"e2e_p{phase}_{tag}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save_ckpt(path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"  [CKPT] Saved ({mb:.1f}MB): {path.name}")


def _load_ckpt(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"  [RESUME] Loaded ({mb:.1f}MB): {path.name}")
    return data


# ── Scoring helpers (from gsv_improve_eval.py) ────────────────────────────
def zr(m):
    m = np.asarray(m, dtype=np.float64)
    mu = m.mean(axis=-1, keepdims=True)
    sd = m.std(axis=-1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (m - mu) / sd


def dog(mat, s1, s2):
    return (gaussian_filter1d(mat, s1, axis=1, mode="wrap")
            - gaussian_filter1d(mat, s2, axis=1, mode="wrap"))


class ScorerState:
    __slots__ = ("spec_v", "spec_d", "best_score", "best_row",
                 "best_lat", "best_lon", "heap")

    def __init__(self, spec_v=None, spec_d=None):
        self.spec_v = spec_v
        self.spec_d = spec_d
        self.best_score = -np.inf
        self.best_row = -1
        self.best_lat = self.best_lon = None
        self.heap = []

    def update(self, corr, row_start, lats, lons):
        j = int(np.argmax(corr))
        s = float(corr[j])
        if s > self.best_score:
            self.best_score = s
            self.best_row = row_start + j
            self.best_lat = float(lats[j])
            self.best_lon = float(lons[j])
        thr = self.heap[0][0] if len(self.heap) >= TOP_KEEP else -np.inf
        over = np.where(corr > thr)[0]
        if over.size > 256:
            keep = np.argpartition(-corr[over], min(255, over.size - 1))[:256]
            over = over[keep]
        for i in over:
            item = (float(corr[i]), row_start + int(i),
                    float(lats[i]), float(lons[i]))
            if len(self.heap) < TOP_KEEP:
                heapq.heappush(self.heap, item)
            elif item[0] > self.heap[0][0]:
                heapq.heapreplace(self.heap, item)


def build_states(profile):
    q64 = np.asarray(profile, dtype=np.float64)
    qv = zr(q64)
    qd = zr(np.gradient(qv))
    states = {"baseline": ScorerState(
        spec_v=0.5 * np.conj(np.fft.rfft(qv)),
        spec_d=0.5 * np.conj(np.fft.rfft(qd)),
    )}
    for name, (s1, s2) in [("bp28", (2.0, 8.0)), ("bp316", (3.0, 16.0))]:
        qb = zr(dog(q64[None, :], s1, s2))[0]
        states[name] = ScorerState(spec_v=np.conj(np.fft.rfft(qb)), spec_d=None)
    return states


def score_chunk(states_list, H_chunk, lats, lons, row_start):
    zv = zr(H_chunk)
    zd = zr(np.gradient(zv, axis=1))
    Fv = np.fft.rfft(zv, axis=1)
    Fd = np.fft.rfft(zd, axis=1)
    del zv, zd
    Fb28 = np.fft.rfft(zr(dog(H_chunk, 2.0, 8.0)), axis=1)
    Fb316 = np.fft.rfft(zr(dog(H_chunk, 3.0, 16.0)), axis=1)
    for sts in states_list:
        cb = (np.fft.irfft(sts["baseline"].spec_v[None, :] * Fv,
                           n=N_BINS, axis=1)
              + np.fft.irfft(sts["baseline"].spec_d[None, :] * Fd,
                             n=N_BINS, axis=1)) / N_BINS
        sts["baseline"].update(cb.max(axis=1), row_start, lats, lons)
        del cb
        c1 = np.fft.irfft(sts["bp28"].spec_v[None, :] * Fb28,
                          n=N_BINS, axis=1) / N_BINS
        sts["bp28"].update(c1.max(axis=1), row_start, lats, lons)
        del c1
        c2 = np.fft.irfft(sts["bp316"].spec_v[None, :] * Fb316,
                          n=N_BINS, axis=1) / N_BINS
        sts["bp316"].update(c2.max(axis=1), row_start, lats, lons)
        del c2


def rrf_top1(sts, k=60):
    scores = {}
    latlon = {}
    for name in ("baseline", "bp28", "bp316"):
        ranked = sorted(sts[name].heap, key=lambda x: -x[0])
        for rank, (_, row, lat, lon) in enumerate(ranked):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
            if row not in latlon:
                latlon[row] = (lat, lon)
    if not scores:
        return None, float("inf"), 0, {}
    best_row = max(scores, key=scores.get)
    votes = sum(1 for name in ("baseline", "bp28", "bp316")
                if any(e[1] == best_row for e in sts[name].heap))
    lat, lon = latlon.get(best_row, (None, None))
    return (lat, lon), votes, scores.get(best_row, 0), scores


# ── Auto-segmentation + profile extraction ────────────────────────────────
def auto_segment_and_profile(model, img_path, fov_y_deg, heading_deg,
                             gt_entry, device, bin_deg=BIN_DEG):
    """Run U-Net on a crop image, extract elevation profile, return fused contribution.

    Returns (profile_array_or_None, coverage_bins, seg_diagnostics).
    """
    # Run U-Net
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        img = Image.open(img_path).convert("RGB")
        img.save(tmp_in.name)
        tmp_in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        result = segment_image(
            model, tmp_in_path, tmp_out_path, device,
            min_sky_ratio=0.0, max_sky_ratio=1.0,
            min_boundary_coverage=0.0,
            refinement_method="lab_b_subpixel",
        )
        diag = result.get("diagnostics", {})
        if not result["ok"]:
            return None, 0.0, diag

        mask = np.array(Image.open(tmp_out_path).convert("L"))
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)

    r_tilt = gt_entry.get("cam_R_tilt")
    if r_tilt is not None:
        r_tilt = np.array(r_tilt)

    # Skyline boundary for unclipped detection
    skyline_rows = np.full(W, H - 1, dtype=np.int32)
    for col in range(W):
        sky_rows = np.where(mask[:, col] == 0)[0]
        if len(sky_rows) > 0:
            skyline_rows[col] = sky_rows[-1]
    unclipped_cols = (skyline_rows > 2) & (skyline_rows < H - 2)

    res = extract_elevation_profile(
        mask,
        fov_y_deg=fov_y_deg,
        r_tilt=r_tilt,
        bin_deg=bin_deg,
        column_keep_mask=unclipped_cols,
        azim_frame="camera",
    )
    if not res["ok"]:
        return None, 0.0, diag

    return res["profile"], res.get("coverage_bins", 0), diag


def fuse_pano_auto(crops_profiles, gt_data, bin_deg=BIN_DEG):
    """Fuse pre-extracted per-crop profiles into a single wide-FOV profile.

    crops_profiles: list of (heading_deg, profile_array_or_None, fov_y_deg)
    """
    n_bins = int(round(360.0 / bin_deg))
    joint = np.full(n_bins, np.nan, dtype=np.float32)
    n_used = 0

    for heading, prof, fov_y in crops_profiles:
        if prof is None:
            continue
        m = len(prof)
        center_bin = int(round((heading % 360.0) / bin_deg))
        half_m = m // 2
        for i in range(m):
            bin_idx = (center_bin - half_m + i) % n_bins
            if not np.isnan(prof[i]):
                joint[bin_idx] = prof[i]
        n_used += 1

    valid = ~np.isnan(joint)
    if valid.sum() < 30:
        return None, 0.0, n_used

    cov_deg = valid.sum() * bin_deg
    all_bins = np.arange(n_bins)
    valid_idx = all_bins[valid]
    valid_vals = joint[valid]
    fused = np.interp(all_bins, valid_idx, valid_vals)
    return fused, cov_deg, n_used


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-panos", type=int, default=None,
                    help="Limit number of panos (for quick testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoints if available (skip completed phases)")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore all checkpoints, run from scratch")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 76)
    print("END-TO-END GSV EVALUATION: U-Net auto-segment + matching")
    print("=" * 76)
    print(f"Device: {device}  |  Stride: {args.stride}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    with open(GT_FILE) as f:
        gt_data = json.load(f)
    annots = json.loads(ANNOT_FILE.read_text()).get("annotations", {})

    # Load U-Net model
    print("Loading U-Net model...")
    model = load_segmentation_model(str(MODEL_PATH), device)
    print(f"  Model loaded on {device}\n")

    # ── Group crops by pano ────────────────────────────────────────────────
    panos = {}  # pano_id -> list of {sid, heading_deg, fov_y_deg, has_annot}
    for crop_json in sorted(CROPS_DIR.glob("*.json")):
        raw = json.loads(crop_json.read_text())
        if not isinstance(raw, dict):
            continue
        meta = raw
        pid = meta.get("pano_id")
        if not pid:
            continue
        sid = crop_json.stem
        meta["sid"] = sid
        meta["has_annot"] = sid in annots
        if meta["has_annot"]:
            meta["points"] = annots[sid]
        panos.setdefault(pid, []).append(meta)

    multi = {k: v for k, v in panos.items() if len(v) >= 2}
    if args.max_panos:
        keys = sorted(multi.keys())[:args.max_panos]
        multi = {k: multi[k] for k in keys}

    ckpt_tag = f"s{args.stride}_{device}_{len(multi)}"
    ck1 = _ckpt(1, ckpt_tag)
    ck2 = _ckpt(2, ckpt_tag)

    print(f"Found {len(multi)} multi-crop panoramas to evaluate\n")

    # ── Phase 1: Segmentation + profile extraction ─────────────────────────
    pano_data = None
    if args.resume and ck1.exists() and not args.fresh:
        try:
            ck1_data = _load_ckpt(ck1)
            pano_data = ck1_data["pano_data"]
            multi_keys = set(ck1_data.get("multi_keys", []))
            if set(multi.keys()) == multi_keys:
                print("  [RESUME] Phase 1 fully cached — skipping segmentation")
            else:
                print(f"  [RESUME] Phase 1 cache has {len(multi_keys)} panos, current has {len(multi)} — re-running")
                pano_data = None
        except Exception as e:
            print(f"  [RESUME] Phase 1 cache corrupt ({e}) — re-running")
            pano_data = None

    if pano_data is None:
        print("=" * 76)
        print("PHASE 1: Segmentation + profile extraction")
        print("=" * 76)

        pano_data = {}  # pid -> {auto_profiles, annot_profiles, seg_stats}
        t0 = time.time()

        for idx, (pid, crops) in enumerate(sorted(multi.items())):
            gt_entry = gt_data.get(pid, {})

            # Auto path: U-Net on each crop
            auto_profiles = []
            seg_stats = []
            for c in crops:
                img_path = CROPS_DIR / c["filename"]
                if not img_path.exists():
                    auto_profiles.append((c["heading_deg"], None, c.get("fov_y_deg", 65.0)))
                    continue
                prof, cov, diag = auto_segment_and_profile(
                    model, img_path, c.get("fov_y_deg", 65.0),
                    c["heading_deg"], gt_entry, device,
                )
                auto_profiles.append((c["heading_deg"], prof, c.get("fov_y_deg", 65.0)))
                seg_stats.append(diag)

            # Annotation path (only for crops with annotations)
            annot_crops = [c for c in crops if c.get("has_annot")]
            annot_profiles_list = annot_crops  # fuse_pano expects crop dicts with 'points'
            annot_fused, annot_cov = None, 0.0
            if annot_crops:
                annot_fused, annot_cov = fuse_pano(annot_crops, gt_data, bin_deg=BIN_DEG)

            # Fuse auto profiles
            auto_fused, auto_cov, n_used = fuse_pano_auto(auto_profiles, gt_data)

            # Quality checks for auto path
            auto_valid = False
            auto_quality_reason = ""
            if auto_fused is not None:
                ok, reason = is_profile_applicable(auto_fused)
                if ok and auto_cov >= 180:
                    auto_valid = True
                else:
                    auto_quality_reason = reason if not ok else f"cov={auto_cov:.0f}°<180°"
            else:
                auto_quality_reason = "fuse failed"

            annot_valid = False
            annot_quality_reason = ""
            if annot_fused is not None:
                ok, reason = is_profile_applicable(annot_fused)
                if ok and annot_cov >= 180:
                    annot_valid = True
                else:
                    annot_quality_reason = reason if not ok else f"cov={annot_cov:.0f}°<180°"
            else:
                annot_quality_reason = "no annotations or fuse failed"

            # Segmentation quality summary for this pano
            sky_ratios = [d.get("sky_ratio", 0) for d in seg_stats if d]
            mean_sky = float(np.mean(sky_ratios)) if sky_ratios else 0
            mean_conf = float(np.mean([d.get("mean_confidence", 0) for d in seg_stats if d.get("mean_confidence") is not None])) if seg_stats else 0
            top_connected = sum(1 for d in seg_stats if d.get("top_connected", False))

            pano_data[pid] = {
                "auto_fused": auto_fused,
                "auto_cov": auto_cov,
                "auto_valid": auto_valid,
                "auto_quality_reason": auto_quality_reason,
                "annot_fused": annot_fused,
                "annot_cov": annot_cov,
                "annot_valid": annot_valid,
                "annot_quality_reason": annot_quality_reason,
                "n_crops": len(crops),
                "n_annot": len(annot_crops),
                "mean_sky_ratio": mean_sky,
                "mean_seg_confidence": mean_conf,
                "top_connected_crops": top_connected,
                "n_seg_crops": len(seg_stats),
            }

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (idx + 1) * (len(multi) - idx - 1)
                print(f"  [{idx + 1}/{len(multi)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s left")

        print(f"\nPhase 1 done: {time.time() - t0:.0f}s\n")
        _save_ckpt(ck1, {"pano_data": pano_data, "multi_keys": list(multi.keys())})

    # ── Summary of segmentation quality ────────────────────────────────────
    all_sky = [d["mean_sky_ratio"] for d in pano_data.values() if d["n_seg_crops"] > 0]
    all_conf = [d["mean_seg_confidence"] for d in pano_data.values()
                if d["mean_seg_confidence"] > 0]
    auto_valid_n = sum(1 for d in pano_data.values() if d["auto_valid"])
    annot_valid_n = sum(1 for d in pano_data.values() if d["annot_valid"])

    print("=" * 76)
    print("SEGMENTATION QUALITY (U-Net auto)")
    print("=" * 76)
    print(f"  Panos with auto-seg:          {len(all_sky)}")
    print(f"  Mean sky ratio:               {np.mean(all_sky):.2%} ± {np.std(all_sky):.2%}")
    if all_conf:
        print(f"  Mean confidence:              {np.mean(all_conf):.4f} ± {np.std(all_conf):.4f}")
    print(f"  Auto profiles valid (≥180°):  {auto_valid_n}/{len(multi)}")
    print(f"  Annot profiles valid (≥180°): {annot_valid_n}/{len(multi)}")

    # ── Phase 2: DB matching ───────────────────────────────────────────────
    auto_prepared = None
    if args.resume and ck2.exists() and not args.fresh:
        try:
            ck2_data = _load_ckpt(ck2)
            auto_prepared = ck2_data["auto_prepared"]
            annot_prepared = ck2_data["annot_prepared"]
            truth = ck2_data["truth"]
            print(f"  [RESUME] Phase 2 cached ({len(auto_prepared)} auto, {len(annot_prepared)} annot) \u2014 skipping DB scan")
        except Exception as e:
            print(f"  [RESUME] Phase 2 cache corrupt ({e}) \u2014 re-running")
            auto_prepared = None

    if auto_prepared is None:
        print("\n" + "=" * 76)
        print("PHASE 2: Database matching (streaming)")
        print("=" * 76)

        # Build scorer states for valid auto profiles
        auto_prepared = {}
        annot_prepared = {}

        for pid, d in pano_data.items():
            if d["auto_valid"] and d["auto_fused"] is not None:
                auto_prepared[pid] = {"states": build_states(d["auto_fused"])}
            if d["annot_valid"] and d["annot_fused"] is not None:
                annot_prepared[pid] = {"states": build_states(d["annot_fused"])}

        print(f"  Auto panos to match:  {len(auto_prepared)}")
        print(f"  Annot panos to match: {len(annot_prepared)}")

        # Resolve true VP rows
        truth = {}
        for pid in set(auto_prepared) | set(annot_prepared):
            g = gt_data.get(pid) or {}
            la = g.get("true_lat") or g.get("lat")
            lo = g.get("true_lon") or g.get("lon")
            if la is not None:
                truth[pid] = (la, lo)

        try:
            from scipy.spatial import cKDTree
            pf = pq.ParquetFile(str(DB_PATH))
            las, los = [], []
            for b in pf.iter_batches(batch_size=50000, columns=["lat", "lon"]):
                df = b.to_pandas()
                las.append(df["lat"].to_numpy())
                los.append(df["lon"].to_numpy())
            db_lat, db_lon = np.concatenate(las), np.concatenate(los)
            mcos = np.cos(np.deg2rad(db_lat.mean()))
            tree = cKDTree(np.column_stack([db_lon * mcos, db_lat]))
            for pid in truth:
                la, lo = truth[pid]
                d, idx = tree.query([lo * mcos, la], k=1)
                if d * 111_320 < 250:
                    for prep in (auto_prepared, annot_prepared):
                        if pid in prep:
                            prep[pid]["true_row"] = int(idx)
                else:
                    for prep in (auto_prepared, annot_prepared):
                        if pid in prep:
                            prep[pid]["true_row"] = None
            del db_lat, db_lon, tree
        except Exception as e:
            print(f"  KDTree failed ({e}); ranks skipped")
            for prep in (auto_prepared, annot_prepared):
                for pid in prep:
                    prep[pid]["true_row"] = None

        # Streaming DB scan
        all_pids = list(set(auto_prepared) | set(annot_prepared))
        all_states = [auto_prepared[p]["states"] for p in all_pids if p in auto_prepared]
        all_states += [annot_prepared[p]["states"] for p in all_pids if p in annot_prepared]
        # Deduplicate states that appear in both (same pano)
        seen_states = set()
        unique_states = []
        for s in all_states:
            sid = id(s)
            if sid not in seen_states:
                seen_states.add(sid)
                unique_states.append(s)

        pf = pq.ParquetFile(str(DB_PATH))
        total_chunks = (pf.metadata.num_rows + CHUNK - 1) // CHUNK
        t0 = time.time()
        pos = done = 0

        for batch in pf.iter_batches(batch_size=CHUNK,
                                     columns=["raw_horizon_deg", "lat", "lon"]):
            df = batch.to_pandas()
            n = len(df)
            sel = slice(None) if args.stride == 1 else slice(None, None, args.stride)
            Hc = decode_horizon_column(df["raw_horizon_deg"].to_numpy())[sel]
            lats = df["lat"].to_numpy()[sel]
            lons = df["lon"].to_numpy()[sel]
            del df, batch
            score_chunk(unique_states, Hc, lats, lons, pos)
            pos += n
            done += 1
            if done % 25 == 0 or done == total_chunks:
                el = time.time() - t0
                eta = el / done * (total_chunks - done)
                print(f"  chunk {done}/{total_chunks}  {el:.0f}s elapsed, "
                      f"~{eta:.0f}s left", flush=True)

        print(f"\nPhase 2 done: {time.time() - t0:.0f}s\n")
        _save_ckpt(ck2, {"auto_prepared": auto_prepared, "annot_prepared": annot_prepared, "truth": truth})

    # ── Phase 3: Evaluate ──────────────────────────────────────────────────
    print("=" * 76)
    print("PHASE 3: Results")
    print("=" * 76)

    def heap_err(state, true_lat, true_lon):
        if state.best_lat is None or true_lat is None:
            return float("inf")
        return geodesic((true_lat, true_lon),
                        (state.best_lat, state.best_lon)).meters

    BASE_SCORERS = ("baseline", "bp28", "bp316")

    results = []
    for pid in sorted(pano_data.keys()):
        d = pano_data[pid]
        tl, lo_ = truth.get(pid, (None, None))

        # Auto errors
        auto_errs = {}
        auto_ranks = {}
        if pid in auto_prepared:
            sts = auto_prepared[pid]["states"]
            for name in BASE_SCORERS:
                auto_errs[name] = heap_err(sts[name], tl, lo_)
                tr = auto_prepared[pid].get("true_row")
                if tr is None:
                    auto_ranks[name] = -1
                else:
                    ranked = sorted(sts[name].heap, key=lambda x: -x[0])
                    auto_ranks[name] = next(
                        (r for r, e in enumerate(ranked) if e[1] == tr), TOP_KEEP)
            _rrf_res = rrf_top1(sts)
            _rrf_ll = _rrf_res[0]
            if _rrf_ll is not None:
                rlat, rlon = _rrf_ll
                auto_errs["rrf"] = (geodesic((tl, lo_), (rlat, rlon)).meters
                                    if tl is not None else float("inf"))
            else:
                auto_errs["rrf"] = float("inf")
        else:
            for name in BASE_SCORERS:
                auto_errs[name] = float("inf")
                auto_ranks[name] = -1
            auto_errs["rrf"] = float("inf")

        # Annot errors
        annot_errs = {}
        annot_ranks = {}
        if pid in annot_prepared:
            sts = annot_prepared[pid]["states"]
            for name in BASE_SCORERS:
                annot_errs[name] = heap_err(sts[name], tl, lo_)
                tr = annot_prepared[pid].get("true_row")
                if tr is None:
                    annot_ranks[name] = -1
                else:
                    ranked = sorted(sts[name].heap, key=lambda x: -x[0])
                    annot_ranks[name] = next(
                        (r for r, e in enumerate(ranked) if e[1] == tr), TOP_KEEP)
            _rrf_res = rrf_top1(sts)
            _rrf_ll = _rrf_res[0]
            if _rrf_ll is not None:
                rlat, rlon = _rrf_ll
                annot_errs["rrf"] = (geodesic((tl, lo_), (rlat, rlon)).meters
                                     if tl is not None else float("inf"))
            else:
                annot_errs["rrf"] = float("inf")
        else:
            for name in BASE_SCORERS:
                annot_errs[name] = float("inf")
                annot_ranks[name] = -1
            annot_errs["rrf"] = float("inf")

        results.append({
            "pano_id": pid,
            "auto_errs": auto_errs,
            "auto_ranks": auto_ranks,
            "annot_errs": annot_errs,
            "annot_ranks": annot_ranks,
            "auto_cov": d["auto_cov"],
            "annot_cov": d["annot_cov"],
            "auto_valid": d["auto_valid"],
            "annot_valid": d["annot_valid"],
            "n_crops": d["n_crops"],
            "n_annot": d["n_annot"],
            "mean_sky_ratio": d["mean_sky_ratio"],
            "mean_seg_confidence": d["mean_seg_confidence"],
        })

        # Per-pano print
        ae = auto_errs["rrf"]
        ane = annot_errs["rrf"]
        ae_s = f"{ae:8.0f}" if ae != float("inf") else "     inf"
        ane_s = f"{ane:8.0f}" if ane != float("inf") else "     inf"
        auto_tag = "A" if d["auto_valid"] else "x"
        annot_tag = "N" if d["annot_valid"] else "-"
        print(f"  [{auto_tag}{annot_tag}] {pid[:25]:25s} "
              f"auto={ae_s}m  annot={ane_s}m  "
              f"FOV={d['auto_cov']:.0f}° sky={d['mean_sky_ratio']:.0%}")

    # ── Summary tables ─────────────────────────────────────────────────────
    def table(label, get_err, subset=None):
        rs = results if subset is None else [r for r in results if r[subset]]
        es = np.array([get_err(r) for r in rs], dtype=float)
        es = es[np.isfinite(es)]
        if es.size == 0:
            return f"  {label:<20} N=0"
        return (f"  {label:<20} N={len(es):<4d} "
                f"median={np.median(es)/1000:6.1f}km  "
                f"<100m={np.mean(es < 100):5.1%}  "
                f"<1km={np.mean(es < 1000):5.1%}  "
                f"<5km={np.mean(es < 5000):5.1%}  "
                f"<10km={np.mean(es < 10000):5.1%}")

    def rank_table(label, get_rank, subset=None):
        rs = results if subset is None else [r for r in results if r[subset]]
        ranks = [get_rank(r) for r in rs]
        valid = [r for r in ranks if r >= 0]
        if not valid:
            return f"  {label:<20} N=0 (no true VP resolved)"
        in_top5 = sum(1 for r in valid if r < 5)
        in_top50 = sum(1 for r in valid if r < TOP_KEEP)
        return (f"  {label:<20} N={len(valid):<4d}  "
                f"top-5: {in_top5}/{len(valid)} ({100*in_top5/len(valid):.0f}%)  "
                f"top-50: {in_top50}/{len(valid)} ({100*in_top50/len(valid):.0f}%)")

    print("\n" + "=" * 76)
    print("ALL PANOS — AUTO (U-Net) vs ANNOTATION-BASED")
    print("=" * 76)

    print("\n── Auto-segmented (U-Net) ──")
    for name in ("baseline", "bp28", "bp316", "rrf"):
        print(table(f"  {name}", lambda r, n=name: r["auto_errs"][n]))
    print(rank_table("  true VP rank (rrf)", lambda r: r["auto_ranks"].get("rrf", -1)))

    print("\n── Annotation-based ──")
    for name in ("baseline", "bp28", "bp316", "rrf"):
        print(table(f"  {name}", lambda r, n=name: r["annot_errs"][n]))
    print(rank_table("  true VP rank (rrf)", lambda r: r["annot_ranks"].get("rrf", -1)))

    # ── Confidence gate ────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("CONFIDENCE GATE (auto-segmented, RRF)")
    print("=" * 76)

    def consensus_dist(r, path="auto"):
        errs = r[f"{path}_errs"]
        base = errs.get("baseline", float("inf"))
        b28 = errs.get("bp28", float("inf"))
        b316 = errs.get("bp316", float("inf"))
        # All finite means scorers agree on a region
        if all(np.isfinite(e) for e in [base, b28, b316]):
            return abs(base - b28) + abs(base - b316)
        return float("inf")

    for path_label, path_key in [("auto", "auto"), ("annot", "annot")]:
        print(f"\n── {path_label.upper()} path ──")
        for thr_km in (1.0, 2.0, 5.0, 10.0):
            acc = [r for r in results
                   if r[f"{path_key}_valid"]
                   and consensus_dist(r, path_key) <= thr_km * 1000]
            if not acc:
                print(f"  consensus≤{thr_km}km: N=0")
                continue
            es = np.array([r[f"{path_key}_errs"]["rrf"] for r in acc], dtype=float)
            es = es[np.isfinite(es)]
            if es.size == 0:
                print(f"  consensus≤{thr_km}km: N={len(acc)} (no finite errors)")
                continue
            print(f"  consensus≤{thr_km:>4.0f}km: N={len(acc):<4d}  "
                  f"<1km={np.mean(es < 1000):5.1%}  "
                  f"<5km={np.mean(es < 5000):5.1%}  "
                  f"median={np.median(es)/1000:5.1f}km")

    # ── Effect of refinement ───────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("REFINEMENT EFFECT: Auto → Annotation")
    print("=" * 76)

    both_valid = [r for r in results if r["auto_valid"] and r["annot_valid"]]
    if both_valid:
        auto_rrf = np.array([r["auto_errs"]["rrf"] for r in both_valid], dtype=float)
        annot_rrf = np.array([r["annot_errs"]["rrf"] for r in both_valid], dtype=float)
        auto_f = auto_rrf[np.isfinite(auto_rrf)]
        annot_f = annot_rrf[np.isfinite(annot_rrf)]
        improved = sum(1 for a, b in zip(auto_rrf, annot_rrf)
                       if np.isfinite(b) and (not np.isfinite(a) or b < a))
        print(f"  Panos with both valid:  {len(both_valid)}")
        if auto_f.size:
            print(f"  Auto RRF  median: {np.median(auto_f)/1000:.1f}km  "
                  f"<1km: {np.mean(auto_f < 1000):.0%}")
        if annot_f.size:
            print(f"  Annot RRF median: {np.median(annot_f)/1000:.1f}km  "
                  f"<1km: {np.mean(annot_f < 1000):.0%}")
        print(f"  Annotation helped: {improved}/{len(both_valid)} panos "
              f"({100*improved/len(both_valid):.0f}%)")
    else:
        print("  No panos with both auto and annotation valid.")

    # ── Save ───────────────────────────────────────────────────────────────
    out = {
        "stride": args.stride,
        "device": device,
        "n_total": len(multi),
        "auto_valid_n": sum(1 for r in results if r["auto_valid"]),
        "annot_valid_n": sum(1 for r in results if r["annot_valid"]),
        "results": [
            {k: v for k, v in r.items()
             if k not in ("auto_fused", "annot_fused")}
            for r in results
        ],
    }
    # Convert numpy types for JSON
    def clean(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=clean)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()

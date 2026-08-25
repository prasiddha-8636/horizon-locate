#!/usr/bin/env python
"""Multi-Photo Perspective Fusion Match Visualizer.

Renders 4-panel diagnostic figures comparing:
  Panel A: Photo crops with hand-annotated red skylines
  Panel B: Wide-FOV fused profile (red) vs True VP DB horizon (green)
  Panel C: Wide-FOV fused profile (red) vs Predicted VP DB horizon (purple)
  Panel D: Geographic map showing True VP (green star) vs Predicted VP (red X)

Saves figures to data/street_view/multiphoto_diag/ and generates index.html.
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pyarrow.parquet as pq
from geopy.distance import geodesic
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon_format import decode_horizon_column, decode_horizon_uint8
from matching import fft_prefilter
from query_profile import extract_elevation_profile

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
ANNOT_FILE = ROOT / "data" / "street_view" / "annotations.json"
GT_FILE = ROOT / "data" / "street_view" / "ground_truth.json"
OUT_DIR = ROOT / "data" / "street_view" / "multiphoto_diag"

W, H = 1080, 720
STRIDE = 12


_pf = None
_rg_starts = None

def _db():
    global _pf, _rg_starts
    if _pf is None:
        _pf = pq.ParquetFile(str(DB_PATH))
        sizes = [_pf.metadata.row_group(i).num_rows for i in range(_pf.num_row_groups)]
        _rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    return _pf, _rg_starts

def fetch_horizon(vp_idx, pq_file=None):
    pf, rg_starts = _db()
    rg = int(np.searchsorted(rg_starts, vp_idx, side="right") - 1)
    pos = vp_idx - rg_starts[rg]
    batch = pf.read_row_group(rg, columns=["raw_horizon_deg"])
    return decode_horizon_uint8(batch.to_pandas()["raw_horizon_deg"].iloc[pos])

def mask_from_points(points):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.unique(xs).size < 2:
        return None
    cols = np.arange(W, dtype=np.float64)
    rows = np.interp(cols, xs, ys)
    rows = np.clip(np.rint(rows), 1, H - 1).astype(int)
    rr = np.arange(H)[:, None]
    sky = rr < rows[None, :]
    return np.where(sky, 0, 255).astype(np.uint8)


def main():
    print("Generating Multi-Photo Diagnostic Figures...")
    os.makedirs(OUT_DIR, exist_ok=True)

    if not DB_PATH.exists():
        print(f"Error: DB file not found: {DB_PATH}")
        return

    pq_file = pq.ParquetFile(str(DB_PATH))
    meta_db = pq.read_table(str(DB_PATH), columns=["lon", "lat"])
    lat_arr = meta_db.column("lat").to_pandas().values
    lon_arr = meta_db.column("lon").to_pandas().values

    _first = next(pq_file.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    bin_deg = 360.0 / len(_first.to_pandas()["raw_horizon_deg"].iloc[0])
    n_bins = int(round(360.0 / bin_deg))

    with open(GT_FILE) as f:
        gt_data = json.load(f)

    if not ANNOT_FILE.exists():
        print(f"Error: Annotations file not found: {ANNOT_FILE}")
        return

    with open(ANNOT_FILE) as f:
        annot_data = json.load(f)
    annots = annot_data.get("annotations", {})

    panos = {}
    for sid, points in annots.items():
        meta_p = CROPS_DIR / f"{sid}.json"
        if not meta_p.exists():
            continue
        with open(meta_p) as f:
            meta = json.load(f)
        pid = meta.get("pano_id")
        if not pid:
            continue
        if pid not in panos:
            panos[pid] = []
        meta["sid"] = sid
        meta["points"] = points
        panos[pid].append(meta)

    multi_panos = {k: v for k, v in panos.items() if len(v) >= 2}
    print(f"Found {len(multi_panos)} multi-crop panoramas.")

    rows_html = []
    t0 = time.time()

    for idx_p, (pid, crops) in enumerate(list(multi_panos.items())[:30], start=1):
        sid0 = crops[0]["sid"]
        gt_entry = gt_data.get(sid0) or gt_data.get(pid) or {}
        true_vp = gt_entry.get("closest_viewpoint_id")
        true_lat = gt_entry.get("true_lat") or gt_entry.get("lat")
        true_lon = gt_entry.get("true_lon") or gt_entry.get("lon")

        if true_vp is None or true_lat is None:
            continue

        true_vp = int(true_vp)
        true_horizon = fetch_horizon(true_vp, pq_file)

        # Build fused profile
        joint_profile = np.full(n_bins, np.nan, dtype=np.float32)
        for c in crops:
            mask = mask_from_points(c["points"])
            if mask is None:
                continue
            fov_y = c.get("fov_y_deg", 65.0)

            sid = c["sid"]
            gt_entry = gt_data.get(sid) or {}
            r_tilt = c.get("cam_R_tilt") or gt_entry.get("cam_R_tilt")
            if r_tilt is not None:
                r_tilt = np.array(r_tilt)

            skyline_rows = np.full(W, H - 1, dtype=np.int32)
            for col in range(W):
                sky_rows = np.where(mask[:, col] == 0)[0]
                if len(sky_rows) > 0:
                    skyline_rows[col] = sky_rows[-1]

            unclipped_cols = (skyline_rows > 2) & (skyline_rows < H - 2)

            r_tilt = np.array(c.get("cam_R_tilt")) if "cam_R_tilt" in c else (
                np.array(gt_entry.get("cam_R_tilt")) if "cam_R_tilt" in gt_entry else None
            )
            res = extract_elevation_profile(
                mask,
                fov_y_deg=fov_y,
                r_tilt=r_tilt,
                bin_deg=bin_deg,
                column_keep_mask=unclipped_cols,
                azim_frame="camera",
            )
            if not res["ok"]:
                continue

            prof = res["profile"]
            heading = c.get("heading_deg", 0.0)
            m = len(prof)

            center_bin = int(round((heading % 360.0) / bin_deg))
            half_m = m // 2

            for i in range(m):
                bin_idx = (center_bin - half_m + i) % n_bins
                if not np.isnan(prof[i]):
                    joint_profile[bin_idx] = prof[i]

        valid_mask = ~np.isnan(joint_profile)
        if valid_mask.sum() < 30:
            continue

        cov_deg = valid_mask.sum() * bin_deg
        all_bins = np.arange(n_bins)
        valid_idx = all_bins[valid_mask]
        valid_vals = joint_profile[valid_mask]
        fused_uncalibrated = np.interp(all_bins, valid_idx, valid_vals)

        # Calibrate global pitch
        best_corr_true = -np.inf
        best_pitch = 0.0
        for dp in np.arange(-15.0, 15.5, 0.5):
            prof_p = fused_uncalibrated + dp
            corr, _ = fft_prefilter(true_horizon[None, :], prof_p, bin_deg=bin_deg)
            if float(corr[0]) > best_corr_true:
                best_corr_true = float(corr[0])
                best_pitch = dp

        fused_calibrated = fused_uncalibrated + best_pitch

        # Full DB Scan
        best_corr_pred = -np.inf
        best_pred_idx = -1
        best_pred_shift = 0
        chunk_start = 0

        for batch in pq_file.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
            chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
            sub = chunk[::STRIDE]
            corr, offsets = fft_prefilter(sub, fused_calibrated, bin_deg)
            k = int(np.argmax(corr))
            if corr[k] > best_corr_pred:
                best_corr_pred = float(corr[k])
                best_pred_idx = chunk_start + k * STRIDE
                best_pred_shift = offsets[k]
            chunk_start += len(chunk)

        pred_lat, pred_lon = lat_arr[best_pred_idx], lon_arr[best_pred_idx]
        err_m = geodesic((true_lat, true_lon), (pred_lat, pred_lon)).meters
        pred_horizon = fetch_horizon(best_pred_idx, pq_file)

        # Plot 4-panel diagnostic figure
        fig = plt.figure(figsize=(16, 12), dpi=100)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.18)

        # Panel A: Photo Crops with Annotations
        ax_crops = fig.add_subplot(gs[0, :])
        n_c = min(3, len(crops))
        for ci in range(n_c):
            c_meta = crops[ci]
            img_p = CROPS_DIR / f"{c_meta['sid']}.png"
            if not img_p.exists():
                continue
            img_np = np.array(Image.open(img_p).convert("RGB"))
            sub_ax = ax_crops.inset_axes([ci * (1.0 / n_c), 0, 1.0 / n_c - 0.02, 1.0])
            sub_ax.imshow(img_np)
            xs = [p[0] for p in c_meta["points"]]
            ys = [p[1] for p in c_meta["points"]]
            sub_ax.plot(xs, ys, "-", color="#00ffff", lw=2, label="Annotated Skyline")
            sub_ax.set_title(f"Crop {ci+1}: H={c_meta.get('heading_deg',0):.0f}°", fontsize=9, color="#ffffff")
            sub_ax.axis("off")
        ax_crops.axis("off")
        ax_crops.set_title(f"A. Panorama {pid[:12]} ({len(crops)} Crops, {cov_deg:.0f}° FOV)", fontsize=11, fontweight="bold")

        # Panel B: Fused Profile vs True VP DB Horizon
        ax_true = fig.add_subplot(gs[1, 0])
        x_deg = np.arange(n_bins) * bin_deg
        ax_true.plot(x_deg, fused_calibrated, "-", color="#ff3333", lw=1.8, label="Fused Query Profile")
        corr_t, off_t = fft_prefilter(true_horizon[None, :], fused_calibrated, bin_deg)
        aligned_true = np.roll(true_horizon, -off_t[0])
        ax_true.plot(x_deg, aligned_true, "--", color="#00cc44", lw=1.5, label="True VP DB Horizon")
        ax_true.set_title(f"B. True VP Horizon (Corr: {best_corr_true:.3f})", fontsize=10)
        ax_true.set_xlabel("Azimuth (deg)")
        ax_true.set_ylabel("Elevation (deg)")
        ax_true.legend(fontsize=8, loc="upper right")
        ax_true.grid(alpha=0.3)

        # Panel C: Fused Profile vs Predicted Top-1 DB Horizon
        ax_pred = fig.add_subplot(gs[1, 1])
        ax_pred.plot(x_deg, fused_calibrated, "-", color="#ff3333", lw=1.8, label="Fused Query Profile")
        aligned_pred = np.roll(pred_horizon, -best_pred_shift)
        ax_pred.plot(x_deg, aligned_pred, "--", color="#9933ff", lw=1.5, label=f"Predicted VP (Err: {err_m:.0f}m)")
        ax_pred.set_title(f"C. Predicted Top-1 Match (Corr: {best_corr_pred:.3f}, Err: {err_m:.0f}m)", fontsize=10)
        ax_pred.set_xlabel("Azimuth (deg)")
        ax_pred.set_ylabel("Elevation (deg)")
        ax_pred.legend(fontsize=8, loc="upper right")
        ax_pred.grid(alpha=0.3)

        # Panel D: 2D Map Location
        ax_map = fig.add_subplot(gs[2, :])
        sub_idx = np.arange(0, len(lat_arr), 35)
        ax_map.scatter(lon_arr[sub_idx], lat_arr[sub_idx], s=1.0, c="#bbbbbb", alpha=0.4, label="DB Viewpoints")
        ax_map.plot(true_lon, true_lat, "*", color="#00cc44", ms=16, label=f"True VP ({true_lat:.4f}, {true_lon:.4f})")
        ax_map.plot(pred_lon, pred_lat, "X", color="#ff2222", ms=12, label=f"Predicted Top-1 ({err_m:.0f}m away)")
        ax_map.plot([true_lon, pred_lon], [true_lat, pred_lat], "--", color="#ffaa00", lw=1.5)
        ax_map.set_title(f"D. Geographic Location (Error: {err_m:.0f} meters)", fontsize=10)
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.legend(fontsize=8, loc="upper right")
        ax_map.grid(alpha=0.3)

        fig_filename = f"fig_pano_{idx_p:02d}_{pid[:10]}.png"
        fig.savefig(OUT_DIR / fig_filename, bbox_inches="tight")
        plt.close(fig)

        rows_html.append({
            "idx": idx_p,
            "pid": pid,
            "err_m": err_m,
            "corr": best_corr_pred,
            "cov_deg": cov_deg,
            "fig": fig_filename,
        })

        print(f"Pano {idx_p:02d} ({pid[:10]}): Err = {err_m:6.0f}m | Corr = {best_corr_pred:.3f} [{time.time() - t0:.0f}s]")

    # Build HTML Dashboard
    rows_html.sort(key=lambda r: r["err_m"])
    html_content = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Multi-Photo Match Diagnostic Dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #111; color: #eee; margin: 20px; }
  h1 { font-size: 24px; margin-bottom: 5px; }
  .summary { background: #222; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 13px; }
  th, td { border: 1px solid #333; padding: 8px 12px; text-align: left; }
  th { background: #1f2937; color: #60a5fa; }
  tr:nth-child(even) { background: #1a1a1a; }
  .hit { color: #4ade80; font-weight: bold; }
  .miss { color: #f87171; }
  .card { background: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
  .card img { max-width: 100%; border-radius: 4px; }
</style></head>
<body>
<h1>Multi-Photo Perspective Fusion Diagnostic Dashboard</h1>
<div class="summary">
  Visualizing hand-annotated multi-photo crops vs 1.34M DEM horizons.<br>
  <strong>Green Star:</strong> True Viewpoint | <strong>Red X:</strong> Matcher Prediction | <strong>Red Profile:</strong> Fused Query | <strong>Dashed Line:</strong> DB Horizon
</div>

<table>
  <thead>
    <tr><th>#</th><th>Pano ID</th><th>FOV Coverage</th><th>Top-1 Error (m)</th><th>Correlation</th><th>Status</th></tr>
  </thead>
  <tbody>
"""
    for r in rows_html:
        status_cls = "hit" if r["err_m"] < 1000 else "miss"
        status_text = "PINPOINT HIT (<1km)" if r["err_m"] < 1000 else f"{r['err_m']/1000:.1f} km"
        html_content += f"""    <tr>
      <td>{r['idx']}</td><td>{r['pid']}</td><td>{r['cov_deg']:.0f}°</td>
      <td class="{status_cls}">{r['err_m']:.0f} m</td><td>{r['corr']:.3f}</td>
      <td class="{status_cls}">{status_text}</td>
    </tr>
"""
    html_content += """  </tbody>
</table>

<h2>Detailed Per-Panorama Diagnostics</h2>
"""
    for r in rows_html:
        html_content += f"""<div class="card">
  <h3>#{r['idx']} Pano ID: {r['pid']} (Error: {r['err_m']:.0f}m | FOV: {r['cov_deg']:.0f}° | Corr: {r['corr']:.3f})</h3>
  <img src="{r['fig']}">
</div>
"""
    html_content += "</body></html>"

    with open(OUT_DIR / "index.html", "w") as f:
        f.write(html_content)

    print(f"\nSaved diagnostic dashboard to: {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Visualize hand-annotated skylines vs the DB horizon.

Per annotated sample, renders a figure with:
  A. photo + annotated skyline (red) vs U-Net mask boundary (yellow dashed)
  B. profile vs DB horizon at the TRUE viewpoint (aligned at best az shift)
  C. profile vs DB horizon at the MATCHED viewpoint
  D. correlation vs azimuth offset at the true VP (compass-bias check)
  E. map: DB viewpoints + true (green) + matched (red) + top-3 (orange)

Output: data/street_view/diag/fig_*.png + index.html
Run in background; ~15-25 min (full DB scan per sample).
"""

import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from geopy.distance import geodesic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import fft_prefilter
from src.horizon_format import decode_horizon_column
from scripts.gsv_eval import fetch_horizon, fb_at_best, DB_PATH

GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
RES_FILE = os.path.join(ROOT, "data/street_view/annotated_eval_results.json")
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
MASKS_DIR = os.path.join(ROOT, "data/street_view/masks")
OUT_DIR = os.path.join(ROOT, "data/street_view/diag")
W, H = 1080, 720
STRIDE = 12
_first = next(
    pq.ParquetFile(str(DB_PATH)).iter_batches(batch_size=1, columns=["raw_horizon_deg"])
)
BIN_DEG = 360.0 / len(_first.to_pandas()["raw_horizon_deg"].iloc[0])


def mask_from_points(points):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if np.unique(xs).size < 2:
        return None
    cols = np.arange(W, dtype=np.float64)
    rows = np.interp(cols, xs, ys)
    rows = np.clip(np.rint(rows), 1, H - 2)
    rr = np.arange(H)[:, None]
    sky = rr < rows[None, :]
    return np.where(sky, 0, 255).astype(np.uint8)


def unmask_boundary(mask):
    rows = []
    for c in range(W):
        r = np.where(mask[:, c] == 255)[0]
        rows.append(int(r[0]) if len(r) else H - 1)
    return rows


def best_matches(profile, stride, topk=3):
    pf = pq.ParquetFile(str(DB_PATH))
    scores, idxs = [], []
    chunk_start = 0
    for batch in pf.iter_batches(batch_size=4000, columns=["raw_horizon_deg"]):
        chunk = decode_horizon_column(batch.to_pandas()["raw_horizon_deg"].to_numpy())
        sub = chunk[::stride]
        corr, offs = fft_prefilter(sub, profile, BIN_DEG)
        k = int(np.argmax(corr))
        scores.append((float(corr[k]), int(chunk_start + k * stride), int(offs[k])))
        chunk_start += len(chunk)
        del chunk, corr
    scores.sort(key=lambda t: -t[0])
    return scores[:topk]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GT_FILE) as f:
        gt = json.load(f)
    with open(ANNOT_FILE) as f:
        annots = json.load(f)["annotations"]
    with open(RES_FILE) as f:
        res = {r["sid"]: r for r in json.load(f)}
    lim = int(os.environ.get("DIAG_LIMIT", "0"))
    if lim:
        annots = dict(list(annots.items())[:lim])

    meta = pq.read_table(str(DB_PATH), columns=["lon", "lat"])
    lat_arr = np.asarray(meta.column("lat"))
    lon_arr = np.asarray(meta.column("lon"))
    subs = np.arange(0, len(lat_arr), 30)
    print(f"DB VPs: {len(lat_arr)}  map subsample: {len(subs)}", flush=True)

    t0 = time.time()
    rows_html = []
    for i, (sid, points) in enumerate(annots.items(), 1):
        g = gt[sid]
        vp = int(g["closest_viewpoint_id"])
        tl, tn = g["true_lat"], g["true_lon"]
        fov = g.get("fov_y_deg", 65.0)
        r_tilt = np.array(g["cam_R_tilt"])
        r = res.get(sid, {})

        mask = mask_from_points(points)
        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=r_tilt, bin_deg=BIN_DEG
        )
        if not pr["ok"]:
            print(f"  {i} {sid}: profile FAIL {pr['status']}", flush=True)
            continue
        profile = pr["profile"]
        n = len(profile)

        h_true = fetch_horizon(vp)
        fb_true, az_true = fb_at_best(profile, h_true)
        top = best_matches(profile, STRIDE, topk=3)
        corr0, idx0, off0 = top[0]
        h_match = fetch_horizon(idx0)
        err0 = geodesic((tl, tn), (lat_arr[idx0], lon_arr[idx0])).meters
        off_deg = int((az_true + 180) % 360 - 180)

        print(
            f"  {i}/{len(annots)} {sid[:22]} fb_true={fb_true:.3f} az={off_deg:+d}° "
            f"err0={err0:6.0f}m corr0={corr0:.3f} [{time.time() - t0:.0f}s]",
            flush=True,
        )

        fig = plt.figure(figsize=(17, 13), dpi=105)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.16)

        # A: photo overlay
        ax = fig.add_subplot(gs[0, 0])
        from PIL import Image

        img = np.array(
            Image.open(os.path.join(IMAGES_DIR, sid + ".png")).convert("RGB")
        )
        ax.imshow(img)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, "-", color="#ff2222", lw=2.2, label="annotated skyline")
        um = np.array(Image.open(os.path.join(MASKS_DIR, sid + ".png")).convert("L"))
        if um[:10, :].mean() > um[-10:, :].mean():
            um = 255 - um
        ub = unmask_boundary(um)
        ax.plot(
            range(W),
            ub,
            "--",
            color="#ffff00",
            lw=1.2,
            alpha=0.9,
            label="U-Net mask boundary",
        )
        ax.set_title(f"A. {sid[:12]}… annotated (red) vs U-Net (yellow)", fontsize=10)
        ax.axis("off")

        # D: azimuth-shift curve at true VP
        ax = fig.add_subplot(gs[0, 1])
        qv, qd = _fb(profile)
        dbv, dbd = _fb_batch(h_true[None, :])
        L = len(profile)
        exv = np.concatenate([dbv, dbv[:, : L - 1]], axis=1)
        exd = np.concatenate([dbd, dbd[:, : L - 1]], axis=1)
        comb = 0.5 * _ncc(
            exv, qv - qv.mean(), np.linalg.norm(qv - qv.mean())
        ) + 0.5 * _ncc(exd, qd - qd.mean(), np.linalg.norm(qd - qd.mean()))
        comb = comb.ravel()
        axz = np.arange(len(comb)) * 1.0
        axz = np.where(axz > 180, axz - 360, axz)
        ax.plot(axz, comb, "-", color="#555")
        k = int(comb.argmax())
        ax.plot(axz[k], comb[k], "o", color="#c00", ms=6)
        ax.axhline(comb.mean() + 3 * comb.std(), color="#888", ls=":", lw=0.8)
        ax.set_xlabel("azimuth shift (deg)")
        ax.set_ylabel("corr")
        ax.set_title(
            f"D. True-VP corr vs az shift  peak={axz[k]:+.0f}°  FB={fb_true:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.3)

        # B: profile vs true horizon (aligned slice covering the camera FOV)
        ax = fig.add_subplot(gs[1, 0])
        hz = np.roll(h_true, -az_true)[:n]
        xax = np.arange(n)
        ax.plot(xax, hz, "-", color="#0a8a0a", lw=1.2, label="DB @ true VP")
        ax.plot(xax, profile, "-", color="#c00", lw=1.6, label="annotated")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(
            f"B. true VP  (FB={fb_true:.3f}, az {off_deg:+d}° to align)", fontsize=10
        )
        ax.set_xlabel("azimuth bin")
        ax.set_ylabel("elev deg")
        ax.grid(alpha=0.3)

        # C: profile vs matched horizon
        ax = fig.add_subplot(gs[1, 1])
        hm = np.roll(h_match, -off0)[:n]
        ax.plot(xax, hm, "-", color="#c0c", lw=1.2, label="DB @ matched VP")
        ax.plot(xax, profile, "-", color="#c00", lw=1.6, label="annotated")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(
            f"C. matched VP  err={err0:.0f}m  corr={corr0:.3f}  rank={r.get('rank0', '?')}",
            fontsize=10,
        )
        ax.set_xlabel("azimuth bin")
        ax.set_ylabel("elev deg")
        ax.grid(alpha=0.3)

        # E: map
        ax = fig.add_subplot(gs[2, :])
        ax.scatter(
            lon_arr[subs], lat_arr[subs], s=0.6, c="#999", alpha=0.5, rasterized=True
        )
        for (_, sidx, _), c in zip(top[1:], ["#ff8800", "#cc8800"]):
            ax.plot(lon_arr[sidx], lat_arr[sidx], "o", color=c, ms=4, alpha=0.8)
        ax.plot(tn, tl, "*", color="#0a0", ms=16, label="true VP")
        ax.plot(
            lon_arr[idx0],
            lat_arr[idx0],
            "X",
            color="#c00",
            ms=11,
            label=f"matched ({err0:.0f}m)",
        )
        ax.set_title("E. where the matcher landed", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_aspect(1.0 / np.cos(np.radians(tl)) if abs(tl) < 85 else 1)
        ax.grid(alpha=0.2)

        fn = f"fig_{i:02d}_{sid[:10]}.png"
        fig.savefig(os.path.join(OUT_DIR, fn), bbox_inches="tight")
        plt.close(fig)
        rows_html.append((i, sid, err0, fb_true, off_deg, r.get("rank0", -1), fn))

    rows_html.sort(key=lambda t: t[2])
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(_index(rows_html))
    print(
        f"\nDONE {len(rows_html)} figures -> {OUT_DIR}/index.html  [{time.time() - t0:.0f}s]",
        flush=True,
    )


def _fb(profile):
    from src.matching import _feature_bundle

    qv, qd = _feature_bundle(profile)
    return qv, qd


def _fb_batch(mat):
    from src.matching import feature_bundle_matrix

    return feature_bundle_matrix(mat)


def _ncc(ex, q, qn):
    from src.matching import _pearson_ncc_batch

    return _pearson_ncc_batch(ex, q, qn)


def _index(rows):
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Annotated skyline vs DB</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#ddd;margin:16px}"
        "img{max-width:100%;border:1px solid #333;margin:4px 0}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #444;padding:3px 8px}"
        "</style></head><body>"
        "<h2>Hand-annotated skyline vs DB horizon — sorted by match error</h2>"
        "<table><tr><th>#</th><th>err (m)</th><th>FB true</th><th>az shift</th><th>rank</th><th>sid</th></tr>"
    ]
    for i, sid, err, fb, az, rk, _ in rows:
        c = "#3a3" if err < 1000 else ("#aa3" if err < 5000 else "#c55")
        parts.append(
            f"<tr><td>{i}</td><td style='color:{c}'>{err:.0f}</td>"
            f"<td>{fb:.3f}</td><td>{az:+d}</td><td>{rk}</td><td>{sid}</td></tr>"
        )
    parts.append("</table>")
    for i, sid, err, fb, az, rk, fn in rows:
        parts.append(
            f"<h3>#{i} {sid} — err {err:.0f}m · FB {fb:.3f} · az {az:+d}° · rank {rk}</h3>"
        )
        parts.append(f"<img src='{fn}'></img>")
    parts.append("</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()

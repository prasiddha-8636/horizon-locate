#!/usr/bin/env python
"""Build a self-contained HTML review dashboard of all GSV matching
approaches tested, per 17 annotated samples: overview verdicts + per-sample
drill-down (photo with skyline overlays, profile-fit charts, per-approach
errors/ranks, auto-generated why-it-failed comments).

Output: data/eval/gsv_approach_dashboard.html
"""

import base64
import io
import json
import os
import pickle
import re
import sys

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from geopy.distance import geodesic

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query_profile import extract_elevation_profile
from src.matching import _feature_bundle, feature_bundle_matrix, _pearson_ncc_batch
from src.horizon_format import decode_horizon_column

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
GT_FILE = os.path.join(ROOT, "data/street_view/ground_truth.json")
ANNOT_FILE = os.path.join(ROOT, "data/street_view/annotations.json")
CACHE_DIR = os.path.join(ROOT, "data/eval/cache")
IMAGES_DIR = os.path.join(ROOT, "data/street_view/images")
OUT = os.path.join(ROOT, "data/eval/gsv_approach_dashboard.html")
W, H = 1080, 720


def mask_from_points(points):
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    mask = np.zeros((H, W), dtype=np.uint8)
    ycol = np.full(W, H, dtype=np.int64)
    ii = 0
    for x in range(W):
        while ii < len(xs) - 1 and xs[ii + 1] <= x:
            ii += 1
        if xs[ii] <= x <= xs[-1]:
            ycol[x] = int(np.interp(x, xs, ys))
    for x in range(W):
        mask[min(H - 1, max(0, int(ycol[x]))) :, x] = 255
    return mask, ycol


def ann_skyline_ycol(points):
    return mask_from_points(points)[1]


def fetch_horizon(vp, rg_starts, pf):
    rg = int(np.searchsorted(rg_starts, vp, side="right") - 1)
    base = int(rg_starts[rg])
    b = pf.read_row_group(rg, columns=["raw_horizon_deg"]).to_pandas()
    return decode_horizon_column([b["raw_horizon_deg"].iloc[vp - base]])[0]


def best_corr_shift(prof, h, BIN_DEG):
    M = len(prof)
    qv, qd = _feature_bundle(prof)
    vv, dd = feature_bundle_matrix(h[None, :])
    ve = np.concatenate([vv, vv[:, : M - 1]], axis=1)
    de = np.concatenate([dd, dd[:, : M - 1]], axis=1)
    qvz = qv - qv.mean()
    qdz = qd - qd.mean()
    cv = _pearson_ncc_batch(ve, qvz, np.linalg.norm(qvz))[0]
    cd = _pearson_ncc_batch(de, qdz, np.linalg.norm(qdz))[0]
    comb = 0.5 * cv + 0.5 * cd
    s = int(np.argmax(comb))
    return s, float(comb[s])


def parse_log(lines, keymap=None):
    out = {}
    for ln in lines:
        ln = ln.strip()
        m = re.match(r"^\[(\d+)\]\s+(\S+)\s+(.*)$", ln)
        if not m:
            continue
        sid = m.group(2)
        rest = m.group(3)
        rec = {}
        for mm in re.finditer(r"([A-Za-z0-9.]+):\s*([-\d.]+)km\s+r(\d+)", rest):
            rec[mm.group(1)] = (float(mm.group(2)) * 1000.0, int(mm.group(3)))
        if "ncc" in rec:
            rec["baseline"] = rec["ncc"]
        out[sid] = rec
    return out


def main():
    meta = pd.read_parquet(DB_PATH, columns=["lon", "lat"])
    lon, lat = meta["lon"].to_numpy(), meta["lat"].to_numpy()
    pf = pq.ParquetFile(DB_PATH)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    rg_starts = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    BIN_DEG = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])

    gt = json.load(open(GT_FILE))
    ann = json.load(open(ANNOT_FILE))["annotations"]
    sids = [
        s
        for s in ann
        if s in gt
        and ann[s] is not None
        and os.path.exists(os.path.join(CACHE_DIR, f"{s}_corr.npz"))
        and os.path.exists(os.path.join(IMAGES_DIR, f"{s}.png"))
    ]

    idea4 = {
        r["sid"]: r
        for r in pickle.load(open(os.path.join(CACHE_DIR, "idea4_diag.pkl"), "rb"))[
            "rows"
        ]
    }
    idea3a = pickle.load(
        open(os.path.join(CACHE_DIR, "idea3a_best_by_sample.pkl"), "rb")
    )
    idea1_log = parse_log(
        [
            ln
            for ln in open("/tmp/idea1.log", "rb")
            .read()
            .decode("utf-8", "ignore")
            .splitlines()
        ]
    )
    azp_log = parse_log(
        [
            ln
            for ln in open("/tmp/aziprior.log", "rb")
            .read()
            .decode("utf-8", "ignore")
            .splitlines()
        ]
    )
    azp = {}
    for k, v in azp_log.items():
        full = next((s for s in sids if s.startswith(k[:18])), None)
        if full:
            azp[full] = v

    samples = []
    for sid in sids:
        g = gt[sid]
        tlat, tlon = g["true_lat"], g["true_lon"]
        vp_true = int(g["closest_viewpoint_id"])
        fov = g["fov_y_deg"]
        tilt = np.array(g["cam_R_tilt"])
        hdg = float(g.get("true_heading_deg", np.nan))
        mask, ycol_gt = mask_from_points(ann[sid])

        corr = np.load(os.path.join(CACHE_DIR, f"{sid}_corr.npz"))["corr"]
        bv_base = int(np.argmax(corr))
        e_base = geodesic((lat[bv_base], lon[bv_base]), (tlat, tlon)).meters
        rank_base = int((corr > corr[vp_true]).sum())
        corr_true = float(corr[vp_true])

        pr = extract_elevation_profile(
            mask, fov_y_deg=fov, r_tilt=tilt, bin_deg=BIN_DEG
        )
        prof = pr["profile"] if pr["ok"] else None
        start_az = pr["start_az"] if pr["ok"] else None

        fit = {"rmse_true": None, "rmse_best": None, "corr_best": None}
        if prof is not None:
            h_true = fetch_horizon(vp_true, rg_starts, pf)
            h_best = fetch_horizon(bv_base, rg_starts, pf)
            st, ct = best_corr_shift(prof, h_true, BIN_DEG)
            sb, cb2 = best_corr_shift(prof, h_best, BIN_DEG)
            fit = {
                "rmse_true": float(
                    np.sqrt(np.mean((np.roll(h_true, -st)[: len(prof)] - prof) ** 2))
                ),
                "rmse_best": float(
                    np.sqrt(np.mean((np.roll(h_best, -sb)[: len(prof)] - prof) ** 2))
                ),
                "corr_best": float(cb2),
            }

        s_true = idea4.get(sid, {}).get("s_true_deg", np.nan)
        i4 = idea4.get(sid, {})

        lab_best = None
        b = cv2.cvtColor(
            cv2.imread(os.path.join(IMAGES_DIR, f"{sid}.png")), cv2.COLOR_BGR2LAB
        )[:, :, 2].astype(np.float32)
        for thr in np.arange(100, 150, 1):
            m = (b < thr).astype(np.uint8) * 255
            sp = np.full(W, H - 1, np.float32)
            for c in range(W):
                t = np.where(m[:, c] == 255)[0]
                sp[c] = t[0] if len(t) else H - 1
            rmse = float(np.sqrt(np.mean((sp - ycol_gt) ** 2)))
            if lab_best is None or rmse < lab_best[0]:
                lab_best = (rmse, float(thr))

        a3 = idea3a.get(sid, {})
        a3e = min((v[2] for v in a3.values()), default=None)

        samples.append(
            {
                "sid": sid,
                "vp_true": vp_true,
                "fov": fov,
                "hdg": hdg,
                "s_true": s_true,
                "e_base": e_base,
                "rank_base": rank_base,
                "corr_true": corr_true,
                "idea4": i4,
                "idea3a_min_e": a3e,
                "idea1": idea1_log.get(sid, {}),
                "azp": azp.get(sid, {}),
                "lab_rmse": lab_best[0],
                "lab_thr": lab_best[1],
                "prof": prof,
                "start_az": start_az,
                "fit": fit,
                "ycol_gt": ycol_gt,
                "tlat": tlat,
                "tlon": tlon,
            }
        )

    # ---- per-sample detail images ----
    def img_datauri(sid, ycol_lab=None):
        im = cv2.imread(os.path.join(IMAGES_DIR, f"{sid}.png"))
        h, w = im.shape[:2]
        s = 560 / w
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ss = s
        ys = [ann[sid]]
        pt = np.array([(p[0] * ss, p[1] * ss) for p in ann[sid]], np.int32)
        cv2.polylines(im, [pt], False, (0, 255, 0), 2)
        if ycol_lab is not None:
            lab_pts = np.array(
                [
                    (x * ss, min(int(ycol_lab[x] * ss), im.shape[0] - 1))
                    for x in range(0, w, 3)
                ],
                np.int32,
            )
            cv2.polylines(im, [lab_pts], False, (0, 165, 255), 1)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    # ---- profile chart ----
    def profile_chart(s):
        fig, ax = plt.subplots(figsize=(7, 2.6), dpi=110)
        if s["prof"] is None:
            ax.text(0.5, 0.5, "profile failed", ha="center")
        else:
            p = s["prof"]
            n = len(p)
            ax.plot(np.arange(n), p, "k-", lw=1.4, label="annotated profile")
            h_true = fetch_horizon(s["vp_true"], rg_starts, pf)
            st = int(round(s["s_true"] / BIN_DEG))
            ax.plot(
                np.arange(n),
                np.roll(h_true, -st)[:n],
                "-",
                lw=1.1,
                alpha=0.85,
                label=f"DB horizon @true VP (shift {s['s_true']:+.1f})",
            )
            h_best = fetch_horizon(
                int(
                    np.argmax(
                        np.load(os.path.join(CACHE_DIR, f"{s['sid']}_corr.npz"))["corr"]
                    )
                ),
                rg_starts,
                pf,
            )
            sb, _ = best_corr_shift(s["prof"], h_best, BIN_DEG)
            ax.plot(
                np.arange(n),
                np.roll(h_best, -sb)[:n],
                "r--",
                lw=0.9,
                label="DB horizon @baseline best VP",
            )
            ax.legend(fontsize=6.5, loc="lower left")
            ax.set_xlabel("bin (0.5°)")
            ax.set_ylabel("elev °")
            ft = s["fit"]
            ftxt = f"  RMSE@true={ft['rmse_true']:.1f}° @best={ft['rmse_best']:.1f}°"
            ax.set_title(
                f"corr@trueVP={s['corr_true']:.3f} vs @bestVP={ft['corr_best']:.3f} "
                f"(s_true={s['s_true']:+.1f}° hdg={s['hdg']:+.1f}°){ftxt}"
            )
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ---- comments ----
    def approach_verdicts(s):
        c = []
        if s["e_base"] < 1000:
            c.append("baseline NCC already <1km.")
        elif s["corr_true"] > 0.7:
            c.append(
                f"true-VP shape match is strong (corr {s['corr_true']:.2f}) but baseline best VP is {s['e_base'] / 1000:.1f}km away: the horizon is NOT discriminative at the winning shift."
            )
        else:
            c.append(
                f"true-VP shape match is weak (corr {s['corr_true']:.2f}): mask/extraction error dominates."
            )
        if np.isfinite(s["s_true"]) and np.isfinite(s["hdg"]):
            d = (s["s_true"] - s["hdg"]) % 360
            if d > 180:
                d -= 360
            if abs(d) > 60:
                c.append(
                    f"azimuth prior broken here (heading {s['hdg']:+.1f}° vs true shift {s['s_true']:+.1f}°, off {d:+.0f}°): tight az-window searches are forced to wrong VPs."
                )
            else:
                c.append(
                    f"azimuth prior OK (heading within {abs(d):.0f}° of true shift)."
                )
        i1 = s["idea1"]
        ft = s["fit"]
        if ft["corr_best"] is not None and ft["corr_best"] > s["corr_true"]:
            c.append(
                f"imposter beat the true VP: the wrong horizon matches the profile BETTER "
                f"(corr {ft['corr_best']:.2f} vs {s['corr_true']:.2f}, RMSE {ft['rmse_best']:.1f}° vs {ft['rmse_true']:.1f}°). "
                "True VP carries a vertical-calibration handicap; 1.3M VPs provide better fakes."
            )
        if "tau0.1" in i1 and i1["tau0.1"][0] < s["e_base"] * 0.5:
            c.append(
                "pinball τ=0.1 beats baseline (asymmetric obstacle penalty helped)."
            )
        elif "tau0.1" in i1 and i1["tau0.1"][0] > s["e_base"] * 2:
            c.append("pinball τ=0.1 much worse than baseline (flat/low-horizon bias).")
        if s["lab_rmse"] > 200:
            c.append(
                f"LAB b* skyline far off (RMSE {s['lab_rmse']:.0f}px): thin top-edge sky, gradual b* boundary."
            )
        if s["idea3a_min_e"] is not None and s["idea3a_min_e"] < s["e_base"] * 0.8:
            c.append("trajectory/neighborhood aggregation helped this sample.")
        return c

    # ---- HTML ----
    rows_html = []
    for s in samples:
        i4 = s["idea4"]
        a = s["azp"]

        def cell(v, good_lt=5000):
            if v is None:
                return "<td class='na'>–</td>"
            k = v / 1000.0
            cls = "good" if v < good_lt else ("med" if v < 20000 else "bad")
            return f"<td class='{cls}'>{k:.1f}</td>"

        i1 = s["idea1"]
        htm = (
            f"""
        <div class='sample' id='{s["sid"]}'>
        <h3>{s["sid"][:20]}</h3>
        <div class='row'>
          <div><img class='photo' src='{img_datauri(s["sid"], None)}'/>
            <div class='cap'>photo + annotated skyline (green)</div></div>
          <div><img class='chart' src='{profile_chart(s)}'/></div>
        </div>
        <table class='perapp'>
        <tr><th>approach</th><th>err km</th><th>true-VP rank</th></tr>
        <tr><td>baseline NCC</td>{cell(s["e_base"])}<td class='r'>{s["rank_base"]:,}</td></tr>
        <tr><td>pinball τ=0.05</td>{cell(i1.get("tau0.05", (None,))[0])}<td class='r'>{i1.get("tau0.05", (0, i1.get("tau0.05", (0, 0))[1]))[1]:,}</td></tr>
        <tr><td>pinball τ=0.1</td>{cell(i1.get("tau0.1", (None,))[0])}<td class='r'>{i1.get("tau0.1", (0, 0))[1]:,}</td></tr>
        <tr><td>pinball τ=0.2</td>{cell(i1.get("tau0.2", (None,))[0])}<td class='r'>{i1.get("tau0.2", (0, 0))[1]:,}</td></tr>
        <tr><td>pinball τ=0.5</td>{cell(i1.get("tau0.5", (None,))[0])}<td class='r'>{i1.get("tau0.5", (0, 0))[1]:,}</td></tr>
        <tr><td>az-prior ±20°</td>{cell(a.get("az20", (None,))[0])}<td class='r'>{a.get("az20", (0, 0))[1]:,}</td></tr>
        <tr><td>az-prior ±45°</td>{cell(a.get("az45", (None,))[0])}<td class='r'>{a.get("az45", (0, 0))[1]:,}</td></tr>
        <tr><td>az-prior ±90°</td>{cell(a.get("az90", (None,))[0])}<td class='r'>{a.get("az90", (0, 0))[1]:,}</td></tr>
        <tr><td>az-prior ±60°</td>{cell(a.get("az60", (None,))[0])}<td class='r'>{a.get("az60", (0, 0))[1]:,}</td></tr>
        <tr><td>azimuth ceiling (GT peek)</td>{cell(i4.get("e_ceil"))}<td class='r'>{i4.get("rank_ceil", 0):,}</td></tr>
        <tr><td>shift0 (metadata heading)</td>{cell(i4.get("e_shift0"))}<td class='r'>{i4.get("rank_shift0", 0):,}</td></tr>
        <tr><td>3a neighborhood best</td>{cell(s["idea3a_min_e"])}<td class='na'>–</td></tr>
        <tr><td>LAB b* (RMSE {s["lab_rmse"]:.0f}px)</td><td class='na'>no match run</td><td class='na'>–</td></tr>
        </table>
        <ul class='comments'>
        """
            + "".join(f"<li>{x}</li>" for x in approach_verdicts(s))
            + "</ul></div>"
        )
        rows_html.append(htm)

    # aggregate overview table
    def agg(name, key, src, which=None):
        errs, ranks = [], []
        for s in samples:
            v = src.get(s["sid"], {}) if which is None else src
            e = v.get(key) if isinstance(v, dict) else None
            if e is not None:
                errs.append(e[0])
                ranks.append(e[1])
        return errs, ranks

    def ovrow(name, errs, ranks, comment, extra=""):
        if not errs:
            return f"<tr><td>{name}</td><td class='na' colspan='7'>not run / n/a</td><td>{comment}</td></tr>"
        e = np.array(errs)
        med = np.median(e)
        t1 = sum(x < 500 for x in e)
        lt1 = sum(x < 1000 for x in e)
        lt5 = sum(x < 5000 for x in e)
        lt10 = sum(x < 10000 for x in e)
        mr = int(np.median(ranks)) if ranks else 0
        cls = "good" if med < 5000 else ("med" if med < 15000 else "bad")
        return (
            f"<tr><td>{name}</td><td>{t1}/{len(e)}</td><td class='{cls}'>{med / 1000:.1f}km</td>"
            f"<td>{lt1}</td><td>{lt5}</td><td>{lt10}</td><td>{mr:,}</td><td>{comment}</td></tr>"
        )

    e_i1_t05, r_i1_t05 = agg("pinball", "tau0.05", idea1_log)
    e_i1_t1, r_i1_t1 = agg("pinball", "tau0.1", idea1_log)
    e_az20, r_az20 = agg("azprior", "az20", azp)
    e_az45, r_az45 = agg("azprior", "az45", azp)
    e_ceil, r_ceil = [], []
    for s in samples:
        if "e_ceil" in s["idea4"]:
            e_ceil.append(s["idea4"]["e_ceil"])
            r_ceil.append(s["idea4"]["rank_ceil"])
    e_sh0, r_sh0 = [], []
    for s in samples:
        if "e_shift0" in s["idea4"]:
            e_sh0.append(s["idea4"]["e_shift0"])
            r_sh0.append(s["idea4"]["rank_shift0"])

    base_errs = [s["e_base"] for s in samples]
    base_ranks = [s["rank_base"] for s in samples]
    a3e = [s["idea3a_min_e"] for s in samples if s["idea3a_min_e"] is not None]
    lab_ok = sum(1 for s in samples if s["lab_rmse"] < 30)

    overview = f"""
    <table class='ov'>
    <tr><th>approach</th><th>top1@500m</th><th>median</th><th>&lt;1km</th><th>&lt;5km</th><th>&lt;10km</th><th>true-VP med rank</th><th>why</th></tr>
    {
        ovrow(
            "baseline NCC (free-shift, 2-feat)",
            base_errs,
            base_ranks,
            "reference. Full-shift search lets wrong VPs win at their own shift; 12k+ VPs outrank the true VP.",
        )
    }
    {
        ovrow(
            "pinball τ=0.05 (asym upper-envelope)",
            e_i1_t05,
            r_i1_t05,
            "small median gain, helps 4/6/8/12, flat-horizon bias kills the one correct sample.",
        )
    }
    {
        ovrow(
            "pinball τ=0.1",
            e_i1_t1,
            r_i1_t1,
            "top1 2/17, but destroys rd3ozg (0.1->12.1km).",
        )
    }
    {
        ovrow(
            "azimuth prior ±20° (crop heading)",
            e_az20,
            r_az20,
            "heading prior unreliable (28.5° med err, 8/17 off >55°) -> forces wrong VPs.",
        )
    }
    {ovrow("azimuth prior ±45°", e_az45, r_az45, "same problem, wider window.")}
    {
        ovrow(
            "azimuth ceiling (perfect shift, GT peek)",
            e_ceil,
            r_ceil,
            "BIGGEST lever: rank 558k->7k, median 13.4->9.4km. But still top1 1/17; needs an azimuth oracle that does not exist (no capture time, unreliable metadata).",
        )
    }
    {
        ovrow(
            "shift0 (pure metadata heading)",
            e_sh0,
            r_sh0,
            "GSV heading is wrong by ~100-175° for 15/17 -> useless as-is.",
        )
    }
    {
        ovrow(
            "3a neighborhood aggregation",
            a3e,
            [0] * len(a3e),
            "no change; failure is not VP-mapping noise.",
        )
    }
    {
        f"<tr><td>RANSAC sub-window consensus</td><td class='na' colspan='6'>0/17 sub-windows agree at the true VP</td><td>20° windows latch spurious shifts 50-200° apart; consensus rejects the true VP.</td></tr>"
    }
    {
        f"<tr><td>LAB b* threshold</td><td class='na' colspan='6'>{lab_ok}/17 skylines &lt;30px RMSE (median {np.median([s['lab_rmse'] for s in samples]):.0f}px)</td><td>b* separates sky(114)/terrain(135) but thin top-edge sky + gradual boundary -> skyline not localizable.</td></tr>"
    }
    {
        f"<tr><td>Deep embeddings (CosPlace/MixVPR)</td><td class='na' colspan='6'>not run</td><td>render↔photo domain gap, no labels (17 samples), does not fix azimuth.</td></tr>"
    }
    {
        f"<tr><td>PnP vertex alignment</td><td class='na' colspan='6'>not run</td><td>same (pos,yaw,pitch) coupling the matcher already solves; intractable without a position prior (GPS-free).</td></tr>"
    }
    {
        f"<tr><td>2D Chamfer</td><td class='na' colspan='6'>not run</td><td>2D slack ≈ re-testing tilt (was neutral); 1D flattening is not the loss.</td></tr>"
    }
    {
        f"<tr><td>Solar azimuth + pinball</td><td class='na' colspan='6'>not run</td><td>no capture time in GSV metadata -> solar azimuth uncomputable.</td></tr>"
    }
    {
        f"<tr><td>SeqSLAM sequence matching</td><td class='na' colspan='6'>not run</td><td>per-pano heading errors are independent (~random) -> deltas are noise until azimuth fixed.</td></tr>"
    }
    </table>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>GSV matching approaches — review dashboard</title>
    <style>
    body {{ font-family: sans-serif; margin: 20px; background: #fafafa; }}
    h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
    table {{ border-collapse: collapse; font-size: 12.5px; }}
    th, td {{ border: 1px solid #ccc; padding: 3px 7px; text-align: right; }}
    th {{ background: #e8e8e8; }} td.na {{ color: #999; text-align: center; }}
    td.good {{ background: #d4edda; }} td.med {{ background: #fff3cd; }} td.bad {{ background: #f8d7da; }}
    td.r {{ font-variant-numeric: tabular-nums; }}
    .sample {{ border: 1px solid #bbb; background: white; margin: 16px 0; padding: 10px; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .photo {{ width: 520px; }} .chart {{ width: 520px; }}
    .cap {{ font-size: 11px; color: #666; }}
    .perapp {{ margin-top: 6px; }} .perapp td:first-child {{ text-align: left; }}
    .comments {{ font-size: 12px; margin: 6px 0 0; padding-left: 18px; }}
    .comments li {{ margin: 2px 0; }}
    </style></head><body>
    <h1>GSV matching — all approaches reviewed (17 annotated samples, 0.5° DB)</h1>
    <h2>1. Overview</h2>{overview}
    <h2>2. Per-sample drill-down</h2>
    {"".join(rows_html)}
    </body></html>"""

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

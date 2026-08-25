#!/usr/bin/env python
"""Generate report figures from the canonical Aug-10 evaluation CSVs.

Outputs go to ~/MinorFinalDefense/src/images/figures/.
Re-run after any new evaluation to refresh figures:
    python scripts/report_figures.py
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/MinorFinalDefense/src/images/figures")
os.makedirs(OUT, exist_ok=True)

FINAL = os.path.join(ROOT, "data", "eval", "final")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "figure.dpi": 150,
    }
)

COLORS = {"synthetic": "#1f77b4", "street_view": "#d62728"}


def load(tag):
    dfs = [
        pd.read_csv(p)
        for p in sorted(glob.glob(os.path.join(FINAL, f"{tag}_chunk*.csv")))
    ]
    df = pd.concat(dfs, ignore_index=True)
    return df["error_m"].dropna().to_numpy()


def accuracy_curve(errors, thresholds):
    return np.array([100.0 * np.mean(errors <= t) for t in thresholds])


# ---------------------------------------------------------------- fig 1+2
syn = load("synthetic")
gsv = load("street_view")

thresholds = np.logspace(1, 4.3, 120)  # 10 m .. ~20 km
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.plot(
    thresholds,
    accuracy_curve(syn, thresholds),
    color=COLORS["synthetic"],
    lw=2.2,
    label=f"Synthetic queries (n={len(syn)})",
)
ax.plot(
    thresholds,
    accuracy_curve(gsv, thresholds),
    color=COLORS["street_view"],
    lw=2.2,
    label=f"GSV panoramas (n={len(gsv)})",
)
for t in (200, 500):
    ax.axvline(t, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.annotate(
        f"{t} m", xy=(t, 2), fontsize=9, color="gray", ha="center", xytext=(t * 1.08, 4)
    )
ax.set_xscale("log")
ax.set_xlabel("Distance error threshold $t$ (m, log scale)")
ax.set_ylabel("Top-1 accuracy within $t$ (%)")
ax.set_title("Top-1 localization accuracy vs.\ error threshold")
ax.set_ylim(0, 102)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_accuracy_thresholds.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 4.2))
for errs, label, c in (
    (syn, "Synthetic", COLORS["synthetic"]),
    (gsv, "GSV panoramas", COLORS["street_view"]),
):
    xs = np.sort(errs)
    ax.plot(
        xs, 100.0 * np.arange(1, len(xs) + 1) / len(xs), lw=2.2, color=c, label=label
    )
    med = np.median(xs)
    ax.axvline(med, color=c, ls="--", lw=1.0, alpha=0.7)
    ax.annotate(
        f"median {med:.0f} m",
        xy=(med, 50),
        color=c,
        fontsize=9,
        xytext=(med * 1.25, 46),
        arrowprops=dict(arrowstyle="->", color=c, alpha=0.6),
    )
ax.set_xscale("log")
ax.set_xlabel("Geodesic distance error (m, log scale)")
ax.set_ylabel("Cumulative fraction of answered queries (%)")
ax.set_title("Error distribution over answered queries")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_error_cdf.png"))
plt.close(fig)

# ---------------------------------------------------------------- fig 3
bench_path = os.path.join(ROOT, "results", "segmentation_benchmark.json")
if os.path.exists(bench_path):
    bench = json.load(open(bench_path))["post_processing_algorithms"]
    names, meds, valid = [], [], []
    pretty = {
        "baseline_raw_unet": "Raw U-Net output",
        "canny_direct": "Canny direct",
        "lab_b_threshold": "LAB $b^*$ threshold",
        "grayscale_fixed_window": "Grayscale fixed window",
        "grayscale_fixed_clahe": "Grayscale + CLAHE",
        "lab_b_subpixel_snap": "Canny-guided sub-pixel snap",
        "lab_b_subpixel_clahe": "Sub-pixel snap + CLAHE",
    }
    for k, v in bench.items():
        if isinstance(v, dict) and "median_error_px" in v:
            names.append(pretty.get(k, k))
            meds.append(v["median_error_px"])
            valid.append(v.get("mean_valid_cols", 0))
    order = np.argsort(meds)[::-1]
    names = [names[i] for i in order]
    meds = [meds[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.barh(
        names,
        meds,
        color=[
            "#d62728"
            if "Raw U-Net" in n
            else (
                "#2ca02c"
                if "sub-pixel snap" == n or "snap" in n and "CLAHE" not in n
                else "#1f77b4"
            )
            for n in names
        ],
    )
    ax.bar_label(bars, fmt="%.1f px", padding=3, fontsize=9)
    ax.set_xlabel("Median boundary error per column (px)")
    ax.set_title("Sky-boundary refinement methods on held-out synthetic views")
    ax.set_xlim(0, max(meds) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_seg_ablation.png"))
    plt.close(fig)

# ---------------------------------------------------------------- fig 4
R_EARTH = 6371000.0
k_refr = 0.13
d_km = np.linspace(0, 60, 300)
drop_m = (d_km * 1000) ** 2 * (1 - k_refr) / (2 * R_EARTH)
visibility_km = 50.0
extinction = 3.912 / visibility_km
contrast = 100.0 * np.exp(-extinction * d_km)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
axes[0].plot(d_km, drop_m, color="crimson", lw=2.2)
axes[0].axvline(30, color="gray", ls="--", lw=1)
axes[0].annotate(
    "30 km ray-trace limit",
    xy=(30, drop_m[np.searchsorted(d_km, 30)]),
    xytext=(33, drop_m[-1] * 0.55),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="gray"),
)
axes[0].set_xlabel("Distance (km)")
axes[0].set_ylabel("Hidden height (m)")
axes[0].set_title("Terrain hidden by Earth curvature\n(includes refraction $k=0.13$)")
axes[1].plot(d_km, contrast, color="teal", lw=2.2)
axes[1].axvline(30, color="gray", ls="--", lw=1)
axes[1].set_xlabel("Distance (km)")
axes[1].set_ylabel("Relative contrast (%)")
axes[1].set_title("Atmospheric contrast decay\n(Koschmieder, visibility 50 km)")
for a in axes:
    a.set_xlim(0, 60)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_physics_limits.png"))
plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)

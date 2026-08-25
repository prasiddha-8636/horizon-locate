#!/usr/bin/env python
"""Generate all report figures from saved result JSONs (no DB access needed).

Outputs PNGs + PDFs into final/figures/.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

SCORERS = ["baseline", "bp28", "bp316", "rrf"]
LABELS = {"baseline": "Baseline (val+grad NCC)",
          "bp28": "Bandpass bp(2,8)",
          "bp316": "Bandpass bp(3,16)",
          "rrf": "RRF fusion"}


def load(name):
    return json.loads((RES / name).read_text())["results"]


def cdf(ax, errs, label, color):
    es = np.sort(np.asarray(errs, dtype=float))
    es = es[np.isfinite(es)]
    y = np.arange(1, es.size + 1) / es.size
    ax.step(es / 1000.0, y, where="post", label=label, color=color, lw=2)


def fig_cdf():
    rs = load("gsv_improve_eval_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#888888", "#1f77b4", "#2ca02c", "#d62728"]
    for ax, subset, title in [
        (axes[0], rs, "All multi-crop panoramas (N=68)"),
        (axes[1], [r for r in rs if r["coverage_deg"] >= 200],
         "Wide-FOV subset \u2265200\u00b0 (N=25)"),
    ]:
        for s, c in zip(SCORERS, colors):
            cdf(ax, [r[f"err_{s}"] for r in subset],
                LABELS[s] if ax is axes[0] else None, c)
        orc = [min(r[f"err_{s}"] for s in SCORERS) for r in subset]
        cdf(ax, orc, "Oracle (best scorer)" if ax is axes[0] else None,
            "#9467bd")
        ax.set_xscale("log")
        ax.set_xlim(10, 5e4)
        ax.set_xlabel("Localization error (km)")
        ax.set_ylabel("Fraction of panoramas")
        ax.set_title(title)
        ax.set_xticks([0.01, 0.1, 1, 10, 50])
        ax.set_xticklabels(["10m", "100m", "1km", "10km", "50km"])
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Skyline-only localization error CDF", y=1.02)
    fig.savefig(FIG / "fig1_error_cdf.png")
    fig.savefig(FIG / "fig1_error_cdf.pdf")
    plt.close(fig)


def fig_confidence():
    rs = load("gsv_improve_eval_results.json")
    wide = [r for r in rs if r["coverage_deg"] >= 200]

    gates = [
        ("All panos\n(no gate)", rs),
        ("Consensus\n(votes \u22653)", [r for r in rs if r["rrf_votes"] >= 3]),
        ("Wide-FOV only", wide),
        ("Wide-FOV +\nconsensus",
         [r for r in wide if r["rrf_votes"] >= 3]),
    ]
    labels = [g[0] for g in gates]
    n = np.array([len(g[1]) for g in gates])
    p100 = np.array([np.mean([r["err_rrf"] < 100 for r in g[1]])
                     if g[1] else 0 for g in gates])
    p1k = np.array([np.mean([r["err_rrf"] < 1000 for r in g[1]])
                    if g[1] else 0 for g in gates])

    x = np.arange(len(gates))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    b1 = ax.bar(x - w / 2, p100 * 100, w, label="< 100 m", color="#d62728")
    b2 = ax.bar(x + w / 2, p1k * 100, w, label="< 1 km", color="#1f77b4")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}%",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=10)
    for i, ni in enumerate(n):
        ax.text(x[i], -7.5, f"N={ni}", ha="center", fontsize=9,
                transform=ax.transData)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Precision (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Confidence gating: precision of RRF-fused matches\n"
                 "(consensus = all 3 scorers place top-1 at same location)")
    ax.legend()
    fig.savefig(FIG / "fig2_confidence_gates.png")
    fig.savefig(FIG / "fig2_confidence_gates.pdf")
    plt.close(fig)


def fig_scatter():
    rs = load("gsv_improve_eval_results.json")
    base = np.array([r["err_baseline"] for r in rs], dtype=float)
    rrf = np.array([r["err_rrf"] for r in rs], dtype=float)
    wide = np.array([r["coverage_deg"] >= 200 for r in rs])

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = (10, 5e4)
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")
    ax.scatter(base[~wide] / 1000, rrf[~wide] / 1000, s=28, alpha=0.6,
               color="#1f77b4", label="FOV < 200\u00b0")
    ax.scatter(base[wide] / 1000, rrf[wide] / 1000, s=36, alpha=0.85,
               color="#d62728", marker="^", label="FOV \u2265 200\u00b0")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims[0] / 1000, lims[1] / 1000)
    ax.set_ylim(lims[0] / 1000, lims[1] / 1000)
    ax.set_xlabel("Baseline error (km)")
    ax.set_ylabel("RRF fusion error (km)")
    ax.set_title("Per-panorama: baseline vs fused matching")
    ax.legend()
    fig.savefig(FIG / "fig3_baseline_vs_rrf.png")
    fig.savefig(FIG / "fig3_baseline_vs_rrf.pdf")
    plt.close(fig)


def fig_fov():
    rs = load("gsv_improve_eval_results.json")
    cov = np.array([r["coverage_deg"] for r in rs], dtype=float)
    err = np.array([min(r[f"err_{s}"] for s in SCORERS) for r in rs],
                   dtype=float)
    ok = np.isfinite(err)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(cov[ok], err[ok] / 1000, c=cov[ok], cmap="viridis",
                    s=40, alpha=0.8)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax.text(60, 1.15, "1 km", color="#d62728", fontsize=9)
    ax.axvline(200, color="gray", ls=":", lw=1)
    ax.text(203, 25, "wide-FOV\ngate", fontsize=9, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("Fused horizon coverage (\u00b0)")
    ax.set_ylabel("Best-scorer error (km)")
    ax.set_title("Coverage of the fused horizon vs achievable accuracy")
    fig.colorbar(sc, label="coverage (\u00b0)")
    fig.savefig(FIG / "fig4_fov_vs_error.png")
    fig.savefig(FIG / "fig4_fov_vs_error.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_cdf();      print("fig1 CDF done")
    fig_confidence(); print("fig2 confidence done")
    fig_scatter();  print("fig3 scatter done")
    fig_fov();      print("fig4 FOV done")
    print(f"\nAll figures saved to {FIG}")

#!/usr/bin/env python
"""Generate evaluation metrics and publication-quality plots.

Produces:
  1. Error distribution CDF (single-crop vs fusion vs oracle)
  2. Accuracy-by-FOV bar chart
  3. Confidence gate precision-recall analysis
  4. Bandpass vs baseline comparison
  5. Off-grid error vs noise/pitch heatmap
  6. Per-sample error scatter (true VP rank vs error)
  7. Summary metrics table (LaTeX-ready)

Usage:
    python scripts/generate_metrics_plots.py [--results DIR] [--output DIR]
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Warning] matplotlib not available — will generate text-only summaries")

# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly)
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#4C72B0",
    "fusion": "#DD8452",
    "single_crop": "#55A868",
    "oracle": "#C44E52",
    "bandpass": "#8172B3",
    "confident": "#937860",
    "rejected": "#DA8BC3",
}


def load_results(results_dir):
    """Load all available result JSONs from the results directory."""
    results = {}
    candidates = [
        "multiphoto_eval_results.json",
        "consensus_eval_results.json",
        "offgrid_eval_results.json",
        "confidence_eval_results.json",
    ]
    for fname in candidates:
        fpath = Path(results_dir) / fname
        if fpath.exists():
            with open(fpath) as f:
                results[fname.replace(".json", "")] = json.load(f)
    return results


def plot_error_cdf(errors_dict, output_dir):
    """CDF of geolocation errors for different methods."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for label, errors in errors_dict.items():
        errors = np.array(errors)
        errors = errors[np.isfinite(errors)]
        if len(errors) == 0:
            continue
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        
        color = COLORS.get(label.split("_")[0], "#333333")
        ax.plot(sorted_err / 1000, cdf, label=f"{label} (N={len(errors)})", 
                color=color, linewidth=2)
    
    # Reference lines
    for thresh_km in [0.1, 1, 5, 10]:
        ax.axvline(thresh_km, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="Median")
    
    ax.set_xlabel("Geolocation Error (km)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title("Error Distribution: CDF by Method", fontsize=14)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 40)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / "error_cdf.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "error_cdf.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved error_cdf.png/pdf")


def plot_fov_accuracy(fov_data, output_dir):
    """Bar chart of accuracy by FOV tier."""
    if not HAS_MPL:
        return
    
    tiers = list(fov_data.keys())
    metrics = ["<100m", "<1km", "<5km", "<10km"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), sharey=True)
    
    x = np.arange(len(tiers))
    width = 0.6
    
    for i, metric in enumerate(metrics):
        values = [fov_data[t].get(metric, 0) for t in tiers]
        axes[i].bar(x, values, width, color=COLORS["fusion"], alpha=0.8)
        axes[i].set_title(metric, fontsize=12)
        axes[i].set_ylabel("Accuracy (%)" if i == 0 else "")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([t.split("(")[0].strip() for t in tiers], 
                                 rotation=30, ha="right", fontsize=9)
        axes[i].set_ylim(0, 100)
        axes[i].grid(True, axis="y", alpha=0.3)
        
        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    
    fig.suptitle("Accuracy by Field-of-View Tier", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fov_accuracy.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "fov_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fov_accuracy.png/pdf")


def plot_confidence_analysis(confident_errors, rejected_errors, output_dir):
    """Confidence gate analysis: precision-recall and error separation."""
    if not HAS_MPL:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Error histograms
    conf_err = np.array(confident_errors)
    rej_err = np.array(rejected_errors)
    
    bins = np.logspace(1, 5, 50)
    axes[0].hist(conf_err / 1000, bins=bins, alpha=0.7, color=COLORS["confident"],
                  label=f"Confident (N={len(conf_err)})", density=True)
    axes[0].hist(rej_err / 1000, bins=bins, alpha=0.5, color=COLORS["rejected"],
                  label=f"Rejected (N={len(rej_err)})", density=True)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Error (km)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Error Distribution by Confidence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: Precision at different thresholds
    all_errors = np.concatenate([conf_err, rej_err])
    all_labels = np.concatenate([np.ones(len(conf_err)), np.zeros(len(rej_err))])
    
    thresholds = np.logspace(1, 5, 100)
    precisions = []
    recalls = []
    for t in thresholds:
        predicted_positive = all_errors <= t
        tp = np.sum(predicted_positive & (all_labels == 1))
        fp = np.sum(predicted_positive & (all_labels == 0))
        fn = np.sum(~predicted_positive & (all_labels == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
    
    axes[1].plot(recalls, precisions, color=COLORS["baseline"], linewidth=2)
    axes[1].set_xlabel("Recall (fraction of true matches found)")
    axes[1].set_ylabel("Precision (fraction of confident matches that are correct)")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    # Add AUC
    auc = np.trapz(precisions, recalls)
    axes[1].text(0.6, 0.2, f"AUC = {auc:.3f}", fontsize=12, 
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_analysis.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "confidence_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confidence_analysis.png/pdf")


def plot_offgrid_heatmap(results, output_dir):
    """Heatmap of error vs noise level and pitch offset."""
    if not HAS_MPL or not results:
        return
    
    noises = sorted(set(r["noise_sigma"] for r in results))
    pitches = sorted(set(r["pitch_offset"] for r in results))
    
    median_grid = np.zeros((len(noises), len(pitches)))
    hit_grid = np.zeros((len(noises), len(pitches)))
    
    for i, n in enumerate(noises):
        for j, p in enumerate(pitches):
            subset = [r["top1_err_m"] for r in results 
                      if r["noise_sigma"] == n and r["pitch_offset"] == p]
            if subset:
                median_grid[i, j] = np.median(subset)
                hit_grid[i, j] = np.mean(np.array(subset) < 1000) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(median_grid / 1000, aspect="auto", cmap="RdYlGn_r",
                           vmin=0, vmax=30)
    axes[0].set_xticks(range(len(pitches)))
    axes[0].set_xticklabels([f"{p}°" for p in pitches])
    axes[0].set_yticks(range(len(noises)))
    axes[0].set_yticklabels([f"{n}°" for n in noises])
    axes[0].set_xlabel("Pitch Offset")
    axes[0].set_ylabel("Noise σ")
    axes[0].set_title("Median Error (km)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(hit_grid, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=100)
    axes[1].set_xticks(range(len(pitches)))
    axes[1].set_xticklabels([f"{p}°" for p in pitches])
    axes[1].set_yticks(range(len(noises)))
    axes[1].set_yticklabels([f"{n}°" for n in noises])
    axes[1].set_xlabel("Pitch Offset")
    axes[1].set_ylabel("Noise σ")
    axes[1].set_title("<1km Accuracy (%)")
    plt.colorbar(im1, ax=axes[1])
    
    fig.suptitle("Off-Grid Matching: Error vs Noise & Pitch", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "offgrid_heatmap.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "offgrid_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved offgrid_heatmap.png/pdf")


def plot_rank_vs_error(ranks, errors, output_dir):
    """Scatter: true VP rank vs geolocation error."""
    if not HAS_MPL:
        return
    
    ranks = np.array(ranks, dtype=float)
    errors = np.array(errors, dtype=float)
    valid = np.isfinite(ranks) & np.isfinite(errors) & (ranks >= 0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(ranks[valid], errors[valid] / 1000, alpha=0.6, s=30,
               color=COLORS["baseline"], edgecolors="white", linewidth=0.5)
    
    # Reference lines
    ax.axhline(1, color="green", linestyle="--", alpha=0.5, label="1km threshold")
    ax.axhline(5, color="orange", linestyle="--", alpha=0.5, label="5km threshold")
    ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="10km threshold")
    
    ax.set_xlabel("True VP Rank (out of 1.34M)", fontsize=12)
    ax.set_ylabel("Geolocation Error (km)", fontsize=12)
    ax.set_title("True VP Rank vs Geolocation Error", fontsize=14)
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / "rank_vs_error.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "rank_vs_error.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved rank_vs_error.png/pdf")


def generate_latex_table(summary_data):
    """Generate LaTeX-ready summary table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Skyline Geolocation Evaluation Summary}",
        r"\label{tab:eval_summary}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & $N$ & Median & $<1$km & $<5$km & $<10$km \\",
        r"\midrule",
    ]
    
    for method, data in summary_data.items():
        n = data.get("N", 0)
        med = data.get("median", float("nan"))
        lt1km = data.get("lt_1km", 0)
        lt5km = data.get("lt_5km", 0)
        lt10km = data.get("lt_10km", 0)
        lines.append(
            f"{method} & {n} & {med/1000:.1f}km & {lt1km:.1f}\\% & {lt5km:.1f}\\% & {lt10km:.1f}\\% \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def print_text_summary(results_dir):
    """Print a text-only summary of all available results."""
    print("\n" + "=" * 72)
    print("METRICS SUMMARY")
    print("=" * 72)
    
    for fname in sorted(Path(results_dir).glob("*.json")):
        try:
            with open(fname) as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            continue
        
        if isinstance(data, list):
            errors = [r.get("top1_err_m", r.get("error_m", float("inf"))) for r in data 
                      if isinstance(r, dict)]
        elif isinstance(data, dict):
            errors = [r.get("top1_err_m", r.get("error_m", float("inf"))) 
                      for r in data.get("results", []) if isinstance(r, dict)]
        else:
            continue
        
        if not errors:
            continue
        
        errors = np.array(errors)
        errors = errors[np.isfinite(errors)]
        if len(errors) == 0:
            continue
        
        print(f"\n{fname.name}:")
        print(f"  N={len(errors)}  Median={np.median(errors):.0f}m  Mean={np.mean(errors):.0f}m")
        print(f"  <100m: {np.mean(errors < 100):.1%}  <1km: {np.mean(errors < 1000):.1%}")
        print(f"  <5km: {np.mean(errors < 5000):.1%}  <10km: {np.mean(errors < 10000):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Generate metrics and plots")
    parser.add_argument("--results", type=str, 
                        default=str(Path(__file__).resolve().parent.parent / "data" / "street_view"),
                        help="Directory containing result JSONs")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data" / "eval" / "figures"),
                        help="Output directory for plots")
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")
    
    # Load all results
    all_results = load_results(results_dir)
    print(f"Found {len(all_results)} result files")
    
    # Text summary
    print_text_summary(results_dir)
    
    if not HAS_MPL:
        print("\n[Skipping plots — matplotlib not available]")
        return
    
    # Generate plots from available data
    print("\nGenerating plots...")
    
    # 1. Error CDF
    # Try to extract errors from multiphoto results
    cdf_data = {}
    if "multiphoto_eval_results" in all_results:
        data = all_results["multiphoto_eval_results"]
        if isinstance(data, dict) and "results" in data:
            for method in ["fusion", "single_crop", "product_ncc", "oracle"]:
                errs = [r.get(f"{method}_error", r.get("error_m", float("inf")))
                        for r in data["results"] if isinstance(r, dict)]
                errs = [e for e in errs if e < 1e6]
                if errs:
                    cdf_data[method] = errs
    
    if cdf_data:
        plot_error_cdf(cdf_data, output_dir)
    
    # 2. Confidence analysis
    if "confidence_eval_results" in all_results:
        data = all_results["confidence_eval_results"]
        if isinstance(data, list):
            conf = [r["top1_err_m"] for r in data if r.get("status") == "confident" and r.get("top1_err_m", float("inf")) < 1e6]
            rej = [r["top1_err_m"] for r in data if r.get("status") == "rejected" and r.get("top1_err_m", float("inf")) < 1e6]
            if conf and rej:
                plot_confidence_analysis(conf, rej, output_dir)
    
    # 3. Off-grid heatmap
    if "offgrid_eval_results" in all_results:
        data = all_results["offgrid_eval_results"]
        if isinstance(data, list):
            plot_offgrid_heatmap(data, output_dir)
    
    # 4. Rank vs error
    if "multiphoto_eval_results" in all_results:
        data = all_results["multiphoto_eval_results"]
        if isinstance(data, dict) and "results" in data:
            ranks = [r.get("true_vp_rank", -1) for r in data["results"] if isinstance(r, dict)]
            errors = [r.get("fusion_error", r.get("error_m", float("inf"))) 
                      for r in data["results"] if isinstance(r, dict)]
            valid = [(r, e) for r, e in zip(ranks, errors) if r >= 0 and e < 1e6]
            if valid:
                plot_rank_vs_error([v[0] for v in valid], [v[1] for v in valid], output_dir)
    
    # Generate LaTeX table
    summary = {}
    for method, data in cdf_data.items():
        errors = np.array(data)
        summary[method] = {
            "N": len(errors),
            "median": float(np.median(errors)),
            "lt_1km": float(np.mean(errors < 1000) * 100),
            "lt_5km": float(np.mean(errors < 5000) * 100),
            "lt_10km": float(np.mean(errors < 10000) * 100),
        }
    
    if summary:
        latex = generate_latex_table(summary)
        latex_path = output_dir / "summary_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\n  Saved summary_table.tex")
        print("\nLaTeX table:")
        print(latex)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run evaluations in manageable chunks, saving results incrementally."""

import json, os, sys, time, warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation import load_ground_truth, run_evaluation

OUT = os.path.join(ROOT, "data", "eval", "final")
os.makedirs(OUT, exist_ok=True)


def eval_chunk(gt_data, masks_dir, tag, batch_ids, chunk_idx):
    """Evaluate a batch of sample IDs, save incrementally."""
    subset_gt = {sid: gt_data[sid] for sid in batch_ids}
    gt_path = f"/tmp/eval_{tag}_chunk{chunk_idx}.json"
    with open(gt_path, "w") as f:
        json.dump(subset_gt, f)

    t0 = time.time()
    df, summary = run_evaluation(
        ground_truth_path=gt_path,
        db_path=os.path.join(
            ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet"
        ),
        masks_dir=masks_dir,
        use_altimeter=True,
        use_compass=True,
        weights=(0.5, 0.5),
        min_std_deg=1.5,
        min_max_elev_deg=1.0,
        spatial_stride=12,
        sample_batch_size=5,
        chunk_rows=4000,
    )
    elapsed = time.time() - t0
    print(
        f"  Chunk {chunk_idx}: {len(batch_ids)} samples, {elapsed:.0f}s ({elapsed / len(batch_ids):.1f}s/sample)"
    )
    df.to_csv(os.path.join(OUT, f"{tag}_chunk{chunk_idx}.csv"), index=False)
    return df, summary


def merge_csvs(tag, n_chunks):
    """Merge chunk CSVs into final."""
    dfs = []
    for i in range(n_chunks):
        p = os.path.join(OUT, f"{tag}_chunk{i}.csv")
        if os.path.exists(p):
            dfs.append(pandas.read_csv(p))
    if dfs:
        merged = pandas.concat(dfs, ignore_index=True)
        merged.to_csv(os.path.join(OUT, f"{tag}_baseline.csv"), index=False)
        return merged
    return None


def compute_summary(df, n_gt):
    """Compute summary from merged results."""
    answered = df.dropna(subset=["error_m"])
    n_ok = len(answered)
    n_nosky = len(df[df["status"] == "NO_SKYLINE"]) if "status" in df.columns else 0
    n_nomatch = len(df[df["status"] == "NO_MATCH"]) if "status" in df.columns else 0

    metrics = {
        "total_gt": n_gt,
        "n_answered": n_ok,
        "n_no_skyline": n_nosky,
        "n_no_match": n_nomatch,
        "coverage_pct": 100.0 * n_ok / max(n_gt, 1),
    }
    if n_ok > 0:
        errs = answered["error_m"].values
        metrics["median_error_m"] = float(np.median(errs))
        metrics["mean_error_m"] = float(np.mean(errs))
        metrics["p90_error_m"] = float(np.percentile(errs, 90))
        for t in [50, 100, 200, 500, 1000, 2000]:
            metrics[f"top1_acc_{t}m"] = float(100.0 * np.mean(errs <= t))
        if "top5_ok" in df.columns:
            metrics["top5_acc_500m"] = float(100.0 * df["top5_ok"].mean())
    return metrics


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # === SYNTHETIC EVAL ===
    print("=" * 50)
    print("SYNTHETIC DATASET EVALUATION")
    print("=" * 50)
    with open(os.path.join(ROOT, "data/synthetic_dataset/ground_truth.json")) as f:
        syn_gt = json.load(f)
    syn_ids = list(syn_gt.keys())
    print(f"Total synthetic samples: {len(syn_ids)}")

    t_start = time.time()
    all_syn_dfs = []
    for i in range(0, len(syn_ids), 10):
        chunk = syn_ids[i : i + 10]
        df_c, _ = eval_chunk(
            syn_gt,
            os.path.join(ROOT, "data/synthetic_dataset/predicted_masks"),
            "synthetic",
            chunk,
            i // 10,
        )
        all_syn_dfs.append(df_c)

    df_syn = pd.concat(all_syn_dfs, ignore_index=True)
    df_syn.to_csv(os.path.join(OUT, "synthetic_baseline.csv"), index=False)
    sum_syn = compute_summary(df_syn, len(syn_ids))
    sum_syn["eval_time_s"] = time.time() - t_start
    with open(os.path.join(OUT, "synthetic_baseline_summary.json"), "w") as f:
        json.dump(sum_syn, f, indent=2, default=str)
    print(
        f"Synthetic DONE: median={sum_syn.get('median_error_m', 0):.0f}m, "
        f"top1@500m={sum_syn.get('top1_acc_500m', 0):.1f}%, "
        f"time={sum_syn['eval_time_s']:.0f}s"
    )

    # === STREET VIEW SUBSET (250 stratified) ===
    print("\n" + "=" * 50)
    print("STREET VIEW SUBSET EVALUATION (250 samples)")
    print("=" * 50)
    with open(os.path.join(ROOT, "data/street_view/ground_truth.json")) as f:
        sv_gt = json.load(f)
    sv_ids = list(sv_gt.keys())

    # Stratified subset
    rng = np.random.default_rng(42)
    lats = np.array([sv_gt[s]["true_lat"] for s in sv_ids])
    lons = np.array([sv_gt[s]["true_lon"] for s in sv_ids])
    lat_bins = np.linspace(lats.min(), lats.max(), 11)
    lon_bins = np.linspace(lons.min(), lons.max(), 11)
    selected = []
    per_cell = max(1, 250 // 100)
    for i in range(10):
        for j in range(10):
            mask = (
                (lats >= lat_bins[i])
                & (lats < lat_bins[i + 1])
                & (lons >= lon_bins[j])
                & (lons < lon_bins[j + 1])
            )
            cells = np.where(mask)[0]
            if len(cells) > 0:
                take = rng.choice(cells, size=min(per_cell, len(cells)), replace=False)
                selected.extend([sv_ids[k] for k in take])
    remaining = [s for s in sv_ids if s not in set(selected)]
    need = 250 - len(selected)
    if need > 0 and remaining:
        extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
        selected.extend(extra)
    subset250 = selected[:250]
    print(f"GSV subset: {len(subset250)} samples")

    t_start = time.time()
    all_sv_dfs = []
    for i in range(0, len(subset250), 10):
        chunk = subset250[i : i + 10]
        df_c, _ = eval_chunk(
            sv_gt,
            os.path.join(ROOT, "data/street_view/masks"),
            "street_view",
            chunk,
            i // 10,
        )
        all_sv_dfs.append(df_c)

    df_sv = pd.concat(all_sv_dfs, ignore_index=True)
    df_sv.to_csv(os.path.join(OUT, "street_view_baseline.csv"), index=False)
    sum_sv = compute_summary(df_sv, len(subset250))
    sum_sv["eval_time_s"] = time.time() - t_start
    with open(os.path.join(OUT, "street_view_baseline_summary.json"), "w") as f:
        json.dump(sum_sv, f, indent=2, default=str)
    print(
        f"GSV DONE: median={sum_sv.get('median_error_m', 0):.0f}m, "
        f"top1@500m={sum_sv.get('top1_acc_500m', 0):.1f}%, "
        f"time={sum_sv['eval_time_s']:.0f}s"
    )

    # === FINAL REPORT ===
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic": sum_syn,
        "street_view_250": sum_sv,
    }
    with open(os.path.join(OUT, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFinal report: {os.path.join(OUT, 'final_report.json')}")

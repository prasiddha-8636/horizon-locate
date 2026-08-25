#!/usr/bin/env python
"""Resume evaluation from last completed chunk. Skips already-done chunks."""

import json, os, sys, time, warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from src.evaluation import load_ground_truth, run_evaluation

OUT = os.path.join(ROOT, "data", "eval", "final")


def done_chunks(tag):
    """Return set of completed chunk indices."""
    done = set()
    for f in os.listdir(OUT):
        if f.startswith(f"{tag}_chunk") and f.endswith(".csv"):
            p = os.path.join(OUT, f)
            if os.path.getsize(p) > 5:
                idx = int(f.replace(f"{tag}_chunk", "").replace(".csv", ""))
                done.add(idx)
    return done


def eval_chunk(gt_data, masks_dir, tag, batch_ids, chunk_idx):
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
    df.to_csv(os.path.join(OUT, f"{tag}_chunk{chunk_idx}.csv"), index=False)
    print(f"  Chunk {chunk_idx}: {len(batch_ids)} samples in {elapsed:.0f}s")
    return df


def merge_and_summarize(tag, n_gt):
    dfs = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith(f"{tag}_chunk") and f.endswith(".csv"):
            p = os.path.join(OUT, f)
            if os.path.getsize(p) > 5:
                dfs.append(pd.read_csv(p))
    if not dfs:
        return {}
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(os.path.join(OUT, f"{tag}_baseline.csv"), index=False)

    answered = merged.dropna(subset=["error_m"])
    n_ok = len(answered)
    m = {
        "total_gt": n_gt,
        "n_answered": n_ok,
        "coverage_pct": 100.0 * n_ok / max(n_gt, 1),
    }
    if n_ok > 0:
        errs = answered["error_m"].values
        m["median_error_m"] = float(np.median(errs))
        m["mean_error_m"] = float(np.mean(errs))
        m["p90_error_m"] = float(np.percentile(errs, 90))
        for t in [50, 100, 200, 500, 1000, 2000]:
            m[f"top1_acc_{t}m"] = float(100.0 * np.mean(errs <= t))
        if "top5_ok" in merged.columns:
            m["top5_acc_500m"] = float(100.0 * merged["top5_ok"].mean())

    with open(os.path.join(OUT, f"{tag}_baseline_summary.json"), "w") as f:
        json.dump(m, f, indent=2, default=str)
    return m


if __name__ == "__main__":
    CHUNK = 10

    # === SYNTHETIC ===
    print("SYNTHETIC — resuming")
    with open(os.path.join(ROOT, "data/synthetic_dataset/ground_truth.json")) as f:
        syn_gt = json.load(f)
    syn_ids = list(syn_gt.keys())
    done_syn = done_chunks("synthetic")
    todo_syn = [i for i in range(len(syn_ids)) if i // CHUNK not in done_syn]
    # group by chunk
    remaining_syn = {}
    for idx in todo_syn:
        ci = idx // CHUNK
        remaining_syn.setdefault(ci, []).append(syn_ids[idx])

    print(
        f"  Already done: {len(done_syn)} chunks, remaining: {len(remaining_syn)} chunks"
    )
    for ci in sorted(remaining_syn):
        eval_chunk(
            syn_gt,
            os.path.join(ROOT, "data/synthetic_dataset/predicted_masks"),
            "synthetic",
            remaining_syn[ci],
            ci,
        )

    sum_syn = merge_and_summarize("synthetic", len(syn_ids))
    print(
        f"  SYNTHETIC: median={sum_syn.get('median_error_m', 0):.0f}m, "
        f"top1@500m={sum_syn.get('top1_acc_500m', 0):.1f}%"
    )

    # === STREET VIEW ===
    print("\nSTREET VIEW — resuming")
    with open(os.path.join(ROOT, "data/street_view/ground_truth.json")) as f:
        sv_gt = json.load(f)
    sv_ids = list(sv_gt.keys())

    rng = np.random.default_rng(42)
    lats = np.array([sv_gt[s]["true_lat"] for s in sv_ids])
    lons = np.array([sv_gt[s]["true_lon"] for s in sv_ids])
    lat_bins = np.linspace(lats.min(), lats.max(), 11)
    lon_bins = np.linspace(lons.min(), lons.max(), 11)
    selected = []
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
                take = rng.choice(
                    cells, size=min(max(1, 250 // 100), len(cells)), replace=False
                )
                selected.extend([sv_ids[k] for k in take])
    remaining_list = [s for s in sv_ids if s not in set(selected)]
    need = 250 - len(selected)
    if need > 0 and remaining_list:
        extra = rng.choice(
            remaining_list, size=min(need, len(remaining_list)), replace=False
        )
        selected.extend(extra)
    subset250 = selected[:250]

    done_sv = done_chunks("street_view")
    remaining_sv = {}
    for idx in range(len(subset250)):
        ci = idx // CHUNK
        if ci not in done_sv:
            remaining_sv.setdefault(ci, []).append(subset250[idx])

    print(
        f"  Already done: {len(done_sv)} chunks, remaining: {len(remaining_sv)} chunks"
    )
    for ci in sorted(remaining_sv):
        eval_chunk(
            sv_gt,
            os.path.join(ROOT, "data/street_view/masks"),
            "street_view",
            remaining_sv[ci],
            ci,
        )

    sum_sv = merge_and_summarize("street_view", len(subset250))
    print(
        f"  GSV: median={sum_sv.get('median_error_m', 0):.0f}m, "
        f"top1@500m={sum_sv.get('top1_acc_500m', 0):.1f}%"
    )

    # === FINAL REPORT ===
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic": sum_syn,
        "street_view_250": sum_sv,
    }
    with open(os.path.join(OUT, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDone! Report: {os.path.join(OUT, 'final_report.json')}")

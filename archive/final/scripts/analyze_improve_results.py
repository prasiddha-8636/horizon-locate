#!/usr/bin/env python
"""Post-hoc analysis of gsv_improve_eval_results.json (no DB rescan).

Produces the report-ready tables:
  * improvement summary per scorer
  * confidence gate via rrf_votes (how many scorers' top-50 contain the
    RRF-fused winner): precision/recall per vote level
"""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "street_view" / "gsv_improve_eval_results.json"


def stats(es):
    es = np.asarray(es, dtype=float)
    es = es[np.isfinite(es)]
    if es.size == 0:
        return "N=0"
    return (f"N={len(es):<3d} median={np.median(es)/1000:6.1f}km  "
            f"<100m={np.mean(es < 100):5.1%}  <500m={np.mean(es < 500):5.1%}  "
            f"<1km={np.mean(es < 1000):5.1%}  <5km={np.mean(es < 5000):5.1%}  "
            f"<10km={np.mean(es < 10000):5.1%}")


def main():
    data = json.loads(JSON_PATH.read_text())
    rs = data["results"]
    stride = data.get("stride", "?")
    print("=" * 78)
    print(f"GSV IMPROVEMENT EVAL — POST-HOC ANALYSIS (stride={stride}, "
          f"{len(rs)} panos)")
    print("=" * 78)

    scorers = ["baseline", "bp28", "bp316", "rrf"]
    print("\n--- Improvement summary ---")
    for s in scorers + ["oracle"]:
        if s == "oracle":
            es = [min(r[f"err_{x}"] for x in scorers) for r in rs]
        else:
            es = [r[f"err_{s}"] for r in rs]
        print(f"{s.upper():<10} {stats(es)}")

    wide = [r for r in rs if r["coverage_deg"] >= 200]
    print(f"\n--- Wide-FOV subset (>=200 deg, N={len(wide)}) ---")
    for s in scorers + ["oracle"]:
        if s == "oracle":
            es = [min(r[f"err_{x}"] for x in scorers) for r in wide]
        else:
            es = [r[f"err_{s}"] for r in wide]
        print(f"{s.upper():<10} {stats(es)}")

    # ---- confidence gates ---------------------------------------------
    print("\n" + "=" * 78)
    print("CONFIDENCE GATES (computed post-hoc, no rescan)")
    print("=" * 78)

    print("\nGate A: rrf_votes == V (scorers whose top-50 contain the "
          "RRF winner)")
    print(f"{'gate':>12} {stats([])}")
    for v in (3, 2):
        acc = [r["err_rrf"] for r in rs if r["rrf_votes"] >= v]
        rej = [r["err_rrf"] for r in rs if r["rrf_votes"] < v]
        print(f"accept>={v}   {stats(acc)}")
        print(f"reject <{v}   {stats(rej)}")

    print("\nGate B: rrf_votes == V AND wide-FOV (>=200 deg)")
    for v in (3,):
        acc = [r["err_rrf"] for r in wide if r["rrf_votes"] >= v]
        print(f"accept>={v}   {stats(acc)}")
        rej = [r["err_rrf"] for r in wide if r["rrf_votes"] < v]
        print(f"reject <{v}   {stats(rej)}")

    # Gate C: best-scorer error proxy — confident if ANY scorer's top-1 is
    # close to another scorer's top-1 region cannot be checked without
    # coordinates; use rank-based proxy instead: true VP ranked in top-50
    # by >=2 scorers.
    print("\nGate C (diagnostic): true VP in top-50 of how many scorers?")
    from collections import Counter
    cnt = Counter()
    for r in rs:
        n_in = sum(r.get(f"rank_{s}", -1) not in (-1, None)
                   and r[f"rank_{s}"] < 50 for s in scorers[:3])
        cnt[n_in] += 1
    for k in sorted(cnt):
        sub = [r["err_rrf"] for r in rs
               if sum(r.get(f"rank_{s}", -1) not in (-1, None)
                      and r[f"rank_{s}"] < 50
                      for s in scorers[:3]) == k]
        print(f"  {k} scorers have true-in-top50: N={cnt[k]:<3d} {stats(sub)}")

    print("\n--- Per-sample detail (accepted by Gate A votes>=3) ---")
    for r in sorted(rs, key=lambda x: x["err_rrf"]):
        if r["rrf_votes"] >= 3:
            hit = "HIT " if r["err_rrf"] < 1000 else "miss"
            print(f"  [{hit}] {r['pano_id'][:20]:20s} FOV={r['coverage_deg']:4.0f}deg  "
                  f"base={r['err_baseline']/1000:6.1f}km  "
                  f"bp28={r['err_bp28']/1000:6.1f}km  "
                  f"bp316={r['err_bp316']/1000:6.1f}km  "
                  f"rrf={r['err_rrf']:8.0f}m")


if __name__ == "__main__":
    main()

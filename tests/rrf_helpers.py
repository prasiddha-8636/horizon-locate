"""Lightweight helpers extracted from archive/scripts/gsv_improve_eval.py
for use by tests/test_core.py.  Kept minimal — only ScorerState and rrf_top1.
"""
import heapq
import numpy as np

TOP_KEEP = 50
BASE_SCORERS = ("baseline", "bp28", "bp316")


class ScorerState:
    """Best hit + bounded top-K heap for one pano under one scorer."""

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


def rrf_top1(sts, k=60):
    """Reciprocal-rank fusion over the three top-K heaps."""
    scores = {}
    for name in BASE_SCORERS:
        ranked = sorted(sts[name].heap, key=lambda x: -x[0])
        for rank, (_, row, _, _) in enumerate(ranked):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank)
    latlon = {}
    for name in BASE_SCORERS:
        for _, row, lat, lon in sts[name].heap:
            if row not in latlon:
                latlon[row] = (lat, lon)
    if not scores:
        return None, float("inf"), {}
    best_row = max(scores, key=scores.get)
    votes = sum(1 for name in BASE_SCORERS
                if any(e[1] == best_row for e in sts[name].heap))
    lat, lon = latlon.get(best_row, (None, None))
    return (lat, lon), votes, scores

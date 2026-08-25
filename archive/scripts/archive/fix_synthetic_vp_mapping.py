#!/usr/bin/env python
"""Recompute closest_viewpoint_id / closest_viewpoint_dist_m for the synthetic GT.

The stored VP mapping in data/synthetic_dataset/ground_truth.json is broken
(e.g. sample 0 points at a VP ~15 km away with dist 0.0 m, while the real
nearest VP is ~50 m away).  This script finds the true nearest DB viewpoint
for every synthetic sample by geodesic distance to its true_lat/true_lon and
rewrites the two fields in place.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation import load_db_metadata

GT_PATH = ROOT / "data" / "synthetic_dataset" / "ground_truth.json"
DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"


def latlon_to_xyz(lat, lon):
    """Unit-sphere XYZ for KDTree nearest-neighbour search (angular proximity)."""
    la = np.radians(lat)
    lo = np.radians(lon)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(a))


def main():
    with open(GT_PATH) as f:
        gt = json.load(f)

    lon, lat, elev_m, n_vp = load_db_metadata(DB_PATH)
    db_xyz = latlon_to_xyz(lat, lon)
    tree = cKDTree(db_xyz)

    print(f"DB viewpoints: {n_vp}")
    print(f"Synthetic samples: {len(gt)}")

    changes = 0
    for sid, g in gt.items():
        tgt = latlon_to_xyz(np.array([g["true_lat"]]), np.array([g["true_lon"]]))
        d, i = tree.query(tgt, k=1)
        vp = int(i[0])
        dist_m = float(haversine_m(g["true_lat"], g["true_lon"], lat[vp], lon[vp]))
        old_vp = g.get("closest_viewpoint_id")
        old_dist = g.get("closest_viewpoint_dist_m")
        if old_vp != vp or old_dist != dist_m:
            changes += 1
        g["closest_viewpoint_id"] = vp
        g["closest_viewpoint_dist_m"] = round(dist_m, 1)

    with open(GT_PATH, "w") as f:
        json.dump(gt, f, indent=4)

    print(f"Updated VP mapping for {changes}/{len(gt)} samples")
    print(
        f"Sample 0 -> vp={gt['0']['closest_viewpoint_id']} "
        f"dist={gt['0']['closest_viewpoint_dist_m']}m"
    )


if __name__ == "__main__":
    main()

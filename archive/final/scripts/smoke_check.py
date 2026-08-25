#!/usr/bin/env python
"""Lightweight import + core logic validation. No network, GPU, or large data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0


def check(description, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS: {description}")
    else:
        FAIL += 1
        print(f"  FAIL: {description}")


# 1. Source imports
print("=== Module imports ===")
try:
    import src.region

    check("src.region", True)
except Exception as e:
    check(f"src.region ({e})", False)

try:
    import src.matching

    check("src.matching", True)
except Exception as e:
    check(f"src.matching ({e})", False)

try:
    import src.query_profile

    check("src.query_profile", True)
except Exception as e:
    check(f"src.query_profile ({e})", False)

try:
    import src.segmentation

    check("src.segmentation", True)
except Exception as e:
    check(f"src.segmentation ({e})", False)

try:
    import src.evaluation

    check("src.evaluation", True)
except Exception as e:
    check(f"src.evaluation ({e})", False)

try:
    import src.config

    check("src.config", True)
except Exception as e:
    check(f"src.config ({e})", False)

# 2. Core logic
print("\n=== Core logic ===")
import numpy as np
from src.matching import _safe_zscore, match_query
from src.query_profile import is_profile_applicable
from src.segmentation import _compute_sky_diagnostics

# Safe zscore
z = _safe_zscore(np.ones(50) * 5.0)
check("safe_zscore flat returns zeros", z.std() < 1e-12)

z2 = _safe_zscore(np.sin(np.linspace(0, 10, 100)) * 5)
check("safe_zscore varied unit std", abs(z2.std() - 1.0) < 0.01)

# is_profile_applicable
check("empty profile rejected", not is_profile_applicable(np.array([]))[0])
check("flat profile rejected", not is_profile_applicable(np.zeros(100))[0])
check("nan profile rejected", not is_profile_applicable(np.full(100, np.nan))[0])
check(
    "good profile accepted",
    is_profile_applicable(np.sin(np.linspace(0, 10, 100)) * 5)[0],
)

# Match query
result = match_query(None, 0.25, np.zeros(100))
check("match empty DB returns INVALID_INPUT", result["status"] == "INVALID_INPUT")

result = match_query(np.zeros((10, 360)), 0.25, np.ones(5))
check("match short query returns INVALID_QUERY", result["status"] == "INVALID_QUERY")

db = np.random.randn(20, 360)
q = np.sin(np.linspace(0, 10, 360)) * 5
result = match_query(db, 1.0, q, spatial_stride=2)
check("match tiny DB runs without error", "matches" in result)

# Diagnostics
mask = np.full((100, 100), 255, dtype=np.uint8)
mask[:40, :] = 0
d = _compute_sky_diagnostics(mask)
check("sky diagnostics ratio correct", abs(d["sky_ratio"] - 0.4) < 0.01)
check("sky diagnostics boundary coverage", d["boundary_coverage"] == 1.0)

print(f"\n{'=' * 40}")
print(f"Results: {PASS} passed, {FAIL} failed")
sys.exit(0 if FAIL == 0 else 1)

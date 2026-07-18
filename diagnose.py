"""
Run this cell-by-cell (or as a script). It prints memory + progress after EVERY
step and flushes immediately. Whatever line printed last before a crash tells us
exactly which operation killed the kernel -- no guessing needed after this.
"""
import sys
import gc
import psutil
import pandas as pd
import numpy as np
from pathlib import Path

proc = psutil.Process()

def mem_mb():
    return proc.memory_info().rss / 1e6

def checkpoint(label):
    print(f"[{label}] RSS = {mem_mb():.1f} MB", flush=True)

checkpoint("start")

# ---- EDIT THESE PATHS ----
near_path = Path("notebooks/02_SkylineDatabase/output/fine_skyline_db.parquet")
far_path = Path("notebooks/02_SkylineDatabase/output/far_100km_db.parquet")  # start with the SMALLEST radius

# 1. File sizes on disk (cheap, no load yet)
for p in [near_path, far_path]:
    size_mb = p.stat().st_size / 1e6 if p.exists() else -1
    print(f"{p.name}: {size_mb:.1f} MB on disk (exists={p.exists()})", flush=True)
checkpoint("after_stat")

# 2. Load near tier's parquet (raw dataframe, before array conversion)
df_near = pd.read_parquet(near_path)
checkpoint("after_read_parquet_near")
print(f"  near rows={len(df_near)}, has raw_horizon_deg={'raw_horizon_deg' in df_near.columns}", flush=True)
if "raw_horizon_deg" in df_near.columns:
    print(f"  near raw_horizon_deg[0] length={len(df_near['raw_horizon_deg'].iloc[0])}", flush=True)

# 3. Convert near tier to numpy matrix (this is the likely spike point)
near_matrix = np.stack(df_near["raw_horizon_deg"].to_numpy())
checkpoint("after_stack_near")
print(f"  near_matrix shape={near_matrix.shape}, dtype={near_matrix.dtype}, "
      f"size={near_matrix.nbytes / 1e6:.1f} MB", flush=True)
del df_near
gc.collect()
checkpoint("after_del_df_near")

# 4. Same for far tier -- THIS is most likely where it dies given large radius
df_far = pd.read_parquet(far_path)
checkpoint("after_read_parquet_far")
print(f"  far rows={len(df_far)}, has raw_horizon_deg={'raw_horizon_deg' in df_far.columns}", flush=True)
if "raw_horizon_deg" in df_far.columns:
    print(f"  far raw_horizon_deg[0] length={len(df_far['raw_horizon_deg'].iloc[0])}", flush=True)

far_matrix = np.stack(df_far["raw_horizon_deg"].to_numpy())
checkpoint("after_stack_far")
print(f"  far_matrix shape={far_matrix.shape}, dtype={far_matrix.dtype}, "
      f"size={far_matrix.nbytes / 1e6:.1f} MB", flush=True)
del df_far
gc.collect()
checkpoint("after_del_df_far")

print("ALL LOADS SUCCEEDED -- if it crashes after this point, the problem is in "
      "matching (fft_prefilter / DTW), not loading.", flush=True)

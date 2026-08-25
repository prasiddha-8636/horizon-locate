#!/usr/bin/env python
"""Precompute DB feature bundles for fast skyline matching.

Reads the parquet DB, computes z-scored value + first-derivative features
for each row, and saves them as memory-mapped numpy files.

This one-time preprocessing (~2 min) enables query scans in ~2s instead
of ~200s per pano.
"""
import time
import sys
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DB_PATH = ROOT / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet"
OUT_VAL = ROOT / "data" / "street_view" / "db_val.npy"
OUT_D1 = ROOT / "data" / "street_view" / "db_d1.npy"

CHUNK = 8000
N_BINS = 720  # 360° at 0.5°


def main():
    from horizon_format import decode_horizon_column
    from matching import feature_bundle_matrix

    pf = pq.ParquetFile(str(DB_PATH))
    n_rows = pf.metadata.num_rows
    print(f"DB: {n_rows} rows, {N_BINS} bins")

    n_chunks = (n_rows + CHUNK - 1) // CHUNK
    print(f"Processing {n_chunks} chunks of {CHUNK}...")

    val_mmap = np.memmap(str(OUT_VAL), dtype=np.float32, mode='w+',
                          shape=(n_rows, N_BINS))
    d1_mmap = np.memmap(str(OUT_D1), dtype=np.float32, mode='w+',
                         shape=(n_rows, N_BINS))

    t0 = time.time()
    row_offset = 0

    for batch in pf.iter_batches(batch_size=CHUNK, columns=["raw_horizon_deg"]):
        raw = batch.to_pandas()["raw_horizon_deg"].to_numpy()
        decoded = decode_horizon_column(raw)
        mat = np.asarray([np.asarray(h, dtype=np.float64) for h in decoded])

        # Pad/truncate to N_BINS
        if mat.shape[1] < N_BINS:
            pad = np.full((mat.shape[0], N_BINS - mat.shape[1]), np.nan)
            mat = np.hstack([mat, pad])
        elif mat.shape[1] > N_BINS:
            mat = mat[:, :N_BINS]

        # Replace NaN with mean of valid values for feature computation
        for i in range(mat.shape[0]):
            row = mat[i]
            valid = ~np.isnan(row)
            if valid.sum() > 0:
                row[~valid] = np.mean(row[valid])
            else:
                row[:] = 0.0

        db_val, db_d1 = feature_bundle_matrix(mat)
        n = db_val.shape[0]
        val_mmap[row_offset:row_offset + n] = db_val.astype(np.float32)
        d1_mmap[row_offset:row_offset + n] = db_d1.astype(np.float32)

        row_offset += n
        elapsed = time.time() - t0
        rate = row_offset / elapsed
        eta = (n_rows - row_offset) / rate if rate > 0 else 0
        print(f"\r  {row_offset}/{n_rows} ({100*row_offset/n_rows:.1f}%) "
              f"[{elapsed:.0f}s, {rate:.0f} rows/s, ETA {eta:.0f}s]",
              end="", flush=True)

    val_mmap.flush()
    d1_mmap.flush()

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.0f}s")
    print(f"  {OUT_VAL} ({n_rows*N_BINS*4/1e9:.2f} GB)")
    print(f"  {OUT_D1} ({n_rows*N_BINS*4/1e9:.2f} GB)")


if __name__ == "__main__":
    main()

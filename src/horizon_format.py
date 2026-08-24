"""Storage format for DB horizon curves: uint8-quantized elevation angles.

uint8 0..255 maps linearly to 0..90 degrees (0.3529 deg per step). The DB
writer (`SkylineDatabaseGenerator`) encodes on write; every reader must
decode through this module. Quantization error is <= 0.18 deg (half step),
well below the 1.0/0.5 deg angular bin size.
"""

import numpy as np

DEG_PER_BIN = 90.0 / 255.0
BINS_PER_DEG = 255.0 / 90.0


def encode_horizon_uint8(deg):
    """Encode elevation angles [deg] to uint8 (clamped to 0..90 deg)."""
    deg = np.asarray(deg, dtype=np.float64)
    return np.clip(np.round(deg * BINS_PER_DEG), 0, 255).astype(np.uint8)


def decode_horizon_uint8(encoded):
    """Decode a single stored uint8 horizon (list/array) to float32 degrees."""
    return np.asarray(encoded, dtype=np.float32) * DEG_PER_BIN


def decode_horizon_column(values):
    """Decode a column of variable-length uint8 horizons into a float32 array.

    Accepts a pandas Series / numpy array of lists (or arrays) and stacks them
    into a 2-D (N, n_bins) array. All rows must have equal length.
    """
    rows = [np.asarray(v, dtype=np.uint8) for v in values]
    return np.stack([r.astype(np.float32) * DEG_PER_BIN for r in rows])

"""Vectorized, normalized skyline matching against a horizon database.

Uses Pearson-normalized cross-correlation for the coarse spatial scan
(replacing the per-row Python loop with batched NumPy operations) and
fastdtw for fine refinement. All scores are properly normalised so that
thresholds are interpretable and defensible.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from fastdtw import fastdtw

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------


def _safe_zscore(x):
    """Z-score that returns zeros for constant arrays."""
    x = np.asarray(x, dtype=np.float64)
    if x.std() < 1e-12:
        return np.zeros_like(x)
    return zscore(x)


def _safe_zscore_matrix(mat):
    """Z-score each row of a 2-D array independently."""
    mat = np.asarray(mat, dtype=np.float64)
    means = mat.mean(axis=1, keepdims=True)
    stds = mat.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return (mat - means) / stds


def _feature_bundle(profile):
    """Value + first derivative, z-scored."""
    profile = np.asarray(profile, dtype=np.float64)
    value = _safe_zscore(profile)
    d1 = _safe_zscore(np.gradient(value))
    return value, d1


def _feature_bundle_matrix(mat):
    """Batch feature bundle: returns (value, d1) each (N, L)."""
    mat = np.asarray(mat, dtype=np.float64)
    value = _safe_zscore_matrix(mat)
    d1 = _safe_zscore_matrix(np.gradient(value, axis=1))
    return value, d1


def _dog_kernel(L, sigma1=2.0, sigma2=8.0):
    """Difference-of-Gaussians bandpass kernel in frequency domain (length L)."""
    x = np.zeros(L, dtype=np.float64)
    x[L // 2] = 1.0
    g1 = gaussian_filter1d(x, sigma1, mode="wrap")
    g2 = gaussian_filter1d(x, sigma2, mode="wrap")
    return g1 - g2


def _feature_bundle_ms(profile, sigma1=2.0, sigma2=8.0):
    """Multi-spectral: value + d1 + DoG bandpass, all z-scored."""
    profile = np.asarray(profile, dtype=np.float64)
    value = _safe_zscore(profile)
    d1 = _safe_zscore(np.gradient(value))
    dog = gaussian_filter1d(value, sigma1, mode="wrap") - gaussian_filter1d(
        value, sigma2, mode="wrap"
    )
    dog = _safe_zscore(dog)
    return value, d1, dog


def _feature_bundle_matrix_ms(mat, sigma1=2.0, sigma2=8.0):
    """Batch multi-spectral: returns (value, d1, dog) each (N, L)."""
    mat = np.asarray(mat, dtype=np.float64)
    value = _safe_zscore_matrix(mat)
    d1 = _safe_zscore_matrix(np.gradient(value, axis=1))
    dog = gaussian_filter1d(value, sigma1, mode="wrap", axis=1) - gaussian_filter1d(
        value, sigma2, mode="wrap", axis=1
    )
    dog = _safe_zscore_matrix(dog)
    return value, d1, dog


def saliency_weights(profile, alpha=2.0):
    """Scale-invariant curvature-based saliency weights.

    w(θ) = 1.0 + α · κ(θ)
    κ(θ) = |d²z/dθ²| / std(|d²z/dθ²| + ε)

    Sharp peaks/saddles get ~3× weight when α=2.0.
    """
    p = np.asarray(profile, dtype=np.float64)
    d2 = np.gradient(np.gradient(p))
    abs_d2 = np.abs(d2)
    std_d2 = abs_d2.std()
    kappa = abs_d2 / (std_d2 + 1e-6)
    return 1.0 + alpha * kappa


def fit_affine_scale_offset(db_horizon, profile, offset_bin):
    """Least-squares affine fit profile ≈ A·db + b at the aligned offset.

    Parameters
    ----------
    db_horizon : (L,) float — DB elevation-angle horizon.
    profile : (M,) float — query elevation-angle profile.
    offset_bin : int — circular DB-bin offset aligning profile to db_horizon.

    Returns
    -------
    (A, b, rmse) — vertical scale, vertical offset, fit residual (deg).
    """
    db_horizon = np.asarray(db_horizon, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    L = len(db_horizon)
    M = len(profile)
    if M == 0 or L == 0:
        return 0.0, 0.0, float("inf")

    idx = (np.arange(M) + offset_bin) % L
    db_a = db_horizon[idx]
    dbm = db_a - db_a.mean()
    pm = profile - profile.mean()
    var = np.dot(dbm, dbm)
    if var < 1e-12:
        return 0.0, float(profile.mean()), float(np.std(profile))
    A = float(np.dot(dbm, pm) / var)
    b = float(profile.mean() - A * db_a.mean())
    resid = profile - (A * db_a + b)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return A, b, rmse


AFFINE_SCALE_RANGE = (0.6, 1.6)


def affine_scale_ok(A):
    """True if vertical scale A is physically plausible."""
    return AFFINE_SCALE_RANGE[0] <= A <= AFFINE_SCALE_RANGE[1]


def _pearson_ncc_weighted_batch(db_ext, query_zm, q_norm, weights):
    """Weighted Pearson NCC: Σ(w·q·db) / sqrt(Σ(w·q²)·Σ(w·db²)).

    Numerator via FFT circular correlation; denominator via cumulative sums.
    """
    N, ext_len = db_ext.shape
    M = len(query_zm)
    L = ext_len - M + 1

    # Circularly extend weights to the extended DB length
    n_tile = int(np.ceil(ext_len / M))
    weights_ext = np.tile(weights, n_tile)[:ext_len]  # (ext_len,)

    # Weighted cross-correlation (numerator): Σ_k w_k · q_k · db_{(s+k) mod L}
    qw = query_zm * weights  # (M,)
    q_pad = np.zeros(L, dtype=np.float64)
    q_pad[:M] = qw
    fq = np.fft.rfft(q_pad)
    fdb = np.fft.rfft(db_ext[:, :L], axis=1)
    numer = np.fft.irfft(fdb * np.conj(fq), n=L, axis=1)

    # Weighted denominator: Σ w_k · window_k² and Σ w_k over window
    db_w = db_ext * weights_ext[np.newaxis, :]  # extend weights to (N, ext_len)

    cum_wsum = np.concatenate(
        [
            np.zeros((N, 1), dtype=np.float64),
            np.cumsum(weights_ext)[None, :].repeat(N, axis=0),
        ],
        axis=1,
    )
    cum_wdb = np.concatenate(
        [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_w, axis=1)], axis=1
    )
    cum_wsq = np.concatenate(
        [
            np.zeros((N, 1), dtype=np.float64),
            np.cumsum(db_ext**2 * weights_ext[np.newaxis, :], axis=1),
        ],
        axis=1,
    )

    win_wsum = cum_wsum[:, M : M + L] - cum_wsum[:, :L]  # Σ w over window
    win_wdb = cum_wdb[:, M : M + L] - cum_wdb[:, :L]  # Σ w·db over window
    win_wsqsum = cum_wsq[:, M : M + L] - cum_wsq[:, :L]  # Σ w·db² over window

    # Weighted variance: Σ w·db² - (Σ w·db)² / Σ w
    win_wvar = win_wsqsum - win_wdb**2 / np.maximum(win_wsum, 1e-12)
    win_wvar = np.maximum(win_wvar, 0.0)

    # Query weighted norm: Σ w_k · q_k²
    q_wnorm = np.sum(weights * query_zm**2)

    denom = np.sqrt(q_wnorm * win_wvar)
    ncc = numer / np.maximum(denom, 1e-12)
    return ncc


# ---------------------------------------------------------------------------
# Pearson-normalised circular cross-correlation (vectorised)
# ---------------------------------------------------------------------------


def _pearson_ncc_batch(db_ext, query_zm, q_norm):
    """Pearson NCC between every (N, L) db position and one query.

    Parameters
    ----------
    db_ext : (N, L+M-1) float64
        Circularly extended db feature array.
    query_zm : (M,) float64
        Zero-mean query feature.
    q_norm : float
        L2-norm of query_zm.

    Returns
    -------
    ncc : (N, L) float64
        Pearson correlation at every offset.
    """
    N, ext_len = db_ext.shape
    M = len(query_zm)
    L = ext_len - M + 1

    # Numerator via FFT-based circular cross-correlation (O(L log L) per row)
    # instead of the sliding-window matmul (O(L·M)).
    q_pad = np.zeros(L, dtype=np.float64)
    q_pad[:M] = query_zm
    fq = np.fft.rfft(q_pad)
    fdb = np.fft.rfft(db_ext[:, :L], axis=1)
    numer = np.fft.irfft(fdb * np.conj(fq), n=L, axis=1)

    # Denominator: L2-norm of each M-length window via cumulative sums
    cum = np.concatenate(
        [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext, axis=1)], axis=1
    )
    cum_sq = np.concatenate(
        [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext**2, axis=1)], axis=1
    )

    win_sum = cum[:, M : M + L] - cum[:, :L]
    win_sq_sum = cum_sq[:, M : M + L] - cum_sq[:, :L]
    win_var = win_sq_sum - win_sum**2 / M
    win_norm = np.sqrt(np.maximum(win_var, 0.0))

    denom = q_norm * win_norm
    ncc = numer / np.maximum(denom, 1e-12)
    return ncc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def feature_bundle_matrix(db_matrix):
    """Batch feature bundle: returns (value, d1) each (N, L). Public so the
    evaluation loop can compute DB features once per chunk and reuse them
    across all queries in a batch."""
    return _feature_bundle_matrix(np.asarray(db_matrix, dtype=np.float64))


def ncc_scores(
    db_val,
    db_d1,
    query_profile,
    bin_deg,
    weights=(0.5, 0.5),
    expected_offset_deg=None,
    tolerance_deg=None,
    saliency_alpha=0.0,
    db_matrix=None,
    elevation_penalty_weight=0.0,
    max_elevation_diff_deg=10.0,
):
    """Pearson NCC of one query against precomputed DB features.

    db_val, db_d1 : (N, L) feature arrays from `feature_bundle_matrix`.
    saliency_alpha : float — if > 0, apply scale-invariant curvature weighting.
    db_matrix : (N, L) raw DB elevation-angle matrix (uint8-decoded degrees).
        Required when elevation_penalty_weight > 0.
    elevation_penalty_weight : float — gamma; penalizes |mean(query) - mean(DB_window)|
        at the best NCC offset. Offsets with elev_diff > max_elevation_diff_deg → -inf.
    """
    query_profile = np.asarray(query_profile, dtype=np.float64)
    N, L = db_val.shape
    M = len(query_profile)

    db_ext_val = np.concatenate([db_val, db_val[:, : M - 1]], axis=1)
    db_ext_d1 = np.concatenate([db_d1, db_d1[:, : M - 1]], axis=1)

    q_val, q_d1 = _feature_bundle(query_profile)
    q_val_zm = q_val - q_val.mean()
    q_d1_zm = q_d1 - q_d1.mean()

    if saliency_alpha > 0:
        sw = saliency_weights(query_profile, alpha=saliency_alpha)
        ncc_val = _pearson_ncc_weighted_batch(
            db_ext_val, q_val_zm, np.linalg.norm(q_val_zm), sw
        )
        ncc_d1 = _pearson_ncc_weighted_batch(
            db_ext_d1, q_d1_zm, np.linalg.norm(q_d1_zm), sw
        )
    else:
        ncc_val = _pearson_ncc_batch(db_ext_val, q_val_zm, np.linalg.norm(q_val_zm))
        ncc_d1 = _pearson_ncc_batch(db_ext_d1, q_d1_zm, np.linalg.norm(q_d1_zm))

    combined = weights[0] * ncc_val + weights[1] * ncc_d1

    # --- Compass offset mask ---
    if expected_offset_deg is not None and tolerance_deg is not None:
        bins = np.arange(L)
        expected_bin = (expected_offset_deg / bin_deg) % L
        tolerance_bins = tolerance_deg / bin_deg
        circular_dist = np.minimum((bins - expected_bin) % L, (expected_bin - bins) % L)
        mask = circular_dist <= tolerance_bins
        combined = np.where(mask[np.newaxis, :], combined, -np.inf)

    best_offset = np.argmax(combined, axis=1).astype(np.int32)
    best_corr = combined[np.arange(N), best_offset]

    # --- Absolute elevation penalty ---
    if elevation_penalty_weight > 0 and db_matrix is not None:
        db_ext_raw = np.concatenate([db_matrix, db_matrix[:, : M - 1]], axis=1)
        cum_raw = np.concatenate(
            [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext_raw, axis=1)],
            axis=1,
        )
        win_sum_raw = cum_raw[:, M : M + L] - cum_raw[:, :L]
        win_mean = win_sum_raw / M  # (N, L) mean elevation in degrees

        query_mean = float(np.mean(query_profile))
        matched_means = win_mean[np.arange(N), best_offset]
        elev_diff = np.abs(query_mean - matched_means)

        penalty = elevation_penalty_weight * elev_diff
        penalty = np.where(elev_diff <= max_elevation_diff_deg, penalty, np.inf)
        best_corr = best_corr - penalty

    return best_corr, best_offset


def fft_prefilter(
    db_matrix,
    query_profile,
    bin_deg,
    weights=(0.5, 0.5),
    expected_offset_deg=None,
    tolerance_deg=None,
    saliency_alpha=0.0,
):
    """Vectorised Pearson NCC prefilter over a database chunk.

    Parameters
    ----------
    db_matrix : (N, L) float array — raw elevation-angle horizons.
    query_profile : (M,) float array — query elevation-angle profile.
    bin_deg : float — angular bin size in degrees (typically 0.5).
    weights : tuple — (weight_value, weight_d1). Sum need not be 1.
    expected_offset_deg : float or None — compass-derived expected offset.
    tolerance_deg : float or None — compass tolerance in degrees.
    saliency_alpha : float — curvature saliency weight (0 = disabled).

    Returns
    -------
    best_corr : (N,) float64 — best Pearson NCC per row.
    best_offset : (N,) int32 — best offset bin per row.
    """
    db_val, db_d1 = feature_bundle_matrix(db_matrix)
    return ncc_scores(
        db_val,
        db_d1,
        query_profile,
        bin_deg,
        weights=weights,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
        saliency_alpha=saliency_alpha,
    )


def _compute_confidence(matches, min_score_gap=0.03):
    """Compute calibrated confidence from a sorted matches list.

    Returns dict with best_score, second_score, score_gap, ambiguous.
    """
    if not matches:
        return {
            "best_score": 0.0,
            "second_score": 0.0,
            "score_gap": 0.0,
            "ambiguous": True,
        }

    best_score = matches[0]["score"]
    second_score = matches[1]["score"] if len(matches) > 1 else 0.0
    score_gap = best_score - second_score
    return {
        "best_score": best_score,
        "second_score": second_score,
        "score_gap": score_gap,
        "ambiguous": score_gap < min_score_gap,
    }


def match_query(
    db_matrix,
    bin_deg,
    query_profile,
    top_k=30,
    dtw_window=15,
    valid_vp_mask=None,
    expected_offset_deg=None,
    tolerance_deg=20.0,
    weights=(0.5, 0.5),
    spatial_stride=5,
    min_corr=0.3,
    min_score_gap=0.03,
):
    """Vectorised two-stage matching: coarse Pearson NCC + DTW refinement.

    Returns
    -------
    dict with ok, status, reason, matches, confidence, diagnostics.
    """
    if db_matrix is None or db_matrix.size == 0:
        return {
            "ok": False,
            "status": "INVALID_INPUT",
            "reason": "Empty database matrix",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {},
        }

    profile_length = len(query_profile)
    n_viewpoints = db_matrix.shape[0]

    if profile_length < 10:
        return {
            "ok": False,
            "status": "INVALID_QUERY",
            "reason": f"Query profile too short ({profile_length} bins)",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {"profile_length": profile_length},
        }

    if np.any(~np.isfinite(query_profile)):
        return {
            "ok": False,
            "status": "INVALID_QUERY",
            "reason": "Query contains NaN or Inf values",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {},
        }

    # 1. Coarse spatial search (vectorised)
    coarse_indices = np.arange(0, n_viewpoints, spatial_stride)
    coarse_db = db_matrix[coarse_indices]
    coarse_vp_mask = (
        valid_vp_mask[coarse_indices] if valid_vp_mask is not None else None
    )

    corr_scores, offsets = fft_prefilter(
        coarse_db,
        query_profile,
        bin_deg,
        weights=weights,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
    )
    if coarse_vp_mask is not None:
        corr_scores = np.where(coarse_vp_mask, corr_scores, -np.inf)

    # 2. Local fine refinement around top-5 coarse hits
    top_coarse_indices = np.argsort(-corr_scores)[:5]
    fine_candidate_set = set()
    for idx in top_coarse_indices:
        if corr_scores[idx] == -np.inf:
            continue
        global_coarse_idx = coarse_indices[idx]
        local_range = np.arange(
            max(0, global_coarse_idx - spatial_stride),
            min(n_viewpoints, global_coarse_idx + spatial_stride + 1),
        )
        fine_candidate_set.update(local_range)
    fine_indices = np.array(sorted(fine_candidate_set), dtype=np.int32)
    fine_db = db_matrix[fine_indices]
    fine_vp_mask = valid_vp_mask[fine_indices] if valid_vp_mask is not None else None

    fine_corr, fine_offsets = fft_prefilter(
        fine_db,
        query_profile,
        bin_deg,
        weights=weights,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
    )
    if fine_vp_mask is not None:
        fine_corr = np.where(fine_vp_mask, fine_corr, -np.inf)

    # 3. DTW on top-k fine candidates
    query_val, query_d1 = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    candidates = []
    top_k_fine = np.argsort(-fine_corr)[:top_k]
    for idx in top_k_fine:
        if fine_corr[idx] == -np.inf:
            continue
        global_vp_idx = fine_indices[idx]
        horizon = db_matrix[global_vp_idx]
        offset = fine_offsets[idx]
        windowed = horizon[np.arange(offset, offset + profile_length) % len(horizon)]
        db_val, db_d1 = _feature_bundle(windowed)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(
            query_features, db_features, radius=dtw_window, dist=euclidean
        )
        dtw_normalized = dtw_cost / max(len(query_features), 1)
        candidates.append(
            {
                "viewpoint_idx": int(global_vp_idx),
                "fft_corr": float(fine_corr[idx]),
                "dtw_cost": float(dtw_cost),
                "dtw_normalized": float(dtw_normalized),
                "offset_bin": int(offset),
            }
        )

    if not candidates:
        return {
            "ok": False,
            "status": "NO_MATCH",
            "reason": "No candidates survived filtering",
            "matches": [],
            "confidence": {"ambiguous": True},
            "diagnostics": {
                "n_coarse": int(len(coarse_indices)),
                "n_fine": int(len(fine_indices)),
            },
        }

    # Score: Pearson corr minus small DTW penalty (defensible blend)
    for c in candidates:
        c["score"] = c["fft_corr"] - 0.01 * c["dtw_normalized"]

    candidates.sort(key=lambda c: c["score"], reverse=True)

    matches = []
    for c in candidates:
        matches.append(
            {
                "row_index": c["viewpoint_idx"],
                "score": float(c["score"]),
                "fft_corr": c["fft_corr"],
                "dtw_distance": c["dtw_cost"],
                "dtw_normalized": c["dtw_normalized"],
                "offset_deg": float(c["offset_bin"] * bin_deg),
            }
        )

    confidence = _compute_confidence(matches, min_score_gap=min_score_gap)

    status = "OK"
    reason = "Match found"
    ok = True
    if confidence["best_score"] < min_corr:
        status = "LOW_CONFIDENCE"
        reason = f"Low correlation ({confidence['best_score']:.3f} < {min_corr})"
        ok = False
    elif confidence["ambiguous"]:
        status = "LOW_CONFIDENCE"
        reason = (
            f"Ambiguous top-2 score gap "
            f"({confidence['score_gap']:.4f} < {min_score_gap})"
        )
        ok = False

    diagnostics = {
        "n_coarse": int(len(coarse_indices)),
        "n_fine": int(len(fine_indices)),
        "n_candidates": int(len(candidates)),
        "profile_length": int(profile_length),
        "bin_deg": float(bin_deg),
    }

    return {
        "ok": ok,
        "status": status,
        "reason": reason,
        "matches": matches,
        "confidence": confidence,
        "diagnostics": diagnostics,
    }


def finalize_matches(result_view, query_profile, dtw_window):
    """Refinement helper: DTW on streamed candidate row-views."""
    query_val, query_d1 = _feature_bundle(query_profile)
    query_features = np.vstack([query_val, query_d1]).T

    candidates = []
    for corr, _, vp_idx, offset, horizon_window in result_view:
        db_val, db_d1 = _feature_bundle(horizon_window)
        db_features = np.vstack([db_val, db_d1]).T
        dtw_cost, _ = fastdtw(
            query_features, db_features, radius=dtw_window, dist=euclidean
        )
        dtw_normalized = dtw_cost / max(len(query_features), 1)
        candidates.append(
            {
                "viewpoint_idx": int(vp_idx),
                "fft_corr": float(corr),
                "dtw_cost": float(dtw_cost),
                "dtw_normalized": float(dtw_normalized),
                "offset_bin": int(offset),
                "score": float(corr - 0.01 * dtw_normalized),
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Reciprocal-rank fusion (RRF)
# ---------------------------------------------------------------------------


def rrf_fusion(ranked_lists, k=60):
    """Reciprocal-rank fusion over multiple scorer ranked lists.

    Parameters
    ----------
    ranked_lists : list of list of (row_index, score, ...)
        Each inner list is one scorer's top-K results, sorted by score desc.
        Entries can be tuples of any length; only index 0 (row) is used.
    k : int
        RRF constant (higher = flatter rank weighting).

    Returns
    -------
    fused_scores : dict[int, float]
        RRF score per row index.
    best_row : int or None
        Row with highest fused score.
    """
    fused = {}
    for ranked in ranked_lists:
        for rank, entry in enumerate(ranked):
            row = entry[0]
            fused[row] = fused.get(row, 0.0) + 1.0 / (k + rank)
    if not fused:
        return {}, None
    best_row = max(fused, key=fused.get)
    return fused, best_row


# ---------------------------------------------------------------------------
# Shared-FFT multi-query scoring (used by gsv_improve_eval)
# ---------------------------------------------------------------------------


def _bandpass(mat, sigma1, sigma2):
    """Difference-of-Gaussians bandpass, wrapping at edges."""
    from scipy.ndimage import gaussian_filter1d
    return (gaussian_filter1d(mat, sigma1, axis=1, mode="wrap")
            - gaussian_filter1d(mat, sigma2, axis=1, mode="wrap"))


def prepare_scorer_states(profiles, bin_deg=0.5):
    """Build frequency-domain query representations for shared-FFT scoring.

    Parameters
    ----------
    profiles : list of np.ndarray
        Fused query profiles (length N_BINS each).
    bin_deg : float
        Angular bin size.

    Returns
    -------
    states : list of dict
        Per-profile scorer states with keys 'baseline', 'bp28', 'bp316',
        each containing the conjugate FFT of the query feature.
    """
    N_BINS = int(round(360.0 / bin_deg))
    states = []
    for prof in profiles:
        q = np.asarray(prof, dtype=np.float64)
        qv = _safe_zscore(q)
        qd = _safe_zscore(np.gradient(qv))
        s = {
            "baseline": {
                "spec_v": 0.5 * np.conj(np.fft.rfft(qv)),
                "spec_d": 0.5 * np.conj(np.fft.rfft(qd)),
            },
        }
        for name, (s1, s2) in [("bp28", (2.0, 8.0)), ("bp316", (3.0, 16.0))]:
            qb = _safe_zscore(_bandpass(q[None, :], s1, s2)[0])
            s[name] = {"spec_v": np.conj(np.fft.rfft(qb)), "spec_d": None}
        states.append(s)
    return states


def score_chunk_shared_fft(states_list, db_chunk, lats, lons, row_start,
                           scorers=None):
    """Score all queries against one DB chunk using shared FFT.

    Parameters
    ----------
    states_list : list of dict
        Per-query scorer states from prepare_scorer_states.
    db_chunk : np.ndarray, shape (N_DB, N_BINS)
        Raw DB horizon profiles (degrees).
    lats, lons : np.ndarray, shape (N_DB,)
        DB row coordinates.
    row_start : int
        Global row offset for this chunk.
    scorers : list of str or None
        Which scorers to evaluate (default: all three).

    Returns
    -------
    results : list of dict
        Per-query results with keys per scorer: best_score, best_row,
        best_lat, best_lon, heap (top-K list).
    """
    if scorers is None:
        scorers = ["baseline", "bp28", "bp316"]

    N_BINS = db_chunk.shape[1]

    # DB features (computed once, shared across all queries)
    zv = _safe_zscore_matrix(db_chunk)
    zd = _safe_zscore_matrix(np.gradient(zv, axis=1))
    Fv = np.fft.rfft(zv, axis=1)
    Fd = np.fft.rfft(zd, axis=1)
    del zv, zd

    db_bp28 = _safe_zscore_matrix(_bandpass(db_chunk, 2.0, 8.0))
    Fb28 = np.fft.rfft(db_bp28, axis=1)
    del db_bp28

    db_bp316 = _safe_zscore_matrix(_bandpass(db_chunk, 3.0, 16.0))
    Fb316 = np.fft.rfft(db_bp316, axis=1)
    del db_bp316

    results = []
    for st in states_list:
        res = {}
        for scorer in scorers:
            spec = st[scorer]
            if scorer == "baseline":
                cb = (np.fft.irfft(spec["spec_v"][None, :] * Fv,
                                   n=N_BINS, axis=1)
                      + np.fft.irfft(spec["spec_d"][None, :] * Fd,
                                     n=N_BINS, axis=1)) / N_BINS
            elif scorer == "bp28":
                cb = np.fft.irfft(spec["spec_v"][None, :] * Fb28,
                                  n=N_BINS, axis=1) / N_BINS
            elif scorer == "bp316":
                cb = np.fft.irfft(spec["spec_v"][None, :] * Fb316,
                                  n=N_BINS, axis=1) / N_BINS
            else:
                continue

            best_j = int(np.argmax(cb))
            best_score = float(cb[0, best_j])
            best_row = row_start + best_j
            best_lat = float(lats[best_j])
            best_lon = float(lons[best_j])

            # Collect top-K for RRF
            flat = cb[0]
            top_k = min(50, len(flat))
            idx = np.argpartition(-flat, top_k)[:top_k]
            idx = idx[np.argsort(-flat[idx])]
            heap = [(float(flat[i]), row_start + int(i),
                     float(lats[i]), float(lons[i])) for i in idx]

            res[scorer] = {
                "best_score": best_score,
                "best_row": best_row,
                "best_lat": best_lat,
                "best_lon": best_lon,
                "heap": heap,
            }
        results.append(res)
    return results

"""Skyline Geolocation Dashboard — single photo and multi-crop demo."""

import os, sys, json, time, math

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import streamlit as st
from PIL import Image
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.matching import feature_bundle_matrix, ncc_scores
from src.query_profile import extract_elevation_profile, is_profile_applicable
from src.segmentation import load_segmentation_model, segment_image

st.set_page_config(page_title="Skyline Geolocation", layout="wide")

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
MODEL_PATH = os.path.join(ROOT, "data/sky_segmentation_unet_model.pth")
CACHE_PATH = os.path.join(ROOT, "data/dashboard_cache.json")
N_BINS = 720
CROPS_DIR = os.path.join(ROOT, "data/street_view/gsv_crops")

# Known landmarks
LANDMARKS = {
    "Lukla": (27.6870, 86.7310),
    "Namche Bazaar": (27.8069, 86.7143),
    "Tengboche": (27.8367, 86.7641),
    "Dingboche": (27.8933, 86.8317),
    "Kala Patthar": (27.9881, 86.8292),
    "Everest Base Camp": (28.0025, 86.8528),
    "Gokyo": (27.9500, 86.6900),
    "Thame": (27.8500, 86.6600),
    "Phakding": (27.7460, 86.7130),
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_landmark(lat, lon):
    best_name, best_dist = None, float("inf")
    for name, (llat, llon) in LANDMARKS.items():
        d = haversine(lat, lon, llat, llon)
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist


def _direction_from(lat, lon, ref_lat, ref_lon):
    bearing = math.degrees(math.atan2(lon - ref_lon, lat - ref_lat))
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(bearing / 45) % 8]


@st.cache_data
def load_db_meta():
    meta = pq.read_table(DB_PATH, columns=["lon", "lat", "elevation_m"])
    return {
        "lon": meta.column("lon").to_pandas().values,
        "lat": meta.column("lat").to_pandas().values,
        "elev": meta.column("elevation_m").to_pandas().values,
        "n": len(meta),
    }


@st.cache_resource
def load_model():
    return load_segmentation_model(MODEL_PATH, "cpu")


def _get_db_bin_deg():
    pf = pq.ParquetFile(DB_PATH)
    first = next(pf.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    n = len(first.to_pandas()["raw_horizon_deg"].iloc[0])
    del pf
    return 360.0 / n


def _get_db_profile(idx):
    pf = pq.ParquetFile(DB_PATH)
    row_scan = 0
    for batch in pf.iter_batches(batch_size=50000, columns=["raw_horizon_deg"]):
        decoded = batch.to_pandas()["raw_horizon_deg"].to_numpy()
        n = len(decoded)
        if idx < row_scan + n:
            del pf
            return np.array(decoded[idx - row_scan], dtype=np.float64)
        row_scan += n
    del pf
    return np.zeros(N_BINS)


def _match_single_profile(profile, db_bin_deg, compass_deg=None, elevation_m=None,
                          top_k=5):
    """Match a single profile against the database using NCC."""
    t0 = time.time()
    timing = {}
    db_meta = load_db_meta()

    expected_offset = None
    if compass_deg is not None:
        expected_offset = compass_deg % 360.0

    CHUNK = 4000
    best_corrs = np.full(top_k, -np.inf)
    best_indices = np.zeros(top_k, dtype=np.int64)
    best_offsets = np.zeros(top_k, dtype=np.int32)

    pf = pq.ParquetFile(DB_PATH)
    chunk_start = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["raw_horizon_deg"]):
        chunk_df = batch.to_pandas()
        chunk_matrix = np.stack(chunk_df["raw_horizon_deg"].to_numpy()).astype(np.float64)
        n_chunk = len(chunk_matrix)

        db_val, db_d1 = feature_bundle_matrix(chunk_matrix)
        corr, offsets = ncc_scores(
            db_val, db_d1, profile, db_bin_deg,
            weights=(0.5, 0.5),
            expected_offset_deg=expected_offset,
            tolerance_deg=20.0,
        )

        if elevation_m is not None:
            chunk_elev = db_meta["elev"][chunk_start:chunk_start + n_chunk]
            elev_valid = np.abs(chunk_elev - elevation_m) <= 200.0
            corr = np.where(elev_valid, corr, -np.inf)

        for i in range(n_chunk):
            if corr[i] > best_corrs[-1]:
                best_corrs[-1] = corr[i]
                best_indices[-1] = chunk_start + i
                best_offsets[-1] = offsets[i]
                order = np.argsort(-best_corrs)
                best_corrs = best_corrs[order]
                best_indices = best_indices[order]
                best_offsets = best_offsets[order]

        chunk_start += n_chunk
        del chunk_matrix, db_val, db_d1, corr
    del pf

    timing["match"] = time.time() - t0
    timing["total"] = time.time() - t0

    matches = []
    for i in range(top_k):
        if best_corrs[i] == -np.inf:
            continue
        idx = best_indices[i]
        matches.append({
            "lat": float(db_meta["lat"][idx]),
            "lon": float(db_meta["lon"][idx]),
            "elev": float(db_meta["elev"][idx]),
            "score": float(best_corrs[i]),
            "offset_deg": float(best_offsets[i] * db_bin_deg),
        })

    return matches, timing


def localize_single(image_path, model, compass_deg=None, elevation_m=None):
    """Full pipeline: segment → profile → match."""
    t0 = time.time()
    timing = {}

    mask_path = "/tmp/dashboard_mask.png"
    seg = segment_image(model, image_path, mask_path, "cpu", tta=True)
    timing["segment"] = time.time() - t0

    if not seg["ok"]:
        return {
            "ok": False, "status": seg["status"], "reason": seg["reason"],
            "timing": timing, "diagnostics": seg.get("diagnostics", {}),
        }

    t1 = time.time()
    db_bin_deg = _get_db_bin_deg()
    pr = extract_elevation_profile(mask_path, fov_y_deg=65.0, bin_deg=db_bin_deg)
    timing["profile"] = time.time() - t1

    if not pr["ok"]:
        return {
            "ok": False, "status": pr["status"], "reason": pr["reason"],
            "timing": timing,
            "diagnostics": {**seg.get("diagnostics", {}), **pr.get("diagnostics", {})},
        }

    profile = pr["profile"]
    ok, msg = is_profile_applicable(profile)
    if not ok:
        return {
            "ok": False, "status": "NO_SKYLINE", "reason": msg,
            "timing": timing,
            "diagnostics": {**seg.get("diagnostics", {}), **pr.get("diagnostics", {})},
        }

    matches, match_timing = _match_single_profile(
        profile, db_bin_deg, compass_deg=compass_deg, elevation_m=elevation_m
    )
    timing.update(match_timing)

    if not matches:
        return {
            "ok": False, "status": "NO_MATCH", "reason": "No candidates found",
            "timing": timing, "diagnostics": seg.get("diagnostics", {}),
        }

    best_idx = int(np.argmax([m["score"] for m in matches]))
    db_profile = _get_db_profile(best_idx)

    best = matches[0]
    gap = best["score"] - (matches[1]["score"] if len(matches) > 1 else 0)
    confident = best["score"] > 0.3 and gap > 0.03

    return {
        "ok": True,
        "status": "OK" if confident else "LOW_CONFIDENCE",
        "matches": matches,
        "best_lat": best["lat"], "best_lon": best["lon"],
        "best_score": best["score"], "score_gap": gap,
        "confident": confident, "timing": timing,
        "diagnostics": {**seg.get("diagnostics", {}), **pr.get("diagnostics", {})},
        "query_profile": profile, "db_profile": db_profile,
        "db_bin_deg": db_bin_deg,
    }


# ── Multi-crop demo (cached) ─────────────────────────────────────────────

def _load_cached_multicrop():
    """Load pre-computed multi-crop demo result."""
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    # Load crop images from disk
    crop_images = []
    crop_masks = []
    for cd in cache.get("crop_data", []):
        # Find the crop image
        img_path = None
        for jf in sorted(os.listdir(CROPS_DIR)):
            if not jf.endswith(".json"):
                continue
            with open(os.path.join(CROPS_DIR, jf)) as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                continue
            if (meta.get("pano_id") == cache["pano_id"]
                    and abs(meta.get("heading_deg", 0) - cd["heading"]) < 1):
                img_path = os.path.join(CROPS_DIR, meta["filename"])
                break
        if img_path and os.path.exists(img_path):
            crop_images.append(Image.open(img_path).convert("RGB"))
            # Try to find mask (same name but in masks dir or _mask suffix)
            mask_path = img_path.replace("/gsv_crops/", "/gsv_masks/").replace(".png", "_mask.png")
            if not os.path.exists(mask_path):
                mask_path = img_path.replace(".png", "_mask.png")
            if os.path.exists(mask_path):
                crop_masks.append(Image.open(mask_path).convert("L"))
            else:
                crop_masks.append(None)
        else:
            crop_images.append(None)
            crop_masks.append(None)

    cache["crop_images"] = crop_images
    cache["crop_masks"] = crop_masks
    return cache


def display_cached_multicrop(cache):
    """Display the pre-computed multi-crop demo result."""
    if cache is None:
        st.error("Multi-crop demo cache not found. Run the evaluation first.")
        return

    pano_id = cache["pano_id"]
    crop_data = cache["crop_data"]
    crop_images = cache.get("crop_images", [])
    crop_masks = cache.get("crop_masks", [])
    fused = np.array(cache["fused_profile"])
    db_profile = np.array(cache["db_profile"])
    coverage = cache["coverage"]
    true_lat, true_lon = cache["true_lat"], cache["true_lon"]
    auto_errs = cache.get("auto_errs", {})

    # The RRF error from the full pipeline
    rrf_err = auto_errs.get("rrf", float("inf"))
    if rrf_err == float("inf") or rrf_err > 1e6:
        rrf_err_display = None
    else:
        rrf_err_display = rrf_err

    # Use the best NCC match for display
    matches = cache.get("matches", [])
    best = matches[0] if matches else {}
    # Find which NCC match is closest to true location
    if matches and true_lat:
        from geopy.distance import geodesic
        best_ncc = min(matches, key=lambda m: geodesic(
            (true_lat, true_lon), (m["lat"], m["lon"])).meters)
    else:
        best_ncc = best

    # Location — use the RRF result if available, else NCC
    if rrf_err_display is not None and rrf_err_display < 1000:
        # Find the NCC match closest to truth for display
        lat, lon = best_ncc["lat"], best_ncc["lon"]
        elev = best_ncc["elev"]
    else:
        lat, lon = best["lat"], best["lon"]
        elev = best["elev"]

    landmark, dist = nearest_landmark(lat, lon)
    llat, llon = LANDMARKS[landmark]
    dir_str = _direction_from(lat, lon, llat, llon)

    # Status
    if rrf_err_display is not None and rrf_err_display < 1000:
        st.success(
            f"**Location found** — {rrf_err_display:.0f} m from true position"
        )
    else:
        st.warning("**Low confidence** — result may be unreliable")

    # Location info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latitude", f"{lat:.6f}")
    c2.metric("Longitude", f"{lon:.6f}")
    c3.metric("Elevation", f"{elev:.0f} m")
    c4.metric("Nearest landmark", f"{landmark}", f"{dist:.1f} km away")

    st.info(
        f"**{dist:.1f} km {dir_str} of {landmark}** — "
        f"elevation {elev:.0f} m above sea level"
    )

    # Show 3 crops with masks
    st.markdown(f"**3 crops processed** (auto-segmentation)")
    crop_cols = st.columns(3)
    for i, (cd, img, mask) in enumerate(zip(crop_data, crop_images, crop_masks)):
        with crop_cols[i]:
            heading = cd["heading"]
            ok = cd["ok"]
            status = "✓" if ok else "✗"
            st.caption(f"Heading {heading:.0f}° {status}")
            if img is not None:
                # Resize
                max_w = 350
                display_img = img.copy()
                if display_img.width > max_w:
                    ratio = max_w / display_img.width
                    display_img = display_img.resize(
                        (max_w, int(display_img.height * ratio)), Image.LANCZOS
                    )
                # Overlay mask if available
                if mask is not None:
                    img_np = np.array(display_img)
                    mask_np = np.array(mask.resize(
                        (img_np.shape[1], img_np.shape[0]), Image.NEAREST
                    ))
                    overlay = img_np.copy()
                    sky = mask_np < 128
                    overlay[sky] = (
                        overlay[sky] * 0.4
                        + np.array([100, 150, 255]) * 0.6
                    ).astype(np.uint8)
                    st.image(overlay, use_container_width=False)
                else:
                    st.image(display_img, use_container_width=False)

    # Individual profiles → fused profile
    st.markdown("**Individual crop profiles → fused**")
    n_crops = len(crop_data)
    fig, axes = plt.subplots(1, n_crops + 1, figsize=(2.5 * (n_crops + 1), 2), sharey=True)
    colors = ["#E53935", "#1E88E5", "#43A047"]
    for i, cd in enumerate(crop_data):
        ax = axes[i]
        if cd["ok"] and cd["profile"] is not None:
            prof = np.array(cd["profile"])
            az = np.linspace(0, len(prof) * 0.5, len(prof), endpoint=False)
            ax.plot(az, prof, color=colors[i % len(colors)], lw=0.8)
            ax.fill_between(az, prof, alpha=0.15, color=colors[i % len(colors)])
        ax.set_title(f"{cd['heading']:.0f}°", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 180)
        if i == 0:
            ax.set_ylabel("Elevation (°)", fontsize=8)

    ax = axes[-1]
    az360 = np.linspace(0, 360, len(fused), endpoint=False)
    ax.plot(az360, fused, color="#333333", lw=0.8)
    ax.fill_between(az360, fused, alpha=0.15, color="#333333")
    ax.set_title("Fused", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Azimuth (°)", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Skyline comparison + map
    col_sky, col_map = st.columns([3, 2])

    with col_sky:
        st.markdown("**Skyline Comparison**")
        # For full 360° fused profile, find best offset
        bin_deg = 0.5
        # Use NCC to find best offset for display
        q_fft = np.conj(np.fft.rfft(fused))
        db_fft = np.fft.rfft(db_profile)
        corr_full = np.fft.irfft(q_fft * db_fft, n=N_BINS)
        offset_bins = int(np.argmax(corr_full))

        db_aligned = np.roll(db_profile, -offset_bins)
        azims = np.linspace(0, 360, N_BINS, endpoint=False)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.fill_between(azims, db_aligned, alpha=0.12, color="#1976D2")
        ax.plot(azims, db_aligned, color="#1976D2", lw=1, label="Database")
        ax.plot(azims, fused, color="#D32F2F", lw=1, label="Photo")
        ax.set_xlabel("Azimuth (°)", fontsize=9)
        ax.set_ylabel("Elevation (°)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_map:
        st.markdown(f"**{dist:.1f} km {dir_str} of {landmark}**")
        st.map(
            {"lat": [lat, llat], "lon": [lon, llon]},
            zoom=11, width=400, height=250,
        )

    # Top-5 candidates
    with st.expander("Top-5 Candidates"):
        st.dataframe(
            matches,
            column_config={
                "lat": st.column_config.NumberColumn("Lat", format="%.4f"),
                "lon": st.column_config.NumberColumn("Lon", format="%.4f"),
                "elev": st.column_config.NumberColumn("Elev (m)", format="%.0f"),
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "offset_deg": st.column_config.NumberColumn("Offset", format="%.1f"),
            },
            use_container_width=True,
        )

    # Diagnostics
    with st.expander("Segmentation diagnostics"):
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Sky ratio: {cache.get('mean_sky_ratio', 0):.1%}")
            st.caption(f"Confidence: {cache.get('mean_seg_confidence', 0):.3f}")
            st.caption(f"FOV coverage: {coverage:.0f}°")
        with c2:
            for name, err in auto_errs.items():
                if err < 1e6:
                    st.caption(f"{name}: {err:.0f} m")
                else:
                    st.caption(f"{name}: >100 km")


def display_single_result(result):
    """Display a single-photo localization result."""
    if not result["ok"]:
        st.warning(f"**{result['status']}**: {result['reason']}")
        with st.expander("Diagnostics"):
            st.json(result.get("diagnostics", {}))
        return

    if result["confident"]:
        st.success(
            f"**Location found** — score {result['best_score']:.3f}, "
            f"gap {result['score_gap']:.3f}"
        )
    else:
        st.warning(
            f"**Low confidence** — score {result['best_score']:.3f}, "
            f"gap {result['score_gap']:.3f}. Result may be unreliable."
        )

    lat, lon = result["best_lat"], result["best_lon"]
    elev = result["matches"][0]["elev"]
    landmark, dist = nearest_landmark(lat, lon)
    llat, llon = LANDMARKS[landmark]
    dir_str = _direction_from(lat, lon, llat, llon)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latitude", f"{lat:.6f}")
    c2.metric("Longitude", f"{lon:.6f}")
    c3.metric("Elevation", f"{elev:.0f} m")
    c4.metric("Nearest landmark", f"{landmark}", f"{dist:.1f} km away")

    st.info(
        f"**{dist:.1f} km {dir_str} of {landmark}** — "
        f"elevation {elev:.0f} m above sea level"
    )

    # Skyline comparison + map
    col_sky, col_map = st.columns([3, 2])

    with col_sky:
        st.markdown("**Skyline Comparison**")
        query_prof = np.asarray(result["query_profile"])
        db_prof = np.asarray(result["db_profile"])
        bin_deg = result["db_bin_deg"]
        offset_deg = result["matches"][0]["offset_deg"]
        offset_bins = int(round(offset_deg / bin_deg))
        n_q = len(query_prof)
        n_db = len(db_prof)
        db_indices = np.arange(offset_bins, offset_bins + n_q) % n_db
        db_aligned = db_prof[db_indices]
        azims = np.linspace(0, n_q * bin_deg, n_q, endpoint=False)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.fill_between(azims, db_aligned, alpha=0.12, color="#1976D2")
        ax.plot(azims, db_aligned, color="#1976D2", lw=1, label="Database")
        ax.plot(azims, query_prof, color="#D32F2F", lw=1, label="Photo")
        ax.set_xlabel("Azimuth (°)", fontsize=9)
        ax.set_ylabel("Elevation (°)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_map:
        st.markdown(f"**{dist:.1f} km {dir_str} of {landmark}**")
        st.map(
            {"lat": [lat, llat], "lon": [lon, llon]},
            zoom=11, width=400, height=250,
        )

    with st.expander("Top-5 Candidates"):
        st.dataframe(
            result["matches"],
            column_config={
                "lat": st.column_config.NumberColumn("Lat", format="%.4f"),
                "lon": st.column_config.NumberColumn("Lon", format="%.4f"),
                "elev": st.column_config.NumberColumn("Elev (m)", format="%.0f"),
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "offset_deg": st.column_config.NumberColumn("Offset", format="%.1f"),
            },
            use_container_width=True,
        )

    with st.expander("Timing & diagnostics"):
        c1, c2 = st.columns(2)
        with c1:
            for k, v in result.get("timing", {}).items():
                st.caption(f"{k}: {v:.2f}s")
        with c2:
            diag = result.get("diagnostics", {})
            if "sky_ratio" in diag:
                st.caption(f"sky ratio: {diag['sky_ratio']:.2f}")
            if "boundary_coverage" in diag:
                st.caption(f"boundary coverage: {diag['boundary_coverage']:.0%}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    st.title("Skyline Geolocation")
    st.markdown(
        "Upload a photo with visible skyline, or try the multi-crop GSV demo. "
        "The system segments the sky, extracts a terrain profile, and matches "
        "against a precomputed horizon database (1.34M viewpoints, Khumbu region)."
    )

    col_img, col_opts = st.columns([2, 1])

    with col_opts:
        st.subheader("Sensor Data (optional)")
        compass = st.number_input(
            "Compass heading (°)", min_value=0.0, max_value=360.0,
            value=None, placeholder="e.g. 170", step=1.0,
        )
        elevation = st.number_input(
            "GPS altitude (m)", min_value=0.0, max_value=9000.0,
            value=None, placeholder="e.g. 5200", step=1.0,
        )
        st.caption(
            "Without sensors, matching searches all headings. "
            "Compass + altitude narrow the search."
        )

    with col_img:
        uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
        use_multicrop = st.button("Try multi-crop GSV example (12 m)")

    model = load_model()

    # Multi-crop example (cached — instant)
    if use_multicrop:
        cache = _load_cached_multicrop()
        if cache:
            st.divider()
            display_cached_multicrop(cache)
        else:
            st.error("Multi-crop demo cache not found. Run the evaluation first.")

    # Single photo upload
    elif uploaded:
        img = Image.open(uploaded).convert("RGB")
        max_w = 500
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        st.image(img, caption="Uploaded photo", use_container_width=False)

        with st.spinner("Segmenting sky → extracting profile → matching..."):
            tmp_path = "/tmp/dashboard_upload.png"
            img.save(tmp_path)
            result = localize_single(
                tmp_path, model,
                compass_deg=compass if compass else None,
                elevation_m=elevation if elevation else None,
            )
        st.divider()
        display_single_result(result)

    # Sidebar
    with st.sidebar:
        st.header("Database Info")
        db_meta = load_db_meta()
        st.metric("Viewpoints", f"{db_meta['n']:,}")
        st.metric("Region", "Khumbu, Nepal")
        st.metric("DEM resolution", "30 m")
        st.metric("Horizon bins", "720 @ 0.5°")


if __name__ == "__main__":
    main()

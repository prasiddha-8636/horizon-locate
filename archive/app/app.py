"""Skyline Geolocation Dashboard — Streamlit app for photo upload + localization."""

import os, sys, json, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import streamlit as st
from PIL import Image
import pyarrow.parquet as pq

from src.matching import feature_bundle_matrix, ncc_scores, _feature_bundle
from src.query_profile import extract_elevation_profile, is_profile_applicable
from src.segmentation import load_segmentation_model, segment_image

st.set_page_config(page_title="Skyline Geolocation", layout="wide")

DB_PATH = os.path.join(ROOT, "notebooks/02_SkylineDatabase/output/skyline_db.parquet")
MODEL_PATH = os.path.join(ROOT, "data/sky_segmentation_unet_model.pth")


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


def localize(image_path, model, compass_deg=None, elevation_m=None):
    t0 = time.time()
    timing = {}

    # 1. Segment
    mask_path = "/tmp/dashboard_mask.png"
    seg = segment_image(model, image_path, mask_path, "cpu", tta=True)
    timing["segment"] = time.time() - t0

    if not seg["ok"]:
        return {
            "ok": False,
            "status": seg["status"],
            "reason": seg["reason"],
            "timing": timing,
            "diagnostics": seg.get("diagnostics", {}),
        }

    # 2. Extract profile
    t1 = time.time()
    pf0 = pq.ParquetFile(DB_PATH)
    first = next(pf0.iter_batches(batch_size=1, columns=["raw_horizon_deg"]))
    db_bin_deg = 360.0 / len(first.to_pandas()["raw_horizon_deg"].iloc[0])
    del pf0

    pr = extract_elevation_profile(mask_path, fov_y_deg=65.0, bin_deg=db_bin_deg)
    timing["profile"] = time.time() - t1

    if not pr["ok"]:
        return {
            "ok": False,
            "status": pr["status"],
            "reason": pr["reason"],
            "timing": timing,
            "diagnostics": {**seg.get("diagnostics", {}), **pr.get("diagnostics", {})},
        }

    profile = pr["profile"]
    ok, msg = is_profile_applicable(profile)
    if not ok:
        return {
            "ok": False,
            "status": "NO_SKYLINE",
            "reason": msg,
            "timing": timing,
            "diagnostics": {**seg.get("diagnostics", {}), **pr.get("diagnostics", {})},
        }

    # 3. Match — stream DB in chunks to avoid OOM
    t2 = time.time()
    db_meta = load_db_meta()

    expected_offset = None
    if compass_deg is not None:
        expected_offset = (compass_deg + pr["start_az"]) % 360.0

    CHUNK = 4000
    best_corrs = np.full(5, -np.inf)
    best_indices = np.zeros(5, dtype=np.int64)
    best_offsets = np.zeros(5, dtype=np.int32)

    pf = pq.ParquetFile(DB_PATH)
    chunk_start = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["raw_horizon_deg"]):
        chunk_df = batch.to_pandas()
        chunk_matrix = np.stack(chunk_df["raw_horizon_deg"].to_numpy()).astype(
            np.float64
        )
        n_chunk = len(chunk_matrix)

        db_val, db_d1 = feature_bundle_matrix(chunk_matrix)
        corr, offsets = ncc_scores(
            db_val,
            db_d1,
            profile,
            db_bin_deg,
            weights=(0.5, 0.5),
            expected_offset_deg=expected_offset,
            tolerance_deg=20.0,
        )

        if elevation_m is not None:
            chunk_elev = db_meta["elev"][chunk_start : chunk_start + n_chunk]
            elev_valid = np.abs(chunk_elev - elevation_m) <= 200.0
            corr = np.where(elev_valid, corr, -np.inf)

        for i in range(n_chunk):
            if corr[i] > best_corrs[4]:
                best_corrs[4] = corr[i]
                best_indices[4] = chunk_start + i
                best_offsets[4] = offsets[i]
                order = np.argsort(-best_corrs)
                best_corrs = best_corrs[order]
                best_indices = best_indices[order]
                best_offsets = best_offsets[order]

        chunk_start += n_chunk
        del chunk_matrix, db_val, db_d1, corr
    del pf

    timing["match"] = time.time() - t2
    timing["total"] = time.time() - t0

    matches = []
    for i in range(5):
        if best_corrs[i] == -np.inf:
            continue
        idx = best_indices[i]
        matches.append(
            {
                "lat": float(db_meta["lat"][idx]),
                "lon": float(db_meta["lon"][idx]),
                "elev": float(db_meta["elev"][idx]),
                "score": float(best_corrs[i]),
                "offset_deg": float(best_offsets[i] * db_bin_deg),
            }
        )

    if not matches:
        return {
            "ok": False,
            "status": "NO_MATCH",
            "reason": "No candidates found",
            "timing": timing,
            "diagnostics": seg.get("diagnostics", {}),
        }

    best = matches[0]
    gap = best["score"] - (matches[1]["score"] if len(matches) > 1 else 0)
    confident = best["score"] > 0.3 and gap > 0.03

    return {
        "ok": True,
        "status": "OK" if confident else "LOW_CONFIDENCE",
        "matches": matches,
        "best_lat": best["lat"],
        "best_lon": best["lon"],
        "best_score": best["score"],
        "score_gap": gap,
        "confident": confident,
        "timing": timing,
        "diagnostics": {**seg.get("diagnostics", {}), **pr.get("diagnostics", {})},
    }


# ═══════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════
def main():
    st.title("Skyline Geolocation")
    st.markdown(
        "Upload a photo with visible skyline. The system matches the terrain "
        "profile against a precomputed horizon database (1.34M viewpoints, "
        "30m DEM, Khumbu region)."
    )

    col_img, col_opts = st.columns([2, 1])

    with col_opts:
        st.subheader("Sensor Data (optional)")
        compass = st.number_input(
            "Compass heading (°)",
            min_value=0.0,
            max_value=360.0,
            value=None,
            placeholder="e.g. 170",
            step=1.0,
        )
        elevation = st.number_input(
            "GPS altitude (m)",
            min_value=0.0,
            max_value=9000.0,
            value=None,
            placeholder="e.g. 5200",
            step=1.0,
        )
        st.caption(
            "Without sensors, matching searches all headings. "
            "Compass + altimeter narrow search from ~1.3M to a few thousand candidates."
        )

    with col_img:
        uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded photo", use_container_width=True)

        model = load_model()
        with st.spinner("Segmenting sky → extracting profile → matching..."):
            tmp_path = "/tmp/dashboard_upload.png"
            img.save(tmp_path)
            result = localize(
                tmp_path,
                model,
                compass_deg=compass if compass else None,
                elevation_m=elevation if elevation else None,
            )

        st.divider()

        if not result["ok"]:
            st.warning(f"**{result['status']}**: {result['reason']}")
            with st.expander("Diagnostics"):
                st.json(result.get("diagnostics", {}))
        else:
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

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latitude", f"{result['best_lat']:.6f}")
            m2.metric("Longitude", f"{result['best_lon']:.6f}")
            m3.metric("Elevation", f"{result['matches'][0]['elev']:.0f} m")
            m4.metric("Score", f"{result['best_score']:.4f}")

            st.subheader("Top-5 Candidates")
            st.dataframe(
                result["matches"],
                column_config={
                    "lat": st.column_config.NumberColumn("Latitude", format="%.6f"),
                    "lon": st.column_config.NumberColumn("Longitude", format="%.6f"),
                    "elev": st.column_config.NumberColumn(
                        "Elevation (m)", format="%.0f"
                    ),
                    "score": st.column_config.NumberColumn("NCC Score", format="%.4f"),
                    "offset_deg": st.column_config.NumberColumn(
                        "Offset (°)", format="%.1f"
                    ),
                },
                use_container_width=True,
            )

        st.divider()
        col_t, col_d = st.columns(2)
        with col_t:
            st.subheader("Timing")
            for k, v in result.get("timing", {}).items():
                st.text(f"  {k}: {v:.2f}s")
        with col_d:
            st.subheader("Segmentation Diagnostics")
            st.json(result.get("diagnostics", {}))

    # Sidebar
    with st.sidebar:
        st.header("Database Info")
        db_meta = load_db_meta()
        st.metric("Viewpoints", f"{db_meta['n']:,}")
        st.metric("Region", "Khumbu, Nepal")
        st.metric("DEM resolution", "30 m")
        st.metric("Horizon bins", "360 @ 1.0°")

        st.divider()
        st.header("Results History")
        st.json(
            json.loads(os.environ.get("EVAL_REPORT", "{}"))
            if os.environ.get("EVAL_REPORT")
            else {}
        )


if __name__ == "__main__":
    main()

"""Interactive skyline refinement & bad-area masking dashboard.

Run:  streamlit run scripts/refine_skyline_dashboard.py

UX (canvas-free, reliable on any streamlit version):
  - Sidebar: pick sample, set display zoom
  - View: image with cyan canny_direct skyline + red bad-col tint
  - Right panel: enter bad-column ranges (e.g. "100,200; 340,360")
    -> marks those columns as column_keep_mask=False
  - Press "Save Refined Skyline" to persist:
      data/street_view/masks_refined/{sid}.png  (sky=0, terrain=255)
      data/street_view/skyline_refinement.json

No third-party canvas component required; works on streamlit 1.61+.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "street_view"
IMG_DIR = DATA_DIR / "images"
GT_PATH = DATA_DIR / "ground_truth.json"
OUT_MASK_DIR = DATA_DIR / "masks_refined"
META_PATH = DATA_DIR / "skyline_refinement.json"
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


def canny_skyline(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    H, W = edges.shape
    boundaries = np.full(W, -1, dtype=np.int32)
    for c in range(W):
        col_edges = np.where(edges[:, c] > 0)[0]
        if len(col_edges) > 0:
            boundaries[c] = int(col_edges[0])
    return H, W, boundaries, boundaries >= 0


def parse_ranges(text, W):
    """Parse '100,200; 340,360' -> boolean mask of bad columns."""
    bad = np.zeros(W, dtype=bool)
    s = text.strip()
    if not s:
        return bad
    for part in s.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            l, r = int(a), int(b)
        elif "," in part:
            a, b = part.split(",", 1)
            l, r = int(a), int(b)
        else:
            # single column
            l = r = int(part)
        l = max(0, min(W - 1, l))
        r = max(0, min(W - 1, r))
        if r < l:
            l, r = r, l
        bad[l : r + 1] = True
    return bad


def interp_boundary(boundaries, keep):
    cols = np.arange(len(boundaries), dtype=np.float64)
    if not np.any(keep):
        return boundaries.copy()
    valid_cols = cols[keep]
    valid_vals = boundaries[keep].astype(np.float64)
    return np.interp(cols, valid_cols, valid_vals).astype(np.int32)


def render_overlay(img_rgb, refined_bnd, keep_mask, bad_mask, gt_pts=None):
    H, W = img_rgb.shape[:2]
    canvas = img_rgb.copy()
    if bad_mask is not None and bad_mask.any():
        canvas[:, bad_mask, :] = (
            canvas[:, bad_mask, :] * 0.4 + np.array([255, 0, 0]) * 0.6
        ).astype(np.uint8)
    for c in range(W):
        b = int(refined_bnd[c])
        if 0 <= b < H:
            canvas[b, c] = [0, 255, 255]
            if b + 1 < H:
                canvas[b + 1, c] = [0, 200, 200]
    if gt_pts is not None and len(gt_pts) > 0:
        for c, r in gt_pts:
            ci, ri = int(c), int(r)
            if 0 <= ci < W and 0 <= ri < H:
                canvas[ri, ci] = [0, 255, 0]
    return canvas


def load_meta():
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def save_meta(meta):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    st.set_page_config(layout="wide", page_title="Skyline Refinement Dashboard")
    st.title("Skyline Refinement & Bad-Area Masking")

    if not IMG_DIR.exists():
        st.error(f"Image directory not found: {IMG_DIR}")
        return

    images = sorted(
        [
            p.name
            for p in IMG_DIR.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    )
    if not images:
        st.error("No images found.")
        return

    with open(GT_PATH) as f:
        gt = json.load(f)
    fname_to_key = {Path(k).stem + Path(k).suffix: k for k in gt.keys()}

    # --- Sidebar: sample + zoom ---
    st.sidebar.header("Sample")
    sid_file = st.sidebar.selectbox("Image", images)
    sid = Path(sid_file).stem
    sid_key = fname_to_key.get(sid_file, sid_file.rsplit(".", 1)[0])
    zoom = st.sidebar.slider(
        "Display width (px)", min_value=320, max_value=1080, value=540, step=20
    )

    img_path = IMG_DIR / sid_file
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    H, W = img_rgb.shape[:2]
    _, _, boundaries, keep = canny_skyline(img_rgb)

    if st.session_state.get("sid") != sid:
        st.session_state.sid = sid
        st.session_state.bad_text = ""

    bad_text = st.sidebar.text_area(
        "Bad column ranges (e.g. 100,200; 340,360; 500-520)",
        value=st.session_state.get("bad_text", ""),
        height=80,
        key=f"bad_text_{sid}",
    )
    bad_mask = parse_ranges(bad_text, W)
    st.session_state.bad_text = bad_text

    effective_keep = keep & ~bad_mask
    if np.any(effective_keep):
        refined_bnd = interp_boundary(boundaries, effective_keep)
    else:
        refined_bnd = boundaries.copy()

    # GT
    gt_pts = None
    if sid_key in gt:
        sgt = gt[sid_key]
        if isinstance(sgt, dict):
            for k in ("skyline", "annotated_skyline", "cols_rows"):
                if k in sgt:
                    gt_pts = sgt[k]
                    break

    # --- Layout ---
    col_view, col_ctrl = st.columns([3, 1])
    with col_view:
        st.subheader(f"{sid_file}  ({W}x{H})")
        overlay = render_overlay(img_rgb, refined_bnd, effective_keep, bad_mask, gt_pts)
        # Resize for display
        disp_h = int(H * zoom / W)
        overlay_pil = Image.fromarray(overlay).resize(
            (zoom, disp_h), Image.Resampling.BILINEAR
        )
        st.image(np.array(overlay_pil), width=zoom, channels="RGB")
        # Legend
        st.markdown(
            "<span style='color:cyan'>**cyan**</span>=skyline · "
            "<span style='color:red'>**red**</span>=bad cols · "
            "<span style='color:lime'>**lime**</span>=annotated GT",
            unsafe_allow_html=True,
        )

    with col_ctrl:
        st.subheader("Stats")
        st.write(f"**Valid canny cols**: {int(keep.sum())}/{W}")
        st.write(f"**Bad cols**: {int(bad_mask.sum())}")
        st.write(f"**Effective keep**: {int(effective_keep.sum())}")

        if gt_pts is not None and len(gt_pts) > 0:
            gt_b = np.full(W, -1, dtype=np.int32)
            for c, r in gt_pts:
                ci, ri = int(c), int(r)
                if 0 <= ci < W:
                    gt_b[ci] = ri
            valid = (refined_bnd >= 0) & (gt_b >= 0)
            if valid.any():
                err = float(
                    np.mean(
                        np.abs(
                            refined_bnd[valid].astype(float) - gt_b[valid].astype(float)
                        )
                    )
                )
                st.metric("Mean |refined − GT| (px)", f"{err:.1f}")
            else:
                st.metric("Mean |refined − GT| (px)", "n/a")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear", key="clear"):
                st.session_state.bad_text = ""
                st.rerun()
        with col_b:
            if st.button("Save Refined Skyline", key="save"):
                saved = np.zeros((H, W), dtype=np.uint8)
                for c in range(W):
                    b = int(refined_bnd[c])
                    if 0 <= b < H:
                        saved[b:, c] = 255
                out_path = OUT_MASK_DIR / f"{sid}.png"
                Image.fromarray(saved).save(out_path)
                meta = load_meta()
                meta[sid] = {
                    "image": sid_file,
                    "bad_cols_count": int(bad_mask.sum()),
                    "valid_cols_count": int(keep.sum()),
                    "effective_keep_count": int(effective_keep.sum()),
                    "bad_text": bad_text,
                    "saved_mask": str(out_path.relative_to(REPO_ROOT)),
                }
                save_meta(meta)
                st.success(f"Saved: {out_path.name}")

        st.divider()
        st.subheader("Saved refinements")
        meta = load_meta()
        sids = sorted(meta.keys())
        if sids:
            st.write(
                f"{len(sids)} samples: "
                + ", ".join(sids[:8])
                + (f" … +{len(sids) - 8}" if len(sids) > 8 else "")
            )
        else:
            st.write("(none)")


if __name__ == "__main__":
    main()

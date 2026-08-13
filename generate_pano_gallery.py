#!/usr/bin/env python3
"""Generate an HTML gallery of all GSV panoramas with auto-seg masks."""
import json, os, base64
from pathlib import Path
from PIL import Image
import numpy as np
import io

BASE = Path("/home/admin/SkylineGeolocation/data/street_view")
IMG_DIR = BASE / "images"
MASK_DIR = BASE / "masks"
RESULTS = BASE / "end_to_end_results.json"
OUT_HTML = Path("/home/admin/SkylineGeolocation/pano_gallery.html")

with open(RESULTS) as f:
    data = json.load(f)
results = data["results"]

# Build lookup
entries = []
for r in results:
    pid = r["pano_id"]
    img_path = IMG_DIR / f"{pid}.png"
    mask_path = MASK_DIR / f"{pid}.png"
    
    # Find best auto error
    auto_errs = r.get("auto_errs", {})
    finite_errs = {k: v for k, v in auto_errs.items() if np.isfinite(v)}
    if finite_errs:
        best_scorer = min(finite_errs, key=finite_errs.get)
        best_err = finite_errs[best_scorer]
    else:
        best_scorer = "none"
        best_err = float("inf")
    
    rrf_err = auto_errs.get("rrf", float("inf"))
    
    entries.append({
        "pid": pid,
        "auto_valid": r.get("auto_valid", False),
        "annot_valid": r.get("annot_valid", False),
        "n_crops": r.get("n_crops", 0),
        "sky_ratio": r.get("mean_sky_ratio", 0),
        "confidence": r.get("mean_seg_confidence", 0),
        "auto_cov": r.get("auto_cov", 0),
        "annot_cov": r.get("annot_cov", 0),
        "best_err": best_err,
        "best_scorer": best_scorer,
        "rrf_err": rrf_err,
        "has_img": img_path.exists(),
        "has_mask": mask_path.exists(),
    })

# Sort: finite errors first (ascending), then inf
entries.sort(key=lambda e: (np.isinf(e["best_err"]), e["best_err"]))

def img_to_b64(path, max_w=300):
    img = Image.open(path)
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def mask_to_b64(path, max_w=400):
    img = Image.open(path).convert("L")
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# Stats
n_finite = sum(1 for e in entries if np.isfinite(e["best_err"]))
n_auto = sum(1 for e in entries if e["auto_valid"])
n_annot = sum(1 for e in entries if e["annot_valid"])

html = f"""<!DOCTYPE html>
<html>
<head>
<title>GSV Pano Gallery ({len(entries)} panos)</title>
<style>
body {{ font-family: monospace; background: #111; color: #eee; margin: 20px; }}
h1 {{ color: #fff; }}
.stats {{ background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 14px; }}
.stats b {{ color: #4fc3f7; }}
.pano {{ 
    border: 1px solid #333; 
    border-radius: 8px; 
    margin: 15px 0; 
    padding: 15px; 
    background: #1a1a1a;
    display: flex;
    gap: 15px;
    align-items: flex-start;
}}
.pano.good {{ border-color: #4caf50; }}
.pano.ok {{ border-color: #ff9800; }}
.pano.bad {{ border-color: #f44336; }}
.pano.neither {{ border-color: #555; }}
.pano-images {{ display: flex; gap: 10px; flex-shrink: 0; }}
.pano-images img {{ max-height: 180px; border-radius: 4px; }}
.pano-info {{ flex: 1; font-size: 13px; line-height: 1.6; }}
.pano-info .pid {{ font-size: 11px; color: #999; word-break: break-all; }}
.err {{ font-size: 18px; font-weight: bold; }}
.err.good {{ color: #4caf50; }}
.err.ok {{ color: #ff9800; }}
.err.bad {{ color: #f44336; }}
.err.none {{ color: #666; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin: 2px; }}
.tag-auto {{ background: #1b5e20; color: #a5d6a7; }}
.tag-annot {{ background: #e65100; color: #ffcc80; }}
.tag-crop {{ background: #0d47a1; color: #90caf9; }}
.filter-bar {{ margin: 15px 0; }}
.filter-bar button {{ 
    background: #333; color: #eee; border: 1px solid #555; 
    padding: 6px 14px; border-radius: 4px; cursor: pointer; margin: 3px;
    font-family: monospace; font-size: 13px;
}}
.filter-bar button:hover {{ background: #444; }}
.filter-bar button.active {{ background: #1565c0; border-color: #42a5f5; }}
.hidden {{ display: none !important; }}
</style>
</head>
<body>
<h1>GSV Pano Gallery &mdash; {len(entries)} panoramas</h1>
<div class="stats">
<b>{n_finite}</b> panos with finite errors &bull; 
<b>{n_auto}</b> auto-valid &bull; 
<b>{n_annot}</b> annot-valid &bull;
RRF median: <b>{np.median([e['rrf_err'] for e in entries if np.isfinite(e['rrf_err'])]):.0f}m</b> (of {sum(1 for e in entries if np.isfinite(e['rrf_err']))} with finite RRF)
</div>

<div class="filter-bar">
<button class="active" onclick="filterAll(this)">All ({len(entries)})</button>
<button onclick="filterClass(this, 'good')">< 100m</button>
<button onclick="filterClass(this, 'ok')">100m - 1km</button>
<button onclick="filterClass(this, 'bad')">> 1km</button>
<button onclick="filterClass(this, 'neither')">No finite error</button>
<button onclick="filterAnnot(this)">Annot-valid only</button>
<button onclick="filterAuto(this)">Auto-valid only</button>
</div>
"""

for i, e in enumerate(entries):
    # Classify
    if np.isinf(e["best_err"]):
        cls = "neither"
    elif e["best_err"] < 100:
        cls = "good"
    elif e["best_err"] < 1000:
        cls = "ok"
    else:
        cls = "bad"
    
    err_cls = cls if cls != "neither" else "none"
    
    # Build image HTML
    img_html = ""
    if e["has_img"] and e["has_mask"]:
        img_b64 = img_to_b64(IMG_DIR / f"{e['pid']}.png")
        mask_b64 = mask_to_b64(MASK_DIR / f"{e['pid']}.png")
        img_html = f'<img src="data:image/jpeg;base64,{img_b64}" title="original"><img src="data:image/png;base64,{mask_b64}" title="auto-seg mask">'
    elif e["has_img"]:
        img_b64 = img_to_b64(IMG_DIR / f"{e['pid']}.png")
        img_html = f'<img src="data:image/jpeg;base64,{img_b64}" title="original"><span style="color:#666">no mask</span>'
    else:
        img_html = '<span style="color:#666">no image</span>'
    
    err_str = f"{e['best_err']:.0f}m ({e['best_scorer']})" if np.isfinite(e["best_err"]) else "inf"
    rrf_str = f"{e['rrf_err']:.0f}m" if np.isfinite(e["rrf_err"]) else "inf"
    
    tags = ""
    if e["auto_valid"]:
        tags += '<span class="tag tag-auto">auto-valid</span>'
    if e["annot_valid"]:
        tags += '<span class="tag tag-annot">annot-valid</span>'
    tags += f'<span class="tag tag-crop">{e["n_crops"]} crops</span>'
    
    html += f"""
<div class="pano {cls}" data-err="{e['best_err']}" data-auto="{e['auto_valid']}" data-annot="{e['annot_valid']}">
  <div class="pano-images">{img_html}</div>
  <div class="pano-info">
    <div class="pid">#{i+1} &mdash; {e['pid']}</div>
    <div><span class="err {err_cls}">{err_str}</span> &nbsp; RRF: <span class="err {err_cls}">{rrf_str}</span></div>
    <div>{tags}</div>
    <div>sky ratio: {e['sky_ratio']:.1%} &bull; confidence: {e['confidence']:.3f} &bull; auto coverage: {e['auto_cov']:.0f}° &bull; annot coverage: {e['annot_cov']:.0f}°</div>
  </div>
</div>
"""

html += """
<script>
function filterAll(btn) {
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.pano').forEach(p => p.classList.remove('hidden'));
}
function filterClass(btn, cls) {
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.pano').forEach(p => {
        if (p.classList.contains(cls)) p.classList.remove('hidden');
        else p.classList.add('hidden');
    });
}
function filterAnnot(btn) {
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.pano').forEach(p => {
        if (p.dataset.annot === 'True') p.classList.remove('hidden');
        else p.classList.add('hidden');
    });
}
function filterAuto(btn) {
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.pano').forEach(p => {
        if (p.dataset.auto === 'True') p.classList.remove('hidden');
        else p.classList.add('hidden');
    });
}
</script>
</body>
</html>
"""

OUT_HTML.write_text(html)
print(f"Gallery written to {OUT_HTML} ({len(entries)} panos)")
print(f"  File size: {OUT_HTML.stat().st_size / 1024 / 1024:.1f} MB")

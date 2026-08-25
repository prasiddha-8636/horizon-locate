#!/usr/bin/env python
"""Fast Paginated Web Gallery for GSV Crops with One-Click Delete Buttons.

Launch:  python scripts/build_crop_gallery.py [--port 8766]
Serves on http://127.0.0.1:8766.
"""

import argparse
import base64
import http.server
import json
import os
import socketserver
import sys
import urllib.parse
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation import (
    load_segmentation_model,
    refine_sky_mask_with_guidance,
)

DEFAULT_PORT = 8766
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
MODEL_PATH = ROOT / "data" / "sky_segmentation_unet_model.pth"

MODEL = None


def get_model():
    global MODEL
    if MODEL is None and MODEL_PATH.exists():
        try:
            MODEL = load_segmentation_model(str(MODEL_PATH), "cpu")
        except Exception:
            pass
    return MODEL


def generate_preview_b64(img_path, max_w=400):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = max_w / float(w)
    new_h = int(h * scale)
    img_sm = img.resize((max_w, new_h), Image.Resampling.BILINEAR)
    img_np = np.array(img_sm)

    model = get_model()
    if model is not None:
        import torch
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        sm = cv2.resize(img_np, (256, 256))
        t = transform(Image.fromarray(sm)).unsqueeze(0)
        with torch.no_grad():
            prob = torch.sigmoid(model(t)).squeeze().numpy()
        prob_res = cv2.resize(prob, (max_w, new_h))
        raw_mask = (prob_res <= 0.70).astype(np.uint8)

        # Ignore Canny edges inside white cloud pixels
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        cloud_mask = (hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 35)

        mask = refine_sky_mask_with_guidance(img_np, raw_mask)
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        mask = np.full((new_h, max_w), 255, dtype=np.uint8)
        for c in range(max_w):
            nz = np.where(edges[:, c] > 0)[0]
            if len(nz) > 0:
                mask[: nz[0], c] = 0

    # Extract 1D sky boundary per column (Sky = 0, Terrain = 255)
    pts = []
    for c in range(max_w):
        sky_rows = np.where(mask[:, c] == 0)[0]
        if len(sky_rows) > 0:
            r = sky_rows[-1]  # bottom-most sky row
        else:
            r = 0  # no sky, mountain at top
        pts.append([c, int(r)])

    # Draw 100% SOLID CONTINUOUS 2PX CYAN LINE using cv2.polylines
    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_np, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    buffered = BytesIO()
    Image.fromarray(img_np).save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GSV Crop Gallery & Cleaner</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }
h1 { margin-bottom: 8px; font-size: 24px; color: #fff; }
.subtitle { color: #aaa; margin-bottom: 16px; font-size: 14px; }
.controls { margin-bottom: 20px; display: flex; gap: 15px; align-items: center; background: #1e1e1e; padding: 12px; border-radius: 8px; flex-wrap: wrap; }
input[type="text"] { padding: 8px 12px; border-radius: 4px; border: 1px solid #444; background: #2a2a2a; color: #fff; width: 300px; }
.stat { background: #2a2a2a; padding: 6px 12px; border-radius: 4px; font-size: 13px; color: #ccc; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; }
.card { background: #1e1e1e; border-radius: 8px; overflow: hidden; border: 1px solid #333; display: flex; flex-direction: column; }
.card img { width: 100%; height: auto; display: block; background: #000; min-height: 200px; }
.card-info { padding: 12px; font-size: 12px; flex: 1; display: flex; flex-direction: column; justify-content: space-between; }
.card-title { font-weight: bold; font-family: monospace; color: #64b5f6; margin-bottom: 6px; word-break: break-all; }
.btn-delete { margin-top: 10px; padding: 8px 12px; background: #c62828; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 12px; width: 100%; }
.btn-delete:hover { background: #e53935; }
.btn-nav { padding: 8px 16px; background: #0f3460; color: #fff; border: 1px solid #333; border-radius: 4px; cursor: pointer; font-weight: bold; }
.btn-nav:hover { background: #1a4a8e; }
.cyan { color: #00ffff; font-weight: bold; }
#toast { position: fixed; bottom: 20px; right: 20px; background: #2e7d32; color: #fff; padding: 10px 20px; border-radius: 6px; display: none; z-index: 100; font-size: 13px; }
</style>
</head>
<body>
<h1>GSV Multi-Photo Crop Gallery & Cleaner</h1>
<div class="subtitle">Review and purge bad crops in <code>data/street_view/gsv_crops/</code> | Overlay: <span class="cyan">cyan line</span> = 1D U-Net skyline</div>

<div class="controls">
    <input type="text" id="search" placeholder="Filter by Pano ID or heading (e.g. h90)..." oninput="onSearch()">
    <button class="btn-nav" onclick="changePage(-1)">← Prev</button>
    <span class="stat">Page <span id="page-num">1</span> of <span id="total-pages">1</span></span>
    <button class="btn-nav" onclick="changePage(1)">Next →</button>
    <div class="stat">Total Crops: <span id="count">0</span></div>
</div>

<div class="grid" id="grid">Loading gallery...</div>
<div id="toast"></div>

<script>
let currentPage = 1;
const limit = 60;
let totalCrops = 0;
let totalPages = 1;

async function loadPage(page = 1) {
    const q = encodeURIComponent(document.getElementById('search').value.trim().toLowerCase());
    const resp = await fetch(`/api/crops?page=${page}&limit=${limit}&q=${q}`);
    const data = await resp.json();

    currentPage = data.page;
    totalCrops = data.total;
    totalPages = data.pages;

    document.getElementById('page-num').textContent = currentPage;
    document.getElementById('total-pages').textContent = totalPages;
    document.getElementById('count').textContent = totalCrops;

    renderGrid(data.items);
}

function renderGrid(list) {
    const grid = document.getElementById('grid');
    if (list.length === 0) {
        grid.innerHTML = '<div style="padding: 40px; color: #888;">No crops found.</div>';
        return;
    }
    grid.innerHTML = list.map(item => `
        <div class="card" id="card-${item.filename}">
            <img src="/api/crop_img?filename=${item.filename}" loading="lazy">
            <div class="card-info">
                <div class="card-title">${item.filename}</div>
                <button class="btn-delete" onclick="deleteCrop('${item.filename}')">DELETE CROP</button>
            </div>
        </div>
    `).join('');
}

let searchTimer;
function onSearch() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadPage(1), 300);
}

function changePage(delta) {
    const newPage = currentPage + delta;
    if (newPage >= 1 && newPage <= totalPages) {
        loadPage(newPage);
        window.scrollTo(0, 0);
    }
}

async function deleteCrop(filename) {
    const resp = await fetch('/api/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ filename: filename })
    });
    const result = await resp.json();
    if (result.ok) {
        const card = document.getElementById(`card-${filename}`);
        if (card) card.remove();
        showToast('Deleted: ' + filename);
    } else {
        alert('Error deleting crop');
    }
}

function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 1500);
}

loadPage(1);
</script>
</body>
</html>"""


class GalleryHandler(http.server.BaseHTTPRequestHandler):
    all_filenames = []

    @classmethod
    def update_file_list(cls):
        if CROPS_DIR.exists():
            cls.all_filenames = sorted(
                [p.name for p in CROPS_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            )
        else:
            cls.all_filenames = []

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif parsed.path == "/api/crops":
            page = int(params.get("page", [1])[0])
            limit = int(params.get("limit", [60])[0])
            q = params.get("q", [""])[0].lower()

            self.update_file_list()
            if q:
                filtered = [f for f in self.all_filenames if q in f.lower()]
            else:
                filtered = self.all_filenames

            total = len(filtered)
            pages = max(1, (total + limit - 1) // limit)
            page = max(1, min(page, pages))

            start = (page - 1) * limit
            end = start + limit
            page_items = [{"filename": f} for f in filtered[start:end]]

            res = {
                "page": page,
                "limit": limit,
                "total": total,
                "pages": pages,
                "items": page_items,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode())

        elif parsed.path == "/api/crop_img":
            filename = params.get("filename", [""])[0]
            img_p = CROPS_DIR / filename
            if img_p.exists():
                b64 = generate_preview_b64(img_p)
                img_data = base64.b64decode(b64)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(img_data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/delete":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            res = self._delete_crop(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode())

    def _delete_crop(self, body):
        filename = body.get("filename")
        if not filename:
            return {"ok": False}
        img_p = CROPS_DIR / filename
        meta_p = CROPS_DIR / f"{img_p.stem}.json"

        deleted = False
        if img_p.exists():
            os.remove(img_p)
            deleted = True
        if meta_p.exists():
            os.remove(meta_p)

        # Record filename in deleted_crops_log.json
        log_p = CROPS_DIR / "deleted_crops_log.json"
        deleted_list = []
        if log_p.exists():
            try:
                with open(log_p) as f:
                    deleted_list = json.load(f)
            except Exception:
                deleted_list = []

        if filename not in deleted_list:
            deleted_list.append(filename)
            with open(log_p, "w") as f:
                json.dump(deleted_list, f, indent=2)

        self.update_file_list()
        return {"ok": deleted}

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Fast Paginated Crop Gallery")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print("Loading segmentation model for skyline overlay...")
    get_model()

    print(f"\n==================================================")
    print(f"Gallery running at: http://127.0.0.1:{args.port}")
    print(f"Click the link above to open!")
    print(f"==================================================\n")

    with socketserver.TCPServer(("0.0.0.0", args.port), GalleryHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()

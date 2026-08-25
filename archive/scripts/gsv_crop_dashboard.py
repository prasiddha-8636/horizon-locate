#!/usr/bin/env python
"""Adaptive GSV Crop Dashboard & Storage-Aware Auto-Cropper.

Launch GUI:   python scripts/gsv_crop_dashboard.py [--port 8765]
Batch Mode:   python scripts/gsv_crop_dashboard.py --auto-detect-all --auto-crop-all [--target-crops 3]
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.streetview_utils import slice_perspective

DEFAULT_PORT = 8765
OUT_W, OUT_H = 1080, 720
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"


def get_pano_list(panos_dir):
    panos_dir = Path(panos_dir)
    panos = []
    for p in sorted(list(panos_dir.glob("*.jpg")) + list(panos_dir.glob("*.png"))):
        panos.append({"id": p.stem, "path": str(p)})
    return panos


def is_valid_sky_crop(img, min_sky_frac=0.15, min_relief=2.0):
    """Check that crop has actual sky at top and is not a ground/bush close-up."""
    arr = np.array(img)
    top_hsv = cv2.cvtColor(arr[: int(OUT_H * 0.25), :, :], cv2.COLOR_RGB2HSV)
    sky_pixel_frac = float(((top_hsv[:, :, 2] > 120) & (top_hsv[:, :, 1] < 55)).mean())

    top_row = arr[:5, :, :].mean(axis=(0, 2))
    has_relief = np.std(top_row) >= min_relief

    return (sky_pixel_frac >= min_sky_frac) and has_relief


HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GSV Crop Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; height: 100vh; }
#app { display: flex; height: 100vh; }
#sidebar { width: 280px; background: #16213e; overflow-y: auto; flex-shrink: 0; border-right: 1px solid #333; }
#pano-list { padding: 4px; }
.pano-item { padding: 6px 10px; cursor: pointer; border-radius: 4px; font-size: 12px; font-family: monospace; display: flex; justify-content: space-between; }
.pano-item:hover { background: #1a1a4e; }
.pano-item.active { background: #0f3460; color: #e94560; }
#viewer { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#top-bar { display: flex; align-items: center; padding: 8px 16px; background: #16213e; border-bottom: 1px solid #333; gap: 16px; font-size: 13px; }
#top-bar button { background: #0f3460; border: 1px solid #333; color: #eee; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; }
#top-bar button.save { background: #e94560; border-color: #e94560; }
#canvas-row { flex: 1; display: flex; overflow: hidden; }
#pano-container { flex: 1; position: relative; overflow: hidden; cursor: grab; }
#pano-canvas { position: absolute; top: 0; left: 0; }
#preview-panel { width: 360px; background: #16213e; border-left: 1px solid #333; display: flex; flex-direction: column; padding: 12px; }
#preview-canvas { width: 100%; border: 1px solid #333; border-radius: 4px; background: #000; }
#toast { position: fixed; bottom: 20px; right: 20px; background: #0f3460; color: #eee; padding: 10px 20px; border-radius: 6px; font-size: 13px; display: none; z-index: 100; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <input type="text" id="search" placeholder="Search panoramas..." oninput="filterPanos()">
    <div id="pano-list"></div>
  </div>
  <div id="viewer">
    <div id="top-bar">
      <span id="pano-id" class="info">Select a panorama</span>
      <button onclick="resetView()">Reset View</button>
      <button class="auto" onclick="autoCrop()">Auto Crop</button>
      <button class="save" onclick="saveCrop()">Save Crop</button>
    </div>
    <div id="canvas-row">
      <div id="pano-container">
        <canvas id="pano-canvas"></canvas>
      </div>
      <div id="preview-panel">
        <h3>PERSPECTIVE PREVIEW</h3>
        <canvas id="preview-canvas" width="360" height="240"></canvas>
      </div>
    </div>
  </div>
</div>
<div id="toast"></div>

<script>
let panos = []; let currentPano = null; let savedCrops = {};
let heading = 0, pitch = 0, fovY = 65, dragging = false, dragStartX = 0, dragStartHeading = 0;
let panoImg = null;

const panoCanvas = document.getElementById('pano-canvas');
const panoCtx = panoCanvas.getContext('2d');
const container = document.getElementById('pano-container');

async function init() {
    const resp = await fetch('/api/panos'); panos = await resp.json();
    const stateResp = await fetch('/api/state'); savedCrops = await stateResp.json();
    renderPanoList(panos);
    if (panos.length > 0) selectPano(0);
}

function renderPanoList(list) {
    const el = document.getElementById('pano-list');
    el.innerHTML = list.map(p => `
        <div class="pano-item ${currentPano && currentPano.id === p.id ? 'active' : ''}" onclick="selectPanoById('${p.id}')">
            <span>${p.id.substring(0, 20)}</span>
            <span class="count">${savedCrops[p.id] || 0} saved</span>
        </div>
    `).join('');
}

function selectPano(idx) { selectPanoById(panos[idx].id); }

function selectPanoById(id) {
    currentPano = panos.find(p => p.id === id);
    if (!currentPano) return;
    document.getElementById('pano-id').textContent = currentPano.id;
    panoImg = new Image();
    panoImg.onload = () => { resizePanoCanvas(); drawPano(); };
    panoImg.src = '/api/pano/' + currentPano.id + '.jpg';
    renderPanoList(panos);
}

function resizePanoCanvas() {
    const rect = container.getBoundingClientRect();
    panoCanvas.width = rect.width * devicePixelRatio;
    panoCanvas.height = (rect.width / 2) * devicePixelRatio;
    panoCanvas.style.width = rect.width + 'px';
    panoCanvas.style.height = (rect.width / 2) + 'px';
    panoCtx.scale(devicePixelRatio, devicePixelRatio);
}

function drawPano() {
    if (!panoImg) return;
    const w = panoCanvas.width / devicePixelRatio, h = panoCanvas.height / devicePixelRatio;
    panoCtx.clearRect(0, 0, w, h);
    panoCtx.drawImage(panoImg, 0, 0, w, h);
}

function filterPanos() {
    const q = document.getElementById('search').value.toLowerCase();
    renderPanoList(panos.filter(p => p.id.toLowerCase().includes(q)));
}

async function saveCrop() {
    if (!currentPano) return;
    const resp = await fetch('/api/crop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ pano_id: currentPano.id, heading_deg: heading, pitch_deg: pitch, fov_y_deg: fovY })
    });
    const result = await resp.json();
    if (result.ok) {
        savedCrops[currentPano.id] = (savedCrops[currentPano.id] || 0) + 1;
        showToast('Saved crop');
        renderPanoList(panos);
    }
}

async function autoCrop() {
    if (!currentPano) return;
    showToast('Auto-cropping adaptive...');
    const resp = await fetch('/api/auto-crop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ pano_id: currentPano.id, target_crops: 3 })
    });
    const result = await resp.json();
    if (result.ok) {
        showToast(`Saved ${result.total} crops`);
        savedCrops[currentPano.id] = result.total;
        renderPanoList(panos);
    }
}

function showToast(msg) {
    const el = document.getElementById('toast');
    el.textContent = msg; el.style.display = 'block';
    setTimeout(() => el.style.display = 'none', 2000);
}

function resetView() { heading = 0; pitch = 0; fovY = 65; drawPano(); }

init();
</script>
</body>
</html>"""


class PanoHandler(http.server.BaseHTTPRequestHandler):
    panos_dir = None
    state_path = None

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/api/panos":
            panos = get_pano_list(self.panos_dir)
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(panos).encode())
        elif self.path == "/api/state":
            state = self._load_state()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(state).encode())
        elif self.path.startswith("/api/pano/"):
            pano_file = self.path.split("/")[-1]
            pano_path = Path(self.panos_dir) / pano_file
            if pano_path.exists():
                self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers()
                with open(pano_path, "rb") as f: self.wfile.write(f.read())
            else:
                self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/api/crop":
            res = self._save_crop(body)
        elif self.path == "/api/auto-crop":
            res = self._auto_crop(body)
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(res).encode())

    def _load_state(self):
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path) as f: return json.load(f)
        return {}

    def _save_state(self, state):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f: json.dump(state, f, indent=2)

    def _forbidden_path(self):
        return Path(os.path.dirname(self.state_path)) / "forbidden_regions.json"

    def _load_forbidden(self):
        p = self._forbidden_path()
        if p.exists():
            with open(p) as f: return json.load(f)
        return {}

    @staticmethod
    def detect_bad_regions(pano_path):
        import cv2
        try:
            pil_img = Image.open(pano_path).convert("RGB")
        except Exception:
            return [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "source": "corrupted_file"}]
        img_rgb = np.array(pil_img)
        Hp, Wp, _ = img_rgb.shape
        scale = 832 / Wp if Wp > 832 else 1.0
        img_sm = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img_rgb
        Hs, Ws, _ = img_sm.shape
        top_h = int(Hs * 0.60)
        top_rgb = img_sm[:top_h, :, :]

        hsv = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2HSV)
        cloud_pixels = (hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 35)
        cloud_col_frac = cloud_pixels.mean(axis=0)

        gray_top = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray_top, (3, 3), 0), 40, 120)
        edge_mask = edges > 0
        edge_mask[: int(top_h * 0.12), :] = False

        bad = (cloud_col_frac > 0.45) | (~edge_mask.any(axis=0))
        regions = []
        in_run, start = False, 0
        for c in range(Ws + 1):
            is_bad = c < Ws and bad[c]
            if is_bad and not in_run: in_run, start = True, c
            elif not is_bad and in_run:
                in_run = False
                if c - start >= int(Ws * 0.03):
                    regions.append({"x1": round(start / Ws, 4), "y1": 0.0, "x2": round(c / Ws, 4), "y2": 1.0, "source": "auto_detect"})
        return regions

    def _auto_crop(self, body):
        pano_id = body["pano_id"]
        target_crops = int(body.get("target_crops", 3))

        pano_path = Path(self.panos_dir) / f"{pano_id}.jpg"
        if not pano_path.exists():
            for ext in [".png", ".jpeg"]:
                p = Path(self.panos_dir) / f"{pano_id}{ext}"
                if p.exists(): pano_path = p; break

        if not pano_path.exists(): return {"ok": False, "error": "Pano not found"}

        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        forbidden = self._load_forbidden().get(pano_id, [])

        def heading_in_forbidden(h_deg):
            for r in forbidden:
                if min(r["x1"], r["x2"]) <= (h_deg / 360.0) <= max(r["x1"], r["x2"]):
                    return True
            return False

        # Count existing valid crops
        existing = list(CROPS_DIR.glob(f"{pano_id}_*.png"))
        saved_count = len(existing)

        if saved_count >= target_crops:
            return {"ok": True, "total": saved_count}

        # Adaptive step fallback: try 90 deg -> 45 deg -> 30 deg -> 15 deg
        attempted_headings = set()
        for step_deg in [90.0, 45.0, 30.0, 15.0]:
            heading = 0.0
            while heading < 360.0:
                if heading not in attempted_headings:
                    attempted_headings.add(heading)
                    if not heading_in_forbidden(heading):
                        try:
                            img = slice_perspective(str(pano_path), heading_deg=heading, pitch_deg=0.0, fov_y_deg=65.0, out_w=OUT_W, out_h=OUT_H)
                        except Exception as e:
                            heading += step_deg
                            continue
                        # Ground / sky filter
                        if is_valid_sky_crop(img, min_sky_frac=0.15, min_relief=2.0):
                            saved_count += 1
                            fname = f"{pano_id}_h{heading:.0f}_p0_n{saved_count}.png"
                            img.save(CROPS_DIR / fname)
                            meta = {"pano_id": pano_id, "heading_deg": heading, "pitch_deg": 0.0, "fov_y_deg": 65.0, "filename": fname}
                            with open(CROPS_DIR / f"{pano_id}_h{heading:.0f}_p0_n{saved_count}.json", "w") as f:
                                json.dump(meta, f, indent=2)

                            if saved_count >= target_crops:
                                state = self._load_state()
                                state[pano_id] = saved_count
                                self._save_state(state)
                                return {"ok": True, "total": saved_count}
                heading += step_deg

        state = self._load_state()
        state[pano_id] = saved_count
        self._save_state(state)
        return {"ok": True, "total": saved_count}

    def _save_crop(self, body):
        pano_id, heading_deg = body["pano_id"], float(body["heading_deg"])
        pano_path = Path(self.panos_dir) / f"{pano_id}.jpg"
        if not pano_path.exists():
            for ext in [".png", ".jpeg"]:
                p = Path(self.panos_dir) / f"{pano_id}{ext}"
                if p.exists(): pano_path = p; break

        img = slice_perspective(str(pano_path), heading_deg=heading_deg, pitch_deg=0.0, fov_y_deg=65.0, out_w=OUT_W, out_h=OUT_H)
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        state = self._load_state()
        n = state.get(pano_id, 0) + 1
        state[pano_id] = n
        self._save_state(state)

        fname = f"{pano_id}_h{heading_deg:.0f}_p0_n{n}.png"
        img.save(CROPS_DIR / fname)
        meta = {"pano_id": pano_id, "heading_deg": heading_deg, "pitch_deg": 0.0, "fov_y_deg": 65.0, "filename": fname}
        with open(CROPS_DIR / f"{pano_id}_h{heading_deg:.0f}_p0_n{n}.json", "w") as f:
            json.dump(meta, f, indent=2)
        return {"ok": True, "filename": fname}

    @classmethod
    def auto_crop_batch(cls, pano_id, panos_dir, state_path, target_crops=3):
        dummy_self = cls.__new__(cls)
        dummy_self.panos_dir = panos_dir
        dummy_self.state_path = state_path
        return dummy_self._auto_crop({"pano_id": pano_id, "target_crops": target_crops})

    def log_message(self, format, *args): pass


def main():
    parser = argparse.ArgumentParser(description="Adaptive GSV Crop Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--panos-dir", type=str, default=str(ROOT / "data" / "street_view" / "panos"))
    parser.add_argument("--auto-detect-all", action="store_true", help="Batch detect bad/cloudy zones in all panos")
    parser.add_argument("--auto-crop-all", action="store_true", help="Batch adaptive crop across all panos")
    parser.add_argument("--target-crops", type=int, default=3, help="Target good crops per pano (default: 3)")
    args = parser.parse_args()

    PanoHandler.panos_dir = args.panos_dir
    PanoHandler.state_path = str(CROPS_DIR / "crop_state.json")

    panos = get_pano_list(args.panos_dir)

    if args.auto_detect_all:
        import tqdm
        forbidden_path = Path(os.path.dirname(PanoHandler.state_path)) / "forbidden_regions.json"
        forbidden = {}
        if forbidden_path.exists():
            with open(forbidden_path) as f: forbidden = json.load(f)

        print(f"Auto-detecting bad/cloudy areas across {len(panos)} panoramas...")
        for p in tqdm.tqdm(panos, desc="Detecting bad skylines"):
            pano_id = p["id"]
            pano_path = Path(p["path"])
            try:
                if pano_path.exists():
                    detected = PanoHandler.detect_bad_regions(pano_path)
                    manual = [r for r in forbidden.get(pano_id, []) if r.get("source") != "auto_detect"]
                    forbidden[pano_id] = manual + detected
            except Exception: forbidden[pano_id] = []

        os.makedirs(os.path.dirname(forbidden_path), exist_ok=True)
        with open(forbidden_path, "w") as f: json.dump(forbidden, f, indent=2)

    if args.auto_crop_all:
        import tqdm
        print(f"\nAdaptive auto-cropping target={args.target_crops} good crops per pano...")
        total_crops = 0
        for p in tqdm.tqdm(panos, desc="Adaptive auto-cropping"):
            res = PanoHandler.auto_crop_batch(p["id"], PanoHandler.panos_dir, PanoHandler.state_path, target_crops=args.target_crops)
            if res.get("ok"): total_crops += res.get("total", 0)
        print(f"\nSuccessfully created/maintained {total_crops} good crops in {CROPS_DIR}!")

    if args.auto_detect_all or args.auto_crop_all:
        return

    print(f"Dashboard running on: http://localhost:{args.port}")
    with socketserver.TCPServer(("", args.port), PanoHandler) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("\nStopped.")


if __name__ == "__main__":
    main()

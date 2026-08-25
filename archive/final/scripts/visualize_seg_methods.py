"""Fixed Side-by-Side Visual Comparison."""

import json, sys, cv2, torch, numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation import load_segmentation_model, refine_sky_mask_with_guidance

DATA_DIR = ROOT / "data" / "street_view"
OUT_DIR = DATA_DIR / "vis_compare"
MODEL_PATH = ROOT / "data" / "sky_segmentation_unet_model.pth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "annotations.json") as f:
    annots = json.load(f)["annotations"]

model = load_segmentation_model(str(MODEL_PATH), "cpu")

import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for sid in sorted(annots.keys()):
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = DATA_DIR / "images" / f"{sid}{ext}"
        if p.exists(): img_path = p; break
    if not img_path: continue

    orig_img = Image.open(img_path).convert("RGB")
    W, H = orig_img.size
    img_np = np.array(orig_img)

    # 1. Canny Panel (Left)
    vis_canny = img_np.copy()
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 150)
    for c in range(W):
        nz = np.where(edges[:, c] > 0)[0]
        if len(nz) > 0:
            r = nz[0]
            vis_canny[max(0, r-1):min(H, r+2), c] = [0, 255, 255] # Cyan = Canny

    for c, r in annots[sid]:
        ci, ri = int(c), int(r)
        if 0 <= ci < W and 0 <= ri < H:
            vis_canny[max(0, ri-1):min(H, ri+2), ci] = [0, 255, 0] # Lime = GT

    # 2. Fixed UNet Panel (Right)
    vis_unet = img_np.copy()
    tensor_img = transform(orig_img).unsqueeze(0)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor_img)).squeeze().numpy()

    # Invert threshold so sky=1, terrain=0
    prob_res = cv2.resize(output, (W, H), interpolation=cv2.INTER_LINEAR)
    raw_mask = (prob_res <= 0.50).astype(np.uint8) # sky=1

    refined_mask = refine_sky_mask_with_guidance(img_np, raw_mask)

    col_nz = (refined_mask != 0) # Terrain is 255
    for c in range(W):
        nz = np.where(col_nz[:, c])[0]
        if len(nz) > 0:
            r = nz[0] - 1
            vis_unet[max(0, r-1):min(H, r+2), c] = [255, 0, 255] # Magenta = U-Net

    for c, r in annots[sid]:
        ci, ri = int(c), int(r)
        if 0 <= ci < W and 0 <= ri < H:
            vis_unet[max(0, ri-1):min(H, ri+2), ci] = [0, 255, 0] # Lime = GT

    combined = np.hstack((vis_canny, vis_unet))
    out_p = OUT_DIR / f"{sid}_side_by_side.jpg"
    Image.fromarray(combined).save(out_p)
    print(f"Saved: {out_p.name}")

print(f"\nRe-generated side-by-side images in {OUT_DIR}")

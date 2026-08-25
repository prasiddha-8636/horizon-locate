"""Dual-head U-Net retraining (mask + per-column confidence).

Corrected version of the Colab notebook logic:
- conf head: 16ch decoder output -> per-column confidence at FULL width (old code
  used stride-32 head -> width 8 vs 256-wide GT = shape crash)
- conf GT: 1.0 default; cloud band that covers the ridge -> 0 for those columns
- HFlip-safe: conf carried through albumentations as additional_targets mask
- cloud augment: band placed over sky+ridge (matches GSV cloud failure mode)

Conventions match src/segmentation.py UnifiedDatasetAug:
mask 0=sky 1=terrain, images BGR->RGB via cv2.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DualHeadUnet(nn.Module):
    """SMP Unet + per-column confidence head on decoder features (full res).

    Uses a forward hook on segmentation_head to capture decoder output,
    version-agnostic across SMP 0.4.x and 0.5.x.
    """

    def __init__(self, encoder_name="tu-mobilenetv3_large_100", in_channels=3):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
        # decoder final block outputs 16ch at full input resolution
        self.conf_head = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self._dec = None
        self._hook = self.unet.segmentation_head.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._dec = inp[0]

    def forward(self, x):
        mask_logits = self.unet(x)
        conf = torch.sigmoid(self.conf_head(self._dec)).squeeze(1).mean(dim=1)  # (B,W)
        return mask_logits, conf


class DualHeadDataset(torch.utils.data.Dataset):
    def __init__(
        self, imgs, masks, is_train=True, input_size=256, cloud_dir=None, cloud_prob=0.5
    ):
        self.imgs = [str(p) for p in imgs]
        self.masks = [str(p) for p in masks]
        self.input_size = input_size
        self.cloud_prob = cloud_prob if is_train else 0.0
        self.cloud_files = []
        if cloud_dir and os.path.isdir(cloud_dir):
            self.cloud_files = sorted(
                os.path.join(cloud_dir, f)
                for f in os.listdir(cloud_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        print(f"  Cloud overlay aug: {len(self.cloud_files)} cloud images")

        if is_train:
            self.transform = A.Compose(
                [
                    A.RandomResizedCrop((input_size, input_size), scale=(0.7, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ],
                additional_targets={"conf_mask": "mask"},
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(input_size, input_size),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2(),
                ],
                additional_targets={"conf_mask": "mask"},
            )

    def __len__(self):
        return len(self.imgs)

    def _cloud_band(self, img, mask, conf):
        """Blend a cloud band over sky+ridge; zero conf where it covers ridge."""
        if not self.cloud_files or random.random() > self.cloud_prob:
            return img, conf
        h, w = img.shape[:2]
        cloud = cv2.imread(random.choice(self.cloud_files), cv2.IMREAD_COLOR)
        if cloud is None:
            return img, conf
        cloud = cv2.cvtColor(cloud, cv2.COLOR_BGR2RGB)
        if cloud.shape[:2] != (h, w):
            cloud = cv2.resize(cloud, (w, h), interpolation=cv2.INTER_AREA)
        # terrain top row per column (ridge position)
        cols = np.arange(w)
        idxs = np.where(mask == 1)
        if len(idxs[0]) == 0:
            return img, conf
        ridge_row = np.full(w, h, dtype=int)
        # per-column min row of terrain
        for c in cols:
            rows = idxs[0][idxs[1] == c]
            if len(rows):
                ridge_row[c] = rows.min()
        ridge_med = int(np.median(ridge_row[ridge_row < h]))
        if ridge_med <= 4:
            return img, conf
        # band centered near the ridge (cloud bank in front of mountain)
        center = min(h - 1, max(0, ridge_med + random.randint(-h // 8, h // 8)))
        y0 = max(0, center - random.randint(h // 10, h // 4))
        y1 = min(h, center + random.randint(0, h // 8))
        alpha = random.uniform(0.4, 0.85)
        img[y0:y1] = (1 - alpha) * img[y0:y1].astype(np.float32) + alpha * cloud[
            y0:y1
        ].astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # columns where cloud covers the ridge -> untrustworthy boundary
        conf[ridge_row < y1] = 0.0
        return img, conf

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = (mask > 10).astype(np.float32)
        h, w = img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        conf = np.ones(w, dtype=np.float32)
        img, conf = self._cloud_band(img, mask, conf)
        conf_2d = np.repeat(conf[None, :], h, axis=0)  # (h,w) for albumentations
        aug = self.transform(image=img, mask=mask, conf_mask=conf_2d)
        conf_out = aug["conf_mask"][0].float()  # (W,) after crop+flip
        return aug["image"], aug["mask"].unsqueeze(0), conf_out


def build_loaders(
    geopose_dir,
    syn_img_dir,
    syn_mask_dir,
    cloud_dir,
    batch_size=8,
    input_size=256,
    cloud_prob=0.5,
):
    try:
        from src.segmentation import load_geopose_split
    except ImportError:

        def _find_photo_path(folder_path):
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                p = os.path.join(folder_path, f"photo{ext}")
                if os.path.exists(p):
                    return p
            return None

        def load_geopose_split(split_file, base_dir):
            images, masks = [], []
            publish_dir = os.path.join(base_dir, "geoPose3K_final_publish")
            if not os.path.exists(split_file) or not os.path.isdir(publish_dir):
                print(f"Warning: split/publish missing: {split_file} {publish_dir}")
                return images, masks
            existing = set(os.listdir(publish_dir))
            with open(split_file) as f:
                folders = [line.strip().strip("/") for line in f if line.strip()]
            for folder in folders:
                if folder not in existing:
                    continue
                fp = os.path.join(publish_dir, folder)
                img_p = os.path.join(fp, "photo.jpg")
                if not os.path.exists(img_p):
                    img_p = _find_photo_path(fp)
                mask_p = os.path.join(fp, "pinhole", "labels_crop.png")
                if img_p and os.path.exists(mask_p):
                    images.append(img_p)
                    masks.append(mask_p)
            return images, masks

    def split_path(name):
        p = os.path.join(geopose_dir, name)
        if os.path.exists(p):
            return p
        p = os.path.join(geopose_dir, "splits", name)
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f"split not found: {name}")

    train_images, train_masks = [], []
    val_images, val_masks = [], []
    tr_img, tr_mask = load_geopose_split(
        split_path("geoPose3K_final_train.txt"), geopose_dir
    )
    va_img, va_mask = load_geopose_split(
        split_path("geoPose3K_final_val.txt"), geopose_dir
    )
    train_images += tr_img
    train_masks += tr_mask
    val_images += va_img
    val_masks += va_mask
    if os.path.isdir(syn_img_dir):
        syn = sorted(f for f in os.listdir(syn_img_dir) if f.lower().endswith(".png"))
        n = len(syn)
        print(f"  {n} synthetic samples")
        for f in syn[: int(n * 0.8)]:
            train_images.append(os.path.join(syn_img_dir, f))
            train_masks.append(os.path.join(syn_mask_dir, f))
        for f in syn[int(n * 0.8) :]:
            val_images.append(os.path.join(syn_img_dir, f))
            val_masks.append(os.path.join(syn_mask_dir, f))
    print(f"  train: {len(train_images)}, val: {len(val_images)}")

    train_ds = DualHeadDataset(
        train_images,
        train_masks,
        is_train=True,
        input_size=input_size,
        cloud_dir=cloud_dir,
        cloud_prob=cloud_prob,
    )
    val_ds = DualHeadDataset(
        val_images,
        val_masks,
        is_train=False,
        input_size=input_size,
        cloud_dir=cloud_dir,
        cloud_prob=0.0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    p = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (union + 1)
    return bce + dice.mean()


def train_epoch(model, loader, opt, device, conf_w=1.0):
    model.train()
    tot = tot_mask = tot_conf = 0.0
    for imgs, masks, confs in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        confs = confs.to(device, non_blocking=True)
        opt.zero_grad()
        logits, conf_pred = model(imgs)
        mask_loss = bce_dice_loss(logits, masks)
        conf_loss = F.binary_cross_entropy(conf_pred, confs)
        loss = mask_loss + conf_w * conf_loss
        loss.backward()
        opt.step()
        tot += loss.item()
        tot_mask += mask_loss.item()
        tot_conf += conf_loss.item()
    n = len(loader)
    return tot / n, tot_mask / n, tot_conf / n


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    tot_mask = tot_conf = 0.0
    for imgs, masks, confs in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        confs = confs.to(device, non_blocking=True)
        logits, conf_pred = model(imgs)
        tot_mask += bce_dice_loss(logits, masks).item()
        tot_conf += F.binary_cross_entropy(conf_pred, confs).item()
    n = len(loader)
    return tot_mask / n, tot_conf / n


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--geopose", default="/content/data/geopose3k")
    ap.add_argument("--syn-img", default="/content/data/synthetic_dataset/images")
    ap.add_argument("--syn-mask", default="/content/data/synthetic_dataset/masks")
    ap.add_argument("--clouds", default="/content/data/clouds")
    ap.add_argument("--out", default="/content/sky_segmentation_dualhead.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cloud-prob", type=float, default=0.5)
    ap.add_argument(
        "--init",
        default=None,
        help="old sky_segmentation_unet_model.pth for finetune init (matched keys)",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| cuda avail:", torch.cuda.is_available())

    train_loader, val_loader = build_loaders(
        args.geopose,
        args.syn_img,
        args.syn_mask,
        args.clouds,
        batch_size=args.batch,
        cloud_prob=args.cloud_prob,
    )
    model = DualHeadUnet().to(device)

    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=device)
        clean = {
            k.replace("module.", "").replace("model.", ""): v for k, v in ckpt.items()
        }
        sd = model.state_dict()
        matched = {k: v for k, v in clean.items() if k in sd and sd[k].shape == v.shape}
        sd.update(matched)
        model.load_state_dict(sd)
        print(
            f"Finetune init from {args.init}: {len(matched)}/{len(sd)} keys transferred"
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    for ep in range(args.epochs):
        t, tm, tc = train_epoch(model, train_loader, opt, device)
        vm, vc = val_epoch(model, val_loader, device)
        sched.step()
        print(
            f"[{ep + 1}/{args.epochs}] train {t:.4f} (m {tm:.4f} c {tc:.4f}) | "
            f"val m {vm:.4f} c {vc:.4f}"
        )
        if vm < best:
            best = vm
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out} (val mask {vm:.4f})")


if __name__ == "__main__":
    main()

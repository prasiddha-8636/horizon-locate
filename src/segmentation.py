"""Sky segmentation utilities using U-Net and OpenCV Canny-edge guidance."""
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from PIL import Image


def refine_sky_mask_with_guidance(img_np, raw_unet_mask):
    """
    Refines raw U-Net sky mask using OpenCV Canny edges to snap 
    boundaries to exact ridgelines, removing snow/rock false positives.
    """
    H, W = raw_unet_mask.shape
    
    # 1. Connected components to keep only sky touching the top boundary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_unet_mask, connectivity=8)
    top_sky = np.zeros_like(raw_unet_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_TOP] == 0 and stats[i, cv2.CC_STAT_AREA] > 100:
            top_sky[labels == i] = 1

    # 2. Extract sharp structural lines (Canny edges)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    refined = np.zeros_like(top_sky)
    for col in range(W):
        sky_rows = np.where(top_sky[:, col] == 1)[0]
        if len(sky_rows) == 0:
            continue
        boundary = sky_rows[-1]
        
        # Snap boundary to the closest Canny edge in a 15px search window
        search_min = max(0, boundary - 15)
        search_max = min(H - 1, boundary + 15)
        edge_rows = np.where(edges[search_min:search_max, col] == 255)[0]
        
        if len(edge_rows) > 0:
            closest_edge = edge_rows[np.argmin(np.abs(edge_rows - (boundary - search_min)))]
            boundary = search_min + closest_edge
            
        refined[:boundary + 1, col] = 1
        
    # Standard convention: Sky = 0 (Black), Terrain = 255 (White)
    return np.where(refined == 1, 0, 255).astype(np.uint8)


def load_segmentation_model(model_path, device):
    """Loads and returns the trained SMP U-Net model."""
    model = smp.Unet(
        encoder_name="tu-mobilenetv3_large_100",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    checkpoint = torch.load(model_path, map_location=device)
    clean_state = {k.replace("module.", "").replace("model.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(clean_state)
    return model.to(device).eval()


def segment_image(model, img_path, mask_output_path, device):
    """Processes a single image, refines it, and saves the resulting sky mask."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    orig_img = Image.open(img_path).convert("RGB")
    W, H = orig_img.size
    
    # 1. Run U-Net inference
    tensor_img = transform(orig_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor_img)).squeeze().cpu().numpy()
        
    # 2. Resize raw probability map back to original aspect ratio
    prob_resized = np.array(Image.fromarray((output * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)) / 255.0
    raw_mask = (prob_resized <= 0.5).astype(np.uint8)
    
    # 3. Apply edge guidance and save
    refined = refine_sky_mask_with_guidance(np.array(orig_img), raw_mask)
    Image.fromarray(refined).save(mask_output_path)


# =============================================================================
# Training utilities
# =============================================================================

import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class UnifiedDatasetAug(Dataset):
    """Albumentations-augmented dataset combining GeoPose3K and synthetic images."""

    def __init__(self, imgs, masks, is_train=True, train_transform=None):
        self.imgs = [str(p) for p in imgs]
        self.masks = [str(p) for p in masks]
        self.is_train = is_train

        if self.is_train:
            self.transform = train_transform if train_transform is not None else A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                # Spatial (varied camera angles and zoom)
                A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-15, 15), p=0.6),
                A.GridDistortion(p=0.3),
                # Color/lighting (mountain shadows, sunset, glare)
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.CLAHE(clip_limit=4.0, p=0.3),
                # Noise/blur (atmospheric haze, motion blur, sensor grain)
                A.GaussNoise(p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 10).astype(np.float32)
        # Resolve any aspect-ratio mismatches (e.g. GeoPose3K pinhole crops)
        h, w, _ = img.shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)


def find_photo_path(folder_path):
    """Return the first photo file found in a GeoPose3K sample folder."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(folder_path, f"photo{ext}")
        if os.path.exists(p):
            return p
    return None


def load_geopose_split(split_file, base_dir):
    """Load image/mask path lists from a GeoPose3K split text file."""
    images, masks = [], []
    if not os.path.exists(split_file):
        print(f"Warning: Split file {split_file} not found.")
        return images, masks
    publish_dir = os.path.join(base_dir, "geoPose3K_final_publish")
    if not os.path.exists(publish_dir):
        print(f"Warning: Publish directory {publish_dir} not found.")
        return images, masks
    # Single listing — 1000x faster than sequential os.path.exists calls
    existing_folders = set(os.listdir(publish_dir))
    with open(split_file) as f:
        folder_names = [line.strip().strip("/") for line in f if line.strip()]
    for folder in folder_names:
        if folder not in existing_folders:
            continue
        folder_path = os.path.join(publish_dir, folder)
        img_p = os.path.join(folder_path, "photo.jpg")
        if not os.path.exists(img_p):
            img_p = find_photo_path(folder_path)
        mask_p = os.path.join(folder_path, "pinhole", "labels_crop.png")
        if img_p and os.path.exists(mask_p):
            images.append(img_p)
            masks.append(mask_p)
    return images, masks


def build_training_loaders(geopose_dir, syn_img_dir, syn_mask_dir, batch_size=8, train_transform=None):
    """Build combined GeoPose3K + synthetic train/val DataLoaders."""
    train_images, train_masks = [], []
    val_images,   val_masks   = [], []

    print("Loading GeoPose3K dataset files...")
    tr_img, tr_mask = load_geopose_split(
        os.path.join(str(geopose_dir), "geoPose3K_final_train.txt"), str(geopose_dir))
    va_img, va_mask = load_geopose_split(
        os.path.join(str(geopose_dir), "geoPose3K_final_val.txt"),   str(geopose_dir))
    train_images.extend(tr_img);  train_masks.extend(tr_mask)
    val_images.extend(va_img);    val_masks.extend(va_mask)

    print("Checking for synthetic dataset...")
    syn_img_dir, syn_mask_dir = str(syn_img_dir), str(syn_mask_dir)
    if os.path.exists(syn_img_dir):
        all_syn = sorted(f for f in os.listdir(syn_img_dir) if f.lower().endswith(".png"))
        n = len(all_syn)
        if n > 0:
            print(f"  Found {n} synthetic samples.")
            split = int(n * 0.8)
            for f in all_syn[:split]:
                train_images.append(os.path.join(syn_img_dir, f))
                train_masks.append(os.path.join(syn_mask_dir, f))
            for f in all_syn[split:]:
                val_images.append(os.path.join(syn_img_dir, f))
                val_masks.append(os.path.join(syn_mask_dir, f))
        else:
            print("  Warning: Synthetic directory is empty.")
    else:
        print("  Warning: Synthetic directory not found. Training on GeoPose3K only.")

    train_loader = DataLoader(
        UnifiedDatasetAug(train_images, train_masks, is_train=True, train_transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        UnifiedDatasetAug(val_images, val_masks, is_train=False),
        batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"  Train samples: {len(train_images)} | Val samples: {len(val_images)}")
    return train_loader, val_loader


def build_sky_model(device):
    """Instantiate the MobileNetV3-backed U-Net for training."""
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="tu-mobilenetv3_large_100",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)


def compute_iou(pred_logits, true_masks, threshold=0.5):
    """Batch IoU from raw logits."""
    import torch
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def bce_dice_loss(pred, target):
    """Combined BCE + soft Dice loss."""
    import torch
    bce = nn.BCEWithLogitsLoss()(pred, target)
    smooth = 1e-6
    probs = torch.sigmoid(pred)
    intersection = (probs * target).sum()
    dice = 1.0 - (2.0 * intersection + smooth) / (probs.sum() + target.sum() + smooth)
    return bce + dice


def train_sky_model(model, train_loader, val_loader, device, save_path, epochs=15, lr=2e-4):
    """Full training loop. Returns (train_losses, val_losses, train_ious, val_ious, lrs)."""
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, train_ious = [], []
    val_losses,   val_ious   = [], []
    lrs = []
    best_val_iou = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = total_iou = 0.0
        lrs.append(optimizer.param_groups[0]["lr"])

        for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = bce_dice_loss(out, msks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_iou  += compute_iou(out, msks)

        train_losses.append(total_loss / len(train_loader))
        train_ious.append(total_iou  / len(train_loader))
        scheduler.step()

        model.eval()
        v_loss = v_iou = 0.0
        with torch.no_grad():
            for imgs, msks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                imgs, msks = imgs.to(device), msks.to(device)
                out = model(imgs)
                v_loss += bce_dice_loss(out, msks).item()
                v_iou  += compute_iou(out, msks)

        val_losses.append(v_loss / len(val_loader))
        val_ious.append(v_iou  / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {train_losses[-1]:.4f}  Train IoU: {train_ious[-1]*100:.2f}%  |  "
              f"Val Loss: {val_losses[-1]:.4f}  Val IoU: {val_ious[-1]*100:.2f}%")

        if val_ious[-1] > best_val_iou:
            best_val_iou = val_ious[-1]
            torch.save(model.state_dict(), str(save_path))
            print(f"  => Checkpoint saved (Val IoU: {best_val_iou*100:.2f}%)")

    print(f"\nTraining complete. Best Val IoU: {best_val_iou*100:.2f}%")
    return train_losses, val_losses, train_ious, val_ious, lrs


def show_augmentation_samples(img_paths, mask_paths, n=6):
    """Plot n random samples side-by-side: Original | Augmented | Augmented Mask."""
    aug = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-15, 15), p=0.6),
        A.GridDistortion(p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.GaussNoise(p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
    ])
    indices = random.sample(range(len(img_paths)), min(n, len(img_paths)))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle("Data Augmentation Samples", fontsize=14, fontweight="bold")
    for row, idx in enumerate(indices):
        img  = cv2.cvtColor(cv2.imread(str(img_paths[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 10).astype(np.uint8) * 255
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = aug(image=img, mask=mask)
        axes[row, 0].imshow(cv2.resize(img, (256, 256)));   axes[row, 0].set_title("Original");         axes[row, 0].axis("off")
        axes[row, 1].imshow(result["image"]);               axes[row, 1].set_title("Augmented");        axes[row, 1].axis("off")
        axes[row, 2].imshow(result["mask"], cmap="gray");   axes[row, 2].set_title("Augmented Mask");   axes[row, 2].axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, val_losses, train_ious, val_ious, lrs):
    """Three-panel plot: loss curves, IoU curves, LR decay."""
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color="crimson",    lw=2)
    plt.plot(epochs_range, val_losses,   label="Val Loss",   color="royalblue",  lw=2, linestyle="--")
    plt.title("BCE + Dice Loss Curve"); plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, [v * 100 for v in train_ious], label="Train IoU", color="crimson",   lw=2)
    plt.plot(epochs_range, [v * 100 for v in val_ious],   label="Val IoU",   color="royalblue", lw=2, linestyle="--")
    plt.title("IoU Curve"); plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, lrs, label="Learning Rate", color="forestgreen", lw=2)
    plt.title("Cosine Annealing LR Decay"); plt.xlabel("Epochs"); plt.ylabel("Learning Rate")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
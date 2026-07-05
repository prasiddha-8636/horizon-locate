"""
SKY SEGMENTATION PIPELINE - Hybrid Guided Segmentation for Mountain Horizons
=============================================================================

This script processes a folder of perspective images, runs U-Net segmentation, 
and applies Canny-edge alignment and top-connected-component constraints 
to output clean, high-precision horizon masks.
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Standard color convention: Sky = 0 (Black), Terrain = 255 (White)
SKY_VAL = 0
TERRAIN_VAL = 255


def refine_sky_mask_with_guidance(img_np, raw_unet_mask):
    """
    Refines raw U-Net binary predictions using Canny edge detection 
    and top-connected-component constraints to eliminate snow/rock false positives.
    """
    H, W = raw_unet_mask.shape
    
    # --- Step 1: Sky-is-at-the-Top Constraint ---
    # Perform Connected Component Analysis on predicted sky pixels (1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_unet_mask, connectivity=8)
    top_connected_sky = np.zeros_like(raw_unet_mask)
    
    for i in range(1, num_labels):
        y_min = stats[i, cv2.CC_STAT_TOP]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Keep the component only if it touches the top edge of the image
        if y_min == 0 and area > 100:
            top_connected_sky[labels == i] = 1

    # --- Step 2: Canny Edge/Gradient Alignment ---
    # Find sharp structural boundaries (the physical skyline)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_edges = cv2.Canny(blurred, 30, 150)
    
    final_sky_mask = np.zeros_like(top_connected_sky)
    
    # Snap each column's sky boundary to the nearest Canny edge
    for col in range(W):
        sky_indices = np.where(top_connected_sky[:, col] == 1)[0]
        if len(sky_indices) == 0:
            continue
        unet_boundary = sky_indices[-1]  # The bottom-most predicted sky pixel
        
        # Search window of 15 pixels above/below the U-Net boundary
        search_window = 15
        search_min = max(0, unet_boundary - search_window)
        search_max = min(H - 1, unet_boundary + search_window)
        
        edge_indices = np.where(canny_edges[search_min:search_max, col] == 255)[0]
        if len(edge_indices) > 0:
            # Snap to the closest edge index in the search window
            closest_edge_rel = edge_indices[np.argmin(np.abs(edge_indices - (unet_boundary - search_min)))]
            final_boundary = search_min + closest_edge_rel
        else:
            final_boundary = unet_boundary
            
        final_sky_mask[:final_boundary + 1, col] = 1
        
    # --- Step 3: Format to Standard Color Convention ---
    # Convert binary mask (Sky=1, Terrain=0) to Grayscale (Sky=0/Black, Terrain=255/White)
    output_mask = np.where(final_sky_mask == 1, SKY_VAL, TERRAIN_VAL).astype(np.uint8)
    
    return output_mask


def run_segmentation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.masks_dir, exist_ok=True)
    
    # Initialize Mobilenetv3 U-Net model
    model = smp.Unet(
        encoder_name="tu-mobilenetv3_large_100",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    
    # Load trained checkpoint weights
    print(f"Loading weights from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        clean_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace("module.", "").replace("model.", "")
            clean_state_dict[name] = v
        model.load_state_dict(clean_state_dict)
        print("✓ Successfully loaded trained UNet weights.")
    except Exception as e:
        raise FileNotFoundError(f"Error loading model weights: {e}")
        
    model.to(device)
    model.eval()
    
    # Standard transformation
    input_size = 256
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_files = sorted([
        f for f in os.listdir(args.images_dir) 
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    print(f"Segmenting {len(image_files)} images from {args.images_dir}...")
    
    with torch.no_grad():
        for file_name in tqdm(image_files, desc="Segmenting"):
            img_path = os.path.join(args.images_dir, file_name)
            mask_path = os.path.join(args.masks_dir, file_name)
            
            # Load and verify image
            orig_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig_img.size
            img_np = np.array(orig_img)
            
            # Forward pass to obtain raw probability map
            tensor_img = transform(orig_img).unsqueeze(0).to(device)
            output = torch.sigmoid(model(tensor_img))
            
            # Resize probability mask back to original resolution first
            prob_mask_pil = Image.fromarray((output.cpu().squeeze().numpy() * 255).astype(np.uint8))
            prob_mask_resized = np.array(prob_mask_pil.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)) / 255.0
            
            # Binarize resized probability map
            raw_unet_mask = (prob_mask_resized <= 0.5).astype(np.uint8)
            
            # Apply structural Canny edge and connected-component guidance
            refined_mask = refine_sky_mask_with_guidance(img_np, raw_unet_mask)
            
            # Save final refined grayscale mask
            mask_pil = Image.fromarray(refined_mask)
            mask_pil.save(mask_path)
            
    print(f"\n✓ Sky segmentation complete! Saved refined masks to: {args.masks_dir}")

def cmd_visualize(args):
    """
    Plots the original perspective image side-by-side with its refined guided sky mask.
    Supports integer index (e.g. 0), full UUID, or partial UUID.
    """
    mask_files = sorted([
        f for f in os.listdir(args.masks_dir) 
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    if not mask_files:
        print(f"Error: No segmented masks found in {args.masks_dir}. Run segment mode first.")
        return

    # Select the sample to visualize
    if args.sample_id:
        target_file = None
        query = args.sample_id.strip()
        
        # 1. Try treating the input as an integer index (e.g., 0, 15, 100)
        try:
            idx = int(query)
            if 0 <= idx < len(mask_files):
                target_file = mask_files[idx]
            else:
                print(f"Error: Index {idx} is out of bounds. Folder contains {len(mask_files)} masks.")
                return
        except ValueError:
            # 2. If not an integer, treat as a direct or partial string UUID
            query_lower = query.lower()
            for f in mask_files:
                f_lower = f.lower()
                name_no_ext = os.path.splitext(f_lower)[0]
                if query_lower == f_lower or query_lower == name_no_ext or f_lower.startswith(query_lower):
                    target_file = f
                    break
                    
        if target_file is None:
            print(f"Error: No mask found matching index, ID, or partial query '{args.sample_id}' in {args.masks_dir}")
            return
    else:
        import random
        target_file = random.choice(mask_files)

    img_path = os.path.join(args.images_dir, target_file)
    mask_path = os.path.join(args.masks_dir, target_file)

    if not os.path.exists(img_path):
        print(f"Error: Original perspective image not found at {img_path}")
        return

    # Load and display
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img)
    axes[0].set_title(f"Original Perspective Image\n(ID/Index: {target_file})", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Guided Sky Mask\n(Terrain=255/White, Sky=0/Black)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guided Sky Segmentation Pipeline.")
    parser.add_argument("--mode", type=str, default="segment", choices=["segment", "visualize"],
                        help="Execution mode: 'segment' (batch processing) or 'visualize' (overlay plot).")
    parser.add_argument("--images_dir", type=str, default="data/street_view/images",
                        help="Input directory containing query images.")
    parser.add_argument("--masks_dir", type=str, default="data/street_view/masks",
                        help="Output directory to save refined masks.")
    parser.add_argument("--model_path", type=str, default="data/sky_segmentation_unet_model.pth",
                        help="Path to the trained UNet PyTorch model state dict.")
    parser.add_argument("--sample_id", type=str, default="",
                        help="Index (e.g. 0), full ID, or partial ID to plot in visualize mode.")

    args = parser.parse_args()
    if args.mode == "segment":
        run_segmentation(args)
    elif args.mode == "visualize":
        cmd_visualize(args)

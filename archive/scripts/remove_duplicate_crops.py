#!/usr/bin/env python3
"""Script to detect and remove exact and visual duplicate crops in data/street_view/gsv_crops/."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) of an image."""
    # Resize to (hash_size + 1, hash_size) grayscale
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.int32)
    # Compare adjacent pixels horizontally
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert boolean array to integer hash
    return sum([2**i for i, v in enumerate(diff.flatten()) if v])


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def compute_small_array(img: Image.Image, size: tuple = (32, 32)) -> np.ndarray:
    """Convert image to normalized float grayscale array for MSE check."""
    resized = img.convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32) / 255.0


def delete_crop_and_meta(img_path: Path, dry_run: bool = False) -> None:
    """Delete crop image and its corresponding JSON metadata file."""
    meta_path = img_path.with_suffix(".json")
    if meta_path.suffix != ".json":
        meta_path = img_path.parent / f"{img_path.stem}.json"

    if dry_run:
        print(f"[DRY-RUN] Would delete: {img_path.name}")
        if meta_path.exists():
            print(f"[DRY-RUN] Would delete metadata: {meta_path.name}")
    else:
        if img_path.exists():
            img_path.unlink()
            print(f"Deleted image: {img_path.name}")
        if meta_path.exists():
            meta_path.unlink()
            print(f"Deleted metadata: {meta_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate crop images.")
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR, help="Path to gsv_crops folder")
    parser.add_argument("--dhash-threshold", type=int, default=2, help="Max Hamming distance for visual match (0-4)")
    parser.add_argument("--mse-threshold", type=float, default=0.005, help="Max MSE for visual match (0.0 to 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Report duplicates without deleting")
    args = parser.parse_args()

    crops_dir = args.crops_dir
    if not crops_dir.exists():
        print(f"Error: Directory {crops_dir} does not exist.")
        return

    image_files = sorted([
        p for p in crops_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])

    print(f"Found {len(image_files)} crop images in {crops_dir}")
    if len(image_files) <= 1:
        print("Not enough images to perform deduplication.")
        return

    # -------------------------------------------------------------
    # PASS 1: Exact Binary Hash Deduplication (SHA256)
    # -------------------------------------------------------------
    print("\n--- Pass 1: Searching for Exact Binary Duplicates ---")
    seen_sha256 = {}
    to_delete = set()

    for p in image_files:
        try:
            h = compute_sha256(p)
            if h in seen_sha256:
                print(f"Exact match found: '{p.name}' is identical to '{seen_sha256[h].name}'")
                to_delete.add(p)
            else:
                seen_sha256[h] = p
        except Exception as e:
            print(f"Error reading {p.name}: {e}")

    print(f"Pass 1 complete. Found {len(to_delete)} exact binary duplicates.")

    # Remove exact duplicates from candidate list for Pass 2
    remaining_files = [p for p in image_files if p not in to_delete]

    # -------------------------------------------------------------
    # PASS 2: Visual / Perceptual Deduplication (dHash + MSE)
    # -------------------------------------------------------------
    print("\n--- Pass 2: Searching for Visual / Near-Duplicates ---")
    image_features = []

    for p in remaining_files:
        try:
            with Image.open(p) as img:
                dh = compute_dhash(img)
                arr = compute_small_array(img)
                image_features.append({
                    "path": p,
                    "dhash": dh,
                    "array": arr,
                })
        except Exception as e:
            print(f"Error processing {p.name}: {e}")

    visual_duplicates = set()
    n = len(image_features)

    for i in range(n):
        f1 = image_features[i]
        p1 = f1["path"]
        if p1 in visual_duplicates or p1 in to_delete:
            continue

        for j in range(i + 1, n):
            f2 = image_features[j]
            p2 = f2["path"]
            if p2 in visual_duplicates or p2 in to_delete:
                continue

            # Compare dHash
            dist = hamming_distance(f1["dhash"], f2["dhash"])
            if dist <= args.dhash_threshold:
                # Confirm with Mean Squared Error (MSE)
                mse = np.mean((f1["array"] - f2["array"]) ** 2)
                if mse <= args.mse_threshold:
                    print(f"Visual match found: '{p2.name}' matches '{p1.name}' (Hamming: {dist}, MSE: {mse:.5f})")
                    visual_duplicates.add(p2)
                    to_delete.add(p2)

    print(f"Pass 2 complete. Found {len(visual_duplicates)} visual duplicates.")

    # -------------------------------------------------------------
    # DELETION
    # -------------------------------------------------------------
    print(f"\nTotal duplicates to remove: {len(to_delete)}")
    for p in sorted(to_delete):
        delete_crop_and_meta(p, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run completed. Re-run without --dry-run to actually delete files.")
    else:
        print("\nDeduplication completed successfully.")


if __name__ == "__main__":
    main()

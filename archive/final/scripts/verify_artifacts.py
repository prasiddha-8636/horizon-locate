"""Check every artifact Colab expects. Reports which exist, which don't."""

import os
from pathlib import Path


def check(
    label: str, path: Path, expect_min_bytes: int = 0, expect_count: int = 0
) -> bool:
    if not path.exists():
        print(f"  MISS  {label}: {path}")
        return False
    if path.is_file():
        size = path.stat().st_size
        if size < expect_min_bytes:
            print(f"  BAD   {label}: {path} ({size} bytes < {expect_min_bytes})")
            return False
        print(f"  OK    {label}: {path.name} ({size / 1e6:.1f} MB)")
        return True
    if path.is_dir():
        n = sum(1 for _ in path.iterdir())
        if expect_count and n < expect_count:
            print(f"  BAD   {label}: {path} ({n} files < {expect_count})")
            return False
        print(f"  OK    {label}: {path.name}/ ({n} files)")
        return True
    return False


def main():
    print("=== Required artifacts ===")
    ok = True
    ok &= check(
        "DB parquet",
        Path("notebooks/02_SkylineDatabase/output/skyline_db.parquet"),
        expect_min_bytes=400_000_000,
    )
    ok &= check(
        "Ground truth",
        Path("data/synthetic_dataset/ground_truth.json"),
        expect_min_bytes=200_000,
    )
    ok &= check(
        "Predicted masks",
        Path("data/synthetic_dataset/predicted_masks"),
        expect_count=300,
    )
    ok &= check(
        "Segmentation model",
        Path("data/sky_segmentation_unet_model.pth"),
        expect_min_bytes=20_000_000,
    )

    print("\n=== Optional ===")
    check("DEM", Path("data/digital_elevation_model/dem_30m.tif"))
    check("Terrain mesh", Path("notebooks/02_SkylineDatabase/output/terrain_mesh.npy"))
    check("Terrain meta", Path("notebooks/02_SkylineDatabase/output/terrain_meta.json"))

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

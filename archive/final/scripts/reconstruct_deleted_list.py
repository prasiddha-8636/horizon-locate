"""Reconstructs deleted crops for target_crops = 3 setup."""

import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANOS_DIR = ROOT / "data" / "street_view" / "panos"
CROPS_DIR = ROOT / "data" / "street_view" / "gsv_crops"
LOG_PATH = CROPS_DIR / "deleted_crops_log.json"

def main():
    if not PANOS_DIR.exists() or not CROPS_DIR.exists():
        print("Error: Directories missing.")
        return

    panos = sorted([p.stem for p in list(PANOS_DIR.glob("*.jpg")) + list(PANOS_DIR.glob("*.png"))])
    existing_crops = set([p.name for p in CROPS_DIR.glob("*.png")])

    deleted_crops = []

    for pid in panos:
        # Find remaining crops for this pano
        pano_crops = [f for f in existing_crops if f.startswith(f"{pid}_h")]
        existing_headings = []
        for f in pano_crops:
            try:
                h = int(f.split("_h")[1].split("_p")[0])
                existing_headings.append(h)
            except Exception:
                pass

        # If pano has some crops but fewer than 3, record missing standard headings
        if 0 < len(existing_headings) < 3:
            standard_headings = [0, 90, 120, 180, 240, 270]
            for sh in standard_headings:
                if sh not in existing_headings:
                    deleted_crops.append(f"{pid}_h{sh}.png")

    if LOG_PATH.exists():
        try:
            with open(LOG_PATH) as f:
                old_log = json.load(f)
                deleted_crops = sorted(list(set(deleted_crops + old_log)))
        except Exception:
            pass

    with open(LOG_PATH, "w") as f:
        json.dump(deleted_crops, f, indent=2)

    print("=" * 60)
    print("RECONSTRUCTED DELETED LIST (TARGET CROPS = 3)")
    print("=" * 60)
    print(f"Total Deleted / Missing Crop Headings Tracked: {len(deleted_crops)}")
    print(f"Saved full list to: {LOG_PATH}")

if __name__ == "__main__":
    main()

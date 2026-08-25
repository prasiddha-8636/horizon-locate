import json
import shutil
from pathlib import Path

annot_path = Path("data/street_view/annotations.json")
backup_path = Path("data/street_view/annotations_single_backup.json")

if annot_path.exists():
    shutil.copy(annot_path, backup_path)
    print(f"Backed up 30 single-photo annotations to: {backup_path}")

# Reset annotations.json for new multi-photo session
fresh_data = {"annotations": {}, "skipped": {}}
with open(annot_path, "w") as f:
    json.dump(fresh_data, f, indent=2)

print("annotations.json reset! Ready for multi-photo annotations.")

"""Resume manifest: track which pipeline stages completed successfully."""
import json
from pathlib import Path
from typing import Optional


class ResumeManifest:
    """Tracks completed pipeline stages so work can be skipped on resume.

    Usage:
        m = ResumeManifest("/tmp/pipeline_state.json")
        if m.done("download_dem"):
            return
        # ... do work ...
        m.mark("download_dem", path="data/dem.tif", size_mb=2600)
        m.save()
    """

    def __init__(self, path):
        self.path = Path(path)
        self.state = {}
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self.state = {}

    def done(self, key: str) -> bool:
        return self.state.get(key, {}).get("status") == "ok"

    def get(self, key: str) -> Optional[dict]:
        return self.state.get(key)

    def mark(self, key: str, status: str = "ok", **meta):
        self.state[key] = {"status": status, **meta}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, default=str))

    def summary(self) -> str:
        lines = [f"Resume manifest: {self.path}"]
        for k, v in self.state.items():
            lines.append(f"  [{v.get('status', '?'):>4}] {k}: {v}")
        return "\n".join(lines)

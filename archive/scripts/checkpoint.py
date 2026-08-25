#!/usr/bin/env python
"""Shared checkpoint/resume utilities for evaluation scripts.

Every script can do:

    from checkpoint import Checkpoint
    ck = Checkpoint("my_eval", phase=1)        # opens data/eval_ckpt/my_eval_p1.pkl
    if ck.exists():
        data = ck.load()
        print(f"[RESUME] Loaded phase 1 from {ck.path}")
    else:
        data = expensive_computation()
        ck.save(data)
        print(f"[SAVE] Saved phase 1 to {ck.path}")

On the next run with --resume, the expensive computation is skipped.

Usage as CLI:
    python -m archive.scripts.checkpoint --list           # show all checkpoints
    python -m archive.scripts.checkpoint --clean my_eval   # delete all phases for my_eval
"""

import json
import pickle
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = ROOT / "data" / "eval_ckpt"


class Checkpoint:
    """Simple pickle-based checkpoint with auto-naming."""

    def __init__(self, name: str, phase: int = 1, *, dir: Path | None = None):
        """
        Args:
            name:  Script identifier, e.g. "end_to_end", "gsv_improve"
            phase: Phase number within that script (1, 2, 3, ...)
            dir:   Override checkpoint directory (default: data/eval_ckpt/)
        """
        self.name = name
        self.phase = phase
        self.ckpt_dir = dir or CKPT_DIR
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.ckpt_dir / f"{name}_p{phase}.pkl"

    def exists(self) -> bool:
        return self.path.exists()

    def load(self):
        """Load checkpoint. Raises FileNotFoundError if missing."""
        with open(self.path, "rb") as f:
            return pickle.load(f)

    def save(self, data):
        """Save checkpoint atomically (write-to-temp then rename)."""
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(self.path)

    def delete(self):
        """Remove checkpoint if it exists."""
        self.path.unlink(missing_ok=True)

    def __repr__(self):
        status = "OK" if self.exists() else "missing"
        return f"Checkpoint({self.name} phase={self.phase} [{status}] @ {self.path})"


def list_checkpoints(dir: Path | None = None):
    """List all existing checkpoints."""
    d = dir or CKPT_DIR
    if not d.exists():
        print(f"No checkpoint directory: {d}")
        return []
    files = sorted(d.glob("*_p*.pkl"))
    if not files:
        print(f"No checkpoints in {d}")
        return []
    print(f"{'Name':<30} {'Phase':>5}  {'Size':>10}  {'Path'}")
    print("-" * 80)
    for f in files:
        parts = f.stem.rsplit("_p", 1)
        name = parts[0]
        phase = int(parts[1]) if len(parts) > 1 else 0
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"{name:<30} {phase:>5}  {size_mb:>8.1f}MB  {f}")
    return files


def clean_checkpoints(name: str | None = None, dir: Path | None = None):
    """Delete checkpoint files. If name given, only delete that script's files."""
    d = dir or CKPT_DIR
    if not d.exists():
        return
    if name:
        pattern = f"{name}_p*.pkl"
    else:
        pattern = "*_p*.pkl"
    for f in d.glob(pattern):
        f.unlink()
        print(f"  deleted {f.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Checkpoint management")
    ap.add_argument("--list", action="store_true", help="List all checkpoints")
    ap.add_argument("--clean", type=str, default=None, help="Delete checkpoints for a script name")
    ap.add_argument("--clean-all", action="store_true", help="Delete ALL checkpoints")
    args = ap.parse_args()

    if args.list:
        list_checkpoints()
    elif args.clean or args.clean_all:
        name = args.clean if not args.clean_all else None
        print(f"Cleaning{' all' if not name else ' ' + name} checkpoints...")
        clean_checkpoints(name)
        print("Done.")
    else:
        ap.print_help()

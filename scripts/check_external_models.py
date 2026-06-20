from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = {
    "SimSwap repository": ROOT / "vendor" / "simswap",
    "SimSwap 512 archive": ROOT / "models" / "swappers" / "simswap512" / "512.zip",
    "GHOST repository": ROOT / "vendor" / "ghost",
    "GHOST 2.0 repository": ROOT / "vendor" / "ghost2",
    "GHOST 2.0 aligner directory": ROOT / "vendor" / "ghost2" / "aligner_checkpoints",
    "GHOST 2.0 blender directory": ROOT / "vendor" / "ghost2" / "blender_checkpoints",
    "SimSwap environment": ROOT / ".environments" / "simswap",
    "GHOST environment": ROOT / ".environments" / "ghost",
    "GHOST 2.0 environment": ROOT / ".environments" / "ghost2",
}

failed = False

for name, path in CHECKS.items():
    exists = path.exists()
    print(f"[{'OK' if exists else 'MISSING'}] {name}: {path}")
    failed = failed or not exists

raise SystemExit(1 if failed else 0)
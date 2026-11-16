"""Verify the competition package before loading any model."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "资源/训练包/robotics-bench-go2-competition-training-resources"


def verify():
    manifest = json.loads((RESOURCES / "MANIFEST.json").read_text())
    failed = []
    count = 0
    for line in (RESOURCES / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (RESOURCES / name.lstrip("*")).resolve()
        if not path.is_relative_to(RESOURCES.resolve()):
            raise ValueError(f"Unsafe resource path: {name}")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failed.append(name)
        count += 1
    if failed:
        raise RuntimeError(f"Resource integrity failed: {failed}")
    return {"version": manifest["version"], "files_verified": count}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))

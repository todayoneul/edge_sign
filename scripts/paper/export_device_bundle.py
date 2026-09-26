"""Copy what runtime_matrix.py needs on another device into one folder (private transfer only).

The ONNX files and AI Hub frames are not in git (size; AI Hub terms forbid
redistribution), so a second measurement device receives them as a bundle:

    <out>/model_space/<every model in runtime_matrix.MODELS>
    <out>/data/aihub_traffic/test/images/...   (same relative paths as the manifest)
    <out>/test_manifest.jsonl                  (first --frames rows, or all)
    <out>/SHA256SUMS                           (checked by run_device_matrix.sh)

Use the bundle folder as --artifact-root and test_manifest.jsonl as --manifest on
the device. Do not publish or commit the bundle.

Usage: python scripts/paper/export_device_bundle.py --artifact-root <checkout> --out <folder> [--frames N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.runtime_matrix import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=None, help="first N test frames (default: all)")
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty {args.out}")
    lines = [line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = lines[: args.frames] if args.frames else lines
    files = sorted({rel for _, rel in MODELS.values()} | {json.loads(line)["image_rel"] for line in lines})
    sums = []
    for i, rel in enumerate(files):
        target = args.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
        sums.append(f"{sha256(target)}  {rel}")
        if (i + 1) % 500 == 0:
            print(f"copied {i + 1}/{len(files)}", flush=True)
    # LF only: shasum -c on macOS reads a trailing CR as part of the file name
    (args.out / "test_manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (args.out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"bundle: {len(MODELS)} models, {len(lines)} frames, {size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()

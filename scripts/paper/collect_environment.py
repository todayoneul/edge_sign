"""Record current hardware/software and immutable hashes of available paper models."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import onnx
import onnxruntime as ort

MODELS = (
    (
        "yolo26_fp32",
        "model_space/yolo_v4_signs_fp32.onnx",
        "runs/detect/edge_sign_v4/weights/best.pt",
        "scripts/export_v4_variants.py:export_fp32",
    ),
    (
        "yolo26_full_qdq",
        "model_space/yolo_v4_signs_int8_static.onnx",
        "model_space/yolo_v4_signs_fp32.onnx",
        "scripts/export_v4_variants.py:export_int8(quant_head=True)",
    ),
    (
        "yolo26_head_excluded_qdq",
        "model_space/yolo_v4_signs_int8_head_excluded.onnx",
        "model_space/yolo_v4_signs_fp32.onnx",
        "scripts/export_v4_variants.py:export_int8(quant_head=False); output renamed historically",
    ),
    (
        "yolo26_fp16",
        "model_space/yolo_v4_signs_fp16.onnx",
        "runs/detect/edge_sign_v4/weights/best.pt",
        "scripts/export_v4_variants.py:export_fp16",
    ),
    (
        "yolo26_checkpoint",
        "runs/detect/edge_sign_v4/weights/best.pt",
        "yolo26n.pt",
        "scripts/train_v4_detector.py",
    ),
    (
        "yolov8s_v3_fp32",
        "model_space/yolov8s_signs_v3_fp32.onnx",
        "runs/detect/edge_sign_v3_lights-4/weights/best.pt",
        "historical export; exact command unrecorded",
    ),
    (
        "yolov8s_v3_head_excluded_qdq",
        "model_space/yolov8s_signs_v3_int8_static.onnx",
        "model_space/yolov8s_signs_v3_fp32.onnx",
        "scripts/quantize_v3_detector.py",
    ),
    (
        "korean_sign_fp32",
        "model_space/korean_sign_net_fp32.onnx",
        "model_space/korean_sign_net_best.pth",
        "scripts/train_korean_classifier.py",
    ),
    (
        "korean_sign_w8a8",
        "model_space/korean_sign_net_w8a8.onnx",
        "model_space/korean_sign_net_fp32.onnx",
        "scripts/train_korean_classifier.py",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onnx_properties(path: Path) -> dict[str, str | int]:
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    return {
        "onnx_opset": ";".join(f"{x.domain or 'ai.onnx'}:{x.version}" for x in model.opset_import),
        "input": ";".join(
            f"{x.name}:{[d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]}"
            for x in graph.input
        ),
        "output": ";".join(
            f"{x.name}:{[d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]}"
            for x in graph.output
        ),
        "quantize_linear_nodes": sum(n.op_type == "QuantizeLinear" for n in graph.node),
        "dequantize_linear_nodes": sum(n.op_type == "DequantizeLinear" for n in graph.node),
        "module23_qdq_nodes": sum(
            n.op_type in ("QuantizeLinear", "DequantizeLinear")
            and re.search(r"model[./]23(?:[/.]|$)", n.name or "") is not None
            for n in graph.node
        ),
        "node_count": len(graph.node),
    }


def model_manifest(artifact_root: Path, output: Path) -> list[dict]:
    root = artifact_root.resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, relative, source, command in MODELS:
        path = root / relative
        row = {
            "model_id": name,
            "filename": path.name,
            "path": relative,
            "status": "present" if path.is_file() else "missing",
            "size_bytes": path.stat().st_size if path.is_file() else "",
            "sha256": sha256(path) if path.is_file() else "",
            "source_checkpoint": source,
            "export_command": command,
            "lineage_status": "script_and_names_only; no saved cryptographic export attestation",
            "onnx_opset": "",
            "input": "",
            "output": "",
            "quantize_linear_nodes": "",
            "dequantize_linear_nodes": "",
            "node_count": "",
            "module23_qdq_nodes": "",
        }
        if path.is_file() and path.suffix == ".onnx":
            row.update(_onnx_properties(path))
        rows.append(row)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _command(args: list[str]) -> str:
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        return (run.stdout or run.stderr).strip() or f"exit={run.returncode}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def _dll_status(name: str) -> str:
    if platform.system() != "Windows":
        return "not applicable"
    try:
        ctypes.WinDLL(name)
        return "loadable"
    except OSError as exc:
        return f"unavailable: {exc}"


def environment(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"cpu={os.environ.get('PROCESSOR_IDENTIFIER', platform.processor())}",
        f"logical_cpus={os.cpu_count()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"onnxruntime={ort.__version__}",
        f"onnxruntime_providers={ort.get_available_providers()}",
        f"onnx={onnx.__version__}",
        f"nvidia_smi={_command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])}",
        f"cublasLt64_12.dll={_dll_status('cublasLt64_12.dll')}",
        f"cudnn64_9.dll={_dll_status('cudnn64_9.dll')}",
    ]
    for browser in ("msedge", "chrome"):
        lines.append(f"{browser}_version={_command([browser, '--version'])}")
    (output / "system_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "python_packages.txt").write_text(
        _command([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("paper_evidence"))
    args = parser.parse_args()
    rows = model_manifest(args.artifact_root, args.output / "models" / "model_manifest.csv")
    environment(args.output / "environment")
    print(f"recorded {sum(row['status'] == 'present' for row in rows)}/{len(rows)} models")


if __name__ == "__main__":
    main()

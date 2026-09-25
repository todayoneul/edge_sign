"""Create the extra static INT8 QDQ detector variants used by the runtime matrix.

Existing model files are never overwritten. Every variant uses the frozen
calibration manifest (150 sorted yolo_signs_v2 val frames), per-channel
symmetric INT8 weights and asymmetric UINT8 activations, matching the original
v3/v4 quantization scripts. Two factors are varied:

* head: quantized (full graph) or excluded (the last model.N module stays FP32)
* bias: INT32 (ORT default) or kept in FP32 (QuantizeBias=False). ORT Web's
  WebGPU kernel rejects INT32 DequantizeLinear without a zero point, so FP32
  bias is the WebGPU-compatible form.

Usage (convnext_env):
    python scripts/paper/quantize_detector_variants.py --artifact-root <checkout>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.evaluate_qdq_detection import _preprocess

SOURCES = {
    "v3": "model_space/yolov8s_signs_v3_fp32.onnx",
    "v4": "model_space/yolo_v4_signs_fp32.onnx",
}
# (family, head_quantized, bias_fp32) -> output file under model_space/
VARIANTS = {
    ("v3", True, False): "model_space/yolov8s_signs_v3_int8_full.onnx",
    ("v3", True, True): "model_space/yolov8s_signs_v3_int8_full_fbias.onnx",
    ("v3", False, True): "model_space/yolov8s_signs_v3_int8_head_excluded_fbias.onnx",
    ("v4", True, True): "model_space/yolo_v4_signs_int8_full_fbias.onnx",
    ("v4", False, True): "model_space/yolo_v4_signs_int8_head_excluded_fbias.onnx",
}


class ManifestCalibration(CalibrationDataReader):
    def __init__(self, root: Path, manifest: Path):
        self.paths = [
            root / json.loads(line)["image_rel"]
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rewind()

    def _frames(self):
        for path in self.paths:
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            yield {"images": _preprocess(image)}

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._frames())


def head_nodes(model_path: Path) -> tuple[str, list[str]]:
    graph = onnx.load(str(model_path)).graph
    indices = {
        int(match.group(1)) for node in graph.node if (match := re.search(r"model\.(\d+)", node.name))
    }
    head = f"model.{max(indices)}"
    # same rule as scripts/export_v4_variants.py: every node whose name contains the head module
    return head, [node.name for node in graph.node if head in node.name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, default=Path("paper_evidence/splits/calibration_manifest.jsonl")
    )
    parser.add_argument("--log", type=Path, default=Path("paper_evidence/models/quantize_variants_log.json"))
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    calibration_rows = sum(1 for line in args.calibration.read_text(encoding="utf-8").splitlines() if line.strip())
    log = []
    for (family, head_quantized, bias_fp32), relative in VARIANTS.items():
        target = root / relative
        if target.exists():
            print(f"skip existing {relative}")
            continue
        source = root / SOURCES[family]
        with tempfile.TemporaryDirectory(prefix="edge_sign_quant_") as tmp:
            prepared = Path(tmp) / "prepared.onnx"
            quant_pre_process(
                input_model_path=str(source),
                output_model_path=str(prepared),
                skip_optimization=False,
                skip_onnx_shape=False,
                skip_symbolic_shape=True,
            )
            head, excluded = head_nodes(prepared)
            quantize_static(
                model_input=str(prepared),
                model_output=str(target),
                calibration_data_reader=ManifestCalibration(root, args.calibration),
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                per_channel=True,
                reduce_range=False,
                nodes_to_exclude=[] if head_quantized else excluded,
                extra_options={
                    "ActivationSymmetric": False,
                    "WeightSymmetric": True,
                    "EnableSubgraph": True,
                    "QuantizeBias": not bias_fp32,
                },
            )
        graph = onnx.load(str(target)).graph
        head_qdq = sum(
            1 for n in graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear") and head in n.name
        )
        int32_inits = sum(1 for t in graph.initializer if t.data_type == onnx.TensorProto.INT32)
        entry = {
            "family": family,
            "head_module": head,
            "head_quantized": head_quantized,
            "bias": "fp32" if bias_fp32 else "int32",
            "source": SOURCES[family],
            "source_sha256": sha256(source),
            "output": relative,
            "output_bytes": target.stat().st_size,
            "output_sha256": sha256(target),
            "excluded_nodes": 0 if head_quantized else len(excluded),
            "head_qdq_nodes": head_qdq,
            "int32_initializers": int32_inits,
            "calibration_manifest": args.calibration.as_posix(),
            "calibration_frames": calibration_rows,
        }
        log.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    if log:
        previous = json.loads(args.log.read_text(encoding="utf-8")) if args.log.exists() else []
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(previous + log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

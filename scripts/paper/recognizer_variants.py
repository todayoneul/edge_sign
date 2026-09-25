"""KoreanSignNet (14-class recognizer) static INT8 QDQ variants and their accuracy.

The existing model_space/korean_sign_net_w8a8.onnx uses ConvInteger, which ORT CPU
cannot execute, so it gives no recognition number. This script builds runnable QDQ
variants with the detector's settings (per-channel symmetric INT8 weights,
asymmetric UINT8 activations, MinMax) and evaluates every variant on the same
independent-test oracle ROIs as evaluate_recognition.py.

Calibration uses 512 ROIs drawn with seed 0 from data/roi_cls/val (validation
sequences; no test frame). Factors, as for the detector:
  head  quantized, or the classifier conv (/fc2_conv/Conv) kept in FP32
  bias  INT32 (ORT default) or FP32 (QuantizeBias=False)
Existing model files are never overwritten.

Usage (convnext_env):
    python scripts/paper/recognizer_variants.py quantize --artifact-root <checkout>
    python scripts/paper/recognizer_variants.py evaluate --artifact-root <checkout>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.evaluate_recognition import _crop, annotation_class

SOURCE = "model_space/korean_sign_net_fp32.onnx"
HEAD = "/fc2_conv/Conv"
VARIANTS = {
    "fp32": SOURCE,
    "fp16": "model_space/korean_sign_net_fp16.onnx",
    "int8_full": "model_space/korean_sign_net_int8_qdq_full.onnx",
    "int8_full_fbias": "model_space/korean_sign_net_int8_qdq_full_fbias.onnx",
    "int8_head_excl": "model_space/korean_sign_net_int8_qdq_head_excluded.onnx",
    "int8_head_excl_fbias": "model_space/korean_sign_net_int8_qdq_head_excluded_fbias.onnx",
}
BUILD = {  # name -> (head_quantized, bias_fp32)
    "int8_full": (True, False),
    "int8_full_fbias": (True, True),
    "int8_head_excl": (False, False),
    "int8_head_excl_fbias": (False, True),
}


def roi_tensor(image: np.ndarray) -> np.ndarray:
    # same as scripts/train_korean_classifier.py: resize to 32, RGB, (x/255 - 0.5) / 0.5
    if image.shape[:2] != (32, 32):
        image = cv2.resize(image, (32, 32))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose((rgb - 0.5) / 0.5, (2, 0, 1))[None]


def calibration_manifest(root: Path, path: Path, count: int) -> list[str]:
    if path.exists():
        return [json.loads(line)["image_rel"] for line in path.read_text(encoding="utf-8").splitlines() if line]
    files = sorted(p.relative_to(root).as_posix() for p in (root / "data/roi_cls/val").rglob("*.jpg"))
    chosen = sorted(np.random.default_rng(0).choice(len(files), count, replace=False))
    rels = [files[i] for i in chosen]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"image_rel": r, "class_id": int(r.split("/")[3])}) + "\n" for r in rels), encoding="utf-8"
    )
    return rels


def quantize(args) -> None:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quant_pre_process, quantize_static

    root = args.artifact_root.resolve(strict=True)
    rels = calibration_manifest(root, args.calibration, args.calibration_count)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def get_next(self):
            return next(self._iter, None)

        def rewind(self):
            self._iter = ({"images": roi_tensor(cv2.imread(str(root / r)))} for r in rels)

    log = []
    for name, (head_quantized, bias_fp32) in BUILD.items():
        target = root / VARIANTS[name]
        if target.exists():
            print(f"skip existing {VARIANTS[name]}")
            continue
        with tempfile.TemporaryDirectory(prefix="edge_sign_rec_quant_") as tmp:
            prepared = Path(tmp) / "prepared.onnx"
            quant_pre_process(str(root / SOURCE), str(prepared), skip_symbolic_shape=True)
            quantize_static(
                model_input=str(prepared),
                model_output=str(target),
                calibration_data_reader=Reader(),
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                per_channel=True,
                reduce_range=False,
                nodes_to_exclude=[] if head_quantized else [HEAD],
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, "QuantizeBias": not bias_fp32},
            )
        graph = onnx.load(str(target)).graph
        entry = {
            "variant": name,
            "head_quantized": head_quantized,
            "bias": "fp32" if bias_fp32 else "int32",
            "source": SOURCE,
            "source_sha256": sha256(root / SOURCE),
            "output": VARIANTS[name],
            "output_bytes": target.stat().st_size,
            "output_sha256": sha256(target),
            "qdq_nodes": sum(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in graph.node),
            "calibration_manifest": args.calibration.as_posix(),
            "calibration_rois": len(rels),
        }
        log.append(entry)
        print(json.dumps(entry))
    if log:
        previous = json.loads(args.log.read_text(encoding="utf-8")) if args.log.exists() else []
        args.log.write_text(json.dumps(previous + log, indent=2) + "\n", encoding="utf-8")


def evaluate(args) -> None:
    root = args.artifact_root.resolve(strict=True)
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite recognition evidence: {out}")
    frames = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    classes = json.loads((root / "data/roi_cls/classes.json").read_text(encoding="utf-8"))["names"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sessions = {k: ort.InferenceSession(str(root / v), opts, providers=["CPUExecutionProvider"]) for k, v in VARIANTS.items()}
    labels, logits = [], {k: [] for k in sessions}
    for frame in frames:
        image = cv2.imread(str(root / frame["image_rel"]))
        annotation = json.loads((root / frame["label_rel"]).read_text(encoding="utf-8"))
        batch = []
        for ann in annotation.get("annotation", []):
            if ann.get("class") not in ("traffic_sign", "traffic_light"):
                continue
            label = annotation_class(ann)
            tensor = _crop(image, ann.get("box", [])) if len(ann.get("box", [])) >= 4 else None
            if label is None or tensor is None:
                continue
            labels.append(label)
            batch.append(tensor)
        if batch:
            x = np.concatenate(batch)
            for k, s in sessions.items():
                logits[k].append(s.run(None, {s.get_inputs()[0].name: x})[0].astype(np.float32))
    y = np.array(labels)
    ref = np.concatenate(logits["fp32"])
    out.mkdir(parents=True, exist_ok=True)
    preds = {}
    for k in sessions:
        z = np.concatenate(logits[k])
        p = z.argmax(1)
        preds[k] = p
        matrix = np.zeros((14, 14), dtype=np.int64)
        np.add.at(matrix, (y, p), 1)
        cos = (z * ref).sum(1) / np.maximum(np.linalg.norm(z, axis=1) * np.linalg.norm(ref, axis=1), 1e-12)
        per_class = [
            {"class_id": c, "class_name": classes[c], "total": int(matrix[c].sum()),
             "accuracy": float(matrix[c, c] / matrix[c].sum()) if matrix[c].sum() else None}
            for c in range(14)
        ]
        valid = [pc["accuracy"] for pc in per_class if pc["accuracy"] is not None]
        result = {
            "variant": k,
            "model": VARIANTS[k],
            "model_bytes": (root / VARIANTS[k]).stat().st_size,
            "model_sha256": sha256(root / VARIANTS[k]),
            "test_manifest_sha256": sha256(args.manifest),
            "evaluation_scope": "oracle ground-truth box classification; independent test sequences",
            "runtime": f"onnxruntime-cpu {ort.__version__}",
            "total_rois": int(len(y)),
            "top1_accuracy": float((p == y).mean()),
            "macro_accuracy": float(np.mean(valid)),
            "top1_retention_vs_fp32": float((p == y).mean() / (ref.argmax(1) == y).mean()),
            "agreement_with_fp32": float((p == ref.argmax(1)).mean()),
            "logit_cosine_vs_fp32_mean": float(cos.mean()),
            "logit_cosine_vs_fp32_min": float(cos.min()),
            "logit_max_abs_diff_vs_fp32": float(np.abs(z - ref).max()),
            "per_class": per_class,
            "confusion_matrix": matrix.tolist(),
        }
        (out / f"{k}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{k}: top1 {result['top1_accuracy']:.6f} retention {result['top1_retention_vs_fp32']:.4f} "
              f"agree {result['agreement_with_fp32']:.4f} cos_min {result['logit_cosine_vs_fp32_min']:.4f}", flush=True)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["roi_index", "true_class_id", *preds])
        for i in range(len(y)):
            writer.writerow([i, int(y[i]), *(int(preds[k][i]) for k in preds)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("quantize")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--calibration", type=Path, default=Path("paper_evidence/splits/recognition_calibration_manifest.jsonl"))
    p.add_argument("--calibration-count", type=int, default=512)
    p.add_argument("--log", type=Path, default=Path("paper_evidence/models/recognizer_variants_log.json"))
    p = sub.add_parser("evaluate")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    p.add_argument("--output", type=Path, default=Path("paper_evidence/recognition/variants"))
    args = parser.parse_args()
    {"quantize": quantize, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()

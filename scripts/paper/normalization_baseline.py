"""Known-mitigation baseline: box-coordinate normalization in the paper's ONNX graphs.

Design and predictions (written before running): paper_evidence/extra/mitigation/DESIGN.md.
Builds normalized FP32 graphs from the paper's FP32 detectors, checks that they reproduce the
original detections, quantizes them to full static INT8 with the paper's settings (same 150
calibration frames, MinMax, per-channel INT8 weights, UINT8 activations, INT32 bias, no node
excluded) and evaluates them on the test set with the paper's decoder.

  post_norm (v3, v4)  vanilla rescaling: x 1/640 after the x stride multiply, before the box-score concat
  pre_norm  (v3, v4)  Ultralytics TensorFlow exporter: distances and anchors scaled by stride/640, no x stride
  fold_norm (v4)      stride/640 folded into the last regression conv of each level (the structure of
                      Moon et al.'s RN, Eq. 6); anchors scaled, no x stride. Not possible for YOLOv8s:
                      the DFL softmax sits between that conv and the distances.

Every variant emits boxes in [0, 1] of the 640 input; the decoder multiplies them by 640 (outside
the quantized graph, as Ultralytics' TFLite backend does). New models go to model_space/*_norm_*.onnx,
results to paper_evidence/extra/mitigation/. Nothing existing is overwritten.

Usage (convnext_env): python scripts/paper/normalization_baseline.py --artifact-root <checkout>
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.runtime_matrix import (
    decode,
    frame_rgb,
    input_tensor,
    load_manifest,
    write_accuracy,
)

OUT = Path("paper_evidence/extra/mitigation")
CALIBRATION = Path("paper_evidence/splits/calibration_manifest.jsonl")
MANIFEST = Path("paper_evidence/splits/test_manifest.jsonl")
SOURCES = {"v3": "model_space/yolov8s_signs_v3_fp32.onnx", "v4": "model_space/yolo_v4_signs_fp32.onnx"}
STEM = {"v3": "model_space/yolov8s_signs_v3", "v4": "model_space/yolo_v4_signs"}
GRAPH = {  # node names checked against the FP32 sources (DESIGN.md section 2)
    "v3": {"head": "model.22", "dist": "/model.22/dfl/Reshape_1", "sub": "/model.22/Sub", "add": "/model.22/Add_1",
           "mul": "/model.22/Mul_2", "concat": "/model.22/Concat_3"},
    "v4": {"head": "model.23", "dist": "/model.23/Concat", "sub": "/model.23/Sub", "add": "/model.23/Add_1",
           "mul": "/model.23/Mul_2", "concat": "/model.23/Concat_3"},
}
VARIANTS = {"v3": ["post_norm", "pre_norm"], "v4": ["post_norm", "pre_norm", "fold_norm"]}
EXISTING = ["fp32", "int8_full", "int8_head_excl", "int8_decode_excl"]  # paper_evidence/runtime/matrix/cpu_<family>_<name>


def build(family: str, variant: str, source: Path, target: Path) -> dict:
    """Normalized FP32 graph; returns what was changed."""
    m = onnx.load(str(source))
    g = m.graph
    names = GRAPH[family]
    nodes = {n.name: n for n in g.node}
    inits = {i.name: i for i in g.initializer}
    mul, concat = nodes[names["mul"]], nodes[names["concat"]]
    if concat.input[0] != mul.output[0]:
        raise SystemExit(f"{family}: {names['concat']} does not take {names['mul']} first")
    stride_name = next(i for i in mul.input if i in inits)
    strides = numpy_helper.to_array(inits[stride_name]).astype(np.float32)  # [1, 8400]: 8, 16, 32 per anchor
    if strides.shape != (1, 8400) or not (np.array_equal(np.unique(strides), [8, 16, 32])):
        raise SystemExit(f"{family}: unexpected stride tensor {strides.shape}")
    norm = (strides / 640.0).astype(np.float32)
    changes: dict = {"stride_tensor": stride_name}

    def add_init(name: str, array: np.ndarray) -> str:
        g.initializer.append(numpy_helper.from_array(array.astype(np.float32), name))
        return name

    def insert_after(producer_output: str, node: onnx.NodeProto) -> None:
        index = next(k for k, n in enumerate(g.node) if producer_output in n.output)
        g.node.insert(index + 1, node)

    if variant == "post_norm":
        scale = add_init("briq_inv640", np.array(1.0 / 640.0, dtype=np.float32))
        node = helper.make_node("Mul", [mul.output[0], scale], ["briq_boxes_norm"], name=f"/{names['head']}/BriqPostNorm")
        insert_after(mul.output[0], node)
        concat.input[0] = "briq_boxes_norm"
        changes["inserted"] = node.name
    else:
        for key in ("sub", "add"):  # anchor constants (grid units) -> anchor * stride / 640
            n = nodes[names[key]]
            anchors = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
            if anchors.shape != (1, 2, 8400):
                raise SystemExit(f"{family}: {n.name} anchors {anchors.shape}")
            n.input[0] = add_init(f"briq_anchors_norm_{key}", anchors * norm[:, None, :])
        if variant == "pre_norm":
            dist = nodes[names["dist"]].output[0]
            scale = add_init("briq_dist_scale", norm[:, None, :])
            node = helper.make_node("Mul", [dist, scale], ["briq_dist_norm"], name=f"/{names['head']}/BriqPreNorm")
            for n in g.node:
                n.input[:] = ["briq_dist_norm" if i == dist else i for i in n.input]
            insert_after(dist, node)
            changes["inserted"] = node.name
        elif variant == "fold_norm":
            if family != "v4":
                raise SystemExit("fold_norm needs a direct regression head (v4)")
            if not (np.all(strides[0, :6400] == 8) and np.all(strides[0, 6400:8000] == 16) and np.all(strides[0, 8000:] == 32)):
                raise SystemExit("v4: anchors are not ordered P3, P4, P5")
            folded = []
            for level, stride in enumerate((8, 16, 32)):
                conv = nodes[f"/model.23/one2one_cv2.{level}/one2one_cv2.{level}.2/Conv"]
                for slot in (1, 2):  # weight, bias
                    arr = numpy_helper.to_array(inits[conv.input[slot]]).astype(np.float32) * (stride / 640.0)
                    conv.input[slot] = add_init(f"briq_fold_{level}_{slot}", arr)
                folded.append(conv.name)
            changes["folded_convs"] = folded
        concat.input[0] = mul.input[0] if mul.input[0] != stride_name else mul.input[1]  # drop x stride
        g.node.remove(mul)
        changes["removed"] = mul.name
    onnx.checker.check_model(m)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(target))
    return changes


def rescale(family: str, out: np.ndarray) -> np.ndarray:
    out = out.copy()
    if family == "v3":  # [1, 6, 8400]: xywh rows 0-3
        out[:, :4, :] *= 640.0
    else:  # [1, 300, 6]: xyxy columns 0-3
        out[..., :4] *= 640.0
    return out


def session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def fp32_check(family: str, original: Path, normalized: Path, root: Path, rows: list[dict], threads: int) -> dict:
    a, b = session(original, threads), session(normalized, threads)
    worst_box, worst_score = 0.0, 0.0
    for row in rows:
        x = input_tensor(frame_rgb(root, row), "float32")
        oa = a.run(None, {a.get_inputs()[0].name: x})[0]
        ob = rescale(family, b.run(None, {b.get_inputs()[0].name: x})[0])
        if family == "v3":
            worst_box = max(worst_box, float(np.abs(oa[:, :4] - ob[:, :4]).max()))
            worst_score = max(worst_score, float(np.abs(oa[:, 4:] - ob[:, 4:]).max()))
        else:
            worst_box = max(worst_box, float(np.abs(oa[..., :4] - ob[..., :4]).max()))
            worst_score = max(worst_score, float(np.abs(oa[..., 4:] - ob[..., 4:]).max()))
    ok = worst_box < 1e-3 and worst_score < 1e-5
    return {"frames": len(rows), "max_abs_box_px": worst_box, "max_abs_score_or_class": worst_score, "passed": ok}


def quantize(source: Path, target: Path, root: Path) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quant_pre_process, quantize_static

    from scripts.paper.quantize_detector_variants import ManifestCalibration

    with tempfile.TemporaryDirectory(prefix="briq_norm_") as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(input_model_path=str(source), output_model_path=str(prepared), skip_optimization=False,
                          skip_onnx_shape=False, skip_symbolic_shape=True)
        quantize_static(model_input=str(prepared), model_output=str(target),
                        calibration_data_reader=ManifestCalibration(root, CALIBRATION),
                        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                        per_channel=True, reduce_range=False, nodes_to_exclude=[],
                        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                       "EnableSubgraph": True, "QuantizeBias": True})


def box_scales(path: Path) -> dict:
    """UINT8 activation scales of the box / concat tensors of a QDQ graph (by tensor name)."""
    g = onnx.load(str(path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    keep = ("Concat_2", "Concat_3", "Mul_2", "briq_", "output0", "dfl/Reshape_1", "/model.23/Concat_output")
    out = {}
    for n in g.node:
        if n.op_type == "QuantizeLinear" and any(k in n.input[0] for k in keep):
            s = inits.get(n.input[1])
            if s is not None and s.size == 1:
                out[n.input[0]] = float(s)
    return out


def evaluate(family: str, model: Path, root: Path, rows: list[dict], folder: Path, meta: dict, threads: int, normalized: bool) -> dict:
    s = session(model, threads)
    name = s.get_inputs()[0].name
    detections = []
    for row in rows:
        out = s.run(None, {name: input_tensor(frame_rgb(root, row), "float32")})[0]
        detections.append(decode(family, rescale(family, out) if normalized else out, row["width"], row["height"]))
    metrics = write_accuracy(folder, rows, detections, meta)
    raw = folder / "predictions.jsonl"
    with raw.open("rb") as src, gzip.open(folder / "predictions.jsonl.gz", "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    raw.unlink()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-frames", type=int, default=64)
    args = parser.parse_args()
    import onnxruntime as ort

    root = args.artifact_root.resolve(strict=True)
    rows = load_manifest(MANIFEST)
    OUT.mkdir(parents=True, exist_ok=True)
    config = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "onnxruntime": ort.__version__, "onnx": onnx.__version__, "platform": platform.platform(),
              "calibration_manifest": CALIBRATION.as_posix(), "calibration_manifest_sha256": sha256(CALIBRATION),
              "test_manifest_sha256": sha256(MANIFEST), "threads": args.threads,
              "quantization": "quant_pre_process + quantize_static QDQ MinMax, per-channel QInt8 weights, QUInt8 "
                              "activations, ActivationSymmetric False, WeightSymmetric True, INT32 bias, no exclusions"}
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log_path = OUT / "models_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for family, variants in VARIANTS.items():
        source = root / SOURCES[family]
        for variant in variants:
            key = f"{family}_{variant}"
            fp32 = root / f"{STEM[family]}_{variant}_fp32.onnx"
            int8 = root / f"{STEM[family]}_{variant}_int8_full.onnx"
            entry = log.get(key, {})
            if not fp32.exists():
                entry["changes"] = build(family, variant, source, fp32)
            entry["fp32_check"] = entry.get("fp32_check") or fp32_check(family, source, fp32, root, rows[: args.check_frames], args.threads)
            print(key, "fp32 check", entry["fp32_check"], flush=True)
            if not entry["fp32_check"]["passed"]:
                raise SystemExit(f"{key}: normalized FP32 graph does not reproduce the original")
            if not int8.exists():
                quantize(fp32, int8, root)
            entry.update(source=SOURCES[family], source_sha256=sha256(source),
                         fp32_model=fp32.relative_to(root).as_posix(), fp32_sha256=sha256(fp32), fp32_bytes=fp32.stat().st_size,
                         int8_model=int8.relative_to(root).as_posix(), int8_sha256=sha256(int8), int8_bytes=int8.stat().st_size,
                         int8_quantize_linear=sum(1 for n in onnx.load(str(int8)).graph.node if n.op_type == "QuantizeLinear"),
                         int8_box_scales=box_scales(int8))
            log[key] = entry
            log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
            for precision, model in (("fp32", fp32), ("int8_full", int8)):
                folder = OUT / f"cpu_{key}_{precision}"
                if (folder / "metrics.json").exists():
                    continue
                meta = {"runtime": "onnxruntime-cpu", "onnxruntime_version": ort.__version__, "threads": args.threads,
                        "model": f"{key}_{precision}", "model_path": model.relative_to(root).as_posix(),
                        "model_sha256": sha256(model), "input_dtype": "float32", "platform": platform.platform(),
                        "boxes": "normalized by the graph, x640 in post-processing"}
                m = evaluate(family, model, root, rows, folder, meta, args.threads, normalized=True)
                print(f"{key} {precision}: mAP50={m['mAP50']:.4f} mAP50_95={m['mAP50_95']:.4f}", flush=True)
    summarize(root)


def summarize(root: Path) -> None:
    matrix = Path("paper_evidence/runtime/matrix")
    lines = ["# Box-coordinate normalization baseline (road test set, 2,417 frames)", "",
             "Retention = mAP@0.5:0.95 / FP32 of the same detector. Existing rows come from paper_evidence/runtime/matrix.", "",
             "| detector | variant | file (MB) | mAP@0.5 | mAP@0.5:0.95 | retention (%) | box/concat scales |",
             "|---|---|---:|---:|---:|---:|---|"]
    log = json.loads((OUT / "models_log.json").read_text(encoding="utf-8"))
    table = {}
    for family in VARIANTS:
        ref = json.loads((matrix / f"cpu_{family}_fp32" / "metrics.json").read_text(encoding="utf-8"))["mAP50_95"]
        rows = []
        for name in EXISTING:
            m = json.loads((matrix / f"cpu_{family}_{name}" / "metrics.json").read_text(encoding="utf-8"))
            path = Path(m.get("model_path", ""))
            size = root / path
            rows.append((name, size.stat().st_size / 1e6 if path.name and size.exists() else None, m, ""))
        for variant in VARIANTS[family]:
            key = f"{family}_{variant}"
            for precision in ("fp32", "int8_full"):
                f = OUT / f"cpu_{key}_{precision}" / "metrics.json"
                if not f.exists():
                    continue
                m = json.loads(f.read_text(encoding="utf-8"))
                size = log[key][f"{precision[:4]}_bytes"] / 1e6
                scales = ", ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in log[key].get("int8_box_scales", {}).items()) if precision != "fp32" else ""
                rows.append((f"{variant} {precision}", size, m, scales))
        table[family] = []
        for name, size, m, scales in rows:
            ret = m["mAP50_95"] / ref * 100
            table[family].append({"variant": name, "MB": size, "mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "retention": ret})
            lines.append(f"| {family} | {name} | {'-' if size is None else f'{size:.2f}'} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} | {ret:.1f} | {scales} |")
    (OUT / "summary.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Where does the full-INT8 detection collapse come from? Activation-side ablation.

weight_only_ablation.py showed that INT8 weights alone keep detections. This script
isolates the activations, starting from the full static QDQ graph (FP32 bias):

  a8sim_all     every activation Q/DQ pair kept, every weight restored to FP32
  a8sim_head    FP32 weights; only activations produced inside the head module
  a8sim_decode  FP32 weights; only the head's decode stage (head nodes outside the
                learned conv branches cv2.*/cv3.*: DFL, dist2bbox, sigmoid, concat,
                TopK/Gather). Its last tensor concatenates box coordinates (0-640)
                with class scores (0-1); one per-tensor UINT8 scale (~2.5) rounds
                every score to 0.
  a8sim_outconcat          FP32 weights; only the Q/DQ pair on that output concat
                           (is this one tensor sufficient for the collapse?)
  a8sim_decode_no_outconcat  FP32 weights; decode-stage activations except the output
                           concat (is it necessary?)
  int8_decode_excl  a real W8A8 QDQ model built with quantize_static that excludes
                only the decode-stage nodes (conv branches of the head stay INT8)

FP32 weights come from re-running the same quant_pre_process on the FP32 source,
whose initializer names match the QDQ graph's "<name>_quantized" tensors.
Accuracy is decoded exactly as runtime_matrix.py and written to
<matrix>/cpu_<family>_<variant>/. Existing files are never overwritten.

Usage: python scripts/paper/activation_ablation.py --artifact-root <checkout>
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.runtime_matrix import (
    MODELS,
    decode,
    frame_rgb,
    input_tensor,
    load_manifest,
    write_accuracy,
)

HEAD = {"v4": "model.23", "v3": "model.22"}
CONV_BRANCH = re.compile(r"cv[23]\.")


def prepared(source: Path, tmp: Path) -> Path:
    from onnxruntime.quantization import quant_pre_process

    out = tmp / f"{source.stem}_prepared.onnx"
    # identical arguments to quantize_detector_variants.py
    quant_pre_process(str(source), str(out), skip_optimization=False, skip_onnx_shape=False, skip_symbolic_shape=True)
    return out


def stage(node_name: str, head: str) -> str:
    if f"/{head}/" not in node_name:
        return "body"
    return "branch" if CONV_BRANCH.search(node_name) else "decode"


def output_concat(g: onnx.GraphProto) -> str:
    """The tensor quantized right before the graph output: DQ(Q(concat)) -> output."""
    producer = {o: n for n in g.node for o in n.output}
    dq = producer[g.output[0].name]
    if dq.op_type != "DequantizeLinear":
        raise ValueError(f"graph output is produced by {dq.op_type}, not DequantizeLinear")
    q = producer[dq.input[0]]
    if q.op_type != "QuantizeLinear" or producer[q.input[0]].op_type != "Concat":
        raise ValueError("expected Concat -> QuantizeLinear -> DequantizeLinear -> output")
    return q.input[0]


def activation_only(qdq: Path, fp32_prepared: Path, head: str, scope: str, target: Path,
                    only: set[str] | None = None) -> dict:
    """FP32 weights with the activation Q/DQ pairs of `scope` kept (or, with `only`, exactly
    the pairs quantizing those tensors; used by decode_tensor_scan.py)."""
    model = onnx.load(str(qdq))
    g = model.graph
    fp = {t.name: t for t in onnx.load(str(fp32_prepared)).graph.initializer}
    inits = {t.name: t for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    out_concat = output_concat(g)
    graph_outputs = {o.name for o in g.output}
    identities = []
    rename: dict[str, str] = {}
    drop: set[int] = set()
    new_inits = []
    restored = dequantized_fallback = 0
    # 1) every DequantizeLinear on an initializer (weights, quantized constants) -> original FP32 tensor
    for n in g.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in inits:
            name = n.input[0].removesuffix("_quantized")
            if name in fp:
                t = onnx.TensorProto()
                t.CopyFrom(fp[name])
                restored += 1
            else:  # not expected; keep the dequantized value so the graph stays valid
                q = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
                s = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
                z = numpy_helper.to_array(inits[n.input[2]]).astype(np.float32) if len(n.input) > 2 else 0
                axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
                shape = [1] * q.ndim
                if np.ndim(s) == 1 and q.ndim > 1:
                    shape[axis] = -1
                t = numpy_helper.from_array(((q - np.reshape(z, shape)) * np.reshape(s, shape)).astype(np.float32), name)
                dequantized_fallback += 1
            t.name = f"{name}__fp32"
            new_inits.append(t)
            rename[n.output[0]] = t.name
            drop.add(id(n))
    # 2) activation Q/DQ pairs outside the requested scope are removed
    kept = removed = 0
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in inits:
            continue
        p = producer.get(n.input[0])
        where = stage(p.name, head) if p is not None else "body"  # graph input -> body
        is_out = n.input[0] == out_concat
        keep = (scope == "all" or (scope == "head" and where != "body") or (scope == "decode" and where == "decode")
                or (scope == "outconcat" and is_out) or (scope == "decode_no_outconcat" and where == "decode" and not is_out))
        if only is not None:
            keep = n.input[0] in only
        if keep:
            kept += 1
            continue
        removed += 1
        drop.add(id(n))
        for dq in consumers.get(n.output[0], []):
            if dq.op_type == "DequantizeLinear":
                drop.add(id(dq))
                if dq.output[0] in graph_outputs:  # keep the output name: Identity(concat) -> output
                    identities.append(onnx.helper.make_node("Identity", [n.input[0]], [dq.output[0]],
                                                            name=f"{dq.name}_fp32_identity"))
                else:
                    rename[dq.output[0]] = n.input[0]
    nodes = [n for n in g.node if id(n) not in drop] + identities
    for n in nodes:
        for k, i in enumerate(n.input):
            while i in rename:
                i = rename[i]
            n.input[k] = i
    for o in g.output:
        if o.name in rename:
            raise ValueError(f"graph output {o.name} would be renamed")
    del g.node[:]
    g.node.extend(nodes)
    g.initializer.extend(new_inits)
    used = {i for n in g.node for i in n.input}
    keep_inits = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    onnx.save(model, str(target))
    return {"weights_restored_fp32": restored, "dequantized_fallback": dequantized_fallback,
            "activation_qdq_kept": kept, "activation_qdq_removed": removed, "output_concat": out_concat}


def decode_excluded(source_prepared: Path, head: str, target: Path, root: Path, calibration: Path) -> dict:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    from scripts.paper.quantize_detector_variants import ManifestCalibration

    g = onnx.load(str(source_prepared)).graph
    excluded = [n.name for n in g.node if stage(n.name, head) == "decode"]
    quantize_static(
        model_input=str(source_prepared), model_output=str(target),
        calibration_data_reader=ManifestCalibration(root, calibration),
        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        per_channel=True, reduce_range=False, nodes_to_exclude=excluded,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, "EnableSubgraph": True},
    )
    return {"excluded_nodes": len(excluded), "output_bytes": target.stat().st_size}


def evaluate(root: Path, rows: list[dict], family: str, model_path: Path, folder: Path, meta: dict, threads: int) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dets = [decode(family, session.run(None, {name: input_tensor(frame_rgb(root, r), "float32")})[0], r["width"], r["height"])
            for r in rows]
    return write_accuracy(folder, rows, dets, {"runtime": "onnxruntime-cpu", "onnxruntime_version": ort.__version__,
                                               "threads": threads, "model_path": model_path.relative_to(root).as_posix(),
                                               "model_sha256": sha256(model_path), "platform": platform.platform(), **meta})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    parser.add_argument("--calibration", type=Path, default=Path("paper_evidence/splits/calibration_manifest.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--families", nargs="+", default=["v4", "v3"])
    parser.add_argument("--scopes", nargs="+", default=["all", "head", "decode", "outconcat", "decode_no_outconcat"])
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    rows = load_manifest(args.manifest)
    log = []
    for family in args.families:
        head = HEAD[family]
        source = root / MODELS[f"{family}_fp32"][1]
        full_qdq = root / MODELS[f"{family}_int8_full_fbias"][1]
        with tempfile.TemporaryDirectory(prefix="edge_sign_act_") as tmp:
            prep = prepared(source, Path(tmp))
            for scope in args.scopes:
                key = f"{family}_a8sim_{scope}"
                folder = args.matrix / f"cpu_{key}"
                if (folder / "metrics.json").exists():
                    print(f"skip existing {folder}")
                    continue
                target = source.with_name(f"{source.stem}_a8sim_{scope}.onnx")
                info = activation_only(full_qdq, prep, head, scope, target)
                meta = {"model": key, "quantization": f"activation-only: weights FP32, activation QDQ scope={scope}",
                        "base_qdq": MODELS[f"{family}_int8_full_fbias"][1], **info}
                m = evaluate(root, rows, family, target, folder, meta, args.threads)
                log.append({**meta, "mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"]})
                print(json.dumps(log[-1]), flush=True)
            key = f"{family}_int8_decode_excl"
            folder = args.matrix / f"cpu_{key}"
            target = source.with_name(f"{source.stem.replace('_fp32', '')}_int8_decode_excluded.onnx")
            if (folder / "metrics.json").exists():
                print(f"skip existing {folder}")
            else:
                info = {} if target.exists() else decode_excluded(prep, head, target, root, args.calibration)
                meta = {"model": key, "quantization": "static W8A8 QDQ, head decode stage excluded (conv branches INT8), INT32 bias",
                        "output_bytes": target.stat().st_size, **info}
                m = evaluate(root, rows, family, target, folder, meta, args.threads)
                log.append({**meta, "mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"]})
                print(json.dumps(log[-1]), flush=True)
    if log:
        path = Path("paper_evidence/models/activation_ablation_log.json")
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        path.write_text(json.dumps(previous + log, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

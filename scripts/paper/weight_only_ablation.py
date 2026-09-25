"""Weight-only INT8 ablation: is the full-QDQ head collapse caused by weights or activations?

Full static QDQ quantizes weights and activations and collapses to mAP 0; the
head-excluded QDQ recovers. This script keeps every activation in FP32 and only
rounds Conv weights to the per-output-channel symmetric INT8 grid (the same weight
scheme as the QDQ variants), in these scopes:

  head        Conv weights of the last model.N module only
  all         every Conv weight in the graph
  head_nodfl  (YOLOv8s) head Conv weights except the fixed DFL integral kernel (dfl/conv),
              to test whether rounding that 0..15 kernel explains a localization loss

If the weight-only models keep FP32 accuracy, the collapse comes from activation
quantization in the head. Graphs stay FP32, so these files say nothing about size
or speed. Outputs: model_space/<source>_w8sim_<scope>.onnx (never overwritten) and
<matrix>/cpu_<family>_w8sim_<scope>/metrics.json, decoded exactly as runtime_matrix.py.

Usage: python scripts/paper/weight_only_ablation.py --artifact-root <checkout>
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.runtime_matrix import MODELS, decode, frame_rgb, input_tensor, load_manifest, write_accuracy


def fake_quant(w: np.ndarray) -> np.ndarray:
    scale = np.abs(w).reshape(w.shape[0], -1).max(axis=1) / 127
    scale = np.where(scale == 0, 1, scale).reshape(-1, *([1] * (w.ndim - 1)))
    return (np.clip(np.round(w / scale), -128, 127) * scale).astype(w.dtype)


def sqnr_db(w: np.ndarray, q: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(w.astype(np.float64) ** 2) / max(np.sum((w - q).astype(np.float64) ** 2), 1e-30)))


def build(source: Path, target: Path, scope: str) -> dict:
    model = onnx.load(str(source))
    graph = model.graph
    index = max(int(m.group(1)) for n in graph.node if (m := re.search(r"model\.(\d+)", n.name)))
    head = f"model.{index}"
    inits = {t.name: t for t in graph.initializer}
    done, sqnrs = 0, []
    for node in graph.node:
        if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] not in inits:
            continue
        if scope in ("head", "head_nodfl") and head not in node.name:
            continue
        if scope == "head_nodfl" and "/dfl/" in node.name:
            continue
        tensor = inits[node.input[1]]
        w = numpy_helper.to_array(tensor)
        q = fake_quant(w)
        sqnrs.append(sqnr_db(w, q))
        tensor.CopyFrom(numpy_helper.from_array(q, tensor.name))
        done += 1
    if not target.exists():
        onnx.save(model, str(target))
    return {"head_module": head, "conv_weights_quantized": done,
            "weight_sqnr_db_mean": float(np.mean(sqnrs)), "weight_sqnr_db_min": float(np.min(sqnrs))}


def main() -> None:
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    rows = load_manifest(args.manifest)
    for family in ("v4", "v3"):
        source = root / MODELS[f"{family}_fp32"][1]
        for scope in ("head", "all") + (("head_nodfl",) if family == "v3" else ()):
            key = f"{family}_w8sim_{scope}"
            folder = args.matrix / f"cpu_{key}"
            if (folder / "metrics.json").exists():
                print(f"skip existing {folder}")
                continue
            target = source.with_name(f"{source.stem}_w8sim_{scope}.onnx")
            info = build(source, target, scope)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = args.threads
            session = ort.InferenceSession(str(target), opts, providers=["CPUExecutionProvider"])
            name = session.get_inputs()[0].name
            dets = [decode(family, session.run(None, {name: input_tensor(frame_rgb(root, r), "float32")})[0],
                           r["width"], r["height"]) for r in rows]
            m = write_accuracy(folder, rows, dets, {
                "runtime": "onnxruntime-cpu", "onnxruntime_version": ort.__version__, "threads": args.threads,
                "model": key, "model_path": target.relative_to(root).as_posix(), "model_sha256": sha256(target),
                "source_sha256": sha256(source), "quantization": "weight-only per-channel symmetric INT8 fake-quant; activations FP32",
                "scope": scope, **info, "platform": platform.platform()})
            print(json.dumps({"model": key, "mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], **info}), flush=True)


if __name__ == "__main__":
    main()

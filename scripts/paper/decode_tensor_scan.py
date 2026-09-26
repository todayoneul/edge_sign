"""Which decode-stage activation tensors collapse detection on their own?

For every activation tensor that the full static QDQ graph (FP32 bias) quantizes inside
the head's decode stage (activation_ablation.stage == "decode"), build a model with FP32
weights and only that tensor's Q/DQ pair, and evaluate mAP on every 10th test frame
(242 frames: a screening subset, not a reported accuracy). The tensor's UINT8 grid is
recorded as well: scale s, zero point z and the representable range [-z s, (255 - z) s].
Values closer than s/2 to zero round to the zero point, so a tensor that carries
class scores (0-1) next to box coordinates (0-640) loses every score when s > 2.

With --confirm, the scan's result is checked on the full test split (write_accuracy,
<matrix>/cpu_<family>_a8sim_decode_no_collapse/): FP32 weights and every decode-stage
activation quantized except the tensors that collapsed detection on their own. If
detection survives, at least one of those tensors is necessary for the collapse.

Usage: python scripts/paper/decode_tensor_scan.py --artifact-root <checkout> [--confirm]
Output: paper_evidence/runtime/matrix/decode_tensor_scan.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.activation_ablation import HEAD, activation_only, output_concat, prepared, stage
from scripts.paper.evaluate_qdq_detection import compute_metrics
from scripts.paper.runtime_matrix import MODELS, decode, frame_rgb, input_tensor, load_manifest


def decode_tensors(qdq: Path, head: str) -> list[dict]:
    g = onnx.load(str(qdq)).graph
    inits = {t.name: t for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    out = output_concat(g)
    found = []
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in inits:
            continue
        p = producer.get(n.input[0])
        if p is None or stage(p.name, head) != "decode":
            continue
        s = float(numpy_helper.to_array(inits[n.input[1]]))
        z = int(numpy_helper.to_array(inits[n.input[2]])) if len(n.input) > 2 else 0
        users = sorted({c.op_type for dq in consumers.get(n.output[0], []) for d in dq.output
                        for c in consumers.get(d, [])})
        found.append({"tensor": n.input[0], "producer": p.name, "producer_op": p.op_type, "consumers": users,
                      "scale": s, "zero_point": z, "range": [-z * s, (255 - z) * s],
                      "is_output_concat": n.input[0] == out})
    return found


def evaluate(model: Path, family: str, frames: list[np.ndarray], rows: list[dict], threads: int) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dets = [decode(family, session.run(None, {name: input_tensor(f, "float32")})[0], r["width"], r["height"])
            for f, r in zip(frames, rows, strict=True)]
    m = compute_metrics([{"classes": r["classes"], "boxes_xyxy": r["boxes_xyxy"]} for r in rows], dets)
    return {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "detections_conf_0.25": sum(
        1 for d in dets for x in d if x["confidence"] >= 0.25)}


def confirm(args: argparse.Namespace, root: Path, all_rows: list[dict]) -> None:
    """Full-test check: all decode activations quantized except the scan's collapsing tensors."""
    from scripts.paper.activation_ablation import evaluate as evaluate_full

    scan = json.loads((args.matrix / "decode_tensor_scan.json").read_text(encoding="utf-8"))
    for family in args.families:
        tensors = scan["families"][family]["tensors"]
        collapsing = {t["tensor"] for t in tensors if t["mAP50_95"] == 0}
        keep = {t["tensor"] for t in tensors} - collapsing
        folder = args.matrix / f"cpu_{family}_a8sim_decode_no_collapse"
        if (folder / "metrics.json").exists():
            print(f"skip existing {folder}")
            continue
        source = root / MODELS[f"{family}_fp32"][1]
        target = source.with_name(f"{source.stem}_a8sim_decode_no_collapse.onnx")
        with tempfile.TemporaryDirectory(prefix="edge_sign_scan_") as tmp:
            prep = prepared(source, Path(tmp))
            info = activation_only(root / MODELS[f"{family}_int8_full_fbias"][1], prep, HEAD[family], "none", target, only=keep)
        meta = {"model": f"{family}_a8sim_decode_no_collapse",
                "quantization": "activation-only: weights FP32, decode-stage activations except the tensors that "
                                "collapsed detection alone in decode_tensor_scan.json",
                "excluded_tensors": sorted(collapsing), **info}
        m = evaluate_full(root, all_rows, family, target, folder, meta, args.threads)
        print(f"{family} decode without {len(collapsing)} collapsing tensors: mAP50 {m['mAP50']:.4f} "
              f"mAP50-95 {m['mAP50_95']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    parser.add_argument("--matrix", type=Path, default=Path("paper_evidence/runtime/matrix"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--families", nargs="+", default=["v4", "v3"])
    parser.add_argument("--confirm", action="store_true", help="only run the full-test necessity check")
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    all_rows = load_manifest(args.manifest)
    if args.confirm:
        confirm(args, root, all_rows)
        return
    index = list(range(0, len(all_rows), args.stride))
    rows = [all_rows[i] for i in index]
    frames = [frame_rgb(root, r) for r in rows]
    result: dict = {"frames": len(rows), "stride": args.stride, "note": "screening subset; see cpu_*/metrics.json for "
                    "the full-test accuracy of the reported variants", "families": {}}
    for family in args.families:
        # FP32 reference on the same frames, from the saved full-test predictions
        packed = args.matrix / f"cpu_{family}_fp32" / "predictions.jsonl.gz"
        saved = [json.loads(x) for x in gzip.decompress(packed.read_bytes()).decode("utf-8").splitlines()]
        ref = compute_metrics([saved[i]["ground_truth"] for i in index], [saved[i]["detections"] for i in index])
        qdq = root / MODELS[f"{family}_int8_full_fbias"][1]
        tensors = decode_tensors(qdq, HEAD[family])
        with tempfile.TemporaryDirectory(prefix="edge_sign_scan_") as tmp:
            prep = prepared(root / MODELS[f"{family}_fp32"][1], Path(tmp))
            for k, t in enumerate(tensors):
                target = Path(tmp) / f"{family}_{k}.onnx"
                activation_only(qdq, prep, HEAD[family], "none", target, only={t["tensor"]})
                t.update(evaluate(target, family, frames, rows, args.threads))
                t["retention"] = t["mAP50_95"] / ref["mAP50_95"]
                target.unlink()
                print(f"{family} {k:2d} {t['producer_op']:>10s} {t['producer'][-40:]:>40s} s={t['scale']:.4f} "
                      f"z={t['zero_point']:3d} range=[{t['range'][0]:.2f},{t['range'][1]:.2f}] "
                      f"mAP50-95={t['mAP50_95']:.4f} ret={t['retention']:.3f}", flush=True)
        result["families"][family] = {"fp32_subset": {"mAP50": ref["mAP50"], "mAP50_95": ref["mAP50_95"]},
                                      "tensors": tensors}
    out = args.matrix / "decode_tensor_scan.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

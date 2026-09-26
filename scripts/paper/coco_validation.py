"""External validation on a standard workload: YOLO11l on the MLPerf COCO safe subset.

Preregistered in paper_evidence/reports/COCO_VALIDATION_PLAN.md (commit 557dcc3) before
any measurement. Subcommands:

  prepare   Safe-subset manifest with the license-URL rule of MLCommons
            filter_coco_safe_images.py / create_safe_annotations.py (expected 1,525
            images), the subset annotation file, and a 500-image calibration manifest
            drawn with seed 0 from the val2017 images outside the subset.
  export    YOLO11l -> FP32 ONNX (opset 14, static 640, simplified, as the Edge-Sign
            detectors), FP16 (FP32 I/O, same converter as the YOLOv8s FP16), static INT8
            QDQ variants with the Edge-Sign settings (quantize_detector_variants.py), and
            the activation-only diagnostics (activation_ablation.py).
  accuracy  ORT CPU inference on the subset: 640 letterbox, Ultralytics validation
            post-processing (conf 0.001, multi-label, class-wise NMS IoU 0.7, max 300),
            pycocotools mAP. Writes <out>/cpu_<variant>/{metrics.json, detections.npz}.
  bootstrap Paired image-level bootstrap of retention vs FP32 (pycocotools accumulate()
            reimplemented with image weights; checked against pycocotools at weight 1).

Existing outputs are never overwritten.

Usage (convnext_env):
    python scripts/paper/coco_validation.py prepare  --artifact-root <checkout>
    python scripts/paper/coco_validation.py export   --artifact-root <checkout>
    python scripts/paper/coco_validation.py accuracy --artifact-root <checkout> [--variants ...]
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256

SAFE_KEYWORDS = [  # MLCommons inference/vision/classification_and_detection/yolo/*.py
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/licenses/by-nd",
    "flickr.com/commons/usage",
    "www.usa.gov",
]
COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
            85, 86, 87, 88, 89, 90]  # YOLO class index -> COCO category_id (MLCommons yolo_ultra_map.py)
DATA = Path("data/coco")
ANNOTATIONS = DATA / "annotations/instances_val2017.json"
SAFE_ANNOTATIONS = DATA / "annotations/instances_val2017_safe.json"
EVIDENCE = Path("paper_evidence/coco")
MODEL_DIR = Path("model_space")
MODELS = {  # variant -> file under model_space/
    "fp32": "yolo11l_coco_fp32.onnx",
    "fp16": "yolo11l_coco_fp16.onnx",
    "int8_full": "yolo11l_coco_int8_full.onnx",
    "int8_full_fbias": "yolo11l_coco_int8_full_fbias.onnx",
    "int8_head_excl": "yolo11l_coco_int8_head_excluded.onnx",
    "int8_head_excl_fbias": "yolo11l_coco_int8_head_excluded_fbias.onnx",
    "int8_decode_excl": "yolo11l_coco_int8_decode_excluded.onnx",
    "a8sim_outconcat": "yolo11l_coco_fp32_a8sim_outconcat.onnx",
    "a8sim_decode_no_outconcat": "yolo11l_coco_fp32_a8sim_decode_no_outconcat.onnx",
}
INT8 = {  # variant -> (scope, bias_fp32); scope: full / head_excl / decode_excl
    "int8_full": ("full", False),
    "int8_full_fbias": ("full", True),
    "int8_head_excl": ("head_excl", False),
    "int8_head_excl_fbias": ("head_excl", True),
    "int8_decode_excl": ("decode_excl", False),
}


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Ultralytics LetterBox(auto=False, center=True): keep aspect, pad with 114."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, r, (left, top)


def input_tensor(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    boxed, r, pad = letterbox(image_bgr)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.transpose(rgb.astype(np.float32) / 255, (2, 0, 1))[None]), r, pad


def postprocess(output: np.ndarray, r: float, pad: tuple[int, int], width: int, height: int,
                conf: float = 0.001, iou: float = 0.7, max_det: int = 300, max_nms: int = 30000) -> list[dict]:
    """Ultralytics validation NMS: multi-label, class-wise, then the top max_det."""
    import torch
    from torchvision.ops import batched_nms

    pred = output.reshape(84, -1).T.astype(np.float32)
    anchors, classes = np.nonzero(pred[:, 4:] > conf)
    if anchors.size == 0:
        return []
    scores = pred[anchors, 4 + classes]
    if scores.size > max_nms:
        keep = np.argsort(-scores)[:max_nms]
        anchors, classes, scores = anchors[keep], classes[keep], scores[keep]
    cx, cy, bw, bh = pred[anchors, 0], pred[anchors, 1], pred[anchors, 2], pred[anchors, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    kept = batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(classes), iou).numpy()
    kept = kept[:max_det]  # batched_nms returns indices sorted by decreasing score
    boxes, scores, classes = boxes[kept], scores[kept], classes[kept]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad[0]) / r, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad[1]) / r, 0, height)
    return [{"category_id": COCO_IDS[int(c)], "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
             "score": float(s)} for b, s, c in zip(boxes, scores, classes, strict=True)]


def save_detections(path: Path, detections: list[dict]) -> None:
    """COCO-format detections as float32 arrays. The model outputs are float32, so this is lossless."""
    np.savez_compressed(
        path,
        image_id=np.array([d["image_id"] for d in detections], dtype=np.int64),
        category_id=np.array([d["category_id"] for d in detections], dtype=np.int16),
        bbox=np.array([d["bbox"] for d in detections], dtype=np.float32).reshape(-1, 4),
        score=np.array([d["score"] for d in detections], dtype=np.float32))


def load_detections(path: Path) -> list[dict]:
    data = np.load(path)
    return [{"image_id": int(i), "category_id": int(c), "bbox": [float(v) for v in b], "score": float(s)}
            for i, c, b, s in zip(data["image_id"], data["category_id"], data["bbox"], data["score"], strict=True)]


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")


def prepare(args: argparse.Namespace, root: Path) -> None:
    subset_path, calib_path = EVIDENCE / "safe_subset_manifest.jsonl", EVIDENCE / "calibration_manifest.jsonl"
    if subset_path.exists() or calib_path.exists():
        raise SystemExit("manifests exist; refusing to overwrite")
    coco = json.loads((root / ANNOTATIONS).read_text(encoding="utf-8"))
    licenses = {lic["id"]: lic for lic in coco["licenses"]}
    safe = [img for img in coco["images"]
            if any(k in licenses.get(img["license"], {}).get("url", "").lower() for k in SAFE_KEYWORDS)]
    safe_ids = {img["id"] for img in safe}
    annotations = [a for a in coco["annotations"] if a["image_id"] in safe_ids]
    out = root / SAFE_ANNOTATIONS
    if not out.exists():
        out.write_text(json.dumps({"info": coco.get("info", {}), "licenses": coco["licenses"], "images": safe,
                                   "annotations": annotations, "categories": coco["categories"]}), encoding="utf-8")

    def row(img: dict) -> dict:
        rel = (DATA / "val2017" / img["file_name"]).as_posix()
        return {"image_id": img["id"], "file_name": img["file_name"], "width": img["width"], "height": img["height"],
                "license": img["license"], "image_rel": rel, "image_sha256": sha256(root / rel)}

    subset = [row(img) for img in sorted(safe, key=lambda i: i["id"])]
    rest = sorted((img for img in coco["images"] if img["id"] not in safe_ids), key=lambda i: i["id"])
    calibration = [row(img) for img in sorted(random.Random(args.seed).sample(rest, args.calibration_images),
                                              key=lambda i: i["id"])]
    write_jsonl(subset_path, subset)
    write_jsonl(calib_path, calibration)
    tools = root / DATA / "mlperf_tools"
    summary = {
        "source": "COCO 2017 val (images.cocodataset.org), license rule of MLCommons inference yolo scripts",
        "val2017_images": len(coco["images"]), "safe_images": len(subset), "safe_annotations": len(annotations),
        "safe_crowd_annotations": sum(a.get("iscrowd", 0) for a in annotations),
        "licenses_in_subset": {licenses[i]["url"]: sum(1 for s in subset if s["license"] == i)
                               for i in sorted({s["license"] for s in subset})},
        "calibration_images": len(calibration), "calibration_seed": args.seed,
        "calibration_overlap_with_subset": len({c["image_id"] for c in calibration} & safe_ids),
        "annotations_sha256": sha256(root / ANNOTATIONS),
        "safe_annotations_sha256": sha256(out),
        "mlcommons_scripts_sha256": {p.name: sha256(p) for p in sorted(tools.glob("*.py"))} if tools.exists() else {},
    }
    (EVIDENCE / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "mlcommons_scripts_sha256"}, indent=2))
    if len(subset) != 1525:
        print(f"WARNING: expected 1,525 safe images, got {len(subset)}")


class Calibration(CalibrationDataReader):
    """onnxruntime CalibrationDataReader over the letterboxed calibration images."""

    def __init__(self, root: Path, rows: list[dict]):
        self.paths = [root / r["image_rel"] for r in rows]
        self.rewind()

    def _frames(self):
        for path in self.paths:
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            yield {"images": input_tensor(image)[0]}

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._frames())


def export(args: argparse.Namespace, root: Path) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quant_pre_process, quantize_static

    from scripts.paper.activation_ablation import activation_only, stage
    from scripts.paper.quantize_detector_variants import head_nodes

    models = root / MODEL_DIR
    log = []
    fp32 = models / MODELS["fp32"]
    if not fp32.exists():
        from ultralytics import YOLO

        weights = root / DATA / "yolo11l.pt"  # Ultralytics downloads the official asset to this path if missing
        yolo = YOLO(str(weights))
        exported = Path(yolo.export(format="onnx", imgsz=640, opset=14, dynamic=False, simplify=True))
        exported.replace(fp32)
        log.append({"variant": "fp32", "weights": str(yolo.ckpt_path), "weights_sha256": sha256(Path(yolo.ckpt_path))})
    fp16 = models / MODELS["fp16"]
    if not fp16.exists():
        from onnxconverter_common import float16

        model = onnx.shape_inference.infer_shapes(onnx.load(str(fp32)))
        half = float16.convert_float_to_float16(model, keep_io_types=True)
        del half.graph.value_info[:]  # same as scripts/export_fp16_detector.py
        onnx.save(half, str(fp16))
        log.append({"variant": "fp16", "converter": "onnxconverter_common.float16, keep_io_types=True"})
    if args.float_only:
        write_export_log(root, log)
        return
    calibration = read_manifest(EVIDENCE / "calibration_manifest.jsonl")
    with tempfile.TemporaryDirectory(prefix="edge_sign_coco_") as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(str(fp32), str(prepared), skip_optimization=False, skip_onnx_shape=False, skip_symbolic_shape=True)
        head, head_names = head_nodes(prepared)
        graph = onnx.load(str(prepared)).graph
        decode_names = [n.name for n in graph.node if stage(n.name, head) == "decode"]
        for variant, (scope, bias_fp32) in INT8.items():
            target = models / MODELS[variant]
            if target.exists():
                print(f"skip existing {target.name}")
                continue
            excluded = {"full": [], "head_excl": head_names, "decode_excl": decode_names}[scope]
            started = time.perf_counter()
            quantize_static(
                model_input=str(prepared), model_output=str(target), calibration_data_reader=Calibration(root, calibration),
                quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                per_channel=True, reduce_range=False, nodes_to_exclude=excluded,
                # same options as quantize_detector_variants.py. Do not add CalibMaxIntermediateOutputs:
                # in ORT 1.23.2 it discards collected ranges instead of merging them. MinMax only keeps
                # per-tensor ReduceMin/ReduceMax outputs, so 500 images fit in memory anyway.
                extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, "EnableSubgraph": True,
                               "QuantizeBias": not bias_fp32})
            log.append({"variant": variant, "head_module": head, "scope": scope, "bias": "fp32" if bias_fp32 else "int32",
                        "excluded_nodes": len(excluded), "calibration_images": len(calibration),
                        "seconds": round(time.perf_counter() - started, 1)})
            print(json.dumps(log[-1]), flush=True)
        for variant, scope in (("a8sim_outconcat", "outconcat"), ("a8sim_decode_no_outconcat", "decode_no_outconcat")):
            target = models / MODELS[variant]
            if target.exists():
                print(f"skip existing {target.name}")
                continue
            info = activation_only(models / MODELS["int8_full_fbias"], prepared, head, scope, target)
            log.append({"variant": variant, "base_qdq": MODELS["int8_full_fbias"], **info})
            print(json.dumps(log[-1]), flush=True)
    write_export_log(root, log)


def write_export_log(root: Path, log: list[dict]) -> None:
    for entry in log:
        path = root / MODEL_DIR / MODELS[entry["variant"]]
        entry.update(output=(MODEL_DIR / MODELS[entry["variant"]]).as_posix(), output_bytes=path.stat().st_size,
                     output_sha256=sha256(path))
    if log:
        log_path = EVIDENCE / "export_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        log_path.write_text(json.dumps(previous + log, indent=2) + "\n", encoding="utf-8")


def accuracy(args: argparse.Namespace, root: Path) -> None:
    import onnxruntime as ort
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rows = read_manifest(EVIDENCE / "safe_subset_manifest.jsonl")
    gt = COCO(str(root / SAFE_ANNOTATIONS))
    for variant in args.variants:
        folder = EVIDENCE / f"cpu_{variant}"
        if (folder / "metrics.json").exists():
            print(f"skip existing {folder}")
            continue
        path = root / MODEL_DIR / MODELS[variant]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        detections, started = [], time.perf_counter()
        for k, row in enumerate(rows):
            image = cv2.imread(str(root / row["image_rel"]))
            tensor, r, pad = input_tensor(image)
            output = session.run(None, {name: tensor})[0]
            for d in postprocess(output, r, pad, row["width"], row["height"]):
                detections.append({"image_id": row["image_id"], **d})
            if (k + 1) % 250 == 0:
                print(f"{variant} {k + 1}/{len(rows)}", flush=True)
        seconds = time.perf_counter() - started
        metrics = {"variant": variant, "model_path": (MODEL_DIR / MODELS[variant]).as_posix(), "model_sha256": sha256(path),
                   "images": len(rows), "detections": len(detections), "runtime": "onnxruntime-cpu",
                   "onnxruntime_version": ort.__version__, "threads": args.threads, "platform": platform.platform(),
                   "preprocess": "letterbox 640 (114), RGB, /255", "postprocess": "conf 0.001, multi-label, NMS 0.7, max 300",
                   "wall_seconds": round(seconds, 1)}
        if detections:
            dt = gt.loadRes(detections)
            evaluator = COCOeval(gt, dt, iouType="bbox")
            evaluator.params.imgIds = [r["image_id"] for r in rows]
            evaluator.evaluate()
            evaluator.accumulate()
            with io.StringIO() as buf:
                stdout, sys.stdout = sys.stdout, buf
                try:
                    evaluator.summarize()
                finally:
                    sys.stdout = stdout
            metrics.update(mAP50_95=float(evaluator.stats[0]), mAP50=float(evaluator.stats[1]),
                           coco_stats=[float(x) for x in evaluator.stats])
        else:
            metrics.update(mAP50_95=0.0, mAP50=0.0, coco_stats=None)
        folder.mkdir(parents=True, exist_ok=True)
        save_detections(folder / "detections.npz", detections)
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: metrics[k] for k in ("variant", "detections", "mAP50_95", "mAP50", "wall_seconds")}), flush=True)


class WeightedCocoAP:
    """pycocotools accumulate() for area 'all' and maxDet 100 with per-image weights.

    COCOeval.evaluate() is run once; a bootstrap resample then repeats each image's
    detections and ground truth by its draw count. With every weight equal to 1 this
    reproduces COCOeval.stats[0] (checked by the caller)."""

    def __init__(self, gt, detections: list[dict], img_ids: list[int]):
        from pycocotools.cocoeval import COCOeval

        evaluator = COCOeval(gt, gt.loadRes(detections), iouType="bbox")
        evaluator.params.imgIds = img_ids
        with io.StringIO() as buf:
            stdout, sys.stdout = sys.stdout, buf
            try:
                evaluator.evaluate()
            finally:
                sys.stdout = stdout
        p = evaluator.params
        self.rec_thrs, self.n_img = p.recThrs, len(p.imgIds)
        n_area, max_det = len(p.areaRng), p.maxDets[-1]
        self.cats = []
        for k in range(len(p.catIds)):
            entries = [(i, e) for i in range(self.n_img)
                       if (e := evaluator.evalImgs[k * n_area * self.n_img + i]) is not None]  # area 'all'
            if not entries:
                continue
            scores = np.concatenate([e["dtScores"][:max_det] for _, e in entries])
            owner = np.concatenate([np.full(len(e["dtScores"][:max_det]), i) for i, e in entries]).astype(np.int64)
            matches = np.concatenate([e["dtMatches"][:, :max_det] for _, e in entries], axis=1)
            ignore = np.concatenate([e["dtIgnore"][:, :max_det] for _, e in entries], axis=1)
            npig = np.zeros(self.n_img, dtype=np.int64)
            for i, e in entries:
                npig[i] = np.count_nonzero(np.asarray(e["gtIgnore"]) == 0)
            self.cats.append((scores, owner, matches.astype(bool), ignore.astype(bool), npig))

    def map50_95(self, weights: np.ndarray) -> float:
        values = []
        for scores, owner, matches, ignore, npig in self.cats:
            total = int((weights * npig).sum())
            if total == 0:
                continue
            idx = np.repeat(np.arange(scores.size), weights[owner])
            order = idx[np.argsort(-scores[idx], kind="mergesort")]
            tps = matches[:, order] & ~ignore[:, order]
            fps = ~matches[:, order] & ~ignore[:, order]
            tp_sum, fp_sum = np.cumsum(tps, axis=1, dtype=float), np.cumsum(fps, axis=1, dtype=float)
            for tp, fp in zip(tp_sum, fp_sum, strict=True):
                recall = tp / total
                precision = tp / (fp + tp + np.spacing(1))
                precision = np.maximum.accumulate(precision[::-1])[::-1]  # envelope, as accumulate()
                inds = np.searchsorted(recall, self.rec_thrs, side="left")
                q = np.zeros(len(self.rec_thrs))
                valid = inds < len(precision)
                q[valid] = precision[inds[valid]]
                values.append(q)
        return float(np.mean(values)) if values else 0.0


def bootstrap(args: argparse.Namespace, root: Path) -> None:
    from pycocotools.coco import COCO

    rows = read_manifest(EVIDENCE / "safe_subset_manifest.jsonl")
    img_ids = [r["image_id"] for r in rows]
    with io.StringIO() as buf:
        stdout, sys.stdout = sys.stdout, buf
        try:
            gt = COCO(str(root / SAFE_ANNOTATIONS))
        finally:
            sys.stdout = stdout

    def load(variant: str) -> WeightedCocoAP:
        dets = load_detections(EVIDENCE / f"cpu_{variant}" / "detections.npz")
        ap = WeightedCocoAP(gt, dets, img_ids)
        exact = json.loads((EVIDENCE / f"cpu_{variant}" / "metrics.json").read_text(encoding="utf-8"))["mAP50_95"]
        fast = ap.map50_95(np.ones(len(img_ids), dtype=np.int64))
        if abs(exact - fast) > 1e-9:
            raise SystemExit(f"{variant}: weighted AP {fast} != pycocotools {exact}")
        return ap

    ref = load("fp32")
    rng = np.random.default_rng(args.seed)
    draws = [np.bincount(rng.integers(0, len(img_ids), len(img_ids)), minlength=len(img_ids))
             for _ in range(args.resamples)]
    ref_values = np.array([ref.map50_95(w) for w in draws])
    out = {"resamples": args.resamples, "seed": args.seed, "unit": "image", "reference": "fp32", "pairs": {}}
    for variant in args.variants:
        ap = load(variant)
        ratios = np.array([ap.map50_95(w) for w in draws]) / ref_values
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        point = ap.map50_95(np.ones(len(img_ids), dtype=np.int64)) / ref.map50_95(np.ones(len(img_ids), dtype=np.int64))
        out["pairs"][variant] = {"retention": point, "ci95_low": float(lo), "ci95_high": float(hi),
                                 "share_below_0.99": float(np.mean(ratios < 0.99)),
                                 "share_below_0.95": float(np.mean(ratios < 0.95))}
        print(variant, json.dumps(out["pairs"][variant]), flush=True)
    (EVIDENCE / "bootstrap_retention.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "export", "accuracy", "bootstrap"])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-images", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=list(MODELS))
    parser.add_argument("--float-only", action="store_true", help="export: only FP32 and FP16")
    parser.add_argument("--resamples", type=int, default=1000, help="bootstrap: resamples")
    args = parser.parse_args()
    root = args.artifact_root.resolve(strict=True)
    {"prepare": prepare, "export": export, "accuracy": accuracy, "bootstrap": bootstrap}[args.command](args, root)


if __name__ == "__main__":
    main()

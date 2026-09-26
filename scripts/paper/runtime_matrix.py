"""Detector precision x runtime matrix: accuracy, latency and operator placement.

All runtimes receive the same input tensor: the frame is resized with cv2 (as in
evaluate_qdq_detection._preprocess) and converted to RGB uint8 640x640 on the
server; each runtime only divides by 255 (and casts to float16 for FP16 inputs).
Differences therefore come from the runtime's kernels, not from preprocessing.

Subcommands
  cpu-accuracy   ORT CPU over the frozen test manifest -> predictions.jsonl + metrics.json
  cpu-speed      ORT CPU latency on one fixed frame (default 1,024 measured iterations)
  serve          HTTP server for runtime_matrix.html (frames, models, results)
  browser        launch Chrome per run and wait for the page to submit its result
  pipeline       detector + recognizer in one page (browser_pipeline.html), each on its
                 own EP, e.g. --configs v4_fp16@webgpu+rec_fp32@wasm

Decoding: YOLO26 (v4) uses evaluate_qdq_detection.decode_v4 (no NMS). YOLOv8s (v3)
uses argmax class, confidence >= 0.001, class-wise NMS IoU 0.7 and at most 300
boxes, applied identically to every v3 variant.

Recognizer (rec_*, KoreanSignNet 32x32) keys are latency/placement only: the input is
the fixed frame resized to 32x32 and normalized as (x/255 - 0.5)/0.5; its accuracy
comes from recognizer_variants.py on the oracle test ROIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.evaluate_qdq_detection import CLASSES, compute_metrics, decode_v4

MODELS = {
    "v4_fp32": ("v4", "model_space/yolo_v4_signs_fp32.onnx"),
    "v4_fp16": ("v4", "model_space/yolo_v4_signs_fp16.onnx"),
    "v4_int8_full": ("v4", "model_space/yolo_v4_signs_int8_static.onnx"),
    "v4_int8_head_excl": ("v4", "model_space/yolo_v4_signs_int8_head_excluded.onnx"),
    "v4_int8_full_fbias": ("v4", "model_space/yolo_v4_signs_int8_full_fbias.onnx"),
    "v4_int8_head_excl_fbias": ("v4", "model_space/yolo_v4_signs_int8_head_excluded_fbias.onnx"),
    "v3_fp32": ("v3", "model_space/yolov8s_signs_v3_fp32.onnx"),
    "v3_fp16": ("v3", "model_space/yolov8s_signs_v3_fp16.onnx"),
    "v3_int8_full": ("v3", "model_space/yolov8s_signs_v3_int8_full.onnx"),
    "v3_int8_head_excl": ("v3", "model_space/yolov8s_signs_v3_int8_static.onnx"),
    "v3_int8_full_fbias": ("v3", "model_space/yolov8s_signs_v3_int8_full_fbias.onnx"),
    "v3_int8_head_excl_fbias": ("v3", "model_space/yolov8s_signs_v3_int8_head_excluded_fbias.onnx"),
    "rec_fp32": ("rec", "model_space/korean_sign_net_fp32.onnx"),
    "rec_fp16": ("rec", "model_space/korean_sign_net_fp16.onnx"),
    "rec_int8_full": ("rec", "model_space/korean_sign_net_int8_qdq_full.onnx"),
    "rec_int8_full_fbias": ("rec", "model_space/korean_sign_net_int8_qdq_full_fbias.onnx"),
    "rec_int8_head_excl": ("rec", "model_space/korean_sign_net_int8_qdq_head_excluded.onnx"),
    "rec_int8_head_excl_fbias": ("rec", "model_space/korean_sign_net_int8_qdq_head_excluded_fbias.onnx"),
    # external validation (coco_validation.py): YOLO11l on the MLPerf COCO safe subset. Latency only here;
    # accuracy uses letterbox + pycocotools in coco_validation.py (paper_evidence/coco/)
    "coco_fp32": ("coco", "model_space/yolo11l_coco_fp32.onnx"),
    "coco_fp16": ("coco", "model_space/yolo11l_coco_fp16.onnx"),
    "coco_int8_full": ("coco", "model_space/yolo11l_coco_int8_full.onnx"),
    "coco_int8_full_fbias": ("coco", "model_space/yolo11l_coco_int8_full_fbias.onnx"),
    "coco_int8_head_excl": ("coco", "model_space/yolo11l_coco_int8_head_excluded.onnx"),
    "coco_int8_head_excl_fbias": ("coco", "model_space/yolo11l_coco_int8_head_excluded_fbias.onnx"),
}
DETECTORS = [k for k, (family, _) in MODELS.items() if family in ("v3", "v4")]
INPUT_SIZE = {"v4": 640, "v3": 640, "rec": 32, "coco": 640}
CHROME = {
    "win32": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "darwin": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
}.get(sys.platform, "google-chrome")
HTML = Path(__file__).with_name("runtime_matrix.html")
PAGES = {"/runtime_matrix.html": HTML, "/browser_pipeline.html": Path(__file__).with_name("browser_pipeline.html")}


# ── shared helpers ────────────────────────────────────────────────────────────
def load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def frame_rgb(root: Path, row: dict, size: int = 640) -> np.ndarray:
    image = cv2.imread(str(root / row["image_rel"]))
    if image is None:
        raise FileNotFoundError(root / row["image_rel"])
    return cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)


def input_tensor(rgb: np.ndarray, dtype: str, family: str = "v4") -> np.ndarray:
    tensor = np.transpose(rgb.astype(np.float32) / 255, (2, 0, 1))[None]
    if family == "rec":
        tensor = (tensor - 0.5) / 0.5
    return tensor.astype(np.float16) if dtype == "float16" else tensor


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= threshold]
    return keep


def decode_v3(output: np.ndarray, *, width: int, height: int, min_conf: float = 0.001) -> list[dict]:
    raw = output.reshape(6, 8400).T
    if not np.isfinite(raw).all():
        raw = raw[np.isfinite(raw).all(axis=1)]
    scores = raw[:, 4:6]
    cls = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    mask = conf >= min_conf
    raw, cls, conf = raw[mask], cls[mask], conf[mask]
    cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * width / 640, 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * height / 640, 0, height)
    detections = []
    for class_id in CLASSES:
        idx = np.where(cls == class_id)[0]
        if idx.size == 0:
            continue
        for k in _nms(boxes[idx], conf[idx], 0.7):
            j = idx[k]
            if boxes[j, 2] > boxes[j, 0] and boxes[j, 3] > boxes[j, 1]:
                detections.append(
                    {
                        "box_xyxy": [float(v) for v in boxes[j]],
                        "confidence": float(conf[j]),
                        "class_id": int(class_id),
                        "class_name": CLASSES[class_id],
                    }
                )
    detections.sort(key=lambda d: -d["confidence"])
    return detections[:300]


def decode(family: str, output: np.ndarray, width: int, height: int) -> list[dict]:
    if family == "v4":
        return decode_v4(output.reshape(1, 300, 6), width=width, height=height)[0]
    return decode_v3(output, width=width, height=height)


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)

    def pct(q: float) -> float:
        k = (len(ordered) - 1) * q
        lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    mean = statistics.fmean(values)
    return {
        "n": len(values),
        "mean_ms": mean,
        "std_ms": statistics.pstdev(values),
        "p50_ms": pct(0.5),
        "p90_ms": pct(0.9),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "fps": 1000 / mean,
    }


def write_accuracy(folder: Path, rows: list[dict], detections: list[list[dict]], meta: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row, dets in zip(rows, detections, strict=True):
            gt = {"classes": row["classes"], "boxes_xyxy": row["boxes_xyxy"]}
            stream.write(
                json.dumps(
                    {"image_rel": row["image_rel"], "lighting": row["lighting"], "ground_truth": gt, "detections": dets},
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics = compute_metrics(
        [{"classes": r["classes"], "boxes_xyxy": r["boxes_xyxy"]} for r in rows], detections
    )
    metrics.update(meta, frames_evaluated=len(rows))
    (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def _session(root: Path, key: str, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / MODELS[key][1]), sess_options=opts, providers=["CPUExecutionProvider"])
    dtype = "float16" if session.get_inputs()[0].type == "tensor(float16)" else "float32"
    return session, dtype, ort.__version__


# ── ORT CPU ───────────────────────────────────────────────────────────────────
def cpu_accuracy(args) -> None:
    root = args.artifact_root.resolve(strict=True)
    rows = load_manifest(args.manifest, args.limit)
    for key in args.models:
        if key not in DETECTORS:
            raise SystemExit(f"{key}: recognizer accuracy is measured by recognizer_variants.py")
        folder = args.output / f"cpu_{key}"
        if (folder / "metrics.json").exists():
            print(f"skip existing {folder}")
            continue
        session, dtype, version = _session(root, key, args.threads)
        name = session.get_inputs()[0].name
        detections = []
        for row in rows:
            out = session.run(None, {name: input_tensor(frame_rgb(root, row), dtype)})[0]
            detections.append(decode(MODELS[key][0], out, row["width"], row["height"]))
        m = write_accuracy(
            folder,
            rows,
            detections,
            {
                "runtime": "onnxruntime-cpu",
                "onnxruntime_version": version,
                "threads": args.threads,
                "model": key,
                "model_path": MODELS[key][1],
                "model_sha256": sha256(root / MODELS[key][1]),
                "input_dtype": dtype,
                "platform": platform.platform(),
            },
        )
        print(f"cpu {key}: mAP50={m['mAP50']:.6f} mAP50_95={m['mAP50_95']:.6f}", flush=True)


def cpu_speed(args) -> None:
    root = args.artifact_root.resolve(strict=True)
    row = load_manifest(args.manifest)[args.frame_index]
    args.output.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        target = args.output / f"cpu_t{args.threads}_{key}.json"
        if target.exists():
            print(f"skip existing {target}")
            continue
        family = MODELS[key][0]
        rgb = frame_rgb(root, row, INPUT_SIZE[family])
        session, dtype, version = _session(root, key, args.threads)
        name = session.get_inputs()[0].name
        trace = []
        for i in range(args.warmup + args.iterations):
            t0 = time.perf_counter()
            tensor = input_tensor(rgb, dtype, family)
            t1 = time.perf_counter()
            session.run(None, {name: tensor})
            t2 = time.perf_counter()
            if i >= args.warmup:
                trace.append({"prepare_ms": (t1 - t0) * 1000, "inference_ms": (t2 - t1) * 1000})
        result = {
            "runtime": "onnxruntime-cpu",
            "onnxruntime_version": version,
            "threads": args.threads,
            "model": key,
            "model_sha256": sha256(root / MODELS[key][1]),
            "model_bytes": (root / MODELS[key][1]).stat().st_size,
            "input_dtype": dtype,
            "frame": row["image_rel"],
            "warmup": args.warmup,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "metrics": {k: summarize([t[k] for t in trace]) for k in ("prepare_ms", "inference_ms")},
            "trace": trace,
        }
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        inf = result["metrics"]["inference_ms"]
        print(f"cpu t{args.threads} {key}: mean {inf['mean_ms']:.2f} p90 {inf['p90_ms']:.2f} ms", flush=True)


# ── browser server ────────────────────────────────────────────────────────────
class State:
    def __init__(self, root: Path, rows: list[dict], output: Path):
        self.root, self.rows, self.output = root, rows, output
        self.frames: dict[tuple[int, int], bytes] = {}
        self.outputs: dict[str, dict[int, np.ndarray]] = {}
        self.lock = threading.Lock()

    def frame(self, index: int, size: int = 640) -> bytes:
        with self.lock:
            if (index, size) not in self.frames:
                self.frames[(index, size)] = frame_rgb(self.root, self.rows[index], size).tobytes()
                if len(self.frames) > 64:
                    self.frames.pop(next(iter(self.frames)))
            return self.frames[(index, size)]


def handler_factory(state: State, isolate: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, mime: str, headers: dict | None = None) -> None:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if isolate:  # crossOriginIsolated -> SharedArrayBuffer -> multi-thread WASM
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in PAGES:
                self._send(PAGES[url.path].read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/info":
                self._send(json.dumps({"frames": len(state.rows)}).encode(), "application/json")
            elif url.path == "/frame":
                i = int(query["i"][0])
                size = int(query.get("size", ["640"])[0])
                row = state.rows[i]
                self._send(state.frame(i, size), "application/octet-stream", {"X-Width": row["width"], "X-Height": row["height"]})
            elif url.path == "/model":
                key = query["key"][0]
                if key not in MODELS:
                    self.send_error(404, "unknown model")
                    return
                path = state.root / MODELS[key][1]
                self._send(path.read_bytes(), "application/octet-stream", {"X-Model-SHA256": sha256(path), "X-Family": MODELS[key][0]})
            else:
                self.send_error(404)

        def do_POST(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                if url.path == "/outputs":
                    run, start, count = query["run"][0], int(query["start"][0]), int(query["count"][0])
                    array = np.frombuffer(body, dtype=np.float32).reshape(count, -1)
                    with state.lock:
                        store = state.outputs.setdefault(run, {})
                        for k in range(count):
                            store[start + k] = array[k].copy()
                elif url.path == "/result":
                    result = json.loads(body)
                    run = result["run"]
                    target = state.output / f"{run}.json"
                    if target.exists():
                        raise FileExistsError(target)
                    if result.get("mode") == "accuracy" and result.get("status") == "completed":
                        family = MODELS[result["model"]][0]
                        store = state.outputs.pop(run, {})
                        n = result["frames"]
                        if sorted(store) != list(range(n)):
                            raise ValueError(f"missing outputs for {run}: {n - len(store)}")
                        rows = state.rows[:n]
                        dets = [decode(family, store[i], rows[i]["width"], rows[i]["height"]) for i in range(n)]
                        meta = {k: result[k] for k in ("runtime", "ep", "ort_web_version", "model", "input_dtype", "adapter", "user_agent") if k in result}
                        metrics = write_accuracy(state.output / run, rows, dets, meta)
                        result["mAP50"], result["mAP50_95"] = metrics["mAP50"], metrics["mAP50_95"]
                    trace = result.pop("trace", None)
                    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    if trace:
                        with (state.output / f"{run}_trace.csv").open("w", newline="", encoding="utf-8") as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                            writer.writeheader()
                            writer.writerows(trace)
                else:
                    self.send_error(404)
                    return
                self._send(b"ok", "text/plain")
            except (KeyError, ValueError, FileExistsError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, *args):
            pass

    return Handler


def serve(args) -> None:
    root = args.artifact_root.resolve(strict=True)
    rows = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer.request_queue_size = 128  # the default backlog of 5 can refuse bursts of local connections
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_factory(State(root, rows, args.output), args.isolate))
    print(f"runtime matrix server on http://127.0.0.1:{args.port} ({len(rows)} frames, isolate={args.isolate})", flush=True)
    server.serve_forever()


def model_input_dtype(root: Path, key: str) -> str:
    import onnx

    graph = onnx.load(str(root / MODELS[key][1]), load_external_data=False).graph
    return "float16" if graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 else "float32"


def _launch(args, run: str, page: str, query: dict) -> dict:
    """Open one headless Chrome on page?query and wait for <output>/<run>.json."""
    import psutil

    target = args.output / f"{run}.json"
    profile = Path(tempfile.mkdtemp(prefix="edge_sign_matrix_"))
    # headed runs keep the window from being throttled when it is covered or unfocused
    mode = (["--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
             "--disable-background-timer-throttling", "--window-size=640,480"] if args.headed else ["--headless=new"])
    command = [args.chrome, *mode, "--enable-unsafe-webgpu", "--no-first-run",
               "--no-default-browser-check", "--disable-background-networking",
               f"--user-data-dir={profile}", f"{args.server_url}/{page}?{urlencode(query)}"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not target.exists():
        time.sleep(0.5)
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    time.sleep(1)
    shutil.rmtree(profile, ignore_errors=True)
    if not target.exists():
        target.write_text(json.dumps({"run": run, "status": "timeout", "ort_web_version": args.ort, "timeout_s": args.timeout}, indent=2) + "\n", encoding="utf-8")
    return json.loads(target.read_text(encoding="utf-8"))


def pipeline(args) -> None:
    root = args.artifact_root.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        (det, det_ep), (rec, rec_ep) = (part.split("@") for part in config.split("+"))
        if MODELS[det][0] != "v4" or MODELS[rec][0] != "rec":
            raise SystemExit(f"{config}: pipeline expects a v4 detector and a rec_* recognizer")
        run = f"pipeline_t{args.wasm_threads}_ort{args.ort.replace('.', '')}_{det}@{det_ep}+{rec}@{rec_ep}"
        if args.tag:  # repeated launches of the same configuration (run-to-run variation)
            run += f"_{args.tag}"
        if (args.output / f"{run}.json").exists():
            print(f"skip existing {run}")
            continue
        result = _launch(args, run, "browser_pipeline.html", {
            "run": run, "ort": args.ort, "det": det, "det_ep": det_ep, "det_dtype": model_input_dtype(root, det),
            "rec": rec, "rec_ep": rec_ep, "rec_dtype": model_input_dtype(root, rec), "frames": args.frames,
            "warmup": args.warmup, "threads": args.wasm_threads, "conf": args.conf})
        m = result.get("metrics", {})
        print(f"{run}: {result.get('status')} "
              + (" ".join(f"{k[:-3]} {m[k]['mean_ms']:.2f}" for k in ("det_ms", "rec_ms", "total_ms")) + f" p90 {m['total_ms']['p90_ms']:.2f}" if m else "")
              + f" {result.get('reason', '')[:160]}", flush=True)


def browser(args) -> None:
    root = args.artifact_root.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        dtype = model_input_dtype(root, key)
        family = MODELS[key][0]
        if family == "rec" and args.mode == "accuracy":
            raise SystemExit(f"{key}: recognizer accuracy is measured by recognizer_variants.py")
        if family == "coco" and args.mode == "accuracy":
            raise SystemExit(f"{key}: COCO accuracy is measured by coco_validation.py")
        for ep in args.eps:
            tag = f"{ep}t{args.wasm_threads}" if ep == "wasm" and args.wasm_threads != 1 else ep
            run = f"{args.mode}_{tag}_ort{args.ort.replace('.', '')}_{key}"
            target = args.output / f"{run}.json"
            if target.exists():
                print(f"skip existing {run}")
                continue
            result = _launch(args, run, "runtime_matrix.html", {
                "run": run, "model": key, "ep": ep, "mode": args.mode, "ort": args.ort, "dtype": dtype,
                "warmup": args.warmup, "iterations": args.iterations, "limit": args.limit or 0,
                "threads": args.wasm_threads, "size": INPUT_SIZE[family], "norm": "pm1" if family == "rec" else "unit"})
            summary = result.get("metrics", {}).get("inference_ms", {})
            print(f"{run}: {result.get('status')} "
                  f"{'mean %.2f p90 %.2f ms' % (summary['mean_ms'], summary['p90_ms']) if summary else ''} "
                  f"{'mAP50_95 %.6f' % result['mAP50_95'] if 'mAP50_95' in result else ''} "
                  f"{result.get('reason', '')[:160]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifact-root", type=Path, required=True)
    common.add_argument("--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl"))
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--models", nargs="+", choices=list(MODELS), default=DETECTORS)
    p = sub.add_parser("cpu-accuracy", parents=[common])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("cpu-speed", parents=[common])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=1024)
    p.add_argument("--frame-index", type=int, default=0)
    p = sub.add_parser("serve", parents=[common])
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--isolate", action="store_true", help="send COOP/COEP so multi-thread WASM is available")
    p = sub.add_parser("browser", parents=[common])
    p.add_argument("--server-url", default="http://127.0.0.1:8791")
    p.add_argument("--eps", nargs="+", choices=["wasm", "webgpu"], default=["wasm", "webgpu"])
    p.add_argument("--mode", choices=["speed", "accuracy", "ops"], default="speed")
    p.add_argument("--ort", default="1.30.0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--wasm-threads", type=int, default=1)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--chrome", default=CHROME)
    p.add_argument("--headed", action="store_true", help="visible window (if headless Chrome has no GPU adapter)")
    p = sub.add_parser("pipeline", parents=[common])
    p.add_argument("--server-url", default="http://127.0.0.1:8791")
    p.add_argument("--configs", nargs="+", required=True, help="det_key@ep+rec_key@ep")
    p.add_argument("--tag", default="", help="suffix for repeated launches, e.g. r2")
    p.add_argument("--ort", default="1.30.0")
    p.add_argument("--frames", type=int, default=256)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--wasm-threads", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--chrome", default=CHROME)
    p.add_argument("--headed", action="store_true", help="visible window (if headless Chrome has no GPU adapter)")
    args = parser.parse_args()
    {"cpu-accuracy": cpu_accuracy, "cpu-speed": cpu_speed, "serve": serve, "browser": browser,
     "pipeline": pipeline}[args.command](args)


if __name__ == "__main__":
    main()

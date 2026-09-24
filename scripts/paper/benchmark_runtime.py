"""Repeatable ONNX Runtime detector benchmark with per-iteration latency traces."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.evaluate_qdq_detection import _preprocess, decode_v4

MODELS = {
    "fp32": "model_space/yolo_v4_signs_fp32.onnx",
    "fp16": "model_space/yolo_v4_signs_fp16.onnx",
    "full_qdq": "model_space/yolo_v4_signs_int8_static.onnx",
    "head_excluded_qdq": "model_space/yolo_v4_signs_int8_head_excluded.onnx",
}
PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def summarize(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty latency trace")
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.pstdev(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps": 1000 / statistics.mean(values),
        "n": len(values),
    }


def benchmark(
    artifact_root: Path,
    manifest: Path,
    output: Path,
    *,
    warmup: int = 10,
    iterations: int = 100,
    threads: int = 4,
    providers: tuple[str, ...] = ("cpu", "cuda"),
    frame_index: int = 0,
) -> dict:
    if warmup < 1 or iterations < 2 or threads < 1:
        raise ValueError("warmup>=1, iterations>=2, threads>=1 required")
    root = artifact_root.resolve(strict=True)
    frames = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    row = frames[frame_index]
    image = cv2.imread(str(root / row["image_rel"]))
    if image is None:
        raise ValueError(f"cannot decode {row['image_rel']}")
    height, width = image.shape[:2]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing benchmark: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for provider in providers:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider: {provider}")
        for name, relative in MODELS.items():
            run_id = f"{provider}_{name}"
            model = root / relative
            config = {
                "run_id": run_id,
                "model": relative,
                "model_sha256": sha256(model),
                "model_size_bytes": model.stat().st_size,
                "test_manifest_sha256": sha256(manifest),
                "frame_image_rel": row["image_rel"],
                "input_resolution": [640, 640],
                "original_resolution": [width, height],
                "warmup_count": warmup,
                "measured_iterations": iterations,
                "intra_op_threads": threads,
                "requested_providers": PROVIDERS[provider],
                "onnxruntime_version": ort.__version__,
                "python": platform.python_version(),
                "OS": platform.platform(),
                "CPU": platform.processor(),
                "timing_scope": "in-memory BGR frame; preprocess + ONNX session.run + decode; excludes disk decode/download/render",
            }
            result: dict = {"config": config, "status": "unsupported"}
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = threads
                t0 = time.perf_counter()
                session = ort.InferenceSession(
                    str(model), sess_options=opts, providers=PROVIDERS[provider]
                )
                config["session_init_ms"] = (time.perf_counter() - t0) * 1000
                actual_providers = session.get_providers()
                config["actual_providers"] = actual_providers
                if provider == "cuda" and "CUDAExecutionProvider" not in actual_providers:
                    raise RuntimeError(
                        "CUDAExecutionProvider unavailable; session fell back to CPU"
                    )
                input_spec = session.get_inputs()[0]
                dtype = np.float16 if input_spec.type == "tensor(float16)" else np.float32
                if input_spec.shape != [1, 3, 640, 640]:
                    raise ValueError(f"unexpected model input: {input_spec.shape}")

                def run_once(session=session, input_spec=input_spec, dtype=dtype) -> dict:
                    start = time.perf_counter()
                    tensor = _preprocess(image).astype(dtype, copy=False)
                    t_pre = time.perf_counter()
                    raw = session.run(None, {input_spec.name: tensor})[0]
                    t_run = time.perf_counter()
                    decoded, _ = decode_v4(raw, width=width, height=height, min_conf=0.25)
                    t_post = time.perf_counter()
                    return {
                        "preprocess_ms": (t_pre - start) * 1000,
                        "inference_ms": (t_run - t_pre) * 1000,
                        "postprocess_ms": (t_post - t_run) * 1000,
                        "detector_total_ms": (t_post - start) * 1000,
                        "detection_count": len(decoded),
                    }

                first = run_once()
                config["first_inference_ms"] = first["inference_ms"]
                for _ in range(warmup - 1):
                    run_once()
                trace = [dict(iteration=i, **run_once()) for i in range(iterations)]
                with (output / f"{run_id}_trace.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                    writer.writeheader()
                    writer.writerows(trace)
                result["status"] = "completed"
                result["metrics"] = {
                    key: summarize([float(x[key]) for x in trace])
                    for key in (
                        "preprocess_ms",
                        "inference_ms",
                        "postprocess_ms",
                        "detector_total_ms",
                    )
                }
                result["detection_count"] = trace[-1]["detection_count"]
            except Exception as exc:  # benchmark must preserve unsupported combinations
                result["reason"] = f"{type(exc).__name__}: {exc}"
            (output / f"{run_id}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            results[run_id] = result
            print(f"{run_id}: {result['status']}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("paper_evidence/splits/test_manifest.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("paper_evidence/runtime/ort"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--providers", nargs="+", choices=list(PROVIDERS), default=["cpu", "cuda"])
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()
    benchmark(
        args.artifact_root,
        args.manifest,
        args.output,
        warmup=args.warmup,
        iterations=args.iterations,
        threads=args.threads,
        providers=tuple(args.providers),
        frame_index=args.frame_index,
    )


if __name__ == "__main__":
    main()

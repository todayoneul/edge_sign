"""Calibration robustness: ONNX Runtime's Percentile and Entropy calibration instead of MinMax.

Two questions (paper 4.2), with predictions written before this script was run:
  1. Is the collapse of the mixed-range output tensor specific to MinMax? (full-graph INT8)
     Prediction: no. A range that keeps the 0-640 coordinates has s >= 2.51 and rounds every
     score below s/2 to zero; a range narrow enough to keep scores clips the coordinates.
  2. Does head-excluded INT8 reach the 99% retention criterion with another method?
     No prior; reported as measured.

Every model is built exactly like quantize_detector_variants.py (quant_pre_process, QDQ, per-channel
symmetric INT8 weights, asymmetric UINT8 activations, INT32 bias, the same 150 calibration frames);
only the calibration method changes. Settings (ONNX Runtime 1.23.2 calibrators and threshold code):
  percentile: 99.999 on 2048 bins, two-sided (CalibTensorRangeSymmetric=False). ORT's default is a
              symmetric |x| range, which wastes half of an asymmetric UINT8 activation's codes.
  entropy:    KL threshold search on 2048 bins with 128 quantized bins (TensorRT-style). quantize_static
              does not forward num_bins, and ORT's default 128/128 leaves a single candidate (the full
              range), so the two values are set on the calibrator. ORT's entropy ranges are symmetric.
ORT's histogram calibrators hold every activation of every calibration frame in memory before building
histograms (tens of GB for YOLOv8s at 150 frames). Here the same histograms are built in two streaming
passes over the calibration frames (global min/max, then fixed-range np.histogram per frame), which
gives the same counts as ORT's single np.histogram over all frames; thresholds then come from ORT's
own compute_percentile / compute_entropy. Test frames are never used.

Output: model_space/<stem>_int8_<scope>_<method>.onnx and paper_evidence/extra/calibration/.
Usage (convnext_env): python scripts/paper/calibration_robustness.py --artifact-root <checkout>
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paper.collect_environment import sha256
from scripts.paper.normalization_baseline import SOURCES, STEM, box_scales, session
from scripts.paper.runtime_matrix import (
    decode,
    frame_rgb,
    input_tensor,
    load_manifest,
    write_accuracy,
)

OUT = Path("paper_evidence/extra/calibration")
CALIBRATION = Path("paper_evidence/splits/calibration_manifest.jsonl")
MANIFEST = Path("paper_evidence/splits/test_manifest.jsonl")
METHODS = ["percentile", "entropy"]
SCOPES = ["full", "head_excl"]


def streaming_collect(self, data_reader) -> None:
    """HistogramCalibrater.collect_data in two streaming passes with ORT's collect_value histogram format.

    ORT stacks all frames and calls np.histogram once with range (-t, t), t = max |x| over all frames.
    Pass 1 finds the global min/max per tensor; pass 2 adds fixed-range histograms frame by frame, which
    yields identical counts without holding every frame's activations."""
    import numpy as np
    from onnxruntime.quantization.calibrate import HistogramCollector

    if self.symmetric:
        raise SystemExit("two-pass streaming reproduces collect_value (symmetric=False) only")
    names = [o.name for o in self.infer_session.get_outputs()]
    wanted = [n for n in names if n in self.tensors_to_calibrate]
    low: dict = {}
    high: dict = {}
    frames = 0
    while (inputs := data_reader.get_next()) is not None:
        outputs = dict(zip(names, self.infer_session.run(None, inputs), strict=True))
        for n in wanted:
            a = outputs[n]
            if a.size:
                lo, hi = np.nanmin(a), np.nanmax(a)
                low[n] = lo if n not in low else min(low[n], lo)
                high[n] = hi if n not in high else max(high[n], hi)
        frames += 1
    if frames == 0:
        raise ValueError("No data is collected.")
    data_reader.rewind()
    threshold = {n: np.array(max(abs(low[n]), abs(high[n])), dtype=low[n].dtype) for n in low}
    hist: dict = {}
    edges: dict = {}
    while (inputs := data_reader.get_next()) is not None:
        outputs = dict(zip(names, self.infer_session.run(None, inputs), strict=True))
        for n, t in threshold.items():
            h, e = np.histogram(outputs[n].flatten(), self.num_bins, range=(-t, t))
            hist[n] = h if n not in hist else hist[n] + h
            edges[n] = e
    collector = HistogramCollector(method=self.method, symmetric=self.symmetric, num_bins=self.num_bins,
                                   num_quantized_bins=self.num_quantized_bins, percentile=self.percentile,
                                   scenario=self.scenario)
    collector.histogram_dict = {n: (hist[n], edges[n], low[n], high[n], threshold[n]) for n in hist}
    self.collector = collector
    self.streamed_frames = frames


@contextlib.contextmanager
def streaming_histograms(settings: dict):
    import importlib

    from onnxruntime.quantization.calibrate import HistogramCalibrater

    # the package re-exports a function named quantize, so "import ...quantize as m" would bind it
    qmod = importlib.import_module("onnxruntime.quantization.quantize")

    original = qmod.create_calibrator

    def create(*args, **kwargs):
        calibrator = original(*args, **kwargs)
        if isinstance(calibrator, HistogramCalibrater):
            if calibrator.method == "entropy":  # not forwarded by quantize_static; 128/128 has one candidate
                calibrator.num_bins, calibrator.num_quantized_bins = 2048, 128
            calibrator.collect_data = types.MethodType(streaming_collect, calibrator)
            settings.update(method=calibrator.method, symmetric=bool(calibrator.symmetric), num_bins=calibrator.num_bins,
                            num_quantized_bins=calibrator.num_quantized_bins, percentile=calibrator.percentile)
        return calibrator

    qmod.create_calibrator = create
    try:
        yield
    finally:
        qmod.create_calibrator = original


def quantize(source: Path, target: Path, root: Path, method: str, head_excluded: bool) -> dict:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_static,
    )

    from scripts.paper.quantize_detector_variants import ManifestCalibration, head_nodes

    started = time.perf_counter()
    settings: dict = {}
    with tempfile.TemporaryDirectory(prefix="briq_calib_") as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(input_model_path=str(source), output_model_path=str(prepared), skip_optimization=False,
                          skip_onnx_shape=False, skip_symbolic_shape=True)
        head, excluded = head_nodes(prepared)
        with streaming_histograms(settings), contextlib.redirect_stdout(io.StringIO()):
            quantize_static(model_input=str(prepared), model_output=str(target),
                            calibration_data_reader=ManifestCalibration(root, CALIBRATION),
                            quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                            per_channel=True, reduce_range=False, nodes_to_exclude=excluded if head_excluded else [],
                            calibrate_method={"percentile": CalibrationMethod.Percentile,
                                              "entropy": CalibrationMethod.Entropy}[method],
                            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True,
                                           "EnableSubgraph": True, "QuantizeBias": True,
                                           "CalibTensorRangeSymmetric": False})
    if settings.get("method") != method:
        raise SystemExit(f"calibrator not patched: {settings}")
    return {"head_module": head, "excluded_nodes": len(excluded) if head_excluded else 0,
            "calibrator": settings, "seconds": round(time.perf_counter() - started, 1)}


def evaluate(family: str, model: Path, root: Path, rows: list[dict], folder: Path, meta: dict, threads: int) -> dict:
    s = session(model, threads)
    name = s.get_inputs()[0].name
    detections = [decode(family, s.run(None, {name: input_tensor(frame_rgb(root, r), "float32")})[0], r["width"], r["height"])
                  for r in rows]
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
    parser.add_argument("--families", nargs="+", default=["v4", "v3"])
    args = parser.parse_args()
    import onnxruntime as ort

    root = args.artifact_root.resolve(strict=True)
    rows = load_manifest(MANIFEST)
    OUT.mkdir(parents=True, exist_ok=True)
    config = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "onnxruntime": ort.__version__, "platform": platform.platform(),
              "calibration_manifest": CALIBRATION.as_posix(), "calibration_manifest_sha256": sha256(CALIBRATION),
              "test_manifest_sha256": sha256(MANIFEST),
              "methods": {"percentile": "ORT PercentileCalibrater: percentile 99.999, 2048 bins, two-sided (symmetric=False)",
                          "entropy": "ORT EntropyCalibrater thresholds with num_bins 2048, num_quantized_bins 128"},
              "streaming": "two passes: global min/max, then fixed-range per-frame histograms (= ORT batch counts)",
              "quantization": "quant_pre_process + quantize_static QDQ, per-channel QInt8 weights, QUInt8 activations, "
                              "ActivationSymmetric False, WeightSymmetric True, INT32 bias"}
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log_path = OUT / "models_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    for family in args.families:
        source = root / SOURCES[family]
        for method in METHODS:
            for scope in SCOPES:
                key = f"{family}_int8_{scope}_{method}"
                target = root / f"{STEM[family]}_int8_{scope}_{method}.onnx"
                entry = log.get(key, {})
                if not target.exists():
                    entry.update(quantize(source, target, root, method, head_excluded=scope == "head_excl"))
                entry.update(method=method, scope=scope, source=SOURCES[family], source_sha256=sha256(source),
                             model=target.relative_to(root).as_posix(), model_sha256=sha256(target),
                             bytes=target.stat().st_size, box_scales=box_scales(target))
                log[key] = entry
                log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
                folder = OUT / f"cpu_{key}"
                if not (folder / "metrics.json").exists():
                    meta = {"runtime": "onnxruntime-cpu", "onnxruntime_version": ort.__version__, "threads": args.threads,
                            "model": key, "model_path": entry["model"], "model_sha256": entry["model_sha256"],
                            "input_dtype": "float32", "platform": platform.platform(), "calibration": method}
                    m = evaluate(family, target, root, rows, folder, meta, args.threads)
                    print(f"{key}: mAP50={m['mAP50']:.4f} mAP50_95={m['mAP50_95']:.4f} ({entry.get('seconds', '-')} s)", flush=True)
    summarize()


def summarize() -> None:
    matrix = Path("paper_evidence/runtime/matrix")
    log = json.loads((OUT / "models_log.json").read_text(encoding="utf-8"))
    lines = ["# Calibration robustness (road test set, 2,417 frames)", "",
             "MinMax rows are the paper's existing models. Retention = mAP@0.5:0.95 / FP32.", "",
             "| detector | scope | calibration | mAP@0.5 | mAP@0.5:0.95 | retention (%) | output-concat / box scales |",
             "|---|---|---|---:|---:|---:|---|"]
    table = []
    for family in ("v3", "v4"):
        ref = json.loads((matrix / f"cpu_{family}_fp32" / "metrics.json").read_text(encoding="utf-8"))["mAP50_95"]
        for scope, existing in (("full", "int8_full"), ("head_excl", "int8_head_excl")):
            m = json.loads((matrix / f"cpu_{family}_{existing}" / "metrics.json").read_text(encoding="utf-8"))
            rows = [("minmax", m, "")]
            for method in METHODS:
                f = OUT / f"cpu_{family}_int8_{scope}_{method}" / "metrics.json"
                if f.exists():
                    key = f"{family}_int8_{scope}_{method}"
                    scales = ", ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in log.get(key, {}).get("box_scales", {}).items())
                    rows.append((method, json.loads(f.read_text(encoding="utf-8")), scales))
            for method, mm, scales in rows:
                ret = mm["mAP50_95"] / ref * 100
                table.append({"family": family, "scope": scope, "calibration": method, "mAP50": mm["mAP50"],
                              "mAP50_95": mm["mAP50_95"], "retention": ret})
                lines.append(f"| {family} | {scope} | {method} | {mm['mAP50']:.4f} | {mm['mAP50_95']:.4f} | {ret:.1f} | {scales} |")
    (OUT / "summary.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

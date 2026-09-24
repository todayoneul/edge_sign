# TIIS Evidence Revalidation Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement the checked tasks. The user supplied the execution order and requires the isolated `paper/tiis-evidence-revalidation` branch.

**Goal:** Produce immutable splits, raw task predictions, repeated latency traces, and a claim-by-claim evidence report for the attached nine-page Edge-Sign v2 PDF.

**Architecture:** Read ignored datasets and models from the original checkout through an explicit `--artifact-root`; write only small manifests, metrics, traces, and scripts to this worktree. Every experiment uses a frozen config and model hash. Historical summaries remain labeled separately from new measurements.

**Tech Stack:** Python 3.10, ONNX Runtime 1.23.2, OpenCV, NumPy, Ultralytics metrics, PyTorch, ONNX, React/ORT Web for browser measurements.

**Spec:** User's P0/P1 instructions in the current task and attached `Edge-Sign_개정원고_v2_검토본.pdf` (nine pages, supplied 2026-09-24).

## Global constraints

- Base is `origin/main` commit `095f3cd`; do not modify `main` or the dirty `docs/paper-draft-ksii-tiis` checkout.
- Never overwrite original ONNX, checkpoint, dataset, or historical experiment output.
- Do not commit dataset images, `.pt`, or `.onnx`; track only manifests, hashes, predictions, metrics, traces, figures, and reports.
- Use sequence-disjoint train/calibration/test lists; label taxonomy is `0=traffic_sign, 1=traffic_light` for the v4 main workload.
- Distinguish training validation, exported ONNX test, detector-only runtime, and full pipeline runtime.

## Review focus

- Missing or corrupt JSON/image: split builder records exclusions and fails if the test set becomes empty.
- Model output layout mismatch: evaluator rejects unsupported shapes instead of silently decoding boxes.
- Calibration/test overlap: split validation fails before inference.
- Unsupported provider/precision: benchmark records `unsupported` with the exact error.
- Interrupted run: writes use unique run directories and appendable JSONL/CSV traces; partial runs are not reported as complete.

## Tasks

### 1. Freeze source and split manifests

**Files:** `scripts/paper/build_test_split.py`, `paper_evidence/splits/{train,calibration,test}_manifest.jsonl`, `paper_evidence/splits/split_summary.json`, `tests/paper/test_split.py`.

- [ ] Parse the original `data/yolo_signs_v2` train labels, 150 ordered val calibration frames, and `data/aihub_traffic/test` JSON without copying images.
- [ ] Record per-image relative path, sequence, day/night, image dimensions, class labels, and pixel boxes where applicable.
- [ ] Assert no image or sequence overlap and summarize images, objects, class counts, exclusions, source and selection rule.
- [ ] Verify `--help`, a small `--limit-test` dry run, deterministic file hashes, then generate full manifests and commit.

### 2. Freeze model/environment lineage

**Files:** `scripts/paper/collect_environment.py`, `paper_evidence/models/model_manifest.csv`, `paper_evidence/environment/`.

- [ ] Hash the three v4 ONNX variants, corresponding checkpoint, and available v3 ONNX files; record bytes, opset, IO shape, graph quantization counts, known/unknown lineage.
- [ ] Capture CPU/GPU/OS/Python/ORT/browser settings without treating current environment as proof of historical measurements.
- [ ] Verify model input/output compatibility and commit the manifest.

### 3. Evaluate three detector variants on one test set

**Files:** `scripts/paper/evaluate_qdq_detection.py`, `tests/paper/test_detection_eval.py`, `paper_evidence/detection/yolo26_{fp32,full_qdq,head_excluded_qdq}/`.

- [ ] Implement one recorded preprocessing and decoding path, IoU matching, AP@0.5/AP@0.5:0.95, precision, recall, and per-class results.
- [ ] Save `config.yaml`, `metrics.json`, `predictions.jsonl`, `environment.txt`, and `command.txt` in distinct run directories.
- [ ] Run a tiny dry run, then all fixed test frames on CPU; compare the same frames and record matched IoU, detection counts, cosine similarity where supported.
- [ ] Commit scripts and small evidence files. Do not claim a failure mode if full-test results differ from historical notes.

### 4. Measure runtime and deployment scope

**Files:** `scripts/paper/benchmark_runtime.py`, browser harness under `scripts/paper/`, `paper_evidence/runtime/`, `tests/paper/test_benchmark.py`.

- [ ] Save warmup and every measured latency sample for CPU FP32/full QDQ/head-excluded QDQ; compute mean/std/p50/p95/FPS and session initialization separately.
- [ ] Attempt CUDA and browser WASM/WebGPU only on supported variants; record `unsupported` and errors when sessions fail.
- [ ] Measure detector-only and compatible detector→ByteTrack→KoreanSignNet pipeline separately on the same frame manifest, with actual deployed model bytes.
- [ ] Commit raw traces/configs and a three-way PASS/FAIL/NOT VERIFIED target table.

### 5. Recognition, tracking decision, figures, and report

**Files:** `scripts/paper/evaluate_recognition.py`, `paper_evidence/{recognition,tracking,figures,reports}/`, `paper_evidence/README.md`.

- [ ] Evaluate available independent KoreanSignNet images with per-class counts and confusion matrix; state when a genuinely independent test set is unavailable.
- [ ] Audit identity annotations; either produce manual-GT evaluation or exclude pseudo-GT tracking scores from main results.
- [ ] Generate only figures backed by new raw JSON/CSV, then write `TIIS_EVIDENCE_REPORT.md` with SUPPORTED/PARTIALLY SUPPORTED/NOT SUPPORTED claims and remaining P0/P1 work.
- [ ] Run script help/dry-run checks, baseline tests, `git diff --check`, review status/log, commit logical units, and push only this branch when verified.

# Additional experiments (2026-09-27)

These folders hold new evidence only. Nothing under `runtime/`, `coco/` or `models/` was changed.
- **Base commit.** Every run records the commit it started from (`09db022`). The new scripts were added to git in the commit that added this folder.
- **Models.** New model files live in `model_space/` (not in git). Their SHA-256 and sizes are logged here.
- **Data.** Calibration used the paper's 150 validation frames (`splits/calibration_manifest.jsonl`). Evaluation used the 2,417 sequence-disjoint test frames (`splits/test_manifest.jsonl`). The two sets share no frames.

| Folder | What | Script | Records |
|---|---|---|---|
| `size_bins/` | Retention per object-size bin. The hypothesis and decision rule were fixed before computing; COCOeval area rules, paired frame bootstrap. | `scripts/paper/size_bin_retention.py` | `size_bins.json` (commit, bins, per-condition AP/retention/CI, model hashes), `summary.md` |
| `mitigation/` | Known mitigation: box-coordinate normalization (vanilla rescaling, Ultralytics-style pre-normalization, stride folded into the regression conv). | `scripts/paper/normalization_baseline.py` | `DESIGN.md` (written before running, with predictions), `config.json`, `models_log.json` (FP32 check, hashes, sizes, QuantizeLinear counts, box/concat scales), `cpu_*/metrics.json` + `predictions.jsonl.gz`, `summary.md`, `size_bins.md`, `bootstrap_retention.json` |
| `calibration/` | ONNX Runtime Percentile (99.999, two-sided) and Entropy (KL, 2048/128 bins) calibration, for full and head-excluded INT8. | `scripts/paper/calibration_robustness.py` | `config.json`, `models_log.json` (calibrator settings, hashes, box scales), `cpu_*/`, `summary.md`, `bootstrap_retention.json` (frame and 25-frame block bootstrap, `bootstrap_extra.py`) |
| `end_to_end_v3/` | Detector → ByteTrack → recognizer accuracy for YOLOv8s. Same protocol as `runtime/end_to_end/`, with the deployed YOLOv8s decode (conf ≥ 0.1, class-wise NMS 0.45). | `scripts/paper/evaluate_end_to_end.py --detectors v3_*` | `summary.json` (model hashes, per-combination counts, retention CI), `per_frame.csv` |

Notes:
- **Calibration without holding all activations.** ORT's histogram calibrators keep every activation of every calibration frame in memory. `calibration_robustness.py` instead builds the same histograms in two streaming passes. It was checked on YOLO26-n with 8 frames: all 391 tensor ranges were identical to ORT's batch path, for both methods.
- **Size-bin script bug fixed before the full run.** The first version assumed a uniform 1/3 input scale. Road frames are resized anisotropically (1920×H → 640×640), so boxes are now mapped per axis. The bin thresholds (8 and 16 px) did not change.

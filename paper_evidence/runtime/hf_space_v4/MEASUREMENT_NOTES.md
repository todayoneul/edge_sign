# Hugging Face Space v4 runtime: run record

The **primary** runs are `20260925_head_excluded_qdq_final/` and
`20260925_fp32_final/`. No pilot runs were made on the Space for v4; the only
earlier runs were local functional smoke tests (2 warm-up + 4 measured frames)
that are not stored here and are not used as results.

## Deployment under test

- Space `gyann/edge-sign`, source commit `f645ad510f3fcbdccab4d8cda21d6de62ccaa4e2`
  on `cpu-basic`. This commit adds only `src/pipeline/paper_v4.py`, the
  opt-in registration in `src/pipeline/app.py`, two `.dockerignore` lines and the
  two YOLO26-n ONNX files to the previous v3 commit
  `9fc2593358a678a5b1597e978a63778bc909fc31`. The v3 demo routes and UI are
  unchanged.
- The route is enabled by the Space variable `EDGE_SIGN_PAPER_V4=1`.
- `/api/paper-v4/status` before the runs: `ready`, ONNX Runtime 1.23.2,
  CPUExecutionProvider, intra-op threads 2, `Linux-6.12.100 ... x86_64`, RSS
  234.7 MB. `os.cpu_count()` reported 16; this is the host count, not the
  `cpu-basic` allocation.
- Model files, identical in the Space repository metadata, the running app and
  the local manifest:

| Model | Bytes | SHA-256 |
|---|---:|---|
| `yolo_v4_signs_fp32.onnx` | 9,805,975 | `4133272b340bc2e788277edce1b4abe3e41126df9d697746a06315fd5603c6f1` |
| `yolo_v4_signs_int8_head_excluded.onnx` | 3,416,372 | `a2ddfe7d07adeb3cc58d413a0936ca3833260e99a4af227b01dea4df7326c9eb` |
| `korean_sign_net_fp32.onnx` | 116,860 | `a5d6584c984fe105c04d22f7ae026ba2fc4be4a97eb67a4997a1c03626ae47ca` |

## Protocol

Identical to the v3 primary runs in `../hf_space_v3/` except for the route:
the public `seoul_daylight.mp4` clip (15 frames, 5.28 source FPS, SHA-256
`68cc9cb6...` in each `config.json`) is looped by the client, offered open-loop
at 10 FPS to `/ws/paper-v4`, with a tracker reset, 10 warm-up frames and 50
measured frames. Each trace keeps 60 ordered responses (frame IDs 1–60). The
head-excluded QDQ run was made first, then FP32, on 2026-09-25 (UTC 12:5x).
Client: Windows 11, Python 3.13.11, OpenCV 4.13.0, websockets 15.0.1.

The v4 route differs from the v3 `/ws/stream` route in decoder, thresholds,
tracker lifetime and recognition (no temporal vote, no OCR); see
`../../reports/SPACE_V4_DEPLOYMENT_READINESS.md`. Results from the two routes
are therefore not a same-condition model comparison.

## Results

| Variant | Send interval mean | Space pipeline mean / p95 | Detector mean | Track mean | Recognize mean | Result delivery FPS | First send → last result FPS | Round trip p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head-excluded QDQ | 99.977 ms | 50.278 / 73.569 ms | 46.723 ms | 2.613 ms | 0.942 ms | 12.182 | 9.662 | 321.5 / 951.7 ms |
| FP32 | 100.264 ms | 77.619 / 98.077 ms | 73.706 ms | 2.957 ms | 0.957 ms | 11.884 | 9.509 | 546.3 / 1,009.8 ms |

Mean track count per measured frame: 4.52 (QDQ), 4.58 (FP32).

An independent recomputation from `trace.jsonl` reproduced the pipeline mean
and p95, round-trip p50 and p95, result delivery FPS, completion FPS and the
measured send interval exactly.

## Interpretation limits

- **The Space was not saturated.** The v4 pipeline (50–78 ms per frame) is
  shorter than the 100 ms offer interval, so completion FPS (9.5–9.7) tracks
  the 10 FPS offer. Result delivery FPS above 10 reflects the warm-up queue
  draining at the start of the measured window. These runs therefore do not
  measure the Space's maximum throughput. The v3 runs, whose pipeline took
  349–456 ms, were saturated; the two are not comparable as capacity figures.
- A server-side pipeline mean of 50.3 ms corresponds to about 20 frames/s for a
  single stream before any network or client cost, below 30 FPS. A 30 FPS
  deployment claim would need a pre-registered saturating workload (for
  example an offer of at least 30 FPS) and the browser path.
- On this `cpu-basic` Linux container the head-excluded QDQ detector was 1.58×
  faster than FP32 (46.7 vs 73.7 ms). The local Windows ORT CPU benchmark
  (`../ort/`) found the opposite (16.1 vs 21.6 ms). Thread counts, CPU and OS
  differ, and each Space figure is a single run on shared hardware, so this is
  an observation of runtime dependence, not a general INT8 speed-up claim.
- Round-trip times include network latency from the client in Korea and client
  queuing. Browser canvas rendering and webcam capture are excluded; the v4
  route has no UI. Detection accuracy was not evaluated in these runs.

#!/usr/bin/env bash
# Runtime matrix on a second device (macOS or Linux), same protocol as the Windows runs:
# operator placement, ORT CPU 1/4T, WASM 1/4T, WebGPU (ORT-Web 1.30; 1.22 for the FP32/FP16
# version check), the browser pipeline with five fresh browser launches for the key
# placements (r1 + r2..r5, as on Windows), and representative YOLO11l (COCO) runtimes.
#
#   bash scripts/paper/run_device_matrix.sh <bundle folder> <output folder> [--with-accuracy]
#   QUICK=1 bash scripts/paper/run_device_matrix.sh <bundle> <scratch folder>   # ~5 min smoke test
#
# <bundle folder> comes from export_device_bundle.py; <output folder> e.g.
# paper_evidence/runtime/matrix_mac. Steps skip results that already exist, so an
# interrupted run can be restarted with the same command. Keep the machine on AC
# power, close other apps, and do not use it while this runs.
set -u
BUNDLE=${1:?bundle folder}
OUT=${2:?output folder}
WITH_ACCURACY=${3:-}
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python3}
MANIFEST="$BUNDLE/test_manifest.jsonl"
RM="$PY scripts/paper/runtime_matrix.py"
COMMON="--artifact-root $BUNDLE --manifest $MANIFEST --output $OUT"
PORT=${PORT:-8791}
URL="--server-url http://127.0.0.1:$PORT"
# full-graph INT8 detectors collapse to mAP 0 on every runtime, so they are not deployable and skipped here
DET="v4_fp32 v4_fp16 v4_int8_head_excl v4_int8_head_excl_fbias v3_fp32 v3_fp16 v3_int8_head_excl v3_int8_head_excl_fbias"
DET_FLOAT="v4_fp32 v4_fp16 v3_fp32 v3_fp16"
DET_INT8="v4_int8_head_excl v4_int8_head_excl_fbias v3_int8_head_excl v3_int8_head_excl_fbias"
REC="rec_fp32 rec_fp16 rec_int8_full rec_int8_full_fbias"
step() { echo "=== $(date +%H:%M:%S) $*"; }
if [ "${QUICK:-}" = "1" ]; then ITER=20; ITER_INT8=3; FRAMES=16; WARM=2; else ITER=1024; ITER_INT8=128; FRAMES=512; WARM=20; fi
mkdir -p "$OUT"

# background CPU load every 10 s (on Windows one busy core slowed WASM by 20%+)
( while true; do echo "--- $(date +%H:%M:%S)"; ps -Ao pcpu,comm -r | head -6; sleep 10; done ) > "$OUT/cpu_load.log" 2>&1 &
SAMPLER=$!

step "0 verify bundle"
(cd "$BUNDLE" && shasum -a 256 -c SHA256SUMS --quiet) || { echo "bundle checksum mismatch"; exit 1; }
{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  if command -v sw_vers >/dev/null; then sw_vers; sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize;
    system_profiler SPHardwareDataType SPDisplaysDataType | grep -vE "Serial|UUID|Provisioning"; fi
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version 2>/dev/null
  $PY -c "import platform,onnxruntime,cv2,numpy;print('python',platform.python_version(),'onnxruntime',onnxruntime.__version__,'opencv',cv2.__version__,'numpy',numpy.__version__)"
} > "$OUT/environment.txt" 2>&1

$RM serve $COMMON --port $PORT --isolate > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER $SAMPLER 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -s "http://127.0.0.1:$PORT/info" >/dev/null && break; sleep 1; done

step "1 WebGPU probe: headless or headed"
HEADED=""
PROBE="$OUT/probe_headless_ops_webgpu_v4_fp32.json"
if [ ! -f "$PROBE" ]; then
  $RM browser $URL $COMMON --models v4_fp32 --eps webgpu --mode ops --timeout 180
  mv "$OUT/ops_webgpu_ort1300_v4_fp32.json" "$PROBE"
fi
if ! $PY - "$PROBE" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
a = r.get("adapter") or {}
sys.exit(0 if r.get("status") == "completed" and a.get("vendor") and not a.get("is_fallback") else 1)
EOF
then
  echo "headless Chrome has no hardware WebGPU adapter; using a visible window (do not cover or minimize it)"
  HEADED="--headed"
fi
echo "HEADED=${HEADED:-no}" >> "$OUT/environment.txt"
B="$RM browser $URL $COMMON $HEADED"

step "2 WebGPU operator placement"
$B --models $DET $REC --eps webgpu --mode ops --ort 1.30.0 --timeout 180
step "3 native ORT CPU latency (1 and 4 threads)"
$RM cpu-speed $COMMON --models $DET $REC --threads 1 --warmup $WARM --iterations $ITER
$RM cpu-speed $COMMON --models $DET $REC --threads 4 --warmup $WARM --iterations $ITER
step "4 WebGPU latency"
$B --models $DET_FLOAT $REC --eps webgpu --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER --timeout 900
$B --models $DET_INT8 --eps webgpu --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER_INT8 --timeout 1800
step "4b ORT-Web version check: FP32 / FP16 on WebGPU 1.22"
$B --models v4_fp32 v4_fp16 --eps webgpu --mode speed --ort 1.22.0 --warmup $WARM --iterations $ITER --timeout 900
$B --models v4_fp32 v4_fp16 --eps webgpu --mode ops --ort 1.22.0 --timeout 180
step "5 WASM latency (1 and 4 threads)"
$B --models $DET $REC --eps wasm --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER --wasm-threads 1 --timeout 3600
$B --models $DET $REC --eps wasm --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER --wasm-threads 4 --timeout 3600
step "6 browser pipeline (detector + recognizer per EP)"
P="$RM pipeline $URL $COMMON $HEADED --ort 1.30.0 --warmup $WARM"
$P --frames $FRAMES --wasm-threads 4 --configs \
  v4_fp32@webgpu+rec_fp32@webgpu v4_fp16@webgpu+rec_fp16@webgpu \
  v4_fp32@webgpu+rec_fp32@wasm v4_fp16@webgpu+rec_fp32@wasm v4_fp16@webgpu+rec_int8_full@wasm \
  v4_fp32@wasm+rec_fp32@wasm v4_int8_head_excl@wasm+rec_int8_full@wasm v4_int8_head_excl_fbias@wasm+rec_int8_full@wasm
$P --frames $((FRAMES / 4)) --wasm-threads 4 --configs v4_int8_head_excl_fbias@webgpu+rec_int8_full@webgpu
$P --frames $FRAMES --wasm-threads 1 --configs \
  v4_fp32@wasm+rec_fp32@wasm v4_int8_head_excl@wasm+rec_int8_full@wasm v4_int8_head_excl_fbias@wasm+rec_int8_full@wasm
step "6b pipeline repeats: four more fresh launches (r2..r5) of the key placements"
KEY="v4_fp16@webgpu+rec_fp32@wasm v4_fp32@webgpu+rec_fp32@wasm v4_fp32@webgpu+rec_fp32@webgpu v4_int8_head_excl@wasm+rec_int8_full@wasm"
for r in r2 r3 r4 r5; do
  $P --frames $FRAMES --wasm-threads 4 --tag $r --configs $KEY
done
step "6c YOLO11l (COCO) representative runtimes (accuracy: same model files, measured on Windows)"
$B --models coco_fp32 coco_fp16 --eps webgpu --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER --timeout 1800
$B --models coco_int8_head_excl --eps webgpu --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER_INT8 --timeout 3600
$B --models coco_fp32 coco_int8_head_excl --eps wasm --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER --wasm-threads 4 --timeout 7200
$B --models coco_int8_head_excl_fbias --eps webgpu --mode ops --ort 1.30.0 --timeout 600
$RM cpu-speed $COMMON --models coco_fp32 coco_int8_head_excl --threads 4 --warmup $WARM --iterations $ITER
if [ "$WITH_ACCURACY" = "--with-accuracy" ]; then
  step "7 WebGPU numeric parity on the full test set"
  $B --models $DET_FLOAT --eps webgpu --mode accuracy --ort 1.30.0 --timeout 3600
fi
step "8 summary"
# accuracy/retention come from the Windows ORT CPU runs of the same model files (same SHA-256)
$PY scripts/paper/summarize_runtime_matrix.py --matrix "$OUT" --accuracy-from paper_evidence/runtime/matrix \
  --artifact-root "$BUNDLE" > "$OUT/summary.md"
$PY scripts/paper/summarize_pipeline_repeats.py --matrix "$OUT" >> "$OUT/summary.md" || echo "pipeline repeat summary skipped"
step "device done"

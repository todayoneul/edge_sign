#!/usr/bin/env bash
# Quiet re-check of the WASM FP32 vs INT8 comparison on the Mac (paper 4.3-4.4).
#
# The first Mac run found no INT8 speedup on WASM, but each model was measured in one launch,
# the FP32 WASM pipeline only once, and other apps were busy during the WASM step. This script
# repeats only that comparison, with a fresh headless Chrome per measurement, a CPU-quiet gate
# before each one, and the FP32 / INT8 order alternating between rounds so slow drift (heat,
# background work) does not favour one precision. It also checks WASM accuracy parity on the Mac.
#
#   caffeinate -dimsu env PYTHON=.venv-bench/bin/python \
#     bash scripts/paper/run_mac_wasm_recheck.sh ~/edge_sign_device_bundle paper_evidence/runtime/matrix_mac_recheck \
#     2>&1 | tee matrix_mac_recheck.log
#   QUICK=1 ... <bundle> /tmp/recheck_quick     # 5-minute smoke test, results not for the paper
#
# Options (environment): ROUNDS (YOLO26-n and pipeline, default 5), ROUNDS_V3 (YOLOv8s, default 3),
# WITH_1T=1 adds three 1-thread rounds for YOLO26-n, QUIET_PCT (default 30) is the CPU % that all
# other processes together (everything except this script's own PIDs, including any personal
# Chrome window) must stay under for three samples in a row before a measurement, QUIET_MAX
# (default 300 s) is how long to wait for that.
# Steps skip results that already exist, so an interrupted run can be restarted with the same command.
set -u
BUNDLE=${1:?bundle folder}
OUT=${2:?output folder}
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python3}
MANIFEST="$BUNDLE/test_manifest.jsonl"
RM="$PY scripts/paper/runtime_matrix.py"
COMMON="--artifact-root $BUNDLE --manifest $MANIFEST --output $OUT"
PORT=${PORT:-8791}
URL="--server-url http://127.0.0.1:$PORT"
ROUNDS=${ROUNDS:-5}
ROUNDS_V3=${ROUNDS_V3:-3}
QUIET_PCT=${QUIET_PCT:-30}
QUIET_MAX=${QUIET_MAX:-300}
if [ "${QUICK:-}" = "1" ]; then ITER=20; FRAMES=16; WARM=2; LIMIT="--limit 32"; ROUNDS=2; ROUNDS_V3=1; QUIET_MAX=30;
else ITER=1024; FRAMES=512; WARM=20; LIMIT=""; fi
step() { echo "=== $(date +%H:%M:%S) $*"; }
tag() { [ "$1" -eq 1 ] && echo "" || echo "--tag r$1"; }  # round 1 untagged, as in the first run
mkdir -p "$OUT"

# background CPU load every 10 s, as in run_device_matrix.sh
( while true; do echo "--- $(date +%H:%M:%S)"; ps -Ao pcpu,comm -r | head -6; sleep 10; done ) > "$OUT/cpu_load.log" 2>&1 &
SAMPLER=$!

# Wait until the rest of the machine is quiet before each measurement: all processes except this
# script, its benchmark server and its load sampler (excluded by PID, never by name) must use
# < QUIET_PCT % CPU in total for three 5-second samples in a row. The measurement Chrome of the
# previous step is already closed here, so any Chrome or python still using CPU (a personal
# browser window, another venv) counts as background load. Gives up after QUIET_MAX seconds and
# logs NOT QUIET (the run continues).
others() {  # "<pid> <%cpu> <command>" of every process except ours, busiest first
  ps -Ao pid=,pcpu=,comm= -r | awk -v skip=" $$ ${SERVER:-} $SAMPLER " \
    'index(skip, " " $1 " ") == 0 && $3 !~ /(^|\/)(ps|awk|sort|head)$/'
}
wait_quiet() {
  local ok=0 waited=0 busy top
  while [ "$ok" -lt 3 ] && [ "$waited" -lt "$QUIET_MAX" ]; do
    busy=$(others | awk '{s += $2} END {printf "%d", s}')
    if [ "$busy" -lt "$QUIET_PCT" ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 5; waited=$((waited + 5))
  done
  top=$(others | head -3 | awk '{c = $3; for (i = 4; i <= NF; i++) c = c " " $i; n = split(c, p, "/"); printf "%s %s; ", $2, p[n]}')
  echo "$(date +%H:%M:%S) before $1: others ${busy}% (waited ${waited}s$([ "$ok" -lt 3 ] && echo ', NOT QUIET')) top: $top" >> "$OUT/quiet.log"
}

step "0 verify bundle"
(cd "$BUNDLE" && shasum -a 256 -c SHA256SUMS --quiet) || { echo "bundle checksum mismatch"; exit 1; }
{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit=$(git rev-parse HEAD)"
  uname -a
  if command -v sw_vers >/dev/null; then sw_vers; sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize; fi
  pmset -g batt 2>/dev/null | head -2
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version 2>/dev/null
  $PY -c "import platform,onnxruntime;print('python',platform.python_version(),'onnxruntime',onnxruntime.__version__)"
  echo "ROUNDS=$ROUNDS ROUNDS_V3=$ROUNDS_V3 WITH_1T=${WITH_1T:-0} QUIET_PCT=$QUIET_PCT QUICK=${QUICK:-0}"
} > "$OUT/environment.txt" 2>&1

$RM serve $COMMON --port $PORT --isolate > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER $SAMPLER 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -s "http://127.0.0.1:$PORT/info" >/dev/null && break; sleep 1; done
B="$RM browser $URL $COMMON --eps wasm --mode speed --ort 1.30.0 --warmup $WARM --iterations $ITER --timeout 3600"
P="$RM pipeline $URL $COMMON --ort 1.30.0 --warmup $WARM --frames $FRAMES --wasm-threads 4 --timeout 1800"

# single-model latency: one fresh Chrome per model and round; odd rounds FP32 first, even rounds INT8 first
pair_rounds() {  # <fp32 key> <int8 key> <rounds> <wasm threads>
  local k first second
  for k in $(seq 1 "$3"); do
    if [ $((k % 2)) -eq 1 ]; then first=$1; second=$2; else first=$2; second=$1; fi
    for m in $first $second; do
      wait_quiet "$m t$4 round $k"
      $B --models "$m" --wasm-threads "$4" $(tag "$k")
    done
  done
}
step "1 YOLO26-n WASM 4T, $ROUNDS rounds"
pair_rounds v4_fp32 v4_int8_head_excl "$ROUNDS" 4
step "2 YOLOv8s WASM 4T, $ROUNDS_V3 rounds"
pair_rounds v3_fp32 v3_int8_head_excl "$ROUNDS_V3" 4
if [ "${WITH_1T:-0}" = "1" ]; then
  step "2b YOLO26-n WASM 1T, 3 rounds"
  pair_rounds v4_fp32 v4_int8_head_excl 3 1
fi

step "3 pipeline FP32 vs INT8 on WASM 4T, $ROUNDS rounds"
FP32_CFG="v4_fp32@wasm+rec_fp32@wasm"
INT8_CFG="v4_int8_head_excl@wasm+rec_int8_full@wasm"
for k in $(seq 1 "$ROUNDS"); do
  if [ $((k % 2)) -eq 1 ]; then order="$FP32_CFG $INT8_CFG"; else order="$INT8_CFG $FP32_CFG"; fi
  for c in $order; do
    wait_quiet "$c round $k"
    $P $(tag "$k") --configs "$c"
  done
done

step "4 WASM numeric parity on the test set (accuracy does not depend on CPU load)"
$RM browser $URL $COMMON --eps wasm --mode accuracy --ort 1.30.0 --wasm-threads 4 --timeout 3600 $LIMIT \
  --models v4_fp32 v4_int8_head_excl v4_int8_head_excl_fbias

step "5 summary"
{
  $PY scripts/paper/summarize_wasm_recheck.py --matrix "$OUT"
  echo; echo "## Browser pipeline: FP32 vs INT8 on WASM 4T, repeated launches"; echo
  $PY scripts/paper/summarize_pipeline_repeats.py --matrix "$OUT"
  echo; echo "## Quiet gate (see quiet.log)"; echo
  echo "measurements: $(grep -c before "$OUT/quiet.log"), not quiet after 5 min: $(grep -c 'NOT QUIET' "$OUT/quiet.log")"
  echo
  $PY scripts/paper/summarize_runtime_matrix.py --matrix "$OUT" --accuracy-from paper_evidence/runtime/matrix \
    --artifact-root "$BUNDLE" | sed -n '/## Browser numeric parity/,$p'
} > "$OUT/summary.md" 2>&1
cat "$OUT/summary.md"
# frame-level predictions are kept compressed in git, as for the other matrices
for f in "$OUT"/accuracy_*/predictions.jsonl; do [ -f "$f" ] && gzip -n -9 -f "$f"; done
step "recheck done"

# v2 분할 학습 완료 후 자동 후속 처리 파이프라인
#
# 사전 조건:
#   - 학습 프로세스 PID(기본 40120)가 백그라운드 실행 중
#   - v1 model_space/yolov8s_signs_*.onnx 는 model_space/v1/ 에 백업 완료
#
# 흐름:
#   1) 학습 완료 대기 (PID polling)
#   2) val 평가 → logs/val_v2split.log
#   3) FP32 ONNX 내보내기 → model_space/yolov8s_signs_fp32.onnx (v1 덮어씀)
#   4) 양자화 E1/E4/E5 → yolov8s_signs_w8a8/w4a16/smoothquant.onnx (v1 덮어씀)
#   5) Static INT8 QDQ → yolov8s_signs_int8_static.onnx (v1 덮어씀)
#   6) 추적 평가 4종 (FP32/W8A8/W4A16/SmoothQuant)
#   7) E2E 종합 평가
#   8) 결과는 logs/ 에 저장 — 문서 업데이트는 사용자 호출 시 수동

param([int]$TrainingPid = 40120)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING  = "utf-8"

$ROOT = "C:\Users\leegy\Desktop\CNN_Quant"
Set-Location $ROOT

$RUN_NAME = "edge_sign_v2_v2split"
$RUN_DIR  = "runs\detect\$RUN_NAME"
$BEST_PT  = "$RUN_DIR\weights\best.pt"
$LOG      = "logs\post_train_v2.log"

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Tee-Object -FilePath $LOG -Append
}

"" | Out-File $LOG -Encoding utf8
Log "=== v2 분할 후속 파이프라인 시작 ==="
Log "Training PID: $TrainingPid"

# ── 1. 학습 완료 대기 ──────────────────────────────────────────────
Log "Step 1/7: Waiting for training PID $TrainingPid ..."
try {
    Wait-Process -Id $TrainingPid -ErrorAction Stop
    Log "Training process exited."
} catch {
    Log "Training PID not found (already exited?). Continuing."
}

if (-not (Test-Path $BEST_PT)) {
    Log "[FATAL] best.pt not found: $BEST_PT"
    Log "Training likely failed. Check logs\train_v2split.log"
    exit 1
}
$bestSize = [math]::Round((Get-Item $BEST_PT).Length / 1MB, 2)
Log "best.pt found: $bestSize MB"

# ── 2. Val 평가 ────────────────────────────────────────────────────
Log "Step 2/7: YOLOv8s validation ..."
python src\detect\yolo_train.py --mode val --weights $BEST_PT --batch_size 32 --device 0 --run_name $RUN_NAME 2>&1 | Out-File logs\val_v2split.log -Encoding utf8 -Append

# ── 3. FP32 ONNX 내보내기 ─────────────────────────────────────────
Log "Step 3/7: Export FP32 ONNX ..."
python src\detect\export_yolo_onnx.py --weights $BEST_PT --output yolov8s_signs_fp32.onnx 2>&1 | Out-File logs\export_v2split.log -Encoding utf8

# ── 4. 양자화 E1/E4/E5 ────────────────────────────────────────────
Log "Step 4/7: Quantization E1 W8A8 / E4 W4A16 / E5 SmoothQuant ..."
python src\quant\run_experiments.py --exp all --weights $BEST_PT --device 0 2>&1 | Out-File logs\quant_v2split.log -Encoding utf8

# ── 5. Static INT8 QDQ ────────────────────────────────────────────
Log "Step 5/7: Static INT8 QDQ quantization ..."
python scripts\quantize_onnx_real.py 2>&1 | Out-File logs\quant_int8_v2.log -Encoding utf8

# ── 6. 추적 평가 ───────────────────────────────────────────────────
Log "Step 6/7: Tracking evaluation (E0/E1/E4/E5 on new test split) ..."
$models = @(
    @{name="E0_FP32";       path="model_space\yolov8s_signs_fp32.onnx"},
    @{name="E1_W8A8";       path="model_space\yolov8s_signs_w8a8.onnx"},
    @{name="E4_W4A16";      path="model_space\yolov8s_signs_w4a16.onnx"},
    @{name="E5_SmoothQuant";path="model_space\yolov8s_signs_smoothquant.onnx"}
)
foreach ($m in $models) {
    if (Test-Path $m.path) {
        Log "  Tracking eval: $($m.name)"
        python src\track\eval_tracking.py --onnx $m.path --quiet 2>&1 | Out-File "logs\track_$($m.name).log" -Encoding utf8
    } else {
        Log "  [SKIP] $($m.path) not found"
    }
}

# ── 7. E2E 종합 평가 ──────────────────────────────────────────────
Log "Step 7/7: E2E comprehensive evaluation ..."
python src\pipeline\eval_e2e.py --n_frames 50 2>&1 | Out-File logs\eval_e2e_v2.log -Encoding utf8

Log "=== Pipeline complete ==="
Log "Outputs:"
Log "  - runs\detect\$RUN_NAME\"
Log "  - logs\val_v2split.log"
Log "  - logs\track_*.log"
Log "  - logs\eval_e2e_v2.log"
Log "  - model_space\yolov8s_signs_*.onnx (v2 replaces v1; v1 in model_space\v1\)"

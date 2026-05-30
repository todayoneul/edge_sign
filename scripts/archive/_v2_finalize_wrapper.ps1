# v2 학습 + 후속 파이프라인이 모두 종료되면 자동으로:
#   1. _finalize_v2_results.py 실행 (로그 → JSON + RESULTS_V2.md)
#   2. git add + commit
#
# 사용: 백그라운드 PowerShell 프로세스로 실행
#   Start-Process powershell -ArgumentList ... scripts\_v2_finalize_wrapper.ps1

param(
    [int]$PipelinePid = 29376
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING  = "utf-8"

$ROOT = "C:\Users\leegy\Desktop\CNN_Quant"
Set-Location $ROOT

$LOG = "logs\finalize_wrapper.log"
"" | Out-File $LOG -Encoding utf8

function Log {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ts] $msg" | Tee-Object -FilePath $LOG -Append
}

Log "=== v2 finalize wrapper 시작 ==="
Log "후속 파이프라인 PID $PipelinePid 종료 대기..."

try {
    Wait-Process -Id $PipelinePid -ErrorAction Stop
    Log "후속 파이프라인 종료 감지."
} catch {
    Log "PID $PipelinePid 이미 종료됨 또는 미발견 - 계속 진행."
}

# 학습 + 평가 결과 파일이 실제로 존재하는지 sanity check
$bestPt = "runs\detect\edge_sign_v2_v2split\weights\best.pt"
if (-not (Test-Path $bestPt)) {
    Log "[WARN] best.pt 없음 — 학습 실패 가능성. 진행은 계속하되 결과 제한적."
}

# 결과 정리 스크립트 실행
Log "Step 1/3: 로그 파싱 + RESULTS_V2.md 생성"
python scripts\_finalize_v2_results.py 2>&1 | Out-File logs\finalize_v2.log -Encoding utf8 -Append

# git status 확인
Log "Step 2/3: git 변경사항 확인"
git status --short 2>&1 | Out-File $LOG -Encoding utf8 -Append

# add + commit
Log "Step 3/3: git commit"
git add docs\RESULTS_V2.md logs\results_v2.json 2>&1 | Out-File $LOG -Encoding utf8 -Append

# 모델 파일과 로그도 함께 커밋 (선택)
if (Test-Path "model_space\yolov8s_signs_fp32.onnx") {
    git add model_space\yolov8s_signs_fp32.onnx model_space\yolov8s_signs_w8a8.onnx model_space\yolov8s_signs_w4a16.onnx model_space\yolov8s_signs_smoothquant.onnx 2>&1 | Out-File $LOG -Encoding utf8 -Append
}

$commitMsg = @"
v2 분할 재학습 + 평가 완료: RESULTS_V2.md 자동 생성

자동 처리 항목:
- YOLOv8s v2 분할 학습 완료 (runs/detect/edge_sign_v2_v2split/)
- FP32 ONNX 내보내기 + W8A8/W4A16/SmoothQuant 양자화 재실행
- Static INT8 QDQ 재생성
- 추적 평가 (E0/E1/E4/E5) - v2 test 시퀀스(주야간 균형) 기준
- E2E 종합 평가 (eval_e2e.py, 50 프레임)
- v1 모델/데이터셋은 model_space/v1/, data/yolo_signs_v1/ 보존

결과 산출물:
- docs/RESULTS_V2.md (자동 생성된 결과 요약)
- logs/results_v2.json (머신 판독 JSON)
- logs/{val,track_*,eval_e2e_v2,quant_*}.log (원본 로그)

다음 단계 (수동):
- README 7.2~7.6 차트/표 v2 결과로 재생성
- EXPERIMENTS.md 본문에 v2 컬럼 통합
"@

git commit -m $commitMsg 2>&1 | Out-File $LOG -Encoding utf8 -Append

Log "=== finalize wrapper 완료 ==="
Log "확인: docs/RESULTS_V2.md, logs/results_v2.json"
Log "git log -1 으로 커밋 확인 가능"

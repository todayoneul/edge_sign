# Missing experiments / evidence recovery

아래는 제안만이며 이번 작업에서 실행하지 않았다. 먼저 과거 로그/백업/정확한 manifest를 복원할 수 있는지 확인한다. 근거가 복원되면 불필요한 재실험을 생략할 수 있다. 논문은 C1/C2/C3 모두를 반드시 주장할 필요가 없다.

## MUST

| 항목 | 필요한 이유/대상 주장 | 필요한 산출물 |
|---|---|---|
| Freeze provenance 복구 | 모든 정량표의 비교 가능성 | checkpoint/ONNX hash, dataset train/val/test 파일 manifest·taxonomy, calibration manifest, exact export/eval CLI·commit·package versions |
| 동일 모델·동일 test의 variant 평가 | C1 및 무손실/열화 주장 | FP32·FP16·full-head QDQ·head-excluded QDQ별 per-image detections/confidence, P/R/mAP50/mAP50-95 및 class별 AP; 동일 preprocessing/postprocessing/conf/IoU |
| Head 관련 통제 ablation | C1/C2의 head-level 민감도와 원인 | backbone-only/head-only/full/none 양자화, weight-only vs activation-only, box vs cls/DFL 분기별 제외; calibration/scale 조건 고정. 인과 주장을 삭제하면 일부 ablation은 SHOULD로 내릴 수 있음 |
| CosSim와 검출 붕괴의 짝지어진 raw 기록 | C2 .9995 및 det0 검증 | 동일 실제 이미지·baseline/full/excluded 파일 hash·raw tensor, global 및 box/cls별 similarity, threshold별 count/conf, mAP. synthetic simulation은 별도 보조 자료 |
| Runtime 반복 timing 및 환경 | C3 56/62/24/2.2, precision 효과 | 동일 모델/입력/해상도/backend 내 precision 비교; 지원 불가 조합은 실제 오류로그; per-run latency와 warmup/runs, provider/browser/adapter/version, mean/median/std/분위수, E2E 범위 |
| OCR/recognizer 원시 평가 복원 | OCR 민감도를 본문 주장할 경우 | 정확한 vocab/split/hash와 FP32/W8A8/W4A16/1-bit별 correct/total, prediction 및 accuracy; 한국14-class와 독일43-class 분리 |
| Tracking 표준 지표 또는 명확한 제외 | IDF1/HOTA를 표준명으로 쓸 경우 | identity GT 및 표준 평가 결과; 불가능하면 proxy로 재명명하고 정량 핵심 기여에서 제외. .295를 쓸 경우 per-sequence counts/predictions와 macro/micro 명시 |
| 불일치 해결 | v2 best/epoch, v4 .748, %/pp, DFL 구조 | 원본 작성 시 평가 조건/로그 회수. 미복구 수치는 DOC_ONLY/CONFLICT로 유지하며 제출 결과표에서 제외 |

## SHOULD

- 독립 반복 latency와 confidence interval, 가능하면 여러 실행일/thermal 상태; GPU synchronization·power·threading 통제.
- 더 많은 독립 주야간 test sequence. 현행 test 야간16/주간2401 프레임은 강한 불균형이며 2개 sequence만으로 도메인 일반화를 주장하기 어렵다.
- calibration 규모/도메인·seed 변화에 대한 민감도. “head가 원인”과 “calibration 부적합”을 구분한다.
- input 640 vs training1280 등 해상도 통제. 동일 데이터셋의 두 모델이라도 모델 크기·학습 설정이 다르면 구조 하나의 인과 효과로 해석하지 않는다.
- 실제 edge 장치(Pi/Jetson/phone 등)에서 quality·latency·메모리·전력 측정. “엣지 로봇 실시간”을 핵심 검증 주장으로 유지하려면 MUST로 승격한다.
- seed 반복 학습은 학습 안정성/모델 비교 주장 시 유용하지만, 기존 freeze 모델의 deployment 분석만 한다면 우선순위를 낮출 수 있다.

## NICE TO HAVE

- 외부 dataset으로 domain shift 확인, 더 많은 detector architecture/runtime 조합.
- profiler/kernel 수준 분석으로 FP16 지연의 이유 확인(커널 미성숙이라는 설명을 본문에 유지하면 SHOULD 이상).
- 정성 검출 사례·error taxonomy·threshold sensitivity plot, 명확한 근거에 연결된 failure 사례.
- 에너지/메모리 peak 및 interactive UI 사용성은 논문 scope가 요구할 때 확장.

계획 단계 수치를 예상 결과로 채우지 않는다. 추가 실험 후에도 train/val/test와 taxonomy가 다른 결과를 동일 정확도 비교표에 합치지 않는다.

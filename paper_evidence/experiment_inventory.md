# 실험 세대 및 검출 학습 evidence inventory

조사일 2026-09-23. 기존 파일의 read-only 메타데이터 조사이며 학습·추론·benchmark를 실행하지 않았다. 아래 CSV 최대 epoch는 보존된 전 epoch의 mAP50-95 argmax이다. best.pt의 존재와 선택 epoch를 구분한다. 체크포인트 역직렬화는 수행하지 않았다. args의 amp:true / half:false만으로 저장 가중치가 FP32임을 보증하지 않는다.

## 실제 보존된 학습 run

| Run | model / dataset | imgsz / batch / planned epochs | 기록 epoch | 최대 mAP50-95 epoch | mAP50 / mAP50-95 / P / R | seed / device |
|---|---|---|---|---|---|---|
| edge_sign_v2_v2split | yolov8s.pt / `C:\Users\leegy\Desktop\CNN_Quant\data\yolo_signs\dataset.yaml` | 640 / 32 / 75 | 1–74 (74 rows) | 54 | 0.5917 / 0.38366 / 0.70214 / 0.53488 | 0 / '0' |

**edge_sign_v2_v2split 근거**: `runs/detect/edge_sign_v2_v2split/results.csv` line 55 (epoch=54); scanned lines 2-75; `runs/detect/edge_sign_v2_v2split/args.yaml:3-29`. 체크포인트 목록:
- `runs/detect/edge_sign_v2_v2split/weights/best.onnx`: 44,747,973 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/best.pt`: 22,509,674 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch0.pt`: 44,854,271 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch10.pt`: 44,855,423 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch20.pt`: 44,856,703 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch30.pt`: 44,857,983 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch40.pt`: 44,859,263 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch50.pt`: 44,860,543 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch60.pt`: 44,861,823 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/epoch70.pt`: 44,863,103 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v2_v2split/weights/last.pt`: 22,509,674 bytes (존재 확인, 내용/선택 epoch 미검증).

| edge_sign_v3_lights-4 | yolov8s.pt / `data\yolo_signs_v2\dataset.yaml` | 1280 / 8 / 40 | 1–34 (34 rows) | 29 | 0.77611 / 0.4457 / 0.79444 / 0.73696 | 0 / '0' |

**edge_sign_v3_lights-4 근거**: `runs/detect/edge_sign_v3_lights-4/results.csv` line 30 (epoch=29); scanned lines 2-35; `runs/detect/edge_sign_v3_lights-4/args.yaml:3-29`. 체크포인트 목록:
- `runs/detect/edge_sign_v3_lights-4/weights/best.onnx`: 44,747,495 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v3_lights-4/weights/best.pt`: 89,579,619 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v3_lights-4/weights/epoch0.pt`: 89,576,291 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v3_lights-4/weights/epoch10.pt`: 89,577,315 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v3_lights-4/weights/epoch20.pt`: 89,578,595 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v3_lights-4/weights/epoch30.pt`: 89,579,875 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v3_lights-4/weights/last.pt`: 89,580,259 bytes (존재 확인, 내용/선택 epoch 미검증).

| edge_sign_v4 | yolo26n.pt / `C:\Users\leegy\Desktop\CNN_Quant\data\yolo_signs_v2\dataset.yaml` | 1280 / 8 / 40 | 1–40 (40 rows) | 28 | 0.75611 / 0.42567 / 0.79293 / 0.69365 | 0 / null |

**edge_sign_v4 근거**: `runs/detect/edge_sign_v4/results.csv` line 29 (epoch=28); scanned lines 2-41; `runs/detect/edge_sign_v4/args.yaml:3-29`. 체크포인트 목록:
- `runs/detect/edge_sign_v4/weights/best.onnx`: 4,971,834 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v4/weights/best.pt`: 5,449,413 bytes (존재 확인, 내용/선택 epoch 미검증).
- `runs/detect/edge_sign_v4/weights/last.pt`: 5,449,413 bytes (존재 확인, 내용/선택 epoch 미검증).

## 문서와 raw의 차이

- v2: README.md:406-408과 docs/EXPERIMENTS.md:59는 best ep56을 주장한다. 그러나 results.csv:55 (epoch54)가 최대 mAP50-95=.38366이다. epoch56(line57)은 mAP50=.59174, mAP50-95=.38351. 따라서 mAP50만을 기준으로 한 설명일 수 있으나 실제 checkpoint selection은 미검증이다. args epochs=75는 계획값이고 기록은 74 epoch다. 최종 평가 .587과 학습 validation .5917도 서로 다른 평가로 유지한다.
- v3: results.csv:30은 epoch29 .77611/.4457로 docs/TRAINING_STATUS.md:17-18과 일치. 기록은 34 epoch이며 계획40 완주가 아니다. 해당 문서:13,21-24에 ep35 진행 중 수동 종료라고 기록되어 있어 patience20 자동 조기종료로 표현하면 부정확하다. 향후 회복 불가능/과적합 확정은 raw curve로 증명되지 않는다.
- v4: results.csv:29 epoch28 .75611/.42567. logs/train_v4.log:66954는 best.pt 검증, :67077은 mAP50=.756/mAP50-95=.426이다. README.md:722 및 docs/EXPERIMENTS.md:390의 .748은 CSV 최종epoch40 값 근처이며 best 검증값으로 쓰면 충돌한다. 최종 epoch와 best validation을 구분해야 한다.
- v4 환경만 직접 로그 확인: logs/train_v4.log:3, Ultralytics8.4.56 / Python3.10.19 / torch2.11.0+cu128 / RTX5070 12227MiB. v2/v3 args device0만으로 GPU 모델을 소급 확정하지 않는다.

## 이전 세대와 확장 실험

| 세대 | 확인되는 내용 | 상태 / 비교 가능성 | 근거 |
|---|---|---|---|
| 초기 GTSDB 단독 | 단독 baseline 진행 기록, run 없음 | VERIFIED_DOC_ONLY; 수치 미확인 | docs/EXPERIMENTS.md:235 |
| 초기 YOLOv8n | train26866, mAP50 .573 | VERIFIED_DOC_ONLY; 현 run과 직접 비교 불가 | docs/EXPERIMENTS.md:241 |
| v1 / edge_sign_v2_e0_full3 | YOLOv8s, train44696, ep57, mAP50 .628 / .437, P.722 R.543 | VERIFIED_DOC_ONLY; run/weights 현재 미보존. yolo_signs_v1 폴더는 있음 | docs/EXPERIMENTS.md:236-242 |
| v2 stratified | 위 보존 run; E0/E1/E4/E5 및 추적 ablation | 각 양자화 보고서는 별도 evidence; 학습CSV와 ORT/task평가를 합치지 않음 | docs/EXPERIMENTS.md:59-72; runs/detect/edge_sign_v2_v2split |
| v3 신호등 분리 | 위 보존 run; 2개 클래스지만 signboard→traffic_light로 의미 변경 | v2와 NOT_COMPARABLE (taxonomy, data, imgsz640→1280 변경) | args.yaml:4,9; 두 dataset.yaml:5-8 |
| Phase11 v3 head excluded INT8 | FP32/INT8 QDQ A/B, 모델 및 script 존재 | 별도 양자화/배포 evidence; 학습 mAP로 양자화 성능 보증 불가 | docs/EXPERIMENTS.md:361-372; scripts/quantize_v3_detector.py |
| Phase12 / v4 YOLO26n | 위 보존 run; head 포함/제외 및 parity 확장 | v3와 YAML경로/imgsz같음. 과거 split hash/manifest가 없어 완전 동일 평가집합 확정 불가 | runs/detect/edge_sign_v4/args.yaml:3-9; docs/EXPERIMENTS.md:388-399 |

## provenance 제한

현재 파일 경로는 재사용 가능한 mutable 경로다. 현재 split 수와 run args 경로의 일치는 당시 exact membership/label bytes 일치 증거가 아니다. gitignored 원본의 당시 manifest/hash, 클래스별 test AP, 반복 seed 실험과 confidence interval은 이 학습 run들에서 확인되지 않았다. 파일 timestamp는 생성 실행의 증명이 아니므로 데이터 생성일로 승격하지 않는다.
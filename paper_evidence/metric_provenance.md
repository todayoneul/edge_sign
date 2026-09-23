# Metric provenance

조사 기준: 2026-09-23, commit `095f3cdfba50e5ad91467a626053700c360b2ee5`. 모든 경로는 저장소 root 상대 경로이다. 현재 파일의 존재/내용과 과거 실행의 동일성은 구별한다. 학습·추론·양자화·benchmark를 실행하지 않았다.

## Evidence 판정

| 상태 | 이 감사에서의 의미 |
|---|---|
| VERIFIED_RAW | 저장된 결과 CSV/log의 해당 값 또는 현재 아티팩트의 직접 관찰(파일 크기, 그래프 구조, 메타데이터 계수)을 확인. 측정 프로토콜의 타당성·역사적 동일성을 자동 보장하지 않음 |
| VERIFIED_DOC_ONLY | 문서/코드 상수에는 값이 있으나 독립된 실행 산출물을 찾지 못함. 논문 검증 수치로 승격 금지 |
| CONFLICT | 원문, 저장 결과, 코드 정의 또는 구조 사이에 구체적인 불일치 존재. 어느 값이 참인지 임의 선택하지 않음 |
| MISSING | 주장을 뒷받침할 필요한 원시 파일/필드 미발견 |
| NOT_COMPARABLE | split/taxonomy/model/runtime/metric 정의가 달라 직접 비교 금지 |

실행 코드는 재현 절차의 근거이며 실행 사실의 근거가 아니다. 결과를 하드코딩한 시각화와 그래프는 독립 반복 실험이 아니다. training validation, exported ONNX validation, test evaluation은 각각 별도 평가이다. CSV 최대 epoch는 `best.pt` 내부가 그 epoch임을 증명하지 않는다.

## 검출 결과와 문서의 연결

| 버전 | 확인된 저장 결과 | 문서 주장 | 판단 |
|---|---|---|---|
| v2 | `runs/detect/edge_sign_v2_v2split/results.csv`: 74개 epoch 행; mAP50-95 최대 epoch54, mAP50 .5917 / mAP50-95 .38366 / P .70214 / R .53488 | README:406-408의 75epoch/best56/.587 | 학습 CSV와 문서 불일치. .587은 별도 CPU ONNX 평가라고도 기록됨(docs/EXPERIMENTS.md:328). CSV .5917로 이를 대체하지 않음. 해당 ONNX val raw log 미발견 |
| v3 | `runs/detect/edge_sign_v3_lights-4/results.csv`: epoch29 .77611/.44570, 완료34행 | docs/EXPERIMENTS.md:340의 .7761/.4457, ep29 | 반올림 일치. `docs/TRAINING_STATUS.md`는 ep34 후 수동 중단을 기록. patience20 자동 조기종료와 혼동 금지 |
| v4 / YOLO26 | `runs/detect/edge_sign_v4/results.csv`: epoch28 .75611/.42567; `logs/train_v4.log` final best validation의 반올림 .756 | docs/EXPERIMENTS.md:390, README:722 .748 | CONFLICT: 문서 .748을 현재 저장된 best validation 값과 일치시킬 근거 부족. 별도 평가였을 가능성은 남겨두며 임의 수정하지 않음 |

상세 모든 설정·행 위치·split 구성은 `experiment_inventory.md`, `dataset_splits.md`, `evidence_index.csv` 참조. v2→v3의 mAP 증감은 class taxonomy와 dataset construction이 달라 성능 개선 증거로 사용하지 않는다.

## Phase 1 원시 결과

- `logs/evaluation_w8a8_ptq.csv:2`: ConvNeXtV2-Nano W8A8 Top-1 **81.24%**, throughput **817.55 FPS**, 기록 Memory **14.90 MB**. 마지막 값은 현재 파일 크기 계수가 아니라 결과 필드이므로 disk size와 별도이다.
- `logs/training_log_w4a16.csv`: 마지막 epoch30 **76.12%**, CSV 최대 epoch8 **76.18%**. README의 76.12는 마지막 행과 일치하나 최대 성능을 의미하지 않는다.
- `logs/training_log_1bit.csv`: 마지막 epoch30 **14.23%**. 최대값은 `evidence_index.csv`의 P1-1bit-max에 별도 기재한다.
- `logs/final_score_report.txt:2-7`: 저장된 aggregate report가 존재한다. SmoothQuant **Recall@1 38.50%, 10.29 ms, FinalScore .8068**; baseline **39.00%, 6.09 ms, .8000**. 이 값은 KoreanOCRNet 정확도나 detection mAP가 아니다.
- baseline classification **81.88%**는 이 로컬 로그 집합에서 독립된 baseline evaluation CSV가 미발견이다. `src/base_W8A8.py:128`의 비교용 문자열은 raw evidence가 아니다.

### Phase 1 해석 제약

`src/final_omnimodal_eval.py:208-225`는 후보 모델 전체의 min/max 정규화 후 `0.6*Perf + 0.2*Speed + 0.2*Mem`을 계산한다. README:227의 “FP16 기준선 대비 정규화”와 정의가 다르다. 후보를 추가/제거하면 점수도 달라지므로 보편적 우수성 지표로 쓰지 않는다.

같은 코드 :107,115,135,178은 BF16으로 변환한다. FP16/W8A8 등의 모델 이름만으로 timing 시 실제 연산 dtype을 단정할 수 없다. :178-189의 timing에는 CUDA synchronization이 없다. :29-63의 memory_mb는 상수이며 실시간 메모리/디스크 측정 코드가 아니다. 저장 report의 존재는 인정하되 측정값의 해석을 제한한다.

## 인식·추적·런타임

- OCR 98.5/98.4/54.6/0.3%, TrafficSignNet 62.8/63.2/49.2/12.8% 및 한국 14-class 80.3%: 아티팩트·코드 존재와 accuracy 원시 실행 기록은 별개. `quantization_evidence.md`의 상태를 따른다.
- MOTA .295, IDF1 .495, HOTA .570: 저장된 framewise track 결과/평가 로그가 누락되어 DOC_ONLY. 구현의 IDF1/HOTA는 표준 metric과 다르므로 그대로 논문에 쓰지 않는다(`tracking_evidence.md`).
- CPU 56.3, WebGPU FP32 62, FP16 24, WASM INT8 2.2 FPS: saved timing trace가 부족한 DOC_ONLY. 정밀도와 backend를 함께 바꾼 비교는 bit-width 효과를 분리하지 못한다(`runtime_evidence.md`).
- data-free weight SQNR/합성 confidence 붕괴는 실제 데이터에서의 task-level 성능을 대체하지 못한다. 기존 분석 스크립트를 이번 감사에서 실행하지 않았다.

## 단위 및 세대 충돌

- README:108의 mAP −11.0%p: 문서 .587→.523에서 계산되는 차이는 **−6.4pp**, 상대 **−10.90%**이다. 원시 평가가 아닌 문서 수치의 산술 검토다.
- README:79의 −0.07%p와 docs/EXPERIMENTS.md:140의 −.0004는 다르다. 후자는 **−.04pp**, baseline .587 대비 약 **−.068%**다.
- README:722의 “본문 §1–§8 모든 정량 결과·시연 v3 기준”은 §6.2가 명시하는 v2 split과 충돌한다.
- 실제 파일 bytes, decimal MB(bytes/1e6), MiB(bytes/2^20), theoretical bit-size를 혼합하지 않는다. `artifact_manifest.csv`와 `quantization_evidence.md`가 현재 파일 크기의 기준이다.

## 감사 범위와 보존

`logs/`, `runs/`, `model_space/`, `models/`, `checkpoints/`, `assets/`의 파일 메타데이터를 `artifact_manifest.csv`에 기록했다. SHA-256은 150,000,000 bytes 미만 파일에만 계산했고 나머지는 NOT_HASHED_SIZE_LIMIT 표시다. mtime은 생성일/실험일 증명이 아니다. 모델 tensor를 변경하거나 체크포인트를 실행하지 않았다.

README/docs/src/scripts/tests, split 메타데이터 및 label, requirements/pyproject/Docker 설정을 조사했다. 원본 영상·이미지 전체를 복사하지 않았다. 접근 제한 캐시, 원본 데이터 전수 내용, 숨은 도구 대화 기록은 실행 증거로 사용하지 않았다. 누락은 “이 로컬 조사 범위에서 미발견”이지 “실험을 수행한 적 없음”의 증명이 아니다.

Sensitive file detected — excluded

`.gitignore`의 `*.md`, `/runs`, `/data`, `model_space/`, `/checkpoints` 패턴 때문에 이 보고서의 Markdown 및 주요 근거는 Git에 자동 포함되지 않는다. 별도 전달/보존 필요. Git commit만으로 모델과 데이터의 과거 상태가 동결되지는 않는다. 이번 작업은 기존 소스/모델/문서를 수정하지 않고 `paper_evidence/`에만 결과를 생성했다.

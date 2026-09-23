# Reproducibility audit

2026-09-23 로컬 스냅샷. COMPLETE=명시한 좁은 범위의 필수 파일/정보 확인, PARTIAL=일부 존재하나 결과의 정확한 재현 연결 부족, MISSING=필요 실행 근거 미발견. 재실험은 수행하지 않았다.

| 항목 | 상태 | 확보된 근거 | 남은 결손 |
|---|---|---|---|
| Dataset split | PARTIAL | 현재 YAML·split 스크립트·파일/label 계수·sequence 교집합(`dataset_splits.md`) | 학습/평가 당시 불변 manifest 및 이미지 hash; mutable path와 taxonomy drift; 주야간 frame 불균형 |
| Training config | COMPLETE (v2/v3/v4 저장 args 범위) | 각 run의 args.yaml, results.csv; v4 train_v4.log | 전체 이전 세대 config 없음; 실제 optimizer auto 선택은 로그 없는 런에서 미확인 |
| Checkpoint | PARTIAL | 세 run의 best/last 및 다수 Phase1 weights, artifact_manifest.csv | 과거 ONNX export와 checkpoint의 연결 hash 없음; 파일명 best와 CSV max의 동일성은 미입증 |
| Quantization config | PARTIAL | detector/recognizer PTQ, static QDQ, QAT/STE 코드; 현행 ONNX 구조 | 각 파일 생성 당시 CLI·code hash·full-head 실패본·변환 로그 |
| Calibration data information | PARTIAL | 스크립트 reader, 디렉터리/default 개수·전처리 | 실제 calibration 이미지 목록/hash·순서·activation ranges·실행별 seed |
| Evaluation code | PARTIAL | detect/recognition/tracking/e2e/parity 코드 | tracking IDF1/HOTA 정의 불일치, pseudo GT, e2e 상수; archive 경로 drift |
| Detection raw metrics | PARTIAL | v2/v3/v4 학습 CSV 및 v4 final val class별 수치 | 양자화 variant별 mAP/predictions, 고정 test 평가; 문서값 충돌 |
| Recognition raw metrics | PARTIAL | Phase1 accuracy/학습/report CSV | 배포 OCR·43-class·한국14-class의 variant별 평가 로그/predictions 없음 |
| Tracking raw metrics | MISSING | 구현 및 문서상 aggregate만 존재 | frame별 prediction, sequence별 counts, identity GT/표준 평가 결과 |
| CPU benchmark | MISSING (핵심 56.3 FPS 주장) | 실행 스크립트·문서 수치 | timing raw array·당시 환경/provider·모델 hash·반복 통계 |
| Browser benchmark | MISSING (핵심 62/24/2.2 FPS 주장) | spike 코드 및 문서 표 | 실행 캡처/로그·raw timings·browser/adapter/driver·EP trace |
| Package versions | PARTIAL | requirements, package-lock, HF pin, v4 학습 로그 | v2/v3 benchmark 당시 exact lock/container digest·CDN 리소스 hash |
| Hardware information | PARTIAL | v4 log RTX5070 12227MiB 학습 장치 | CPU/RAM/OS/browser/driver 및 benchmark 당시 정보 |
| Random seed | PARTIAL | args.yaml seed0/deterministic; split 생성 규칙 | 모든 calibration/timing 입력의 seed 및 실행별 기록 |

## 재현성에 영향을 주는 구체적 문제

1. 코드 commit은 gitignore 모델·데이터·runs의 상태를 동결하지 않는다. `artifact_manifest.csv`는 이번 감사 때의 크기/hash를 보존하며 과거 실행 연결을 새로 만들어내지 않는다.
2. `data/yolo_signs_v1/dataset.yaml`의 path는 `data/yolo_signs`를 가리킨다. 디렉터리 이름만 보고 과거 v1 split을 재현한다고 판단하면 안 된다.
3. 현재 `prepare_dataset.py` taxonomy와 보존 YAML 사이 drift가 있다. 데이터셋 버전별 클래스 매핑을 고정해야 한다.
4. archive로 옮긴 일부 스크립트의 parent.parent는 root가 아닌 scripts를 가리킨다. README 명령 존재만으로 바로 재현 가능하다고 판정하지 않는다.
5. `src/pipeline/eval_e2e.py`는 task metric을 새로 측정하지 않고 상수를 사용한다. 이 스크립트를 실행해 얻은 표로 양자화 정확도를 검증할 수 없다.
6. 설정 파일은 의도를, 실행 로그는 수행을 기록한다. 계획 epoch와 완료 epoch, 자동 patience 종료와 수동 중단을 구분한다.

## 감사 자체의 제약

소스/모델/기존 문서를 변경하지 않았다. 모델 실행, training, quantization, benchmark, web search, git commit/push는 수행하지 않았다. 원본 데이터를 복제하지 않았고 텍스트/메타데이터 집계와 ONNX graph의 정적 검사를 수행했다. Python checkpoint pickle 실행으로 과거 성능을 추론하지 않았다. `.pytest_cache`와 사용자 Git global ignore에 읽기 제한 경고가 있었으며, 해당 캐시는 논문 실행 근거로 사용하지 않았다.

Sensitive file detected — excluded

# YOLO26 v4 웹 측정 준비 상태 (2026-09-24)

**현재 상태: 로컬 경로 검증 완료, 공개 Hugging Face Space 미배포.** 아래 로컬 smoke는 기능 확인용이며 논문의 배포 FPS가 아니다. 기존 공개 Space는 계속 YOLOv8s v3 소스 commit `9fc2593358a678a5b1597e978a63778bc909fc31`을 실행한다.

## 준비한 격리 경로

- `EDGE_SIGN_PAPER_V4=1`일 때만 `/api/paper-v4/status`와 `/ws/paper-v4`를 등록한다. 기존 `/api/status`, `/ws/stream`, `/detection/` 및 v3 variant는 그대로 사용한다.
- v4 전용 WebSocket 연결마다 별도의 ByteTracker를 만들고 `reset` 명령으로 초기화한다.
- FP32와 head-excluded QDQ는 같은 640×640 RGB/255 전처리, YOLO26 `[1,300,6]` xyxy/conf/class 디코더, 외부 NMS 없음, confidence ≥0.1의 ByteTrack 입력을 사용한다. Tracker 설정은 `track_thresh=.25`, `low_thresh=.1`, `match_thresh=.8`, `track_buffer=30`이다.
- KoreanSignNet 14-class는 예측 track ROI를 32×32로 잘라 학습 때의 정규화를 적용한다. 이 경로는 브라우저 렌더를 포함하지 않는다.
- `benchmark_hf_space.py`에 상태·WebSocket 경로 옵션을 추가해 기존 v3 측정 명령을 유지하면서 v4 경로도 측정할 수 있게 했다.

## 모델 계보와 크기

| 모델 | 실제 파일 B | SHA-256 |
|---|---:|---|
| `yolo_v4_signs_fp32.onnx` | 9,805,975 | `4133272b340bc2e788277edce1b4abe3e41126df9d697746a06315fd5603c6f1` |
| `yolo_v4_signs_int8_head_excluded.onnx` | 3,416,372 | `a2ddfe7d07adeb3cc58d413a0936ca3833260e99a4af227b01dea4df7326c9eb` |
| `korean_sign_net_fp32.onnx` | 116,860 | `a5d6584c984fe105c04d22f7ae026ba2fc4be4a97eb67a4997a1c03626ae47ca` |

선택 구성의 합계는 FP32 9,922,835 B, QDQ 3,533,232 B다. 두 detector와 recognizer를 A/B 측정용으로 함께 싣는 경우 13,339,207 B다. 기존 v3 모델까지 같은 Space에 싣는 전체 저장 용량과 선택 구성 크기를 혼용하지 않는다.

## 수행한 기능 검증

- `tests/paper/test_space_v4.py`: v4 디코더의 평가 코드 일치와 WebSocket `reset`/양 variant의 첫 프레임 응답 확인. **2 passed**.
- 로컬 FastAPI 상태: `ready`, ONNX Runtime **1.23.2**, CPU EP, intra-op thread 2, 두 detector와 recognizer SHA가 위 목록과 일치.
- 로컬 샘플 2 warm-up + 4 measured frame의 open-loop smoke에서 trace, config, metrics가 생성됐다. 4프레임 값은 throughput 추정에 사용하지 않는다. 로컬 Windows `convnext_env`에서 처음 `cv2`와 ORT의 중복 OpenMP 로드로 프로세스가 종료됐고, 프로젝트의 기존 로컬 실행 설정인 `KMP_DUPLICATE_LIB_OK=TRUE`로 다시 실행해 기능을 확인했다. 이 설정의 로컬 수치를 Linux Space 성능으로 취급하지 않는다.
- React 빌드와 기존 31개 Vitest가 통과했다. `npm run lint`는 변경 전 코드의 `react-hooks/set-state-in-effect` 등 11개 오류로 실패했다. 이번 수정 위치의 TypeScript 빌드 오류는 없었다.

## v4 출력 형식 감사 결과와 반영

독립 감사(V4 shape audit)는 **모델 파일만 v4로 바꾸면 서버가 v4를 올바르게 실행하지 못한다**고 판정했다. 항목별 반영 상태는 다음과 같다.

| 감사 지적 | 반영 |
|---|---|
| v3 파서(`e2e_pipeline.py`의 `postprocess_yolo`)는 `[1,6,8400]` 중심좌표·클래스 점수 + NMS를 가정하므로, v4의 `[1,300,6]` xyxy/conf/class를 넣으면 박스와 클래스 1 신뢰도가 손상된다 | v3 파서를 쓰지 않는다. `paper_v4.py`의 `_decode_v4`가 출력 shape를 검사하고, 논문 평가 코드(`evaluate_qdq_detection.decode_v4`)와 같은 결과를 내는지 테스트로 확인한다 |
| v4 모델 경로를 명시적으로 선택하고 `.dockerignore`에서 v4 파일을 허용해야 한다 | `paper_v4.py`의 `DETECTORS`가 두 파일명을 고정하고, 세션 로드 시 출력이 `[1,300,6]`이 아니면 거부한다. `.dockerignore`에 두 파일을 추가했다 |
| 브라우저 온디바이스 모델 URL이 v3로 고정돼 있다(`Viewport.tsx`의 `startCaptureInference`) | **변경하지 않았다.** v4 경로는 서버 WebSocket 측정 전용이며, 같은 Space에 올려도 웹 화면의 서버·온디바이스 모드는 계속 v3를 사용한다. v4 온디바이스 측정에는 TypeScript 디코더와 모델 URL 변경이 별도로 필요하다 |
| `/api/status`에 v4를 표시하고, 운영 임계값이 논문 v4 평가와 다름을 기록해야 한다 | 기존 `/api/status`는 v3 그대로 두고, v4는 별도 `/api/paper-v4/status`의 `generation: "YOLO26-n v4"`와 `decoder`·`tracking` 필드로 구분한다. 임계값 차이는 아래 표와 같다 |

| 설정 | 공개 데모 v3 (`/ws/stream`) | 논문 v4 경로 (`/ws/paper-v4`) |
|---|---|---|
| 검출기 출력 | `[1,6,8400]`, 중심좌표 + 클래스 점수 | `[1,300,6]`, xyxy/conf/class (top-300 내장) |
| 검출 후처리 | conf ≥ 0.15 (`app.py`에서 지정), 클래스별 NMS IoU 0.45 | conf ≥ 0.1, 외부 NMS 없음 |
| ByteTrack | track_thresh 0.5, low_thresh 0.1, match 0.8, buffer 30, frame_rate 30 | track_thresh 0.25, low_thresh 0.1, match 0.8, buffer 30, frame_rate 30 |
| 인식 | KoreanSignNet + 트랙별 8프레임 시간 투표. v3 검출 클래스에 간판이 없어 OCR 분기는 호출되지 않음 | 현재 프레임의 KoreanSignNet 14-class, 시간 투표·OCR 없음 |
| 추적기 상태 | 서버 전역 파이프라인 공유 | WebSocket 연결마다 새 ByteTracker, `reset` 지원 |

따라서 두 경로의 FPS나 트랙 수를 같은 조건의 비교로 쓰지 않는다.

## 재검증 (2026-09-25)

Codex 작업 중단 후 같은 워크트리에서 다시 확인했다. 수치는 기능 확인용이며 논문 결과로 쓰지 않는다.

- `pytest tests/`: **22 passed, 1 skipped**. 건너뛴 1건은 v3 variant ONNX가 이 워크트리에 없어서 생긴 기존 `test_pipeline_variants.py`이며, v4 테스트 2건은 모두 통과했다.
- `ruff check`·`ruff format --check`(변경 파일)와 `mypy src/pipeline`(strict) 통과.
- `web_modern`: `npm run build` 성공, Vitest **31 passed**.
- 로컬 서버(`EDGE_SIGN_PAPER_V4=1`, `EDGE_SIGN_CPU_ONLY=1`)에서 `/api/paper-v4/status`가 `ready`였고 세 모델의 크기·SHA-256이 위 표와 일치했다. 이 워크트리에는 v3 검출기 ONNX가 없어 `/api/status`는 검출기 없이 응답했다. **v3 경로 회귀는 Space에서 확인해야 한다.**
- `benchmark_hf_space.py`로 두 variant를 각각 warm-up 2 + 측정 4프레임 실행했다. 두 실행 모두 `route=/ws/paper-v4`로 기록됐고, 추적기 초기화로 frame_id가 1–6으로 다시 시작했으며, trace의 variant가 요청과 일치했다.
- 실행 조건: 측정 스크립트는 `datetime.UTC`를 쓰므로 **Python 3.11 이상**이 필요하다(3.10인 `convnext_env`에서는 import 실패, 클라이언트 Python 3.13으로 실행). Windows Git Bash에서는 `/api/...` 인자가 Windows 경로로 바뀌므로 `MSYS_NO_PATHCONV=1`을 붙인다.

## 공개 Space 측정 전 확인할 것

1. 현재 공개 Space에 v4 전용 경로를 추가할지 배포 방식을 정한다. 기존 Space의 새 커밋은 재빌드·재시작을 일으킨다. 별도 신규 Space는 계정의 유료 플랜 요건에 걸릴 수 있다.
2. Space 저장소에 이 브랜치의 `src/pipeline/paper_v4.py`, `app.py`, `.dockerignore`, benchmark 코드와 **해시가 맞는** v4 ONNX 두 파일을 반영하고 `EDGE_SIGN_PAPER_V4=1`로 기동한다. 모델은 논문 Git 저장소에 커밋하지 않는다.
3. `/api/paper-v4/status`에서 Linux RSS, 실제 CPU/ORT, 파일 해시와 준비 상태를 확인한다. v3 `/api/status`와 `/detection/` 회귀도 확인한다.
4. 같은 샘플과 protocol로 v4 FP32/QDQ를 각각 10 warm-up + 50회 이상 측정하고 원시 trace를 별도 폴더에 보존한다. 공개 2 vCPU에 다른 방문자가 있으면 혼잡 요인으로 기록한다.
5. 브라우저의 카메라/영상 캡처 및 렌더까지 포함한 측정은 별도 수행한다. v4 WebSocket 측정만으로 배포 30 FPS를 PASS라고 판정하지 않는다.

현재 공개 Space v3 실측은 [HF_SPACE_V3_MEASUREMENT.md](HF_SPACE_V3_MEASUREMENT.md), 논문 주장 판정은 [TIIS_EVIDENCE_REPORT.md](TIIS_EVIDENCE_REPORT.md)에 있다.

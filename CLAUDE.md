# Edge-Sign: 초경량 온디바이스 간판/표지판 인식 시스템

## 프로젝트 개요

Edge-Sign은 엣지 디바이스에서 실시간으로 한글 간판과 교통표지판을 **검출 + 추적 + 인식**하는 시스템입니다.
극한의 신경망 양자화(W8A8, W4A16, SmoothQuant, 1-Bit)를 파이프라인 각 단계에 적용하여, <15MB 총 모델 크기로 30+ FPS 실시간 추론을 목표로 합니다.

### 연구 질문
> 검출+추적+인식 파이프라인에 단계별 양자화를 적용했을 때, 어떤 단계가 가장 민감하며, 엣지에서 실시간 구동이 가능한가?

---

## 프로젝트 단계

### Phase 1 (완료): 분류 양자화 기초 연구
- ConvNeXtV2-Nano 백본에 6가지 양자화 방법 비교
- W8A8 SmoothQuant 최고 성능 (Final Score 0.8068)
- ONNX 추출 및 CPU 추론 검증 완료

### Phase 2 (진행 중): 검출 + 추적 + 인식 파이프라인
- YOLOv8-Nano 검출기 + ByteTrack 추적기 + 분기 인식기
- 파이프라인 각 단계별 양자화 실험 (8개 구성)
- 웹 실시간 시연

### Phase 3 (계획): 주행 Q&A 결론 데모
- 엣지 파이프라인(YOLOv8n-INT8 + OCR-INT8) → 구조화 JSON → Groq LLM API → 자연어 답변
- 논문/발표 결론 섹션 시연 시스템

---

## 디렉토리 구조

```
CNN_Quant/
├── CLAUDE.md                    # 이 파일 - 프로젝트 진입점
├── docs/
│   ├── ROADMAP.md               # 단계별 로드맵 + 진행 상태
│   ├── ARCHITECTURE.md          # 파이프라인 아키텍처 + 설계 결정
│   └── EXPERIMENTS.md           # 양자화 실험 매트릭스 + 결과
│
├── src/                         # Python 소스 코드
│   ├── model.py                 # TrafficSignNet (65K params, 교통표지판 분류)
│   ├── korean_ocr_model.py      # KoreanOCRNet (700K params, 한글 OCR)
│   ├── base_W8A8.py             # W8A8 PTQ 구현
│   ├── base_train_w4a16_qat.py  # W4A16 QAT 학습
│   ├── base_train_1bit_kd.py    # 1-Bit 이진화 + 지식증류
│   ├── multimodal_w8a8_smoothquant.py  # SmoothQuant 구현
│   ├── export_onnx.py           # ONNX 내보내기 (opset 14)
│   ├── quantize_int8.py         # ONNX Runtime INT8 양자화
│   ├── final_omnimodal_eval.py  # 종합 평가 프레임워크
│   │
│   ├── detect/                  # [Phase 2] 검출 모듈
│   │   ├── prepare_dataset.py   # GTSDB/AI Hub 프레임 → YOLO 포맷 변환
│   │   ├── yolo_train.py        # YOLOv8n 학습/평가/추론
│   │   ├── export_yolo_onnx.py  # ONNX 내보내기 + INT8 양자화
│   │   └── train_traffic_sign_net.py  # TrafficSignNet GTSDB 학습 + ONNX 내보내기
│   ├── track/                   # [Phase 2] 추적 모듈
│   │   ├── bytetrack.py         # ByteTrack (Kalman+IoU, 8-dim) — 구현 완료
│   │   ├── eval_tracking.py     # MOT 평가 (MOTA/IDF1/HOTA)
│   │   └── run_tracking_ablation.py  # E1/E4/E5 추적 ablation 일괄 실행
│   ├── pipeline/                # [Phase 2] E2E 파이프라인
│   │   ├── e2e_pipeline.py      # 검출+추적+인식 통합 파이프라인
│   │   ├── eval_e2e.py          # E0~E7 전체 구성 종합 평가 (FPS + Final Score)
│   │   ├── qa_bridge.py         # LLM 컨텍스트 빌더 + Groq API 래퍼
│   │   ├── sources.py           # [SP1] FrameSource 추상화 (Image/VideoFile/UrlStream) — 모든 코덱 디코딩
│   │   ├── session.py           # [SP1] 서버 스트림 세션 매니저 (단일 세션 + 재생 제어)
│   │   ├── app.py               # FastAPI 서버 (WS /ws/stream·/ws/session, /api/ingest, /api/qa SSE)
│   │   └── logging_config.py    # [SP-A] 중앙 loguru 설정 + stdlib 인터셉트 (서버 startup 1회 호출)
│   └── quant/                   # [Phase 2] 파이프라인 양자화
│       ├── quantize_yolo.py     # W8A8/W4A16/SmoothQuant PTQ 구현
│       ├── run_experiments.py   # E1/E4/E5 검출기 양자화 실험 일괄 실행
│
├── Dockerfile                   # [Phase 11] HF Spaces (Docker, CPU) 이미지
├── .dockerignore                # 빌드 컨텍스트 슬림화 (필요 ONNX 4개만 선별)
├── requirements-hf.txt          # [Phase 11] HF Space 슬림 CPU 의존성
├── spaces/README.md             # [Phase 11] HF Space YAML 헤더 + 배포 안내
├── pyproject.toml               # [SP-A] ruff·mypy·pytest 설정 (mypy strict: src/pipeline)
├── requirements-dev.txt         # [SP-A] dev 도구 (ruff·mypy·pytest)
│
├── web_modern/                  # [SP-C] 웹 프론트엔드 (React 19 + Vite + TS) — 유일한 프론트
│   ├── src/                     # 검출+추적+인식+Q&A 콘솔: components·hooks·store·lib·styles
│   │   │                        #   양자화 A/B(PerfStrip)·단계 stage_ms·BYOK Q&A·통합 seek
│   │   │                        #   서버⇄온디바이스 토글(Controls) — 추론을 브라우저 WebGPU로
│   │   ├── lib/byteTrack.ts     # [SP-C] ByteTrack TS 포팅(클라 추적) — bytetrack.py 골든 검증
│   │   ├── lib/clientPipeline.ts # [SP-C] 온디바이스 검출+추적(ORT-Web) → FrameResult (서버 렌더 재사용)
│   │   └── hooks/useClientPipeline.ts # [SP-C] ORT-Web 로드(WebGPU ESM)+모델 fetch+추론
│   ├── public/
│   │   ├── ocr/                 # Phase 1 한글 OCR 캔버스 데모(ORT-Web) — /detection/ocr/ 로 체험
│   │   └── spike/               # [SP-C] 브라우저 온디바이스 타당성 스파이크 — /detection/spike/
│   │                            #   ORT-Web WebGPU YOLO FPS 실측(웹캠/파일/합성 프레임)
│   ├── index.html · vite.config.ts · package.json · tailwind.config.cjs
│   └── dist/                    # 빌드 산출물(gitignore) — FastAPI가 /detection/ 로 서빙
│                                #   (구 web/ vanilla 데모는 web_modern으로 통합·제거됨)
│
├── AIhub/                       # AI Hub 원본 데이터 (.gitignore 제외)
│   ├── 신호등-도로표지판 인지 영상(수도권)/  # TAR 압축 (9시퀀스, 110,900 JPG 프레임)
│   ├── 030.야외 실제 촬영 한글 이미지/      # 이미 해제됨 (30,141 JPG+JSON 쌍)
│   ├── 다양한 형태의 한글 문자 OCR/         # ZIP 압축 (인쇄체+필기체, 39.6GB)
│   └── 교통사고 블랙박스/                   # 미사용 (참고용 보관)
│
├── scripts/                     # 현행 데이터 준비 + 검증 스크립트
│   ├── extract_frames.py        # [Phase 2] AI Hub TAR 해제 + 시퀀스 분할 + 서브샘플링
│   ├── build_demo_video.py      # [Phase 7] test JPG 시퀀스 → H.264 mp4 합성 (검증/시연용) → data/demo_videos/
│   ├── prepare_korean_traffic.py # [Phase 8] AI Hub → 검출기(yolo_signs_v2) + 분류기 ROI(roi_cls) 단일패스 생성
│   ├── train_korean_classifier.py # [Phase 9] 한국 표지판/신호등 14클래스 분류기 학습 → korean_sign_net
│   ├── check_gpu_ort.py         # [Phase 10] onnxruntime-gpu CUDA EP 검증
│   ├── check_codec_matrix.py    # [Phase 10] H.264/MPEG-4/HEVC 서버 디코딩 검증
│   ├── quantize_v3_detector.py  # [Phase 11] v3 검출기 Static INT8(QDQ) — 헤드 제외, A/B용
│   ├── export_fp16_detector.py  # [SP-C] v3 검출기 FP32→FP16 ONNX(22MB) — 브라우저 WebGPU용
│   │                            #   (스파이크 실측: INT8은 WebGPU 불가, fp16은 지원·크기 절반)
│   ├── export_bytetrack_golden.py # [SP-C] ByteTrack 골든 출력 생성 → byteTrack.ts 검증 fixture
│   ├── analyze_quant_collapse.py # [Phase 12] 붕괴 '원인' 분석(data-free) — OCR=비트폭/헤드=활성화 규명, README §8.3
│   └── archive/                 # 종료된 Phase 1·4·5 실험·플롯·벤치마크·다운로드 스크립트 보관
│                                #   (plot_pareto/sensitivity/v2_extras/detection_samples,
│                                #    benchmark_pipeline, quantize_onnx_real, download_*, export_* 등)
├── checkpoints/                 # 학습 체크포인트 (.pth, gitignore)
├── models/                      # Phase 1 내보낸 모델 (safetensors, ONNX) — git 추적
├── model_space/                 # Phase 2/3 ONNX 모델 (검출/추적/인식, gitignore)
├── logs/                        # 실험 데이터 (학습 곡선 CSV, final_score_report.txt)
├── data/                        # 데이터셋·매핑·FAISS 인덱스 (gitignore)
└── assets/                      # 시각화 이미지
```

---

## 핵심 명령어

```bash
# 환경 설치
pip install -r requirements.txt

# Phase 1 - 기존 양자화 평가
python src/base_model.py              # FP16 기준선
python src/base_W8A8.py               # W8A8 PTQ
python src/final_omnimodal_eval.py    # 종합 평가

# Phase 2 - 데이터 준비
python scripts/archive/download_gtsdb.py       # GTSDB 다운로드 (이미 완료, 보관됨)

# 신호등-도로표지판 TAR 해제 + 시퀀스 분할 + 서브샘플링
python scripts/extract_frames.py --dry_run     # 분할 계획 미리보기
python scripts/extract_frames.py \
  --input "AIhub/신호등-도로표지판 인지 영상(수도권)/Validation" \
  --output data/aihub_traffic \
  --sample_rate 6                        # TAR 해제 (30fps→5fps 서브샘플, 시퀀스 단위 분할)

# YOLO 포맷 변환
python src/detect/prepare_dataset.py --source gtsdb           # GTSDB → YOLO 포맷
python src/detect/prepare_dataset.py --source aihub_traffic   # 신호등-도로표지판 → YOLO 포맷
python src/detect/prepare_dataset.py --source aihub_signboard # 야외 한글 간판 → YOLO 포맷
python src/detect/prepare_dataset.py --source all             # 전체 합산

# Phase 2 - 검출 학습
python src/detect/yolo_train.py --mode train --epochs 100  # YOLOv8n 학습
python src/detect/yolo_train.py --mode val                  # 평가
python src/detect/export_yolo_onnx.py --weights best.pt     # ONNX 내보내기

# Phase 2 - 전체 파이프라인
python src/pipeline/e2e_pipeline.py \
  --yolo model_space/yolov8n_signs_fp32.onnx \
  --ocr  model_space/korean_ocr_net_w8a8.onnx \
  --input data/aihub_traffic/val/   # E2E 추론 (JSON 출력)

python src/quant/run_experiments.py   # 양자화 실험 실행

# Phase 3 - 주행 Q&A 데모 서버
cp .env.example .env                  # GROQ_API_KEY 설정

# 시연/검증용 동영상 합성 (학습 미사용 test 시퀀스 → H.264 mp4)
# test 시퀀스는 여러 위치 스냅샷의 몽타주라, 기본 모드는 '같은 위치' 연속 구간을
# 개별 짧은 클립으로 분할 (각 ~3초, 박스 안정적, 일시정지하며 Q&A).
python scripts/build_demo_video.py --list                  # 사용 가능한 test 시퀀스 나열
python scripts/build_demo_video.py --fps 5 --top_n 12       # 위치별 클립 분할(기본)
# → data/demo_videos/<seq>_clips/clip_01.mp4 ... (검출 품질 확인 후 선별 사용)
python scripts/build_demo_video.py --full --fps 15         # (옵션) 시퀀스 전체 단일 영상

# 프론트 빌드(필수) — FastAPI는 web_modern/dist를 /detection/ 으로 서빙
(cd web_modern && npm ci && npm run build)   # → web_modern/dist (OCR 데모 public/ocr→dist/ocr 포함)
# (프론트 개발 시: cd web_modern && npm run dev → http://localhost:5173, /api·/ws는 8000으로 프록시)

uvicorn src.pipeline.app:app --reload --port 8000
# 브라우저 → http://localhost:8000/detection/   (헤더 'OCR 데모' → /detection/ocr/ 한글 OCR 캔버스)
# [SP-C] 컨트롤 바 '서버 ⇄ 온디바이스' 토글 → 검출+추적+인식을 브라우저 WebGPU로 직접 실행
#   (fp32/fp16 토글, /detection/spike/ 는 EP·FPS 실측 스파이크, /api/labels 인식 메타).
#   온디바이스 fp16 사용 시: KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_fp16_detector.py
# [SP1] 범용 입력: 웹캠·이미지·URL/RTSP·모든 코덱 영상 지원.
#   - H.264 등 브라우저 호환 영상/웹캠 → 클라 캡처(/ws/stream), 클라가 박스 렌더
#   - MPEG-4 등 비호환 코덱·URL·이미지 → 서버 인제스트(/api/ingest → /ws/session),
#     서버가 디코딩, 클라가 동일 스타일로 박스 렌더 (라벨링 두 모드 일치)
# 검증: python scripts/check_gpu_ort.py / python scripts/check_codec_matrix.py
# 테스트: python -m pytest tests/   (FrameSource·세션·ingest·variant·BYOK)
# 주의: 서버는 convnext_env에서 실행 — groq·onnxruntime-gpu·yt-dlp 설치 필요.
#   YOLO 재학습 시 KMP_DUPLICATE_LIB_OK=TRUE + --workers 0.
#   주의: conda run이 불안정하면 env python 직접 호출
#     ("$CONDA/envs/convnext_env/python.exe"). 셸 env는 set이 아닌 export/inline prefix.

# Phase 11 — 다이나믹 시연 + HF Space
# v3 검출기 INT8(QDQ, 헤드 제외 — 헤드 양자화 시 검출 붕괴) 생성:
KMP_DUPLICATE_LIB_OK=TRUE python scripts/quantize_v3_detector.py --bench
#   → model_space/yolov8s_signs_v3_int8_static.onnx (18MB, 검출 near-lossless)
#   서버가 v3 fp32+int8_static 둘 다 로드 → 웹 FP32⇄INT8 A/B 토글 노출.
# 양자화 A/B·단계 stage_ms·BYOK Q&A는 web_modern/src/ 에 구현 (React).
# HF Spaces (Docker, CPU 전용) — 멀티스테이지 빌드(node 빌더가 dist 생성) 로컬 검증:
docker build -t edge-sign . && docker run --rm -p 7860:7860 edge-sign
#   → http://localhost:7860/detection/  (EDGE_SIGN_CPU_ONLY=1, 모델 LFS 동봉은 spaces/README.md)

# SP-A — 코드 품질 (ruff·mypy·loguru, convnext_env에서 실행)
pip install -r requirements-dev.txt          # dev 도구 (ruff·mypy·pytest)
ruff format . && ruff check .                 # 포맷 + 린트 (E,F,I,UP,B,W,C4 / line 100 / E501 무시)
mypy src/pipeline                             # 엄격 타입 게이트 (서빙 경계만; ML 경계 완화)
python -m pytest tests/                       # 회귀 (20 passed: 16 + 로깅 4)
# 운영 로그=loguru(logger.*), CLI/__main__ 출력=print 유지. 레벨 EDGE_SIGN_LOG_LEVEL(기본 INFO). 핫루프 무로깅.
# 주의: 셸 기본 python은 base(deps 없음) — 반드시 convnext_env python으로 위 명령 실행.
```

---

## 에이전트 지침

### 문서 관리 규칙
1. **CLAUDE.md** (이 파일): 새 모듈/스크립트 추가 시 디렉토리 구조와 명령어 업데이트
2. **docs/ROADMAP.md**: 태스크 완료 시 `[x]` 체크, 계획 변경 시 항목 수정, 날짜 기록
3. **docs/ARCHITECTURE.md**: 설계 결정 변경/추가 시 업데이트, 이유 반드시 기록
4. **docs/EXPERIMENTS.md**: 실험 실행 시 결과 셀 채우기, 새 실험 추가 시 행 추가

### 코드 작성 규칙
- 기존 양자화 코드(`src/base_W8A8.py`, `src/base_train_w4a16_qat.py` 등)의 함수/클래스를 최대한 재활용
- ONNX 내보내기는 항상 opset 14 + TorchScript 모드 사용 (`export_onnx.py` 참조)
- 평가 코드는 `final_omnimodal_eval.py`의 Final Score 공식 사용: `0.6*Perf + 0.2*Speed + 0.2*Mem`
- 메인 웹 콘솔은 `web_modern/`(React/Vite/TS) — 기존 컴포넌트·hooks·store 패턴 따르기
- 온디바이스 ORT-Web 데모 패턴은 `web_modern/public/ocr/app.js`(한글 OCR) 참조

### 기술 스택
- **ML**: PyTorch 2.11+cu128, Ultralytics (YOLOv8), timm, transformers
- **양자화**: 커스텀 PTQ/QAT 구현(W4A16,W8A8), ONNX Runtime quantization
- **추론**: ONNX Runtime (CPU), ONNX Runtime Web (WASM/WebGPU)
- **웹**: FastAPI + WebSocket (서버), React 19 + Vite + TypeScript + zustand (web_modern), ONNX Runtime Web (OCR 데모), PWA
- **추적**: ByteTrack (Kalman + IoU), BoT-SORT (ReID 옵션)
- **Q&A**: Groq Python SDK (`groq`), Llama 3.3 70B (무료 티어), SSE 스트리밍
- **코드 품질 [SP-A]**: ruff(린트+포맷), mypy(`src/pipeline` strict), loguru(구조화 로깅), pytest — 설정 `pyproject.toml`, dev 의존성 `requirements-dev.txt`

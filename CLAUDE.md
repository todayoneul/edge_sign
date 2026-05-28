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
- 엣지 파이프라인(YOLOv8n-INT8 + OCR-INT8) → 구조화 JSON → Claude API → 자연어 답변
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
│   │   ├── qa_bridge.py         # LLM 컨텍스트 빌더 + Claude API 래퍼
│   │   └── app.py               # FastAPI 백엔드 서버 (WebSocket + SSE)
│   └── quant/                   # [Phase 2] 파이프라인 양자화
│       ├── quantize_yolo.py     # W8A8/W4A16/SmoothQuant PTQ 구현
│       ├── run_experiments.py   # E1/E4/E5 검출기 양자화 실험 일괄 실행
│
├── web/                         # 웹 프론트엔드
│   ├── index.html               # 한글 OCR 캔버스 데모
│   ├── app.js                   # ONNX Runtime Web 추론
│   ├── aihub/                   # OpenPose 데모
│   └── detection/               # [Phase 2] 검출+추적 + Q&A 데모
│       ├── index.html           # 검출 뷰 + 채팅 UI
│       └── app.js               # WebSocket 프레임 전송 + SSE Q&A
│
├── AIhub/                       # AI Hub 원본 데이터 (.gitignore 제외)
│   ├── 신호등-도로표지판 인지 영상(수도권)/  # TAR 압축 (9시퀀스, 110,900 JPG 프레임)
│   ├── 030.야외 실제 촬영 한글 이미지/      # 이미 해제됨 (30,141 JPG+JSON 쌍)
│   ├── 다양한 형태의 한글 문자 OCR/         # ZIP 압축 (인쇄체+필기체, 39.6GB)
│   └── 교통사고 블랙박스/                   # 미사용 (참고용 보관)
│
├── scripts/                     # 데이터 수집/전처리 스크립트
│   ├── extract_frames.py        # [Phase 2] AI Hub TAR 해제 + 시퀀스 분할 + 서브샘플링
├── checkpoints/                 # 학습 체크포인트 (.pth)
├── models/                      # 내보낸 모델 (safetensors, ONNX)
├── logs/                        # 학습 로그 (CSV)
├── data/                        # 데이터 매핑, FAISS 인덱스
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
python scripts/download_gtsdb.py               # GTSDB 다운로드 (이미 완료)

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
  --ocr  web/korean_ocr_quant.onnx \
  --input data/aihub_traffic/val/   # E2E 추론 (JSON 출력)

python src/quant/run_experiments.py   # 양자화 실험 실행

# Phase 3 - 주행 Q&A 데모 서버
cp .env.example .env                  # ANTHROPIC_API_KEY 설정
uvicorn src.pipeline.app:app --reload --port 8000
# 브라우저 → http://localhost:8000/detection/
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
- 웹 코드는 기존 `web/app.js`의 ONNX Runtime Web 패턴 따르기

### 기술 스택
- **ML**: PyTorch 2.11+cu128, Ultralytics (YOLOv8), timm, transformers
- **양자화**: 커스텀 PTQ/QAT 구현(W4A16,W8A8), ONNX Runtime quantization
- **추론**: ONNX Runtime (CPU), ONNX Runtime Web (WASM/WebGPU)
- **웹**: FastAPI + WebSocket (서버), ONNX Runtime Web (클라이언트), PWA
- **추적**: ByteTrack (Kalman + IoU), BoT-SORT (ReID 옵션)
- **Q&A**: Anthropic Python SDK (`anthropic`), Claude Haiku API, SSE 스트리밍

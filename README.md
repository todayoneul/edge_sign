<div align="center">

# 🚦 Edge-Sign

### 초경량 On-device 교통표지판·신호등 인식 시스템

**Detection → Tracking → Recognition → Scene Q&A**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/)
[![WebGPU](https://img.shields.io/badge/WebGPU-On--device-4285F4?style=flat-square)](https://www.w3.org/TR/webgpu/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](LICENSE)

[**Live Demo**](https://huggingface.co/spaces/gyann/edge-sign) ·
[**Technical Notes**](docs/TECHNICAL_REPORT.md) ·
[**Experiments**](docs/EXPERIMENTS.md) ·
[**Release**](https://github.com/todayoneul/edge_sign/releases/tag/v1.0.1)

</div>

Edge-Sign은 도로 영상에서 **교통표지판과 신호등을 실시간으로 검출·추적·인식**하고, 구조화된 인식 결과를 LLM 기반 주행 Q&A까지 연결한 프로젝트입니다. 단순 정확도뿐 아니라 **모델 크기, CPU latency, 브라우저 runtime, 양자화 안정성**을 함께 측정했습니다.

<p align="center">
  <img src="assets/v3/v3_detection_sample.jpg" width="78%" alt="Edge-Sign detection sample">
</p>

## 핵심 결과

| Metric | Result | Context |
|---|---:|---|
| **INT8 model size** | **11.7 MB** | Static INT8 QDQ, server CPU |
| **CPU inference** | **56.3 FPS** | 30 FPS 목표 초과 |
| **Browser on-device** | **62 FPS** | FP32 + WebGPU |
| **Traffic detection mAP@0.5** | **0.776** | Phase 3, traffic sign / light |
| **Korean sign/light classifier** | **80.3%** | 14-class validation |

> 이론적 INT 최소 크기 5.6 MB와 실제 배포 파일 11.7 MB를 구분해 기록합니다.

## 무엇을 검증했나

- **W8A8은 검출기에서 거의 무손실**이지만 W4A16은 추적·인식 성능을 크게 떨어뜨렸습니다.
- **낮은 정밀도 = 항상 빠름**이 아니었습니다. 브라우저에서는 FP32/WebGPU가 INT8/WASM보다 훨씬 빨랐습니다.
- YOLO 검출 헤드는 full INT8에서 붕괴할 수 있어, tensor cosine similarity보다 **실제 detection count / confidence** 검증이 중요했습니다.
- 최종 배포는 환경에 따라 **Browser: FP32/WebGPU**, **Server CPU: INT8 Static QDQ**로 분리했습니다.

## 데이터 파이프라인 & 품질 관리

모델 결과의 신뢰도는 데이터 품질에서 나온다고 보고, 구축·정제·분할·검증 전 구간을 직접 설계했습니다.

| 단계 | 핵심 작업 | 결과 |
|---|---|---|
| **구축** | 이기종 3개 공개 데이터셋(GTSDB·AI Hub 2종)의 상이한 어노테이션 스키마(xyxy·COCO xywh·gt.txt)를 단일 YOLO 포맷으로 정규화 통합 | train 26,866 / val 4,667 |
| **정제·필터링** | 인접 프레임 서브샘플(30→5fps)로 중복 제거, type-불명·비대상 어노테이션 필터링 | 110,900 → 18,488 프레임 |
| **분할** | 촬영 시퀀스(TAR) 단위 분할로 인접 프레임의 train/val 데이터 리크 차단 | leak-free split |
| **품질 관리** | 초기 분할의 주간 검증 누락(도메인 편향)을 발견 → 주·야간 stratified 재분할 | 신뢰 가능한 기준선 mAP 0.587 |

- **검증 지표의 맹점 제거:** 출력 CosSim 0.9995인데 실제 검출 0건인 함정을 발견 → 텐서 유사도 대신 **실프레임 detection count·confidence**를 판정 기준으로 채택.
- **원인 기반 실패 분석:** 붕괴를 관찰에 그치지 않고 원인별로 분해 — OCR은 비트폭(고-fan-in 레이어 SQNR 12–13 dB), 검출 헤드는 활성화 이상치(초과첨도 49). (`scripts/analyze_quant_collapse.py`)
- **학습 = 추론 전처리 일치:** 분류기 학습·추론 전처리 불일치로 인한 분포 shift를 제거해 val 정확도 80.3% 확보.

## Architecture

```mermaid
flowchart LR
    A[Video / Webcam] --> B[YOLO Detection]
    B --> C[ByteTrack]
    C --> D[Sign & Light Recognition]
    D --> E[FrameResult JSON]
    E --> F[Overlay]
    E --> G[LLM Scene Q&A]

    B -. Browser .-> H[ONNX Runtime Web / WebGPU]
    B -. Server .-> I[ONNX Runtime / INT8]
```

## Demo

온라인 데모에서는 샘플 영상·웹캠을 이용해 **Server ↔ On-device**, **FP32 ↔ INT8** 경로를 비교할 수 있습니다.

▶ **[Hugging Face Space에서 실행](https://huggingface.co/spaces/gyann/edge-sign)**

브라우저 호환 H.264 입력은 WebGPU 경로를 사용할 수 있고, 비호환 코덱은 서버 디코딩 경로로 자동 fallback됩니다.

## Quick Start

```bash
pip install -r requirements.txt

# pipeline server
uvicorn src.pipeline.app:app --port 8000

# tests
python -m pytest tests/
```

양자화 재현:

```bash
python scripts/archive/quantize_onnx_real.py
python scripts/archive/benchmark_pipeline.py --pipe_only
```

## Project Map

| Path | Role |
|---|---|
| `src/detect/` | YOLO training / export |
| `src/track/` | ByteTrack / tracking evaluation |
| `src/quant/` | quantization experiments |
| `src/pipeline/` | E2E inference, API, session |
| `web_modern/` | React + WebGPU demo |
| `assets/` | experiment plots / qualitative samples |
| `docs/` | detailed experiments and design notes |

## 더 자세히 보기

- [Experiment details](docs/EXPERIMENTS.md)
- [Technical report / 이전 상세 README 안내](docs/TECHNICAL_REPORT.md)
- [Third-party licenses](THIRD_PARTY_NOTICES.md)
- [AGPL-3.0 License](LICENSE)

### Scope

현재의 “Edge” 측정은 **Browser/WASM/WebGPU 및 Server CPU runtime** 기준입니다. Jetson·Raspberry Pi 등 전용 edge hardware 실측은 포함하지 않으며, tracking MOTA의 절대값보다 **quantization에 따른 상대 열화와 배포 trade-off**를 주요 분석 대상으로 삼았습니다.

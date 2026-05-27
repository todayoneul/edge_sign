# Edge-Sign: 초경량 온디바이스 간판/표지판 인식 시스템
**(Edge-Sign: Ultra-Lightweight On-Device Signboard & Traffic Sign Recognition System)**

## 프로젝트 개요 (Project Overview)

**Edge-Sign**은 엣지 디바이스에서 실시간으로 한글 간판과 교통표지판을 **검출 → 추적 → 인식**하는 시스템입니다.  
극한의 신경망 양자화(W8A8, W4A16, SmoothQuant, 1-Bit)를 파이프라인 각 단계에 적용하여,  
**< 15 MB 총 모델 크기**로 **30+ FPS 실시간 추론**을 목표로 합니다.

### 연구 질문
> 검출+추적+인식 파이프라인에 단계별 양자화를 적용했을 때, **어떤 단계가 가장 민감**하며, 엣지에서 실시간 구동이 가능한가?

---

## 목차 (Table of Contents)
- [1. 핵심 방법론: 신경망 압축](#1-핵심-방법론-신경망-압축-core-compression-methodology)
- [2. 실험 환경 및 데이터셋](#2-실험-환경-및-데이터셋-experimental-setup--dataset)
- [3. Phase 1 — 압축 성능 평가 및 파레토 프론티어](#3-phase-1--압축-성능-평가-및-파레토-프론티어)
- [4. Phase 1 — 옴니모달(VLM) 한계 검증](#4-phase-1--옴니모달vlm-한계-검증)
- [5. 종합 평가 및 최적 모델 선정 (Final Score)](#5-종합-평가-및-최적-모델-선정-final-score)
- [6. Phase 2 — 검출+추적+인식 파이프라인 설계](#6-phase-2--검출추적인식-파이프라인-설계)
- [7. Phase 2 양자화 실험 매트릭스](#7-phase-2-양자화-실험-매트릭스)
- [8. 웹 배포 아키텍처](#8-웹-배포-아키텍처)
- [9. 재현 가이드 (Reproduction Guide)](#9-재현-가이드-reproduction-guide)

---

## 1. 핵심 방법론: 신경망 압축 (Core Compression Methodology)

### 1.1. 8-Bit PTQ (Post-Training Quantization)
학습이 완료된 모델의 가중치를 256개의 구간(-128 ~ 127)으로 선형 맵핑합니다.  
재학습 없이 즉각적인 메모리 절감(14.9 MB)이 가능하며, FP16 대비 **0.64%p 미만의 성능 하락**을 보였습니다.

### 1.2. 4-Bit QAT & Custom STE
가중치를 16개의 구간(-8 ~ 7)으로 압축할 때 발생하는 Weight Collapse를 극복하기 위해 **QAT(양자화 인지 학습)** 를 도입했습니다.  
미분 불가능한 양자화 함수의 그레디언트를 통과시키기 위해 **Straight-Through Estimator (STE)** 를 직접 설계했습니다.

$$\text{Forward: } W_q = \text{Clamp}(\text{Round}(W / \Delta), -8, 7) \times \Delta$$

$$\text{Backward: } \frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q} \quad (\text{if } W \in [-8, 7] \text{ else } 0)$$

### 1.3. 1-Bit Binarization & Bit-Packing
모든 CNN 필터 가중치를 +1과 -1로 이진화하며, **채널별 L1 Norm**을 스케일 팩터로 활용합니다.  
`numpy.packbits`로 8개의 이진 가중치를 1개의 `uint8`에 패킹하여 **1.99 MB 달성**했습니다.

### 1.4. Knowledge Distillation (KD)
1-Bit 환경의 정보 병목을 극복하기 위해 FP16 교사 모델의 소프트 라벨(KL Divergence)을 혼합합니다.

$$L_{KD} = \alpha \cdot T^2 \cdot D_{KL}\!\left( \sigma\!\left(\frac{Z_S}{T}\right) \| \sigma\!\left(\frac{Z_T}{T}\right) \right) + (1-\alpha) \cdot CE(Z_S, y)$$

---

## 2. 실험 환경 및 데이터셋 (Experimental Setup & Dataset)

### 2.1. Phase 1 — 분류 모델 사전학습
| 항목 | 내용 |
| :--- | :--- |
| **Architecture** | ConvNeXtV2-Nano (`convnextv2_nano.fcmae_ft_in1k`) |
| **Pre-train Dataset** | ImageNet-1K (1.2M images, 1000 classes) |
| **Hardware** | NVIDIA RTX 5070 12 GB / PyTorch 2.x |

```bash
pip install -r requirements.txt
```

### 2.2. Phase 2 — 검출+추적+인식 데이터셋

| 데이터셋 | 설명 | 용도 |
| :--- | :--- | :--- |
| [AI Hub 신호등·도로표지판 인지 영상(수도권)](https://aihub.or.kr/) | 9 시퀀스, 110,900 프레임 (30 fps) | YOLOv8n 검출 학습 |
| [AI Hub 야외 실제 촬영 한글 이미지](https://aihub.or.kr/) | 30,141 JPG+JSON 쌍 | 간판 OCR 학습 |
| [GTSDB](https://benchmark.ini.rub.de/gtsdb_news.html) | 독일 교통표지판 벤치마크 | 교통표지판 검출 보강 |

---

## 3. Phase 1 — 압축 성능 평가 및 파레토 프론티어

| 모델 (Quantization) | 메모리 (MB) | Top-1 Acc (%) | 비고 |
| :--- | :---: | :---: | :--- |
| **Baseline (FP16)** | 125.0 | 81.88 | Hugging Face Pre-trained |
| **W8A8 (PTQ)** | 14.9 | 81.24 | Zero-shot Calibration |
| **W4A16 (QAT)** | 14.92 | 76.12 | Custom STE |
| **1-Bit (QAT + KD)** | 1.99 | 14.23 | Bit-packing, Teacher-Student KD |

*(1.99 MB 환경의 14.23% 정확도는 물리적 정보 한계치를 정량화한 결과이며, 무작위 확률(0.1%) 대비 140배 이상의 성능을 지식 증류로 방어한 수치입니다.)*

---

## 4. Phase 1 — 옴니모달(VLM) 한계 검증

엣지 환경에서 VLM 아키텍처의 적합성을 검증하기 위해 1-Bit 압축 환경에서 선행 실험을 수행했습니다.

### 4.1. 1-Bit × 멀티모달 공간 얼라인먼트 붕괴

CLIP(openai/clip-vit-base-patch32)의 의미론적 공간을 1-Bit ConvNeXt-Nano에 매핑할 때 발생하는 얼라인먼트 붕괴 현상을 관찰했습니다.

![Omni-Modal Alignment Progress](./assets/mm_all_progress.png)

- **FP16 / 8-Bit / 4-Bit:** 10 에포크 이내 코사인 유사도 0.88~0.90 안정 수렴
- **1-Bit:** 정보 병목으로 0.80 부근에서 수렴 한계

### 4.2. 프로젝션 헤드 아키텍처 분석

| 평가 지표 (Recall@K) | 1-Bit (Linear Head) | 1-Bit (MLP Head) |
| :--- | :---: | :---: |
| **Recall@1** | **14.20%** | 11.30% (▼2.90%p) |
| **Recall@5** | **31.30%** | 28.50% (▼2.80%p) |
| **Recall@10** | **41.60%** | 38.90% (▼2.70%p) |

> **결론:** 극단적 1-Bit 희소성 환경에서는 단순한 Linear Head가 복잡한 MLP보다 더 강건합니다.

---

## 5. 종합 평가 및 최적 모델 선정 (Final Score)

$$\text{Final Score} = 0.6 \times \text{PerfNorm} + 0.2 \times \text{SpeedNorm} + 0.2 \times \text{MemNorm}$$

각 항은 FP16 기준선 대비 정규화된 값이며 상한을 1.0으로 고정합니다.

![Inference Latency Comparison](./assets/mm_latency_comparison.png)
![Pareto Frontier](./assets/mm_final_pareto.png)

| 모델 | Recall@1 (%) | Latency (ms) | Memory (MB) | Final Score |
| :--- | :---: | :---: | :---: | :---: |
| **W8A8 SmoothQuant PTQ** | 38.50 | 10.29 | 30.70 | **0.8068** |
| FP16 Baseline | 39.00 | 6.09 | 125.00 | 0.8000 |
| W4A16 QAT | 34.80 | 9.97 | 14.92 | 0.7628 |
| W8A8 QAT | 36.80 | 12.28 | 14.90 | 0.7314 |
| 1-Bit (Linear Head) | 14.20 | 9.02 | 1.99 | 0.3680 |
| 1-Bit (MLP Head) | 11.30 | 8.51 | 1.99 | 0.3218 |

**→ W8A8 SmoothQuant를 Phase 2 파이프라인의 인식 백본으로 채택합니다.**

### 5.1. ONNX 배포 성과

- **ONNX Export:** `opset_version=14` + TorchScript 익스포터로 Shape Inference Error 해결
- **ONNX Runtime INT8 동적 양자화:** FP32 60.70 MB → **15.61 MB** (압축률 3.9×)
- **순수 CPU 추론:** `ONNX Runtime (CPUExecutionProvider)`만으로 테스트 이미지 Rank-1 신뢰도 41.18% 정확 분류

---

## 6. Phase 2 — 검출+추적+인식 파이프라인 설계

### 6.1. 전체 파이프라인 구조

```
영상 입력 (대시캠 / 거리 영상 / 웹캠, 640×480)
         │
         ▼
┌─────────────────────────────┐
│ 1단계: YOLOv8-Nano 검출기   │  3.2M params, FP16 ~6.3 MB
│ 클래스: signboard           │  입력: 640×640 RGB
│         traffic_sign        │  출력: bbox, confidence, class
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 2단계: ByteTrack 추적기      │  모델 파라미터 없음 (Kalman + IoU)
│ ablation: BoT-SORT + ReID   │  (E6 실험: OSNet-x0.25 ReID 양자화)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 3단계: 클래스별 분기 인식기  │
│  signboard  → KoreanOCRNet  │  700K params, 2350 한글 문자 클래스
│    ROI 크롭: 64×64 gray     │
│  traffic_sign → TrafficNet  │  65K params, 12 교통표지판 클래스
│    ROI 크롭: 32×32 RGB      │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 결과 조합 + 오버레이 출력    │
│ Track ID + bbox             │
│ 간판: OCR 텍스트            │
│ 표지판: 분류 라벨            │
└─────────────────────────────┘
```

### 6.2. 모델 선택 근거

| 구성 요소 | 선택 모델 | 선택 근거 |
| :--- | :--- | :--- |
| 검출기 | YOLOv8-Nano (3.2M) | Ultralytics ONNX/양자화 지원 성숙. 엣지 예산 충족 |
| 추적기 (기본) | ByteTrack | 추가 파라미터 없음. 검출기 양자화 효과를 순수하게 분리 가능 |
| 추적기 (ablation) | BoT-SORT + OSNet-x0.25 | ReID 백본 양자화 효과를 E6 실험에서 측정 |
| 간판 OCR | KoreanOCRNet (700K) | Phase 1 양자화 실험 완료. 신규 학습 불필요 |
| 교통표지판 분류 | TrafficSignNet (65K) | 동일 — Phase 1 재활용 |

### 6.3. 데이터 파이프라인

AI Hub 데이터셋의 원천 데이터는 **동영상 시퀀스**입니다.  
프레임 단위 랜덤 분할 시 동일 동영상의 인접 프레임이 train/val에 동시 노출되어 **데이터 리크**가 발생합니다.  
이를 방지하기 위해 **시퀀스 단위 분할**을 채택합니다.

```
AI Hub validation 동영상 (~40 GB, ~30 fps)
         │
         ▼
scripts/extract_frames.py
  --sample_rate 6          # 6× 다운샘플: 30fps→5fps, 시각적 중복 제거
  --split_by sequence      # 동영상 경계 단위 train/val 분리 (프레임 리크 없음)
         │
         ├── train sequences (80%) → data/aihub_traffic/frames/train/
         └── val sequences   (20%) → data/aihub_traffic/frames/val/
                                     + data/aihub_traffic/val_videos/  ← 추적 평가·시연용
         │
         ▼
src/detect/prepare_dataset.py --source aihub_traffic
  # AI Hub JSON 어노테이션 → YOLO bbox 포맷 (.txt)
         │
         ▼  (GTSDB 합산: --source all)
data/yolo_signs/dataset.yaml  →  YOLOv8n 학습
```

| 분할 방식 | 데이터 리크 | 이유 |
| :--- | :---: | :--- |
| 프레임 단위 랜덤 분할 | **발생** | 동일 동영상 인접 프레임이 train/val에 동시 존재 |
| **시퀀스 단위 분할 (채택)** | **없음** | train/val 동영상이 완전히 분리됨 |

---

## 7. Phase 2 양자화 실험 매트릭스

파이프라인의 각 단계를 독립적으로 양자화하여 **단계별 양자화 민감도**를 정량화합니다.

| ID | 검출기 | 추적기 | 간판 OCR | 교통 분류 | 예상 총 크기 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **E0** | FP16 | ByteTrack | FP16 | FP16 | ~10 MB |
| **E1** | W8A8 | ByteTrack | FP16 | FP16 | ~8 MB |
| **E2** | FP16 | ByteTrack | W8A8 | W8A8 | ~7 MB |
| **E3** | W8A8 | ByteTrack | W8A8 | W8A8 | ~5 MB |
| **E4** | W4A16 | ByteTrack | W4A16 | W4A16 | ~3 MB |
| **E5** | SmoothQuant | ByteTrack | SmoothQuant | SmoothQuant | ~6 MB |
| **E6** | W8A8 | BoT-SORT (W8A8 ReID) | W8A8 | W8A8 | ~7 MB |
| **E7** | W4A16 | ByteTrack | 1-Bit | 1-Bit | ~2 MB |

### 평가 지표

- **검출:** mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **추적:** MOTA, IDF1, HOTA, ID Switches
- **인식:** OCR 문자/단어 정확도, 표지판 Top-1/Top-5
- **종합:** 총 모델 크기, FPS (CPU), FPS (ONNX Runtime Web), Final Score

$$\text{Final Score} = 0.6 \times \frac{\text{인식률}_i}{\text{인식률}_{E0}} + 0.2 \times \frac{\text{Latency}_{E0}}{\text{Latency}_i} + 0.2 \times \min\!\left(1, \frac{\text{크기}_{E0}}{\text{크기}_i}\right)$$

실험 결과는 `docs/EXPERIMENTS.md`에 기록하며, 완료 후 Pareto Frontier 차트를 생성합니다.

---

## 8. 웹 배포 아키텍처

### 모드 1: 전체 클라이언트 사이드 추론 (목표)

```
브라우저 (ONNX Runtime Web)
┌──────────────────────────────────┐
│  Camera API → 프레임 캡처         │
│  → YOLOv8n ONNX (WASM/WebGPU)   │
│  → ByteTrack (순수 JavaScript)   │
│  → ROI 크롭                       │
│  → OCR / 분류 ONNX (WASM)        │
│  → Canvas 오버레이 렌더링          │
└──────────────────────────────────┘
총 모델 페이로드 목표: < 15 MB
```

### 모드 2: 서버 어시스트 (Fallback)

```
브라우저                        FastAPI 서버
┌────────────┐   WebSocket    ┌──────────────────┐
│ Camera     │ ────────────►  │ YOLOv8n          │
│ 프레임      │                │ + ByteTrack      │
│            │ ◄────────────  │ + 인식기          │
│ 결과 표시   │   JSON 결과    │ (CPU / GPU)      │
└────────────┘                └──────────────────┘
```

참조 구현: `web/app.js` (ONNX Runtime Web 클라이언트), `web/detection/` (검출+추적 데모)

---

## 9. 재현 가이드 (Reproduction Guide)

### 환경 설치

```bash
pip install -r requirements.txt
```

### Phase 1 — 분류 양자화

```bash
python src/base_model.py                    # FP16 기준선
python src/base_W8A8.py                     # W8A8 PTQ
python src/base_train_w4a16_qat.py          # W4A16 QAT
python src/base_train_1bit_kd.py            # 1-Bit + KD
python src/multimodal_w8a8_smoothquant.py   # SmoothQuant
python src/final_omnimodal_eval.py          # 종합 평가 (Final Score)
```

### Phase 1 — ONNX 추출 및 CPU 추론

```bash
python src/export_onnx.py     # PyTorch → ONNX (opset 14, TorchScript)
python src/quantize_int8.py   # ONNX Runtime INT8 동적 양자화
```

### Phase 2 — 데이터 준비

```bash
# GTSDB 다운로드 및 변환
python scripts/download_gtsdb.py
python src/detect/prepare_dataset.py --source gtsdb

# AI Hub 동영상 → 프레임 (시퀀스 단위 분할)
python scripts/extract_frames.py \
  --input  "AIhub/신호등-도로표지판 인지 영상(수도권)/Validation" \
  --output data/aihub_traffic \
  --sample_rate 6

# 어노테이션 변환 및 통합
python src/detect/prepare_dataset.py --source aihub_traffic
python src/detect/prepare_dataset.py --source aihub_signboard
python src/detect/prepare_dataset.py --source all
```

### Phase 2 — 검출 학습

```bash
python src/detect/yolo_train.py --mode train --epochs 100
python src/detect/yolo_train.py --mode val
python src/detect/export_yolo_onnx.py --weights best.pt
```

### Phase 2 — E2E 파이프라인 및 양자화 실험 (진행 예정)

```bash
python src/pipeline/e2e_pipeline.py    # 전체 파이프라인 추론
python src/quant/run_experiments.py    # E0–E7 실험 일괄 실행
```

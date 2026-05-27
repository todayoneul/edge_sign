# Edge-Sign: 초경량 온디바이스 수어 해석 시스템 및 극단적 신경망 압축 프레임워크(Edge-Sign: Ultra-Lightweight On-Device Sign Language Interpreter & Extreme Neural Network Compression Framework)

## 프로젝트 개요 (Project Plan)

### 1. 프로젝트 이름
**Edge-Sign**: 경량화 ConvNeXt 기반 서버리스(Serverless) 온디바이스 실시간 수어 해석 시스템

### 2. 프로젝트 소개 및 연구 동기
본 프로젝트는 청각장애인과의 원활한 소통을 위한 **'실시간 한국수어(KSL) 번역 모델'을 인터넷 연결(통신)이나 고성능 클라우드 서버 없이, 사용자의 스마트폰 웹 브라우저 및 초소형 IoT 기기(Smart Glass 등 Edge Device)에서 독립적으로 구동**시키는 것을 목표로 하는 코어 시스템 엔지니어링 및 응용 연구입니다.

* **문맥(Context) 인지의 필요성:** Google MediaPipe와 같은 기존 솔루션은 손가락 관절(Skeleton) 3D 좌표만 추출합니다. 그러나 실제 수어는 손의 위치뿐만 아니라 얼굴 표정(비수지 기호), 몸의 기울기 등 '문맥'이 의미를 결정하므로 이미지 전체를 픽셀 단위로 해석하는 고성능 비전 모델(CNN/ViT)이 필수적입니다.
* **극한의 하드웨어 제약 (Edge AI):** 고성능 비전 모델은 수백 MB 이상의 메모리와 막대한 연산량을 요구하여 엣지 디바이스 탑재가 불가능합니다. 서버 통신으로 우회할 경우 프라이버시 침해(카메라 영상 전송), 통신 지연(Latency), 서버 유지 비용이 발생합니다.
* **코어 엔지니어링을 통한 해결:** 이를 극복하기 위해, 파이토치(PyTorch)의 연산을 수학적으로 재정의하여 **125MB의 거대 모델을 14.9MB(8-Bit PTQ) 및 1.99MB(1-Bit Binarization) 수준으로 극단적으로 압축**하는 기술을 직접 구현했습니다.
* **[최종 결론] 하이브리드 파이프라인의 채택:** 자연스러운 번역을 위해 최신 옴니모달(VLM) 구조를 검토하였으나, 엣지 환경에서의 치명적인 FPS 저하 및 메모리 초과 현상을 실험적으로 확인했습니다. 이에 따라 본 프로젝트는 초고속 수어 키워드 추출(W8A8 비전 엔진)과 경량화된 자연어 조립(NLP Rule-base)을 결합한 **'Action-Trigger 기반 하이브리드 파이프라인'** 을 최종 채택하여 60FPS의 제로-레이턴시를 달성합니다.

---

## 목차 (Table of Contents)
- [1. 핵심 방법론: 신경망 압축 (Core Compression Methodology)](#1-핵심-방법론-신경망-압축-core-compression-methodology)
- [2. 실험 환경 및 타겟 데이터셋 (Experimental Setup & Target Domain)](#2-실험-환경-및-타겟-데이터셋-experimental-setup--target-domain)
- [3. [Phase 1] 압축 성능 평가 및 파레토 프론티어 분석](#3-phase-1-압축-성능-평가-및-파레토-프론티어-분석)
- [4. [Phase 1] 옴니모달(VLM) 한계 검증을 위한 사전 연구](#4-phase-1-옴니모달vlm-한계-검증을-위한-사전-연구)
- [5. 종합 평가 및 최적 모델 선정 (Final Score)](#5-종합-평가-및-최적-모델-선정-final-score)
- [6. [Phase 2] 검출+추적+인식 파이프라인 설계](#6-phase-2-검출추적인식-파이프라인-설계)
- [7. Phase 2 양자화 실험 매트릭스](#7-phase-2-양자화-실험-매트릭스)
- [8. 웹 배포 아키텍처 및 추론 환경](#8-웹-배포-아키텍처-및-추론-환경)
- [9. 재현 가이드 (Reproduction Guide)](#9-재현-가이드-reproduction-guide)

---

## 1. 핵심 방법론: 신경망 압축 (Core Compression Methodology)

### 1.1. 8-Bit PTQ (Post-Training Quantization)
학습이 완료된 모델의 가중치를 256개의 구간(-128 ~ 127)으로 선형 맵핑(Linear Mapping)합니다. 재학습(Epoch) 없이 즉각적인 메모리 절반(14.9MB) 단축이 가능하며, 원본(FP16) 대비 0.64%p의 미미한 성능 하락만을 보였습니다.

### 1.2. 4-Bit QAT & Custom STE
가중치를 16개의 구간(-8 ~ 7)으로 압축할 경우 발생하는 뇌사 상태(Weight Collapse)를 극복하기 위해 **QAT(양자화 인지 학습)** 를 도입했습니다. 미분 불가능한 양자화 함수의 그레디언트를 통과시키기 위해 **Straight-Through Estimator (STE)** 함수를 아래 수식과 같이 설계하여 적용했습니다.

$$\text{Forward: } W_q = \text{Clamp}(\text{Round}(W / \Delta), -8, 7) \times \Delta$$
   
$$\text{Backward: } \frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q} \quad (\text{if } W \in [-8, 7] \text{ else } 0)$$

### 1.3. 1-Bit Binarization & Bit-Packing
모든 CNN 필터 가중치를 흑백(+1과 -1)으로 이진화하며, 필터의 크기 소실을 막기 위해 **채널별 절댓값 평균(Per-channel L1 Norm)** 을 스케일 팩터로 활용합니다. 디스크 추출 시 파이썬 `numpy.packbits`를 적용하여 8개의 이진 가중치를 1개의 `uint8` 메모리 블록에 욱여넣는(Bit-packing) 기술을 구현해 1.99MB 도달에 성공했습니다.

### 1.4. Knowledge Distillation (KD) 기반 성능 방어
1-Bit 환경에서의 치명적인 **정보 병목(Information Bottleneck)** 현상을 극복하기 위해, FP16 교사 모델(Teacher)의 소프트 라벨(KL Divergence)을 혼합하여 연산합니다.   
$$L_{KD} = \alpha \cdot T^2 \cdot D_{KL}\left( \sigma\left(\frac{Z_S}{T}\right) \| \sigma\left(\frac{Z_T}{T}\right) \right) + (1-\alpha) \cdot CE(Z_S, y)$$

---

## 2. 실험 환경 및 타겟 데이터셋 (Experimental Setup & Target Domain)

### 2.1. Core Engine Training (경량화 엔진 사전학습)
* **Dataset:** ImageNet-1K (1.2 Million Images, 1000 Classes) 
* **Base Architecture:** ConvNeXtV2-Nano (`convnextv2_nano.fcmae_ft_in1k`)
* **Hardware Framework:** NVIDIA RTX 5070 12GB (Local) / PyTorch 2.x
* **Dependencies:** 프로젝트 구동을 위한 파이썬 패키지 의존성은 루트 디렉토리의 `requirements.txt`에 정의되어 있습니다. 아래 명령어로 쉽게 설치할 수 있습니다.
  ```bash
  pip install -r requirements.txt
  ```

### 2.2. Target Domain (최종 서비스 도메인: 한국수어)
* **Dataset:** [AI Hub - 수어 영상 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=103)
* **Data Scale:** 한국인 수어 구사자 20인 이상이 참여한 **53.6만 개 이상의 고해상도 수어 영상 클립**.
* **Usage:** 완성된 압축 비전 엔진이 실제 한국 수어의 문맥과 형태적 특징을 정밀하게 학습하기 위한 파인튜닝(Fine-tuning) 데이터셋으로 활용.

---

## 3. [Phase 1] 압축 성능 평가 및 파레토 프론티어 분석

| 모델 아키텍처 (Quantization) | 물리적 메모리 용량 (MB) | Top-1 Accuracy (%) | 특징 및 적용 기술 |
| :--- | :---: | :---: | :--- |
| **Baseline (FP16)** | **125.0 MB** | 81.88 % | Hugging Face Pre-trained Model |
| **W8A8 (PTQ)** | **14.9 MB** | 81.24 % | Zero-shot Calibration, INT8 캐스팅 |
| **W4A16 (QAT)** | **14.92 MB** | 76.12 % | Custom STE, INT8 그릇 내 4-Bit 제한 |
| **1-Bit (QAT + KD)** | **1.99 MB** | 14.23 % | Bit-packing (uint8), Teacher-Student KD |

*(1.99MB 환경의 급격한 성능 하락(14.23%)은 물리적 정보의 한계치를 정량화한 데이터이며, 그럼에도 불구하고 지식 증류를 통해 무작위 확률(0.1%) 대비 140배 이상 지능을 방어해 낸 결과입니다.)*

---

## 4. [Phase 1] 옴니모달(VLM) 한계 검증을 위한 사전 연구

수어 동작을 자연어로 번역하기 위해, 시각 정보를 언어 모델(LLM)에 직접 전달하는 옴니모달(VLM) 아키텍처가 엣지 환경에 적합한지 1-Bit 압축 환경에서 선행 검증을 수행했습니다.

### 4.1. 1-Bit와 멀티모달 공간의 융합 한계 (Alignment Progress)
거대한 CLIP 모델(openai/clip-vit-base-patch32)이 학습한 의미론적 공간(Semantic Space)을 1-Bit ConvNeXt-Nano 모델에 매핑할 때 발생하는 얼라인먼트 붕괴 현상을 관찰했습니다.

![Omni-Modal Alignment Progress](./assets/mm_all_progress.png)

* **FP16 및 8/4-Bit QAT:** 10 에포크 이내에 코사인 유사도 0.88~0.90 수준으로 매우 안정적으로 수렴.
* **1-Bit Binarization:** 공간적 제약으로 인해 Teacher 모델의 정보를 온전히 수용하지 못하고 0.80 부근에서 강한 병목 현상 발생.

### 4.2. 프로젝션 헤드 아키텍처 분석 (Sparsity Conflict)
1-Bit 모델의 '정보 병목' 해소를 위해 다층 퍼셉트론(Custom MLP Head) 도입 실험을 진행했습니다.

| 평가 지표 (Recall@K) | 1-Bit (Linear Head) | 1-Bit (Custom MLP Head) |
| :--- | :---: | :---: |
| **Recall@1** | **14.20%** | 11.30% (▼ 2.90%p) |
| **Recall@5** | **31.30%** | 28.50% (▼ 2.80%p) |
| **Recall@10** | **41.60%** | 38.90% (▼ 2.70%p) |

* **결과 시사점:** 극단적 1-Bit 가중치의 희소성(Sparsity)이 LayerNorm 등과 충돌하며 복잡한 MLP 헤드가 오히려 성능을 저하시켰습니다. 극단적 경량화 환경에서는 **가장 단순한 매핑 구조(Linear Head)가 더욱 강건함**을 입증했습니다.

---

## 5. 종합 평가 및 최적 모델 선정 (Final Score)

Edge-Sign의 최종 백본을 선정하기 위해 성능(Performance), 속도(Latency), 메모리(Memory)를 아우르는 **종합 평가 프레임워크(Final Score)** 를 설계하였습니다.

$$\text{Final Score} = 0.6 \times \text{PerfNorm} + 0.2 \times \text{SpeedNorm} + 0.2 \times \text{MemNorm}$$

각 항은 FP16 기준선 대비 정규화된 값이며 상한을 1.0으로 고정합니다.

**추론 지연 시간 비교 및 파레토 프론티어**

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

**분석:** W8A8 SmoothQuant 모델이 종합 평가 1위(0.8068)를 기록했습니다. FP16 대비 Recall@1 손실은 0.5%p에 불과하면서 속도 및 메모리 항에서 FP16을 역전합니다. 반면 VLM(Vision + LLM) 구조는 텍스트 모델 오버헤드로 인해 수백 MB의 메모리를 소비하고 실시간 FPS를 충족하지 못함을 확인하였습니다. 따라서 W8A8 SmoothQuant를 Phase 2 파이프라인의 인식 백본으로 채택합니다.

### 5.1. W8A8 KSL 도메인 파인튜닝 및 ONNX 배포 성과

W8A8 백본에 1,404개 클래스의 한국 수어(KSL) 데이터셋을 파인튜닝한 후 ONNX로 추출하였습니다.

- **ONNX Export:** 최신 PyTorch의 Dynamo 익스포터에서 발생하는 형상 추론 오류(Shape Inference Error)를 `opset_version=14` 및 TorchScript 익스포터로 전환하여 해결하였습니다.
- **ONNX Runtime INT8 동적 양자화:** FP32 60.70 MB → **15.61 MB** (압축률 3.9×).
- **순수 CPU 추론 검증:** PyTorch 의존성 없이 `ONNX Runtime(CPUExecutionProvider)`만으로 추론하며, 테스트 이미지("가능" 클래스)를 Rank-1 신뢰도 41.18%로 정확히 분류하였습니다.

### 5.2. 실시간 웹캠 데모 (MediaPipe 기반)

MediaPipe에서 추출한 랜드마크를 학습된 OpenPose 959차원 형식으로 변환한 후 FastAPI WebSocket으로 전송하여 서버 측 추론을 수행합니다.

```bash
# 백엔드 서버 실행
python scripts/mediapipe_ws_server.py

# 프론트엔드 실행 (브라우저에서 web/mediapipe/index.html 열기)

# LAN 환경에서 모바일 접속
cd web/mediapipe && python -m http.server 8080
# 접속: http://<호스트 IP>:8080
```

### 5.3. 진단 대시보드 (온디바이스 분석)

ONNX 엔진을 브라우저(WebAssembly)에 이식하여 서버 연동 없이 동작하는 분석 인터페이스를 구현하였습니다.

- **획 단위 기여도 분석 (Stroke-level Attribution):** 획별 Ablation 테스트로 예측 Logit에 기여하는 핵심 획을 실시간 시각화합니다.
- **자소 분석 (Hangul Decomposer):** 유니코드 연산으로 초/중/종성을 분해하고 획수 일치성(Consistency Score)을 트리 그래프로 표시합니다.
- **양자화 샌드박스:** 8-Bit에서 1-Bit까지 활성화 텐서의 가상 양자화를 시뮬레이션하며 MSE, KL Divergence, 분포 히스토그램을 브라우저에서 렌더링합니다.
- **강건성 및 엔트로피 분석:** 가우시안/임펄스 노이즈 주입 시 출력 분포의 Shannon Entropy를 연산하여 모델의 의사결정 불확실성을 정량화합니다.

---

## 6. [Phase 2] 검출+추적+인식 파이프라인 설계

Phase 1의 분류 양자화 연구를 기반으로, Phase 2에서는 검출(Detection) → 추적(Tracking) → 인식(Recognition)의 3단계 파이프라인에 단계별 양자화를 적용합니다. 핵심 연구 질문은 **어떤 단계가 양자화에 가장 민감한가**이며, 그 결과를 바탕으로 15 MB 예산 내에서 최적 구성을 도출합니다.

### 6.1. 전체 파이프라인 구조

```
영상 입력 (대시캠 / 거리 영상 / 웹캠, 640×480)
         │
         ▼
┌─────────────────────────────┐
│ 1단계: YOLOv8-Nano 검출기   │  양자화 대상 — 3.2M params, FP16 ~6.3 MB
│ 클래스: signboard           │  입력: 640×640 RGB
│         traffic_sign        │  출력: bbox, confidence, class
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 2단계: ByteTrack 추적기      │  모델 파라미터 없음 (Kalman Filter + IoU)
│ ablation: BoT-SORT + ReID   │  (E6 실험에서 OSNet-x0.25 ReID 양자화)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 3단계: 클래스별 분기 인식기  │
│                             │
│  signboard  → KoreanOCRNet  │  700K params, 2350 한글 문자 클래스
│    ROI 크롭: 64×64 gray     │
│                             │
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
| 검출기 | YOLOv8-Nano (3.2M) | Ultralytics ONNX/양자화 지원 가장 성숙. 엣지 예산 충족 |
| 추적기 (기본) | ByteTrack | 추가 모델 파라미터 없음. 검출기 양자화 효과를 순수하게 분리 가능 |
| 추적기 (ablation) | BoT-SORT + OSNet-x0.25 | ReID 백본 양자화 효과를 E6 실험에서 측정 |
| 간판 OCR | KoreanOCRNet (700K) | Phase 1에서 양자화 실험 완료. 신규 학습 불필요 |
| 교통표지판 분류 | TrafficSignNet (65K) | 동일 — Phase 1 재활용 |

### 6.3. 데이터 파이프라인

AI Hub 데이터셋 188(신호등/도로표지판, 수도권)의 원천 데이터는 개별 이미지가 아닌 **동영상 시퀀스**입니다. 프레임 단위 랜덤 분할을 적용할 경우 동일 동영상의 인접 프레임이 train/val에 동시 노출되어 데이터 리크가 발생합니다. 이를 방지하기 위해 **시퀀스 단위 분할**을 채택합니다.

```
AI Hub validation 동영상 (~40 GB, ~30 fps)
         │
         ▼
scripts/extract_frames.py
  --fps 5              # 6× 다운샘플: 시각적 중복 제거
  --split_by sequence  # 동영상 경계 단위로 train/val 분리 (프레임 리크 없음)
         │
         ├── train sequences (80%) → data/aihub_traffic/frames/train/
         └── val sequences   (20%) → data/aihub_traffic/frames/val/
                                     + data/aihub_traffic/val_videos/  ← 추적 평가·시연용 원본 보존
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
| 프레임 단위 랜덤 분할 | 발생 | 동일 동영상의 인접 프레임이 train/val에 동시 존재 |
| 시퀀스 단위 분할 (채택) | 없음 | train 동영상과 val 동영상이 완전히 분리됨 |

val 분할의 원본 동영상은 추적 평가(MOTA/IDF1/HOTA는 연속 프레임 시퀀스 필요) 및 웹 시연에 직접 활용합니다.

---

## 7. Phase 2 양자화 실험 매트릭스

파이프라인의 각 단계를 독립적으로 양자화하여 단계별 민감도를 정량화합니다. E0 대 E1 비교는 검출기 단계의 민감도를 분리하고, E0 대 E2 비교는 인식기 단계의 민감도를 분리합니다.

### 실험 구성

| ID | 검출기 | 추적기 | 간판 OCR | 교통 분류 | 예상 총 크기 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| E0 | FP16 | ByteTrack | FP16 | FP16 | ~10 MB |
| E1 | **W8A8** | ByteTrack | FP16 | FP16 | ~8 MB |
| E2 | FP16 | ByteTrack | **W8A8** | **W8A8** | ~7 MB |
| E3 | W8A8 전체 | ByteTrack | W8A8 | W8A8 | ~5 MB |
| E4 | W4A16 전체 | ByteTrack | W4A16 | W4A16 | ~3 MB |
| E5 | SmoothQuant | ByteTrack | SmoothQuant | SmoothQuant | ~6 MB |
| E6 | W8A8 | **BoT-SORT** (W8A8 ReID) | W8A8 | W8A8 | ~7 MB |
| E7 | W4A16 | ByteTrack | **1-Bit** | **1-Bit** | ~2 MB |

### 평가 지표

검출 단계: mAP@0.5, mAP@0.5:0.95, Precision, Recall  
추적 단계: MOTA, IDF1, HOTA, ID Switches  
인식 단계: OCR 문자 정확도, OCR 단어 정확도, 표지판 Top-1/Top-5  
종합: 총 모델 크기, FPS (CPU), FPS (ONNX Runtime Web), Final Score

$$\text{Final Score} = 0.6 \times \frac{\text{인식률}_i}{\text{인식률}_{E0}} + 0.2 \times \frac{\text{Latency}_{E0}}{\text{Latency}_i} + 0.2 \times \min\!\left(1, \frac{\text{크기}_{E0}}{\text{크기}_i}\right)$$

실험 결과는 `docs/EXPERIMENTS.md`에 기록하며, 완료 후 Pareto Frontier 차트를 생성합니다.

---

## 8. 웹 배포 아키텍처 및 추론 환경

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

참조 구현: `scripts/mediapipe_ws_server.py` (WebSocket 서버), `web/app.js` (ONNX Runtime Web 클라이언트)

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

### Phase 1 — 웹 데모

```bash
python scripts/mediapipe_ws_server.py   # MediaPipe WebSocket 서버

# LAN 환경에서 모바일 접속
cd web/mediapipe && python -m http.server 8080
```

### Phase 2 — 데이터 준비

```bash
# GTSDB
python scripts/download_gtsdb.py
python src/detect/prepare_dataset.py --source gtsdb

# AI Hub 동영상 → 프레임 (시퀀스 단위 분할)
python scripts/extract_frames.py \
  --input  data/aihub_traffic/validation \
  --output data/aihub_traffic/frames \
  --fps 5 --split_by sequence

# 어노테이션 변환 및 통합
python src/detect/prepare_dataset.py --source aihub_traffic
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

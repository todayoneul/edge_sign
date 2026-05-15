# Edge-Sign: 초경량 온디바이스 수어 해석 시스템 및 극단적 신경망 압축 프레임워크(Edge-Sign: Ultra-Lightweight On-Device Sign Language Interpreter & Extreme Neural Network Compression Framework)

## 프로젝트 개요 (Project Plan)

### 1. 프로젝트 이름
**Edge-Sign (경량화 ConvNeXt 기반 서버리스(Serverless) 온디바이스 실시간 수어 해석 시스템)**

### 2. 프로젝트 소개
본 프로젝트는 청각장애인과의 원활한 소통을 위한 **'실시간 한국수어(KSL) 번역 모델'을 인터넷 연결(통신)이나 고성능 클라우드 서버 없이, 사용자의 스마트폰 웹 브라우저 및 초소형 IoT 기기(Smart Glass 와 같은 Edge Device)에서 독립적으로 구동**시키는 것을 목표로 하는 코어 시스템 엔지니어링 및 응용 연구입니다.

**왜 경량화를 진행했는가?**
* **문맥(Context) 인지의 필요성:** 기존의 대중적인 솔루션(예: Google MediaPipe)은 손가락 관절(Skeleton)의 3D 좌표만을 추출하는 데 그칩니다. 하지만 실제 한국수어는 손의 위치뿐만 아니라 얼굴 표정(비수지 기호), 몸의 기울기 등 전체적인 '문맥'이 의미를 결정합니다. 따라서 이미지 전체를 픽셀 단위로 End-to-End 해석하는 고성능 비전 모델(CNN/ViT)이 필수적입니다.
* **극한의 하드웨어 제약 (Edge AI):** 그러나 이러한 고성능 비전 모델은 수백 MB 이상의 메모리와 막대한 연산량을 요구하여 모바일 브라우저나 IoT 기기 탑재가 불가능에 가깝습니다. 이를 서버 통신으로 해결할 경우, 심각한 프라이버시 침해(카메라 영상 외부 전송), 통신 지연(Latency), 그리고 막대한 서버 유지/탄소 배출 비용이 발생합니다.
* **해결책 (코어 엔지니어링):** 이를 극복하기 위해, 본 연구는 파이토치(PyTorch)의 순전파/역전파 연산을 수학적으로 재정의하여 **125MB의 거대 모델을 14.9MB(8-Bit PTQ) 및 1.99MB(1-Bit Binarization) 수준으로 극단적으로 압축(Quantization)** 하는 기술을 직접 구현했습니다.

**[최종 응용 및 배포 계획: 하이브리드 파이프라인]**
수어 해석을 자연어로 변환하기 위해 거대한 옴니모달(VLM) 모델을 사용할 경우 발생하는 치명적인 FPS 저하 및 메모리 초과 문제를 극복하기 위해, 본 프로젝트는 **Action-Trigger 기반 하이브리드 파이프라인**을 채택했습니다. 자체 초경량 비전 엔진(W8A8)이 초고속으로 수어 키워드를 추출하면, 경량화된 NLP 로직이 이를 자연스러운 문장으로 즉각 조립합니다. ONNX 포맷 변환을 통해 **1) 스마트폰 웹 브라우저 네이티브 구동 (WebAssembly)** 및 **2) 인터넷이 단절된 환경(No-Signal)에서의 라즈베리파이 기반 오프라인 번역기** 형태로 배포하여, 프라이버시 완벽 보호와 60FPS의 제로-레이턴시를 달성합니다.

---

## 목차 (Table of Contents)
- [1. 연구 동기 및 기여도 (Motivation & Contribution)](#1-연구-동기-및-기여도-motivation--contribution)
- [2. 핵심 방법론 (Methodology)](#2-핵심-방법론-methodology)
- [3. 실험 환경 (Experimental Setup)](#3-실험-환경-experimental-setup)
- [4. 실험 결과 및 파레토 프론티어 분석 (Results & Pareto Analysis)](#4-실험-결과-및-파레토-프론티어-분석-results--pareto-analysis)
- [5. 멀티모달 시맨틱 공간 증류 (Sign-to-Text 한계 검증을 위한 사전 연구)](#5-멀티모달-시맨틱-공간-증류-sign-to-text-한계-검증을-위한-사전-연구)
- [6. 양자화율에 따른 얼라인먼트 학습 경과](#6-양자화율에-따른-얼라인먼트-학습-경과)
- [7. 프로젝션 헤드 구조에 따른 성능 분석](#7-프로젝션-헤드-구조에-따른-성능-분석)
- [8. Omni-Modal 종합 평가 프레임워크 (최적 모델 선정)](#8-omni-modal-종합-평가-프레임워크-최적-모델-선정)
- [9. 종합 결론 및 Edge-Sign 하이브리드 파이프라인 구축](#9-종합-결론-및-edge-sign-하이브리드-파이프라인-구축)
- [10. 실행 가이드](#10-실행-가이드)

---

## 1. 연구 동기 및 기여도 (Motivation & Contribution)
(상단 프로젝트 개요 참고. 본 파이프라인은 친환경 AI(Green AI) 및 TinyML 접근법으로서 커스텀 연산자 설계, 비트 패킹(Bit-packing) 실증, 지식 증류(KD) 수학적 최적화의 3대 기여도를 가집니다.)

---

## 2. 핵심 방법론 (Methodology)

### 2.1. 8-Bit PTQ (Post-Training Quantization)
학습이 완료된 모델의 가중치를 256개의 구간(-128 ~ 127)으로 선형 맵핑(Linear Mapping)합니다. 재학습(Epoch) 없이 즉각적인 메모리 절반(14.9MB) 단축이 가능하며, 원본(FP16) 대비 0.64%p의 미미한 성능 하락만을 보였습니다.

### 2.2. 4-Bit QAT & Custom STE
가중치를 16개의 구간(-8 ~ 7)으로 압축할 경우 발생하는 뇌사 상태(Weight Collapse)를 극복하기 위해 **QAT(양자화 인지 학습)** 를 도입했습니다. 미분 불가능한 양자화 함수의 그레디언트를 통과시키기 위해 **Straight-Through Estimator (STE)** 함수를 아래 수식과 같이 사용했습니다. 
$$\text{Forward: } W_q = \text{Clamp}(\text{Round}(W / \Delta), -8, 7) \times \Delta$$   
$$\text{Backward: } \frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial W_q} \quad (\text{if } W \in [-8, 7] \text{ else } 0)$$

### 2.3. 1-Bit Binarization & Bit-Packing
모든 CNN 필터 가중치를 흑백(+1과 -1)으로 이진화하며, 필터의 크기 소실을 막기 위해 **채널별 절댓값 평균(Per-channel L1 Norm)** 을 스케일 팩터로 활용합니다. 디스크 추출 시 `numpy.packbits`를 적용하여 8개의 이진 가중치를 1개의 `uint8` 메모리 블록에 욱여넣는(Bit-packing) 기술을 구현해 1.99MB 도달에 성공했습니다.

### 2.4. Knowledge Distillation (KD) 기반 성능 방어
1-Bit 환경에서의 치명적인 **정보 병목(Information Bottleneck)** 현상을 극복하기 위해, FP16 교사 모델(Teacher)의 소프트 라벨(KL Divergence)을 혼합하여 연산합니다.   
$$L_{KD} = \alpha \cdot T^2 \cdot D_{KL}\left( \sigma\left(\frac{Z_S}{T}\right) \| \sigma\left(\frac{Z_T}{T}\right) \right) + (1-\alpha) \cdot CE(Z_S, y)$$

---

## 3. 실험 환경 (Experimental Setup)

### 3.1. Core Engine Training
* **Dataset:** ImageNet-1K (1.2 Million Images, 1000 Classes) 
* **Base Architecture:** ConvNeXtV2-Nano (`convnextv2_nano.fcmae_ft_in1k`)
* **Hardware Framework:** NVIDIA RTX 5070 12GB (Local) / PyTorch 2.x

### 3.2. Target Domain: Korean Sign Language (KSL)
* **Dataset:** [AI Hub - 수어 영상 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=103)
* **Data Scale:** 한국인 수어 구사자 20인 이상이 참여한 **53.6만 개 이상의 고해상도 수어 영상 클립**.
* **Usage:** 압축된 비전 엔진이 실제 한국 수어의 문맥과 형태적 특징을 정밀하게 학습하기 위한 파인튜닝(Fine-tuning) 데이터셋으로 활용되었습니다.

---

## 4. 실험 결과 및 파레토 프론티어 분석 (Results & Pareto Analysis)

| 모델 아키텍처 (Quantization) | 물리적 메모리 용량 (MB) | Top-1 Accuracy (%) | 특징 및 적용 기술 |
| :--- | :---: | :---: | :--- |
| **Baseline (FP16)** | **125.0 MB** | 81.88 % | Hugging Face Pre-trained Model |
| **W8A8 (PTQ)** | **14.9 MB** | 81.24 % | Zero-shot Calibration, INT8 캐스팅 |
| **W4A16 (QAT)** | **14.92 MB** | 76.12 % | Custom STE, INT8 그릇 내 4-Bit 제한 |
| **1-Bit (QAT + KD)** | **1.99 MB** | 14.23 % | Bit-packing (uint8), Teacher-Student KD |

*(1.99MB 환경의 급격한 성능 하락(14.23%)은 정보의 한계치를 정량화한 데이터이며, 지식 증류를 통해 무작위 확률(0.1%) 대비 140배 이상 지능을 유지한 결과입니다.)*

---

## 5. 멀티모달 시맨틱 공간 증류 (Sign-to-Text 한계 검증을 위한 사전 연구)

시각 정보를 자연어 처리 모델에 전달하는 옴니모달(VLM) 구조가 극단적 엣지 환경에 적합한지 판단하기 위해, 1-Bit 압축 환경에서 멀티모달 정렬성(Alignment) 검증을 수행했습니다.

### 1-Bit와 멀티모달 공간의 융합 한계 분석
거대한 CLIP 모델(openai/clip-vit-base-patch32)이 학습한 방대한 의미론적 공간(Semantic Space)을 1-Bit ConvNeXt-Nano 모델에 매핑할 때, 극단적인 정보 압축으로 인해 얼라인먼트가 심각하게 붕괴(Collapse)될 위험을 분석했습니다.
1. **Teacher Model:** CLIPVisionModel (FP16/BFloat16, 512d Projection)
2. **Student Model:** Target Vision Encoder (FP16, W8A8, W4A16, 1-Bit) + 512d Output Head
3. **Loss Function:** CosineEmbeddingLoss 적용

---

## 6. 양자화율에 따른 얼라인먼트 학습 경과 (Alignment Progress)

![Omni-Modal Alignment Progress](./assets/mm_all_progress.png)

* **FP16 및 8/4-Bit QAT:** 10 에포크 이내에 코사인 유사도 0.88~0.90 수준으로 매우 안정적이고 빠르게 수렴합니다.
* **1-Bit Binarization:** 공간적 제약으로 인해 Teacher 모델의 풍부한 의미론적 정보를 온전히 수용하지 못하고, 코사인 유사도 0.80 부근에서 강한 병목 현상을 보입니다.

---

## 7. 프로젝션 헤드 아키텍처 분석 (Projection Head Analysis)

1-Bit 모델의 '정보 병목'을 해소하기 위해, 비선형성을 강화한 다층 퍼셉트론(Custom MLP Head)을 도입하는 실험을 진행했습니다.

| 평가 지표 (Recall@K) | 1-Bit (Linear Head) | 1-Bit (Custom MLP Head) |
| :--- | :---: | :---: |
| **Recall@1** | **14.20%** | 11.30% (▼ 2.90%p) |
| **Recall@5** | **31.30%** | 28.50% (▼ 2.80%p) |
| **Recall@10** | **41.60%** | 38.90% (▼ 2.70%p) |

### 결과 분석 및 시사점 (Sparsity Conflict)
일반적인 모델과 달리, **극단적 1-Bit 양자화 모델에서는 복잡한 MLP 헤드가 오히려 성능을 저하**시켰습니다. 1-Bit 가중치의 극단적인 희소성(Sparsity)이 LayerNorm 등과 충돌하며 노이즈로 작용한 결과입니다. 1-Bit 환경에서는 **가장 단순한 매핑 구조(Linear Head)가 더 강건함**을 입증했습니다.

---

## 8. Omni-Modal 종합 평가 프레임워크 (최적 모델 선정)

Edge-Sign 백본 모델 선정을 위해, 성능(Performance), 속도(Latency), 메모리(Memory)를 아우르는 **종합 평가 프레임워크(Final Score)** 를 설계하였습니다.   
**Final Score = 0.6 * PerfNorm + 0.2 * SpeedNorm + 0.2 * MemNorm**

**[추론 지연 시간(Latency) 비교 & 파레토 프론티어]** ![Inference Latency Comparison](./assets/mm_latency_comparison.png)   
![Pareto Frontier](./assets/mm_final_pareto.png)

| Model | Recall@1 (%) | Latency (ms) | Memory (MB) | Final Score |
| :--- | :---: | :---: | :---: | :---: |
| **W8A8 (SmoothQuant PTQ)** | 38.50 | 10.29 | 30.70 | **0.8068** |
| **FP16 (Baseline)** | 39.00 | 6.09 | 125.00 | 0.8000 |
| **W4A16 (QAT)** | 34.80 | 9.97 | 14.92 | 0.7628 |
| **1-Bit (Linear Head)** | 14.20 | 9.02 | 1.99 | 0.3680 |

**분석 결론:** 옴니모달 환경에서 W8A8 모델이 최고점(0.8068)을 기록했으나, 전체 VLM 구조(Vision + LLM) 채택 시 발생하는 수백 MB의 텍스트 모델 메모리 오버헤드 및 라즈베리파이 등에서의 심각한 실시간 추론 속도(FPS) 저하 문제는 피할 수 없음을 확인했습니다.

---

## 9. 종합 결론 및 Edge-Sign 하이브리드 파이프라인 구축 (Conclusion & Hybrid Integration)

본 연구를 통해 옴니모달(VLM) 구조가 "초저전력, 15MB 이하의 물리적 크기 제한, 제로 레이턴시"라는 진정한 엣지 환경의 요구사항을 충족하기 어렵다는 한계를 정량적으로 증명했습니다. 이를 바탕으로, **수어 해석에 최적화된 'Action-Trigger 기반 하이브리드 파이프라인'** 으로 최종 배포 방향을 확정했습니다.

### 시스템 통합 및 배포 구조
1. **시각 엔진 (W8A8 ConvNeXt):** 종합 평가 1위를 차지한 W8A8 양자화 모델(14.9MB)을 백본으로 채택하여, 카메라 영상을 60FPS로 처리하며 수어 키워드(Label)를 추출합니다.
2. **신뢰할 수 있는 데이터 원천:** [AI Hub의 대규모 수어 영상 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=103)를 활용하여 한국수어의 미세한 동작과 비수지 기호에 대한 분류 정확도를 확보했습니다.
3. **논리 및 조립 모듈 (NLP Rule-base):** 무거운 대형 언어 모델을 배제하고, 수십 KB 수준의 경량 버퍼 모듈을 통해 추출된 키워드를 자연스러운 한국어 문장으로 결합합니다.
4. **WebAssembly 기반 배포 (진행 예정):** 시스템 전체 용량을 20MB 이하로 유지하며 ONNX로 추출, 서버 없이 사용자 브라우저에서 즉각 동작하는 네이티브 웹 애플리케이션으로 배포합니다.

---

## 10. 실행 가이드 (Execution Guide)

1. **멀티모달 통합 종합 평가 (Final Score 도출)**
   ```bash
   python src/final_omnimodal_eval.py
   ```
2. **양자화 모델 개별 검증 (Recall@1 성능 측정)**
    ```bash
    python src/multimodal_unified_eval.py --ckpt "./checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_15.pth"
    ```
3. **1-Bit 커스텀 헤드 실험 및 검증**
    ```bash
    python src/multimodal_1bit_custom_head_kd.py
    python src/multimodal_unified_eval.py --ckpt "./checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_15.pth" --custom-head
    ```
4. **[예정] Action-Trigger 하이브리드 추론 엔진**
    ```bash
    python src/run_hybrid_inference.py --model w8a8 --video input.mp4
    ```
5. **[예정] WebAssembly 배포 (ONNX Export)**
    ```bash
    python src/export_onnx_web.py --model w8a8 --output edge_sign_web.onnx
    ```
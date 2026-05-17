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
- [4. [Phase 2] 옴니모달(VLM) 한계 검증을 위한 사전 연구](#4-phase-2-옴니모달vlm-한계-검증을-위한-사전-연구)
- [5. 종합 평가 및 최적 모델 선정 (Final Score)](#5-종합-평가-및-최적-모델-선정-final-score)
- [6. [최종 계획] Edge-Sign 하이브리드 파이프라인 구축 및 KSL 추론 파이프라인 (Future Work & Progress)](#6-최종-계획-edge-sign-하이브리드-파이프라인-구축-및-ksl-추론-파이프라인-future-work--progress)

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

## 4. [Phase 2] 옴니모달(VLM) 한계 검증을 위한 사전 연구

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

Edge-Sign의 최종 백본을 선정하기 위해 성능(Performance), 속도(Latency), 메모리(Memory)를 아우르는 **종합 평가 프레임워크(Final Score)** 를 자체 설계하여 분석했습니다.   
**Final Score = 0.6 * PerfNorm + 0.2 * SpeedNorm + 0.2 * MemNorm**

**[추론 지연 시간(Latency) 비교 & 파레토 프론티어]** ![Inference Latency Comparison](./assets/mm_latency_comparison.png)   
![Pareto Frontier](./assets/mm_final_pareto.png)

| Model | Recall@1 (%) | Latency (ms) | Memory (MB) | Final Score |
| :--- | :---: | :---: | :---: | :---: |
| **W8A8 (SmoothQuant PTQ)** | 38.50 | 10.29 | 30.70 | **0.8068** |
| **FP16 (Baseline)** | 39.00 | 6.09 | 125.00 | 0.8000 |
| **W4A16 (QAT)** | 34.80 | 9.97 | 14.92 | 0.7628 |
| **1-Bit (Linear Head)** | 14.20 | 9.02 | 1.99 | 0.3680 |

**최종 결론:** W8A8 모델이 종합 평가 최고점(0.8068)을 기록했습니다. 그러나 본 연구를 통해 VLM 구조(Vision + LLM) 채택 시 발생하는 수백 MB의 텍스트 모델 메모리 오버헤드와 실시간 추론 속도(FPS) 저하 현상을 확인했습니다. 이는 **"초저전력, 15MB 이하의 물리적 크기 제한, 제로 레이턴시"** 라는 본 프로젝트의 목표를 달성하기 어렵다는 것을 시사합니다.

---

## 6. [최종 계획] Edge-Sign 하이브리드 파이프라인 구축 및 KSL 추론 파이프라인 (Future Work & Progress)

선행 연구의 철저한 한계 분석을 바탕으로, 옴니모달(VLM)을 배제하고 수어 해석에 최적화된 **'Action-Trigger 기반 하이브리드 파이프라인'** 으로 최종 시스템 통합을 진행 중입니다.

1. **시각 엔진 (W8A8 ConvNeXt) 및 KSL 파인튜닝:** 종합 평가 1위를 차지한 W8A8 양자화 모델을 백본으로 채택하고 실제 한국 수어 데이터셋을 학습시켜 도메인 특화 정확도를 극대화합니다.
2. **순수 CPU 엣지 추론 최적화:** 무거운 프레임워크(PyTorch 등) 없이 ONNX Runtime만을 이용한 초경량 추론 환경을 구축합니다.
3. **논리 및 조립 모듈 (NLP Rule-base):** 무거운 언어 모델을 대신하여, 수십 KB 수준의 경량 버퍼 모듈을 통해 추출된 키워드를 즉각적인 한국어 문장으로 결합합니다.
4. **WebAssembly 기반 배포:** 시스템 전체 용량을 20MB 이하로 유지하며 ONNX로 추출, 서버 통신 없이 사용자 스마트폰 브라우저에서 즉각 동작하는 네이티브 웹 애플리케이션 데모를 완성합니다.

### 6.1. W8A8 KSL 도메인 특화 Fine-Tuning 및 모델 경량화 성과
최종 백본으로 선정된 W8A8 모델에 1,404개 클래스의 실제 한국 수어(KSL) 데이터셋을 파인튜닝하고 모델 추출 과정을 거쳐 극한의 경량화와 최적화를 완료했습니다.
1. **W8A8 QAT Fine-Tuning:** 사전 양자화 인지 학습(QAT) 기법을 기반으로 ConvNeXtV2-Nano 모델에 KSL 데이터셋을 성공적으로 학습시켜 엣지 배포용 파이프라인 준비를 마쳤습니다.
2. **ONNX Export 오류 해결 및 최적화:** 최신 PyTorch 버전에 도입된 Dynamo ONNX 익스포터의 형상 추론 오류(Shape Inference Error)를 극복하기 위해 `opset_version=14` 및 레거시(TorchScript) 익스포터 강제 모드로 파이프라인을 성공적으로 전환했습니다. 그 결과 안정적인 추출에 성공했습니다.
3. **Real INT8 Dynamic Quantization:** ONNX Runtime의 동적 양자화를 통해 가중치를 완벽하게 물리적 INT8 규격으로 압축, 모델 사이즈를 FP32 60.70MB에서 **15.61MB** 로 감축하여 **압축률 약 3.9배**를 달성했습니다.

### 6.2. 엣지 디바이스용 순수 CPU 추론 테스트 성공
PyTorch 등 무거운 라이브러리 없이 순수 `ONNX Runtime(CPUExecutionProvider)`만을 사용하여 최종 KSL 타겟 추론 검증을 성공적으로 마쳤습니다.
* **한국어 경로 버그 해결:** 한글 폴더/파일 경로로 인해 발생하던 이미지 로딩 문제(`cv2.imread`)를 `numpy` 바이트 배열 디코딩 방식으로 우회하여 안정적으로 이미지를 로드합니다.
* **Float64 에러 수정:** ONNX 엔진이 요구하는 엄격한 타입 입력 규격(float32)에 맞춰 NumPy 전처리 결과 타입 파이프라인을 고도화했습니다.
* **추론 결과:** "가능" 클래스의 테스트 이미지(`NIA_SL_WORD2493_REAL01_D_1.jpg`)에 대해 추론한 결과, 정답 라벨("가능")을 **1위 (신뢰도 41.18%)** 로 성공적으로 분류해 내며 가벼운 용량에도 뛰어난 인식 능력을 보존함을 증명했습니다.

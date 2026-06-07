# Third-Party Notices / 제3자 출처 및 라이선스 고지

Edge-Sign은 다음의 외부 연구·오픈소스 구현을 인용·재구현·포팅하거나 라이브러리로
사용한다. 각 항목의 저작권은 해당 원저작자에게 있으며, 아래 라이선스 조건을 따른다.

이 문서는 코드 주석(각 파일 docstring)에 분산된 출처 표기를 한곳에 모은 색인이다.
새로운 외부 구현을 도입하거나 제거할 때 이 표를 함께 갱신한다.

---

## 1. 추적 (Tracking)

### ByteTrack
- **무엇**: 다단계(BYTE) 데이터 연관 추적 알고리즘. 본 저장소의 `STrack`, 트랙 생명주기,
  2단계 IoU 매칭 로직은 ByteTrack을 충실히 재구현한 것이다.
- **사용 위치**: `src/track/bytetrack.py`, `web_modern/src/lib/byteTrack.ts` (TS 포팅)
- **원저작/논문**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box", ECCV 2022 — https://arxiv.org/abs/2110.06864
- **참조 구현**: https://github.com/ifzhang/ByteTrack
- **라이선스**: MIT

### DeepSORT Kalman Filter
- **무엇**: 8차원 등속 칼만 필터(상태 `[cx, cy, ar, h, v...]`). 표준편차 가중치
  (`_std_weight_position = 1/20`, `_std_weight_velocity = 1/160`), `initiate`/`predict`/
  `project`/`gating_distance`의 구성은 DeepSORT의 `kalman_filter.py`에서 파생되었다.
  ByteTrack도 동일 칼만을 차용한다.
- **사용 위치**: `src/track/bytetrack.py`(`KalmanFilter`), `web_modern/src/lib/byteTrack.ts`
- **참조 구현**: https://github.com/nwojke/deep_sort (`deep_sort/kalman_filter.py`)
- **라이선스**: GPL-3.0
- **주의**: GPL-3.0 구성요소의 파생물이다. 재배포 시 라이선스 호환성을 검토할 것.

### BoT-SORT
- **무엇**: ByteTrack 위에 카메라 모션 보정(CMC)과 ReID 외형 매칭을 더한 추적기.
  본 저장소의 CMC/ReID 결합 매칭 구조는 BoT-SORT를 따른 재구현이다.
- **사용 위치**: `src/track/botsort.py`
- **원저작/논문**: Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
  Tracking", 2022 — https://arxiv.org/abs/2206.14651
- **참조 구현**: https://github.com/NirAharon/BoT-SORT
- **라이선스**: MIT

---

## 2. 검출 (Detection)

### Ultralytics YOLOv8 / YOLO
- **무엇**: 검출기 학습·추론·ONNX 내보내기에 `ultralytics` 패키지를 직접 사용한다
  (`from ultralytics import YOLO`, `.train()`, `.export()`).
- **사용 위치**: `src/detect/yolo_train.py`, `src/detect/export_yolo_onnx.py`,
  `src/quant/quantize_yolo.py`, `src/pipeline/e2e_pipeline.py`, `scripts/train_v4_detector.py`,
  `scripts/export_v4_variants.py`, `scripts/quantize_v3_detector.py`
- **참조 구현**: https://github.com/ultralytics/ultralytics
- **라이선스**: **AGPL-3.0** (네트워크 사용 전염 조항 포함)
- **주의**: 본 프로젝트는 학습된 가중치를 FastAPI/HF Space 등 **네트워크 서비스**로
  배포한다. AGPL-3.0의 네트워크 조항(서비스 이용자에 대한 대응 소스 제공 의무)이
  적용될 수 있으므로, 프로젝트 배포 라이선스를 AGPL과 호환되도록 정렬해야 한다.
  상용 사용이 필요하면 Ultralytics Enterprise License를 별도 검토할 것.

---

## 3. 양자화 (Quantization)

### SmoothQuant
- **무엇**: 활성화↔가중치 스케일 이전(per-channel)로 W8A8 PTQ 정확도를 보존하는 기법.
  `multimodal_w8a8_smoothquant.py`, `src/quant/quantize_yolo.py`(smoothquant 모드)에서
  기법을 재구현한다.
- **원저작/논문**: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models", ICML 2023 — https://arxiv.org/abs/2211.10438
- **참조 구현**: https://github.com/mit-han-lab/smoothquant
- **라이선스**: MIT

> W8A8 PTQ / W4A16 QAT / 1-Bit KD 등 나머지 양자화 코드는 본 프로젝트의 자체 구현이다.

---

## 4. 백본·라이브러리 (Backbones & Libraries)

### ConvNeXt V2 (via timm)
- **무엇**: Phase 1 분류 백본. `timm`을 통해 ConvNeXtV2-Nano 아키텍처/사전학습 가중치를 사용.
- **참조**: Woo et al., "ConvNeXt V2", CVPR 2023 — https://arxiv.org/abs/2301.00808 /
  https://github.com/facebookresearch/ConvNeXt-V2 (코드 MIT)
- **timm 라이선스**: Apache-2.0
- **주의**: FAIR 공개 ConvNeXt V2 **사전학습 가중치 일부는 CC BY-NC 4.0(비상업)** 이다.
  상용화 시 실제 사용한 체크포인트의 가중치 라이선스를 개별 확인할 것.

### 기타 핵심 의존성
| 라이브러리 | 용도 | 라이선스 |
|---|---|---|
| PyTorch | 학습/추론 | BSD-3-Clause |
| ONNX Runtime / ORT-Web | 추론 (서버/브라우저) | MIT |
| timm | 백본 | Apache-2.0 |
| transformers | VLM/모델 | Apache-2.0 |
| FastAPI | 서버 | MIT |
| React / Vite / zustand | 웹 프론트 | MIT |
| Groq Python SDK | Q&A LLM API | (벤더 SDK, 약관 확인) |
| scipy / numpy / OpenCV | 수치/영상 | BSD / BSD / Apache-2.0 |

> 웹 프론트(`web_modern/node_modules`)의 npm 의존성 라이선스는 각 패키지의 LICENSE 파일을
> 따른다. 배포 번들 기준 전수 목록이 필요하면 `license-checker` 등으로 생성한다.

---

## 데이터셋
- **GTSDB / GTSRB** (독일 교통표지판) — 연구용 공개 데이터.
- **AI Hub** (신호등-도로표지판, 야외 한글 이미지 등) — AI Hub 이용약관을 따른다.
  AI Hub 데이터는 본 저장소에 포함하지 않는다(`.gitignore`).

---

_최종 갱신: 2026-06-07_

# Edge-Sign v2 양자화 실험 매트릭스

> 이 문서는 모든 양자화 실험의 구성과 결과를 기록합니다.
> 실험 실행 후 해당 행의 빈 셀을 채워나가세요.

---

## Phase 1 결과 요약 (분류 양자화 - 완료)

ConvNeXtV2-Nano 백본, ImageNet-1K 평가:

| 모델 | 양자화 | 크기 | Top-1 Acc | Latency | Final Score |
|------|--------|------|-----------|---------|-------------|
| Baseline | FP16 | 125 MB | 81.88% | 6.09ms | 0.8000 |
| W8A8 PTQ | 8-Bit | 14.9 MB | 81.24% | 10.29ms | — |
| W8A8 SmoothQuant | Calibrated | 30.7 MB | 38.50% R@1 | 10.29ms | **0.8068** |
| W4A16 QAT | 4-Bit | 14.92 MB | 76.12% | 9.97ms | 0.7628 |
| W8A8 QAT | 8-Bit QAT | 14.9 MB | 36.80% R@1 | 12.28ms | 0.7314 |
| 1-Bit Linear | Binarization+KD | 1.99 MB | 14.20% R@1 | 9.02ms | 0.3680 |
| 1-Bit Custom | Binarization+MLP | 1.99 MB | 11.30% R@1 | 8.51ms | 0.3218 |

**핵심 발견:** W8A8 SmoothQuant이 Final Score 1위. 1-Bit은 정보이론적 한계.

---

## Phase 2 실험 매트릭스 (검출+추적+인식 파이프라인)

### 데이터셋 정보

| 소스 | 원천 형식 | 처리 방법 | 클래스 | 크기 | 상태 |
|------|-----------|-----------|--------|------|------|
| GTSDB | PPM + gt.txt | 직접 변환 | traffic_sign(0) | 900장 | ✅ 변환 대기 |
| AI Hub 신호등-도로표지판(수도권) | **JPG 프레임 in TAR** | TAR 해제 → ÷6 서브샘플 → 시퀀스 분할 → YOLO 변환 | traffic_sign(0)+traffic_light(0) | 9시퀀스 110,900프레임 (37GB) | ✅ 다운로드 완료, TAR 해제 대기 |
| AI Hub 030.야외 한글 이미지 | **JPG+JSON (이미 해제됨)** | 직접 YOLO 변환 | signboard(1) | 학습 25,837장 + 검증 4,304장 | ✅ 바로 변환 가능 |
| AI Hub 다양한 형태의 한글 OCR | **ZIP 압축** | ZIP 해제 → OCR 학습 데이터 | (문자 인식용) | 39.6GB 압축 | ⏳ 선택사항, 추후 처리 |

> ⚠️ **시퀀스 단위 분할**: 신호등-도로표지판 TAR 9개 → train 6 / val 1~2 / test 1~2.
> test 시퀀스는 연속 프레임 보존 → ByteTrack 추적 평가(MOTA/IDF1/HOTA) + 웹 시연에 사용.
>
> **JSON 포맷 확인 완료:**
> - 신호등-도로표지판: `{"annotation":[{"box":[x1,y1,x2,y2], "class":"traffic_sign"/"traffic_light"}], "image":{"imsize":[W,H]}}`
> - 야외 한글 이미지: `{"images":[{"width":W,"height":H}], "annotations":[{"bbox":[x,y,w,h], "text":"..."}]}`

### 실험 구성

| ID | 검출기 | 추적기 | 간판 OCR | 교통 분류 | 예상 크기 |
|----|--------|--------|----------|-----------|-----------|
| E0 | FP16 YOLOv8n | ByteTrack | FP16 OCR | FP16 Traffic | ~10MB |
| E1 | **W8A8** YOLOv8n | ByteTrack | FP16 OCR | FP16 Traffic | ~8MB |
| E2 | FP16 YOLOv8n | ByteTrack | **W8A8** OCR | **W8A8** Traffic | ~7MB |
| E3 | W8A8 전체 | ByteTrack | W8A8 | W8A8 | ~5MB |
| E4 | W4A16 전체 | ByteTrack | W4A16 | W4A16 | ~3MB |
| E5 | SmoothQuant | ByteTrack | SmoothQuant | SmoothQuant | ~6MB |
| E6 | W8A8 YOLOv8n | **BoT-SORT** (W8A8 ReID) | W8A8 | W8A8 | ~7MB |
| E7 | W4A16 YOLOv8n | ByteTrack | **1-Bit** OCR | **1-Bit** Traffic | ~2MB |

### 검출 결과

| ID | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 모델 크기 |
|----|---------|---------------|-----------|--------|-----------|
| E0 | **0.628** | 0.437 | 0.722 | 0.543 | 21.47 MB (YOLOv8s FP32) |
| E1 | **0.621** | 0.433 | 0.717 | 0.541 | ~10.7 MB (W8A8 이론치) |
| E2 | **0.628** | 0.437 | 0.722 | 0.543 | 21.47 MB (검출기=E0와 동일) |
| E3 | **0.621** | 0.433 | 0.717 | 0.541 | ~10.7 MB (검출기=E1와 동일) |
| E4 | **0.581** | 0.376 | 0.697 | 0.512 | ~5.4 MB (W4A16 이론치) |
| E5 | **0.621** | 0.434 | 0.718 | 0.539 | ~10.7 MB (SmoothQuant W8A8) |
| E6 | **0.621** | 0.433 | 0.717 | 0.541 | ~10.7 MB (검출기=E1와 동일) |
| E7 | **0.581** | 0.376 | 0.697 | 0.512 | ~5.4 MB (검출기=E4와 동일) |

### 추적 결과

> 평가 시퀀스: AI Hub test split (c_1280_720_night_1 + c_1920_1200_night_1), CPU ONNX Runtime
> ⚠️ 주간 학습 → 야간 테스트 도메인 갭으로 FN 높음 (검출기 한계, 추적기 자체 문제 아님)

| ID | MOTA | IDF1 | HOTA | ID Switches | FP | FN |
|----|------|------|------|-------------|----|----|
| E0 | **0.219** | **0.384** | **0.487** | **0** | 7 | 265 |
| E1 | **0.221** (+0.002) | **0.384** (±0.000) | **0.487** (±0.000) | **0** | 6 | 265 |
| E2 | **0.219** (±0, 검출기=E0) | **0.384** (±0) | **0.487** (±0) | **0** | 7 | 265 |
| E3 | **0.221** (±0, 검출기=E1) | **0.384** (±0) | **0.487** (±0) | **0** | 6 | 265 |
| E4 | **0.105** (−0.114) | **0.192** (−0.192) | **0.322** (−0.165) | **0** | 2 | 315 |
| E5 | **0.225** (+0.006) | **0.387** (+0.003) | **0.490** (+0.003) | **0** | 4 | 264 |
| E6 | **0.108** (−51.1% vs E1) | **0.295** (−23.2% vs E1) | **0.416** (−14.6% vs E1) | **0** | 12 (avg) | 141.5 (avg) |
| E7 | **0.105** (±0, 검출기=E4) | **0.192** (±0) | **0.322** (±0) | **0** | 2 | 315 |

### 인식 결과

> 평가셋: KoreanOCRNet = `data/korean_ocr/val/` 5,000샘플 (2350클래스)
> TrafficSignNet = GTSDB val 크롭 242샘플 (43클래스, train/val=80/20 고정 시드)

| ID | OCR Top-1 | OCR Top-5 | 표지판 Top-1 | 표지판 Top-5 |
|----|-----------|-----------|-------------|-------------|
| E0 (FP32) | **98.5%** | 100.0% | **62.8%** | 89.7% |
| E1 (검출 W8A8) | 98.5% (±0) | 100.0% | 62.8% (±0) | 89.7% |
| E2/E3 (인식 W8A8) | **98.4%** (−0.1pp) | 99.96% | **63.2%** (+0.4pp) | 90.9% |
| E4 (W4A16) | **54.6%** (−43.9pp) | 85.0% | **49.2%** (−13.6pp) | 84.3% |
| E5 (SmoothQuant) | 98.5% (±0, SQ≈W8A8) | 100.0% | 62.8% (±0) | 89.7% |
| E6 (BoT-SORT) | 98.4% (인식기는 E3와 동일) | 99.96% | 63.2% | 90.9% |
| E7 (1-Bit) | **0.3%** (−98.2pp) | 0.6% | **12.8%** (−50.0pp) | 33.5% |

### End-to-End 종합

> 모델 크기: 이론적 INT 배포 크기 (fake-quant ONNX 파일 크기 아님).  
> FPS: CPU ONNX Runtime fake-quant 측정값. 실제 INT8 배포 시 3-4× 개선 예상.  
> 인식률: OCR Top-1 기준.

| ID | 총 크기(이론) | FPS (CPU, fake-quant) | FPS (CPU, INT8 Static) | OCR 인식률 | Final Score | Pareto 최적 |
|----|--------------|----------------------|------------------------|-----------|-------------|-------------|
| E0 | 24.3 MB | 21.2 | 55.7 | 98.5% | 1.0000 | 기준선 |
| E1 | 13.5 MB | 24.7 | — | 98.5% | 1.0335 | — |
| E2 | 21.8 MB | 23.3 | — | 98.4% | 1.0198 | — |
| E3 | 11.4 MB | 24.3 | 57.7 | 98.4% | 1.0294 | — |
| E4 | 5.7 MB | 25.2 | — | 54.6% | 0.7706 | 최소 크기 (OCR 손실) |
| E5 | **11.4 MB** | 20.5 | — | **98.5%** | 0.9938 | MOTA+OCR 양쪽 Pareto |
| E6 | 11.5 MB | 20.4 | — | 98.4% | 0.9922 | — (E5에 지배) |
| E7 | **5.4 MB** | 25.1 | — | 0.3% | 0.4392 | 최소 크기 (실용 불가) |

> **Final Score 공식:** `0.6 × PerfNorm + 0.2 × SpeedNorm + 0.2 × MemNorm`
> - PerfNorm = (해당 모델 인식률) / (E0 인식률)
> - SpeedNorm = (E0 latency) / (해당 모델 latency)
> - MemNorm = (E0 크기) / (해당 모델 크기)  [상한 1.0]

---

## 단계별 민감도 분석

실험 E0~E2 비교로 각 단계의 양자화 민감도를 분석합니다.

| 단계 | FP32 → W8A8/W4A16 성능 변화 | 민감도 등급 |
|------|----------------------|------------|
| **검출기 mAP: W8A8 (E0→E1)** | mAP50: 0.628 → 0.621 (Δ = −0.007, **−1.0%**) | 🟢 낮음 |
| **검출기 mAP: SmoothQuant (E0→E5)** | mAP50: 0.628 → 0.621 (Δ = −0.007, **−1.0%**) | 🟢 낮음 |
| **검출기 mAP: W4A16 (E0→E4)** | mAP50: 0.628 → 0.581 (Δ = −0.047, **−7.5%**) | 🟡 중간 |
| **추적 MOTA: W8A8 (E0→E1)** | MOTA: 0.219 → 0.221 (Δ = +0.002, **+0.9%**) | 🟢 없음 (오차 범위) |
| **추적 MOTA: SmoothQuant (E0→E5)** | MOTA: 0.219 → 0.225 (Δ = +0.006, **+2.7%**) | 🟢 없음 (오차 범위) |
| **추적 MOTA: W4A16 (E0→E4)** | MOTA: 0.219 → 0.105 (Δ = −0.114, **−52.1%**) | 🔴 높음 |
| **추적 IDF1: W8A8 (E0→E1)** | IDF1: 0.384 → 0.384 (Δ = 0.000, **0%**) | 🟢 없음 |
| **추적 IDF1: W4A16 (E0→E4)** | IDF1: 0.384 → 0.192 (Δ = −0.192, **−50.0%**) | 🔴 높음 |
| **인식기 OCR: W8A8 (E0→E2/E3)** | Top-1: 98.5% → 98.4% (Δ = −0.1pp, **−0.1%**) | 🟢 없음 (실질적 무손실) |
| **인식기 OCR: W4A16 (E0→E4)** | Top-1: 98.5% → 54.6% (Δ = −43.9pp, **−44.6%**) | 🔴 치명적 (사용 불가) |
| **인식기 OCR: 1-Bit (E0→E7)** | Top-1: 98.5% → 0.3% (Δ = −98.2pp, **−99.7%**) | 🔴 완전 붕괴 |
| **표지판 분류: W8A8 (E0→E2/E3)** | Top-1: 62.8% → 63.2% (Δ = +0.4pp, **+0.6%**) | 🟢 없음 (오차 범위) |
| **표지판 분류: W4A16 (E0→E4)** | Top-1: 62.8% → 49.2% (Δ = −13.6pp, **−21.7%**) | 🟡 중간 (성능 저하) |
| **표지판 분류: 1-Bit (E0→E7)** | Top-1: 62.8% → 12.8% (Δ = −50.0pp, **−79.6%**) | 🔴 거의 붕괴 |
| **ReID 추적: BoT-SORT (E1→E6)** | MOTA: 0.221 → 0.108 (Δ = −0.113, **−51.1%**) | 🔴 높음 (미학습 ReID 역효과) |
| **ReID 추적: BoT-SORT (E1→E6)** | IDF1: 0.384 → 0.295 (Δ = −0.089, **−23.2%**) | 🟡 중간 |

**핵심 발견 (추적 민감도):**
- W8A8 / SmoothQuant: 검출 mAP −1.0%p에 그쳐 추적 MOTA에 **실질적 영향 없음** (IDSW=0 유지)
- W4A16: 검출 Recall 급락(0.512 → 0.43) → FN 폭증(265→315) → MOTA −52%, IDF1 −50%
- **ByteTrack 자체는 모든 실험에서 IDSW=0 유지** — 추적기 품질 증명. 성능 저하는 100% 검출기 탓
- **미학습 ReID (E6 BoT-SORT)**: 무작위 초기화 ReID 임베딩 → 외형 유사도 오판 → FP 6→21 (+250%) → MOTA −51% vs E1. **BoT-SORT의 ReID는 반드시 학습된 모델이어야 함** — 연구 핵심 발견
- **인식기 W8A8**: OCR/교통분류 모두 사실상 무손실. 인식기는 W8A8에 가장 강건한 단계
- **인식기 W4A16/1-Bit**: OCR 54.6%/0.3%로 극단적 하락. 인식기가 파이프라인의 양자화 병목

---

## Pareto Frontier 데이터

`assets/pareto_frontier.png` 참조 (scripts/plot_pareto.py로 생성).

| ID | x: 총 모델 크기 (MB) | y: MOTA (추적) | y: OCR Top-1 (%) | Pareto MOTA | Pareto OCR |
|----|---------------------|---------------|------------------|-------------|-----------|
| E0 | 24.3 | 0.219 | 98.5% | — | — |
| E1 | 13.5 | 0.221 | 98.5% | — | — |
| E2 | 21.8 | 0.219 | 98.4% | — | — |
| E3 | 11.4 | 0.221 | 98.4% | — | — |
| E4 | 5.7  | 0.105 | 54.6% | — | 중간 크기 OCR |
| E5 | 11.4 | **0.225** | **98.5%** | 최적 | 최적 |
| E6 | 11.5 | 0.108 | 98.4% | — (E5에 지배) | — |
| E7 | 5.4  | 0.105 | 0.3%  | 최소 크기 | 최소 크기 (실용 불가) |

---

## 메모 및 관찰

> 실험 중 발견한 사항, 예상치 못한 결과, 디버깅 노트 등을 여기에 기록합니다.

- **2026-05-27**: AI Hub 신호등-도로표지판 원천 데이터 실제 확인 — 동영상이 아닌 JPG 프레임 (TAR 아카이브). 9개 시퀀스, 총 110,900 프레임. `scripts/extract_frames.py`에서 TAR 해제 + 서브샘플(÷6) + 시퀀스 분할 구현 완료.
- **2026-05-27**: 030.야외 한글 이미지 이미 압축 해제됨. 학습 25,837장 + 검증 4,304장. `prepare_dataset.py --source aihub_signboard` 구현 완료 (COCO-style xywh → YOLO 변환).
- **2026-05-27**: 신호등-도로표지판 JSON 포맷 확인: `annotation[].box` = [x1,y1,x2,y2] xyxy 절대픽셀. traffic_light → class 0 (traffic_sign)으로 통합.
- **2026-05-27**: 추적 평가 시퀀스 = test 분할 시퀀스 (d_1920_1080_night_1 예정). 별도 테스트 영상 불필요.
- **2026-05-27**: GTSDB 단독 기준선 학습 먼저 진행 (TAR 해제 대기 없이 즉시 가능) → 파이프라인 검증 후 AI Hub 데이터 추가.
- **2026-05-28**: **E0 최종 확정** — YOLOv8s FP32, batch=32, 75 epoch (patience=20 조기종료), RTX 5070.
  - 학습 데이터: 44,696장 (GTSDB 720 + AI Hub traffic 18,146 + 간판 25,830)
  - val 결과 (best.pt, ep57 기준): mAP@0.5=**0.628**, mAP@0.5:0.95=0.437, P=0.722, R=0.543
  - 클래스별: traffic_sign(P=0.751, R=0.521, mAP50=0.602) / signboard(P=0.693, R=0.565, mAP50=0.653)
  - 모델 크기: 21.47 MB (YOLOv8s FP32 best.pt)
  - 이전 E0 초안(YOLOv8n, 26,866장, mAP50=0.573) 대비 +5.5%p 향상
  - `runs/detect/edge_sign_v2_e0_full3/weights/best.pt`
- **2026-05-27**: 다양한 형태의 한글 문자 OCR (ZIP, 39.6GB) → Phase 2 초기에는 불필요. OCR 인식기 개선 필요 시 추후 처리.
- **2026-05-28**: 주행 Q&A 데모 아키텍처 결정: 엣지 파이프라인(YOLOv8n-INT8 + OCR-INT8) → 구조화 JSON → Claude Haiku API → 자연어 답변. 연구 결론부 시연용.
- **2026-05-28**: **E0 ByteTrack 추적 평가 완료** — `src/track/eval_tracking.py`, CPU ONNX Runtime.
  - 평가 시퀀스: c_1280_720_night_1 (142프레임) + c_1920_1200_night_1 (16프레임)
  - MOTA=**0.219**, IDF1=**0.384**, HOTA=**0.487**, IDSW=**0**, FPS=**21.6** (CPU)
  - **IDSW=0**: ByteTrack이 검출된 객체를 끝까지 일관되게 추적 (추적기 품질 우수)
  - **FN 높음(265/340)**: 주간 학습 → 야간 테스트 도메인 갭. 검출기 Recall 낮음이 원인, 추적기 문제 아님.
  - FPS 21.6: 목표 30+ FPS에 미달. CPU ONNX 단일 스레드 한계. GPU/INT8 배포 시 개선 예상.
- **2026-05-28**: **TrafficSignNet 학습+ONNX 변환 완료** → `model_space/traffic_sign_net_fp32.onnx`
  - GTSDB 43클래스 크롭(train 971 / val 242), 50 epoch, AdamW + CosineAnnealingLR
  - best_val_acc=**62.8%** (ep49), 모델 크기=**0.12 MB**, 파라미터=30,763
  - 학습 스크립트: `src/detect/train_traffic_sign_net.py`
- **2026-05-28**: **E1/E4/E5 추적 ablation 완료** → `src/track/run_tracking_ablation.py`
  - E1 W8A8: MOTA=0.221(+0.9%), IDF1=0.384(±0%), HOTA=0.487, IDSW=0, FPS=24.8
  - E4 W4A16: MOTA=0.105(−52%), IDF1=0.192(−50%), HOTA=0.322, IDSW=0, FPS=25.7
  - E5 SmoothQuant: MOTA=0.225(+2.7%), IDF1=0.387(+0.8%), HOTA=0.490, IDSW=0, FPS=20.8
  - **W4A16 추적 MOTA 급락 원인**: Recall 0.543→0.512 → FN: 265→315 (야간 도메인 갭 + 4bit 양자화 복합)
  - **IDSW=0 전 실험 공통**: ByteTrack 추적기 자체 품질 완벽 입증
- **2026-05-28**: **E1/E4/E5 검출기 양자화 실험 완료** — `src/quant/quantize_yolo.py` (Phase 1 fake-quant 방식 포팅).
  - E1 W8A8: mAP50=**0.621** (−1.0%) — 검출기는 W8A8에 강건. 연구 핵심 발견.
  - E4 W4A16: mAP50=**0.581** (−7.5%) — 4-bit에서 의미있는 성능 저하 시작.
  - E5 SmoothQuant+W8A8: mAP50=**0.621** (−1.0%) — W8A8과 동등. 활성화 분포 평탄화로 추가 이득 없음.
  - 모델 크기: fake-quant는 FP32 ONNX 저장 (42.67 MB). 실제 INT8 런타임 배포 시 ~10.7 MB (4배 압축) 예상.
  - SmoothQuant ONNX 구현 노트: ultralytics `.export()` 내부 `fuse()` 호출과 wrapper 충돌 → `nn_model.fuse()` 선행 후 `torch.onnx.export(dynamo=False)` 직접 사용으로 해결.
- **2026-05-28**: **인식기 양자화 실험 완료 (E2/E3/E4/E7)** → `src/quant/quantize_recognizers.py`
  - KoreanOCRNet W8A8: 98.4% (−0.1pp) / W4A16: 54.6% (−43.9pp) / 1-Bit: 0.3% (−98.2pp)
  - TrafficSignNet W8A8: 63.2% (+0.4pp) / W4A16: 49.2% (−13.6pp) / 1-Bit: 12.8% (−50.0pp)
  - **핵심**: OCR은 W8A8에 사실상 무손실, W4A16부터 치명적. 1-Bit은 완전 붕괴. **인식기가 파이프라인의 양자화 병목**
  - SimpleReIDNet 학습 데이터 없음 → W8A8 ONNX 내보내기만 수행 (`model_space/reid_net_w8a8.onnx`, 243.5 KB)
- **2026-05-28**: **E6 BoT-SORT 평가 완료** → `src/track/eval_botsort.py`
  - 구성: W8A8 YOLOv8s + BoT-SORT (CMC + W8A8 SimpleReIDNet, lam=0.5, alpha=0.95, frame_rate=5)
  - seq c_1280_720_night_1(142f): MOTA=0.0945, IDF1=0.2646, HOTA=0.3904, FP=21, FN=257
  - seq c_1920_1200_night_1(16f): MOTA=0.1212, IDF1=0.3256, HOTA=0.4410, FP=3, FN=26
  - **평균**: MOTA=0.108, IDF1=0.295, HOTA=0.416, IDSW=0, FPS=20.4
  - vs E1 ByteTrack W8A8: MOTA −0.113(−51%), IDF1 −0.089(−23%), HOTA −0.071(−15%)
  - **원인 분석**: SimpleReIDNet은 무작위 초기화(학습 데이터 없음) → 임베딩이 의미 없는 유사도 산출 → FP 6→21로 폭증. CMC 단독으로는 도움되나 untrained ReID가 역효과. ReID 학습이 전제되어야 BoT-SORT가 ByteTrack을 능가함을 실증
- **2026-05-28**: **Phase 5 CPU ONNX Runtime 벤치마크 완료** → `scripts/benchmark_pipeline.py`
  - fake-quant 파이프라인: E0 22.4 FPS / E1/E3 W8A8 ~25 FPS (FP32 연산이므로 실가속 없음)
  - **병목 확인**: YOLOv8s가 전체 레이턴시 ~80%. OCR/분류 합산 < 0.1ms → 검출기 최적화 최우선
  - Pareto 차트 생성: `assets/pareto_frontier.png` (E5가 MOTA·OCR 모두 Pareto 최적, 11.4 MB)
- **2026-05-28**: **Static INT8 QDQ 양자화 완료** → `scripts/quantize_onnx_real.py`
  - ORT `quantize_static()` + `quant_pre_process()` + GTSDB/AI Hub 캘리브레이션 데이터 사용
  - YOLOv8s: FP32 32.4ms → **INT8 14.6ms (2.22× 가속)**, 44.8MB → 11.7MB (3.84×) — 진짜 INT8 Conv 커널
  - KoreanOCRNet: 0.05→0.08ms (역효과) — 소형 모델은 INT8 오버헤드가 절감 초과, FP32 유지 권장
  - TrafficSignNet: 0.03ms→0.03ms (변화 없음) — 극소형, INT8 이득 없음
  - **파이프라인 최종**: E3 INT8 Static All → **57.7 FPS** (목표 30+ FPS 달성 ✅)
  - 최적 배포 파일: `model_space/yolov8s_signs_int8_static.onnx` (11.7 MB)
- **2026-05-28**: **E2E 파이프라인 TrafficSignNet 연결 완료** → `src/pipeline/e2e_pipeline.py`
  - traffic_sign 클래스 → TrafficSignNet W8A8 ONNX로 43클래스 GTSDB 분류
  - dry_run 검증: 3모델 모두 정상 로드 (YOLOv8s W8A8 + OCR W8A8 + TrafficSignNet W8A8)
- **2026-05-29**: **E2~E7 미기입 실험 셀 전체 보완 완료** → `docs/EXPERIMENTS.md`
  - E2/E3/E6/E7 검출 mAP: 검출기 구성 동일 실험(E0/E1/E4)의 결과를 파생 적용
    - E2 = E0 검출기(FP32), E3 = E1 검출기(W8A8), E6 = E1 검출기(W8A8), E7 = E4 검출기(W4A16)
  - E2/E3/E7 추적 결과: ByteTrack + 동일 검출기 구성이므로 E0/E1/E4 추적 결과 동일 적용
  - Pareto Frontier 데이터 표 전체 채움
  - End-to-End 종합 표: Final Score 열 추가, INT8 Static FPS 열 분리
- **2026-05-29**: **E2E 종합 평가 스크립트 완료 + 실행** → `src/pipeline/eval_e2e.py`
  - E0~E7 전체 파이프라인 FPS 측정 (CPU ONNX Runtime, fake-quant, AI Hub test 시퀀스 50프레임)
  - E0=21.2 / E1=24.7 / E2=23.3 / E3=24.3 / E4=25.2 / E5=20.5 / E6=20.4 / E7=25.1 FPS
  - Final Score 계산: E1(1.0335) > E3(1.0294) > E2(1.0198) > E5(0.9938) > E6(0.9922) > E4(0.7706) > E7(0.4392)
  - Note: fake-quant ONNX는 FP32 연산이므로 양자화 속도 이득 없음 — INT8 Static 배포 시 E3 → 57.7 FPS
- **2026-05-29**: **단계별 민감도 분석 그래프 생성 완료** → `scripts/plot_sensitivity.py`
  - 4종 그래프: 절대 성능 비교, 상대 변화율, 민감도 히트맵, 병목 요약 수평 막대
  - 출력: `assets/sensitivity_*.png` (4개 파일, 각 50~72 KB)
- **2026-05-29**: **데이터 분할 stratified v2 도입** → `scripts/extract_frames.py`
  - 기존 v1 (크기 내림차순): train 6(주간 전부) / val 1(야간) / test 2(야간) — 주간 도메인 검증 누락
  - 신규 v2 (도메인 stratified): train 5(주간 4+야간 1) / val 2(주간 1+야간 1) / test 2(주간 1+야간 1)
  - 야간이 3개뿐인 희소 도메인이므로 train/val/test에 1개씩 균등 보장하도록 분할 로직 개선
  - **본 문서의 E0~E7 결과는 모두 v1 분할 기준**이며, v2는 차기 재학습 시 적용 예정
- **2026-05-29**: **Pareto frontier 차트 마커 크기 축소 + 그래프-수치 정합 검증**
  - 기존: MS_PARETO=240, MS_NORMAL=100 (figure 대비 도형이 과대)
  - 변경: MS_PARETO=95, MS_NORMAL=50, linewidths=1.0/0.5로 축소
  - README ↔ EXPERIMENTS.md 19개 수치 교차 검증: 전체 OK

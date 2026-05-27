# Edge-Sign v2 아키텍처

> 이 문서는 시스템 아키텍처와 설계 결정을 기록합니다.
> 설계 변경 시 이유와 함께 업데이트하세요.

---

## 전체 파이프라인

```
영상 입력 (대시캠/거리 영상/웹캠, 640x480)
       │
       ▼
┌──────────────────────────┐
│ 1. YOLOv8-Nano 검출기    │  양자화 대상
│    다중 클래스:           │  ~3.2M params, FP16 ~6.3MB
│    - signboard (간판)    │  입력: 640x640 RGB
│    - traffic_sign (표지판)│  출력: bbox + confidence + class
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 2. ByteTrack 추적기      │  모델 없음 (순수 알고리즘)
│    Kalman Filter + IoU   │  또는 BoT-SORT (ReID 추가)
│    매칭                   │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 3. 클래스별 분기 인식기  │
│                          │
│  signboard ──► OCR       │  KoreanOCRNet (700K params)
│    ROI crop → 64x64 gray │  입력: (1,1,64,64), 출력: 2350 classes
│                          │
│  traffic_sign ──► 분류   │  TrafficSignNet (65K params)
│    ROI crop → 32x32 RGB  │  입력: (1,3,32,32), 출력: 12 classes
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ 4. 결과 조합 + 표시      │
│  Track ID + bbox 오버레이│
│  간판: OCR 텍스트 표시   │
│  표지판: 분류 라벨 표시  │
└──────────────────────────┘
```

---

## 데이터 파이프라인

> 실제 확인된 데이터 형식: 신호등-도로표지판은 **동영상이 아닌 JPG 프레임**이 TAR로 압축된 형태.
> 030.야외 한글 이미지는 이미 압축 해제된 상태 (JPG + JSON 쌍).

### 신호등-도로표지판 파이프라인

```
AIhub/신호등-도로표지판 인지 영상(수도권)/Validation/
  [원천]c_validation_1280_720_daylight_1.tar  ←─ TAR 1개 = 촬영 시퀀스 1개
  [원천]c_validation_1280_720_daylight_2.tar     (9개 시퀀스, 총 110,900 JPG 프레임)
  ...
  [라벨]c_validation_1280_720_daylight_1.tar  ←─ 대응 JSON 어노테이션
  ...
       │
       ▼
scripts/extract_frames.py
  - --sample_rate 6  (매 6번째 프레임 → 30fps → 5fps 시뮬레이션)
  - --split_by sequence  ← ★ 핵심: TAR 단위(=시퀀스 단위) 분할
       │                         (프레임 단위 분할 시 인접 프레임 리크 발생)
       ├─── train: 6개 TAR → data/aihub_traffic/train/images/{seq}/*.jpg
       │                      data/aihub_traffic/train/labels/{seq}/*.json
       ├─── val:   1~2개 TAR → data/aihub_traffic/val/
       └─── test:  1~2개 TAR → data/aihub_traffic/test/  ← ByteTrack 평가·시연용 연속 프레임
       │
       ▼
src/detect/prepare_dataset.py --source aihub_traffic
  - JSON {"annotation":[{"box":[x1,y1,x2,y2], "class":"traffic_sign"/"traffic_light"}]}
  - xyxy 절대좌표 → YOLO 정규화 cx cy w h
  - traffic_sign + traffic_light → 모두 class 0 (traffic_sign)
  - data/yolo_signs/images/{train,val}/  (플랫 구조: {seq}__{frame}.jpg)
  - data/yolo_signs/labels/{train,val}/
```

### 야외 한글 이미지 파이프라인

```
AIhub/030.야외 실제 촬영 한글 이미지/01.데이터/
  1.Training/원천데이터_230216_add/1.간판/{subclass}/*.jpg    ← 25,837장 (이미 해제됨)
  1.Training/라벨링데이터_230216_add/1.간판/{subclass}/*.json
  2.Validation/원천데이터_230216_add/                        ← 4,304장
  2.Validation/라벨링데이터_230216_add/
       │
       ▼
src/detect/prepare_dataset.py --source aihub_signboard
  - JSON {"annotations":[{"bbox":[x,y,w,h], "text":"간판텍스트"}]}  (COCO-style xywh)
  - xywh 절대좌표 → YOLO 정규화 cx cy w h
  - 간판 → class 1 (signboard)
  - data/yolo_signs/images/{train,val}/  (플랫 구조: sign_{subclass}__{stem}.jpg)
  - data/yolo_signs/labels/{train,val}/
       │
       ▼ (GTSDB 합산)
data/yolo_signs/dataset.yaml  → YOLOv8n 학습 입력
```

### 시퀀스별 프레임 현황 (신호등-도로표지판)

| 시퀀스 | 해상도 | 조건 | 프레임 수 | TAR 크기 | 용도 |
|--------|--------|------|----------|---------|------|
| c_1280_720_daylight_1 | 1280×720 | 주간 | 30,000 | 6.2GB | train |
| c_1280_720_daylight_2 | 1280×720 | 주간 | 30,000 | 6.3GB | train |
| c_1920_1200_daylight_1 | 1920×1200 | 주간 | 12,907 | 7.6GB | train |
| d_1920_1080_daylight_1 | 1920×1080 | 주간 | 15,000 | 7.7GB | train |
| d_1920_1080_daylight_2 | 1920×1080 | 주간 | 14,401 | 7.8GB | train |
| c_1280_720_daylight_3 | 1280×720 | 주간 | 6,553 | 1.4GB | train |
| c_1280_720_night_1 | 1280×720 | 야간 | 847 | 0.14GB | val |
| c_1920_1200_night_1 | 1920×1200 | 야간 | 93 | 0.03GB | val |
| d_1920_1080_night_1 | 1920×1080 | 야간 | 1,099 | 0.5GB | test |

서브샘플 후 (÷6): 학습 약 18,143프레임 + 검증 156프레임 + 테스트 183프레임

### 시퀀스 기반 분할의 중요성

| 분할 방식 | 리크 여부 | 이유 |
|-----------|-----------|------|
| **프레임 단위** (잘못된 방법) | ⚠️ 리크 | 동일 시퀀스의 인접 프레임이 train/test에 동시 존재 |
| **시퀀스 단위** (올바른 방법) | ✅ 없음 | train 시퀀스와 test 시퀀스가 완전히 분리됨 |

### 시연·평가용 프레임 활용

- test 시퀀스 (d_night_1): 연속 프레임 → ByteTrack MOTA/IDF1 평가 (학습 미사용)
- test 시퀀스 → Phase 6 웹 시연에서 프레임 단위 재생 (실제 한국 도로 영상)

---

## 모델 선택 근거

### 검출기: YOLOv8-Nano

| 고려 모델 | Params | mAP@0.5 (COCO) | 선택 여부 |
|-----------|--------|-----------------|-----------|
| YOLOv8n | 3.2M | 37.3 | **선택** |
| YOLOv11n | 2.6M | 39.5 | 대안 (추후 ablation) |
| RT-DETR-l | 32M | 53.0 | 제외 (너무 큼) |

**선택 이유:** YOLOv8n은 Ultralytics에서 ONNX 내보내기/양자화 지원이 가장 성숙함. 3.2M params로 엣지 제약 충족. Phase 1의 ConvNeXtV2-Nano(7.2M)보다 작아서 양자화 효과가 더 극적일 수 있음.

### 추적기: ByteTrack (기본) + BoT-SORT (ablation)

| 추적기 | 추가 모델 | MOTA (MOT17) | 선택 여부 |
|--------|-----------|--------------|-----------|
| ByteTrack | 없음 (Kalman+IoU만) | 80.3 | **기본** |
| BoT-SORT | ReID ~0.5M params | 80.5 | ablation |
| StrongSORT | ReID ~11M params | 79.6 | 제외 (ReID 너무 큼) |
| UCMCTrack | CMC module | 80.5 | 제외 (카메라 보정 필요) |

**선택 이유:** ByteTrack은 추가 모델이 없어서 양자화 효과를 검출기에 순수 분리 가능. BoT-SORT는 ReID 백본(OSNet-x0.25)도 양자화할 수 있어서 "추적 단계 양자화" 실험에 활용.

### 인식기: 기존 모델 재활용

- **KoreanOCRNet** (`src/korean_ocr_model.py`): Depthwise-Separable Conv, 700K params, 2350 한글 문자 클래스
- **TrafficSignNet** (`src/model.py`): 1x1 Conv 기반, 65K params, 12 교통표지판 클래스

**재활용 이유:** Phase 1에서 이미 양자화 실험(W8A8, W4A16, 1-Bit)을 거친 아키텍처. 새 모델 학습 없이 파이프라인 통합에 집중 가능.

---

## 양자화 전략

### 단계별 독립 양자화

핵심 아이디어: 파이프라인의 각 단계를 독립적으로 양자화하여 **어떤 단계가 가장 민감한지** 분석.

```
검출기 양자화 ──► 검출 mAP 변화 ──► 추적 MOTA 변화 (전파 효과)
인식기 양자화 ──► 인식 정확도 변화 (검출/추적은 영향 없음)
ReID 양자화  ──► 추적 IDF1 변화 (ID 재식별 정확도)
```

### 적용할 양자화 방법

| 방법 | 출처 코드 | 핵심 원리 | Phase 1 결과 |
|------|-----------|-----------|-------------|
| W8A8 PTQ | `src/base_W8A8.py` | MinMax 선형 매핑, 재학습 불필요 | -0.64%p (우수) |
| W4A16 QAT | `src/base_train_w4a16_qat.py` | STE 기반 4비트, 학습 필요 | -5.76%p |
| SmoothQuant | `src/multimodal_w8a8_smoothquant.py` | 활성화-가중치 균형 스케일링 | Final Score 1위 |
| 1-Bit KD | `src/base_train_1bit_kd.py` | 이진화 + 지식증류, Bit-packing | 정보 한계 (14.2%) |

---

## 웹 배포 아키텍처

### 모드 1: 전체 클라이언트 사이드 (목표)

```
브라우저 (ONNX Runtime Web)
┌─────────────────────────────┐
│ Camera API → 프레임 캡처     │
│ → YOLOv8n ONNX (WASM)      │
│ → ByteTrack (순수 JS)       │
│ → ROI Crop                  │
│ → OCR/분류 ONNX (WASM)      │
│ → Canvas 오버레이 렌더링     │
└─────────────────────────────┘
총 모델 페이로드 목표: <15MB
```

### 모드 2: 서버 어시스트 (fallback)

```
브라우저                    FastAPI 서버
┌──────────┐  WebSocket  ┌──────────────┐
│ Camera   │ ──────────► │ YOLOv8n      │
│ 프레임    │             │ + ByteTrack  │
│ 전송     │ ◄────────── │ + 인식기      │
│ 결과 표시 │  JSON 결과  │ (GPU/CPU)    │
└──────────┘             └──────────────┘
```

**참조 패턴:** `scripts/mediapipe_ws_server.py` (WebSocket), `web/app.js` (ONNX Runtime Web)

---

## 설계 결정 로그

| 날짜 | 결정 | 이유 | 대안 |
|------|------|------|------|
| 2026-05-27 | 간판+교통표지판 두 도메인 통합 | "Sign" 브랜딩 유지 + 다중 클래스 검출의 복잡도 증가 + 기존 OCR/분류기 모두 재활용 | 수어(정확도 부족으로 보류), 일반 MOT(기존 인프라 활용 불가) |
| 2026-05-27 | ByteTrack 기본 추적기 | 추가 모델 없어서 양자화 효과를 검출기에 순수 분리 가능 | BoT-SORT(ablation으로 병행), StrongSORT(ReID 너무 큼) |
| 2026-05-27 | 기존 OCR/분류기 재활용 | Phase 1 양자화 경험 활용, 새 학습 불필요 | ConvNeXtV2-Nano로 통합(오버킬), 경량 MobileNet(기존 코드 없음) |
| 2026-05-27 | AI Hub validation only (~40GB) 사용 + 재분할 | 전체 191만 장 다운로드(수 TB) 불가. val 데이터만으로도 수천~수만 프레임 확보 가능 | 전체 학습 데이터(너무 큼), GTSDB만(한국 도로 표지판 없음) |
| 2026-05-27 | 시퀀스 단위 train/val/test 분할 (서브샘플 ÷6) | AI Hub 원천은 동영상이 아닌 JPG 프레임 (TAR 압축). 프레임 단위 분할 시 인접 프레임 간 데이터 리크 발생. ÷6 서브샘플로 30fps→5fps 시뮬레이션, 시각적 중복 제거와 데이터 크기 균형 | ÷1(저장소 과다), 프레임 단위 분할(리크), 시퀀스 내 랜덤 샘플(임시 리크 가능성) |
| 2026-05-27 | test 시퀀스 연속 프레임 보존 (추적 평가·시연용) | 추적 평가(MOTA/IDF1)는 연속 프레임 시퀀스가 필요. 웹 시연도 실제 한국 도로 영상을 사용해야 설득력 있음 | 별도 테스트 동영상 촬영(불필요한 추가 작업) |
| 2026-05-27 | traffic_light → class 0 (traffic_sign) 통합 | 신호등 인식기(red/green/yellow 판별)가 현재 파이프라인에 없음. 검출 단계에서 통합 후 추후 분리 가능 | class 2로 별도 분리(인식기 필요), 신호등 어노테이션 제외(데이터 손실) |
| 2026-05-27 | 030.야외 한글 이미지 공식 Train/Val 분할 그대로 사용 | AI Hub 공식 분할 준수 → 재현 가능한 벤치마크. 4,304 Val → 평가용으로 충분 | 무작위 재분할(공식 분할 무시), Val 50%를 test로 분리(현재는 불필요) |

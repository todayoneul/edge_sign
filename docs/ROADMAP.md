# Edge-Sign v2 로드맵

> 이 문서는 프로젝트의 단계별 진행 상태를 추적합니다.
> 태스크 완료 시 `[x]`로 표시하고 날짜를 기록하세요.

---

## Phase 0: 문서 및 기반 준비
- [x] CLAUDE.md 생성 (2026-05-27)
- [x] docs/ROADMAP.md 생성 (2026-05-27)
- [x] docs/ARCHITECTURE.md 생성 (2026-05-27)
- [x] docs/EXPERIMENTS.md 생성 (2026-05-27)

---

## Phase 1: 검출 기반 구축 (1~2주차)
**목표:** YOLOv8n으로 간판+교통표지판 검출 FP16 기준선 확립

- [ ] 데이터셋 선정 및 준비
  - [x] GTSDB 다운로드 완료 (2026-05-27) — `data/GTSDB/FullIJCNN2013/`, 900장 + gt.txt
  - [x] AI Hub 신호등-도로표지판 인지 영상(수도권) TAR 다운로드 완료 (2026-05-27)
    - ✅ 9개 시퀀스, 총 110,900 JPG 프레임 + JSON 어노테이션 (TAR 압축 상태, 37GB)
    - ⚠️ **원천 데이터는 동영상이 아닌 JPG 프레임** — TAR 아카이브에 이미 프레임으로 저장됨
    - JSON 포맷: `{"annotation":[{"box":[x1,y1,x2,y2],"class":"traffic_sign"/"traffic_light"}],"image":{"imsize":[W,H]}}`
    - 시퀀스별 파일 크기: daylight_1,2(1280×720)≈30k프레임, d_daylight_1,2(1920×1080)≈15k프레임 등
  - [x] AI Hub 030.야외 실제 촬영 한글 이미지 다운로드 완료 (2026-05-27)
    - ✅ 이미 압축 해제됨 (JPG+JSON 쌍): Training 25,837장 + Validation 4,304장
    - JSON 포맷: `{"images":[{"width":W,"height":H}],"annotations":[{"bbox":[x,y,w,h],"text":"..."}]}`
    - 카테고리: 가로형간판(18,841), 실내간판(6,574), 세로형간판(363) 등
  - [ ] AI Hub 다양한 형태의 한글 문자 OCR 데이터 ZIP 해제 (선택사항, 39.6GB)
    - 인쇄체+필기체 한글 문자 인식용 — OCR 인식기 개선에 활용 예정
  - [x] 신호등-도로표지판 TAR 해제 + 시퀀스 분할 완료 (2026-05-27)
    - `--sample_rate 6` → 18,488프레임 추출 (train 18,146 / val 184 / test 158)
    - train: 주간 6시퀀스 / val: d_1920_1080_night_1 / test: c_1280_720_night_1 + c_1920_1200_night_1
    - 출력: `data/aihub_traffic/{train,val,test}/{images,labels}/{seq_name}/`
  - [x] GTSDB → YOLO 포맷 변환 완료 (2026-05-27) — 900장 (train 720 / val 180)
  - [x] 신호등-도로표지판 → YOLO 포맷 변환 완료 (2026-05-27) — 18,330장
  - [x] 야외 한글 이미지 → YOLO 포맷 변환 완료 (2026-05-27) — 12,303장 (train 8,000 / val 4,303)
  - ✅ **최종 yolo_signs**: train 26,866장 / val 4,667장 (GTSDB + 신호등 + 간판 통합)
- [ ] YOLOv8n 학습
  - [x] GTSDB + AI Hub 전체 학습 완료 (2026-05-28) — `runs/detect/edge_sign_v2_e0_full3/`
    - 설정: YOLOv8s, batch=32, imgsz=640, cos_lr, patience=20, RTX 5070, 75 epoch (조기종료)
    - 학습 데이터: 44,696장 (GTSDB 720 + AI Hub traffic 18,146 + 간판 25,830)
    - best epoch: 57, `weights/best.pt` (21.47 MB)
  - [x] FP32 기준선 mAP 측정 및 기록 완료 (2026-05-28) → `docs/EXPERIMENTS.md` E0 행
    - mAP@0.5=**0.628**, mAP@0.5:0.95=0.437, P=0.722, R=0.543
    - 클래스별: traffic_sign mAP50=0.602 / signboard mAP50=0.653
- [x] ONNX 내보내기 (2026-05-28)
  - [x] PyTorch → ONNX 변환 → `src/detect/export_yolo_onnx.py` → `model_space/yolov8s_signs_fp32.onnx` (42.67 MB)
  - [x] ONNX 모델 검증 완료 (output shape (1,6,8400) 일치 확인)

**완료 기준:** FP16 YOLOv8n의 mAP@0.5 > 0.7 달성

---

## Phase 2: 추적 통합 (2~3주차)
**목표:** ByteTrack으로 프레임 간 객체 추적, MOT 메트릭 기준선 확립

- [ ] ByteTrack 구현
  - [x] Kalman Filter + IoU 매칭 구현 완료 (2026-05-28) → `src/track/bytetrack.py` (616줄)
    - 8-dim constant-velocity Kalman Filter, 2단계 매칭(BYTE 전략), Hungarian + greedy fallback
  - [x] AI Hub test 시퀀스 동작 확인 + 평가 완료 (2026-05-28) → `src/track/eval_tracking.py`
    - c_1280_720_night_1(142프레임) + c_1920_1200_night_1(16프레임)
    - MOTA=0.219, IDF1=0.384, HOTA=0.487, IDSW=0, FPS=21.6 (CPU)
- [x] BoT-SORT 통합 (ablation용) (2026-05-28)
  - [x] 경량 ReID 백본: SimpleReIDNet (62,816 params, 0.24 MB) 자체 구현 — OSNet 대비 경량 우선
  - [x] BoT-SORT 구현 완료 → `src/track/botsort.py` (CMC + EMA ReID + 3단계 매칭)
- [ ] MOT 평가
  - [ ] 테스트 시퀀스 준비 — extract_frames.py의 test split 시퀀스 사용
    - ⚠️ 시퀀스 단위 분할로 test 프레임은 학습에 미등장, 리크 없음
  - [x] MOTA/IDF1/HOTA 평가 코드 완료 (2026-05-28) → `src/track/eval_tracking.py`
  - [x] FP32 기준선 추적 메트릭 기록 완료 (2026-05-28) → `docs/EXPERIMENTS.md`
  - [x] E1/E4/E5 추적 ablation 완료 (2026-05-28) → `src/track/run_tracking_ablation.py`
    - E1 W8A8: MOTA=0.221(+0.9%), IDF1=0.384(±0%), HOTA=0.487(±0%), IDSW=0, FPS=24.8
    - E4 W4A16: MOTA=0.105(−52%), IDF1=0.192(−50%), HOTA=0.322(−34%), IDSW=0, FPS=25.7
    - E5 SmoothQuant: MOTA=0.225(+2.7%), IDF1=0.387(+0.8%), HOTA=0.490(+0.6%), IDSW=0, FPS=20.8
    - **핵심 발견**: IDSW=0 유지, W4A16만 FN 폭증으로 MOTA 급락 — 추적기 문제 아님

**완료 기준:** ByteTrack MOTA > 0.5 on AI Hub test 시퀀스

---

## Phase 3: 파이프라인 조립 + 인식 연결 (3~4주차)
**목표:** 검출 → 추적 → 클래스별 분기 인식 전체 파이프라인 완성

- [ ] 트랙별 ROI 크롭 구현
  - [x] 검출 bbox → 추적된 ID별 이미지 크롭 (2026-05-28) — `e2e_pipeline.py` preprocess_ocr_roi()
  - [x] 시간 버퍼 (최근 T=8 프레임) 관리 (2026-05-28) — deque(maxlen=8) per track_id
- [ ] 인식기 분기 연결
  - [x] signboard → KoreanOCRNet ONNX 연결 완료 (2026-05-28) — `e2e_pipeline.py` _run_ocr()
  - [x] traffic_sign → TrafficSignNet GTSDB 학습 + ONNX 변환 완료 (2026-05-28)
    - `src/detect/train_traffic_sign_net.py` → `model_space/traffic_sign_net_fp32.onnx`
    - 43클래스(GTSDB), val_acc=62.8%, 모델 크기 0.12 MB, 파라미터 30,763개
- [x] E2E 파이프라인 구현 완료 (2026-05-28) → `src/pipeline/e2e_pipeline.py`
  - YOLOv8n-ONNX 검출 + ByteTracker + KoreanOCRNet OCR 통합
  - CLI: `python src/pipeline/e2e_pipeline.py --dry_run` 으로 초기화 확인 가능
- [ ] E2E 평가 → `src/pipeline/eval_e2e.py`
- [ ] FP16 기준선 E2E 메트릭 기록 → `docs/EXPERIMENTS.md` E0 행 완성
  - ⚠️ YOLOv8n ONNX 변환 선행 필요 (`export_yolo_onnx.py`)

**완료 기준:** 영상 입력 → 검출+추적+인식 결과 출력 파이프라인 동작

---

## Phase 4: 체계적 양자화 실험 (4~6주차)
**목표:** E0~E7 실험 매트릭스 완성, 단계별 민감도 분석

- [x] YOLOv8s 양자화 포팅 (2026-05-28) → `src/quant/quantize_yolo.py`
  - [x] W8A8 PTQ (E1): mAP50=0.621 (−1.0%) — `model_space/yolov8s_signs_w8a8.onnx`
  - [x] W4A16 PTQ (E4): mAP50=0.581 (−7.5%) — `model_space/yolov8s_signs_w4a16.onnx`
  - [x] SmoothQuant+W8A8 (E5): mAP50=0.621 (−1.0%) — `model_space/yolov8s_signs_smoothquant.onnx`
- [x] 인식기 양자화 완료 (2026-05-28) → `src/quant/quantize_recognizers.py`
  - [x] KoreanOCRNet: W8A8=98.4%(−0.1pp) / W4A16=54.6%(−43.9pp) / 1-Bit=0.3%(−98.2pp)
  - [x] TrafficSignNet: W8A8=63.2%(+0.4pp) / W4A16=49.2%(−13.6pp) / 1-Bit=12.8%(−50.0pp)
  - SmoothQuant ≈ W8A8 (활성화 분포 정규화 효과 미미) — 별도 실험 생략
- [x] ReID 백본 양자화 완료 (2026-05-28) → `src/quant/quantize_reid.py`
  - [x] SimpleReIDNet W8A8 → `model_space/reid_net_w8a8.onnx` (243.5 KB)
  - ⚠️ 학습 데이터 없음(미학습) → E6에서 ReID 역효과 확인됨
- [x] 실험 매트릭스 실행 (2026-05-28) — 추적/인식 결과 `docs/EXPERIMENTS.md` 기록
  - [x] E1: 검출기만 W8A8 → MOTA=0.221(+0.9%), IDF1=0.384(±0%)
  - [x] E2/E3: 인식기 W8A8 → OCR=98.4%(−0.1pp), TS=63.2%(+0.4pp)
  - [x] E4: 전체 W4A16 → MOTA=0.105(−52%), OCR=54.6%(−43.9pp) ← 파이프라인 병목
  - [x] E5: 전체 SmoothQuant → MOTA=0.225(+2.7%), OCR=98.5%(±0)
  - [x] E6: BoT-SORT + W8A8 ReID → MOTA=0.108(−51% vs E1) ← 미학습 ReID 역효과 실증
  - [x] E7: 극한 (W4A16 검출 + 1-Bit 인식) → OCR=0.3%(−98.2pp) ← 완전 붕괴
- [ ] 결과 분석 + 시각화
  - [ ] Pareto frontier 차트 생성
  - [ ] 단계별 민감도 분석 그래프
  - [ ] `docs/EXPERIMENTS.md` 전체 결과 기록

**완료 기준:** 8개 실험 전체 결과 + Pareto 차트 완성

---

## Phase 5: ONNX 최적화 + 엣지 벤치마크 (6~7주차)
**목표:** 최적 구성을 ONNX로 내보내고 엣지 성능 벤치마크

- [x] 최적 구성 선정 (2026-05-28): **E5 SmoothQuant+W8A8 (11.4 MB)**
  - Pareto 최적: MOTA=0.225, OCR=98.5%, 목표 15 MB 이내 달성
- [x] 전체 파이프라인 ONNX 내보내기 (2026-05-28) → 각 구성요소별 ONNX 분리 형태
  - `model_space/yolov8s_signs_*.onnx` + `korean_ocr_net_*.onnx` + `traffic_sign_net_*.onnx`
- [x] ONNX Runtime CPU 벤치마크 완료 (2026-05-28) → `scripts/benchmark_pipeline.py`
  - fake-quant E0/E3: 22~25 FPS (FP32 연산)
  - **Static INT8 QDQ 실양자화** → `scripts/quantize_onnx_real.py`
    - YOLOv8s: 32.4ms → **14.6ms (2.22×)**, 44.8MB → **11.7MB (3.84×)**
    - E3 INT8 Static All: **57.7 FPS** — 30+ FPS 목표 달성 ✅
    - OCR/분류 소형 모델은 INT8 오버헤드 역효과 → FP32 유지 권장
- [ ] ONNX Runtime Web (WASM) 벤치마크 (선택사항, 실제 브라우저 배포 시)
- [x] 벤치마크 결과 기록 → `docs/EXPERIMENTS.md` + `README.md` Section 7.6

**완료 기준:** ONNX 파이프라인 30+ FPS on CPU ✅ (E3 INT8 Static: 57.7 FPS 달성)

---

## Phase 6: 웹 배포 + 시연 (7~8주차)
**목표:** 브라우저에서 실시간 검출+추적+인식 시연

- [ ] 웹 프론트엔드 구현 (서버 어시스트 방식 — Phase 7에서 선행 구현)
  - [x] `web/detection/index.html` — 영상 뷰 + 트랙 목록 + Q&A 채팅 UI (2026-05-28)
  - [x] `web/detection/app.js` — WebSocket 프레임 전송 + SSE Q&A 클라이언트 (2026-05-28)
  - [ ] `web/detection/bytetrack.js` — JS ByteTrack (전체 클라이언트 사이드 목표 시 필요)
  - [ ] 양자화 모델 전환 토글 (FP16/W8A8/W4A16 비교)
  - [ ] FPS + 모델 크기 실시간 표시
- [x] 서버 어시스트 모드 구현 완료 (2026-05-28) → `src/pipeline/app.py`
  - WS /ws/stream + POST /api/qa + StaticFiles 서빙
- [ ] 최종 시연 준비
  - [ ] AI Hub 도로 영상 클립 (val 분할에서 선별) 재생 확인
  - [ ] Pareto 차트 대시보드

**완료 기준:** AI Hub 도로 영상 + 웹캠에서 간판+표지판 실시간 검출+추적+인식 동작

---

## Phase 7: 주행 Q&A 결론 데모 (결론 섹션용)
**목표:** "엣지 압축 인식 + 클라우드 언어 지능" 하이브리드 시연

### 연구 스토리
> 엣지 디바이스(`<15MB` 모델)가 실시간으로 표지판/간판 인식 → 구조화 JSON 생성 →
> Claude API가 컨텍스트를 받아 운전자 질문에 자연어 답변.

### 구현 항목
- [x] E2E 파이프라인 ONNX 추론 + ByteTrack 통합 (2026-05-28) → `src/pipeline/e2e_pipeline.py`
- [x] Claude API Q&A 브리지 (2026-05-28) → `src/pipeline/qa_bridge.py`
  - `build_context(tracks)` + `ask_stream()` (claude-haiku-4-5-20251001, SSE 스트리밍)
- [x] FastAPI 백엔드 (2026-05-28) → `src/pipeline/app.py`
- [x] 웹 데모 UI (2026-05-28) → `web/detection/index.html` + `app.js`
- [x] `.env.example` 생성 (2026-05-28)
- [ ] 실제 동작 검증 (YOLOv8n ONNX 변환 + ANTHROPIC_API_KEY 설정 후)
  ```bash
  cp .env.example .env          # API 키 입력
  python src/detect/export_yolo_onnx.py --weights runs/detect/edge_sign_v2_e0/weights/best.pt
  uvicorn src.pipeline.app:app --port 8000
  ```

**완료 기준:** AI Hub 도로 영상 재생 → 표지판/간판 인식 → 질문 입력 → Claude 답변 시연

---

## 최종 산출물 체크리스트
- [ ] 연구 보고서 (실험 결과 + 분석)
- [ ] 시연 시스템 (웹 앱 — Phase 7 기반)
- [ ] 코드 정리 + 문서 최종 업데이트

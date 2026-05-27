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
  - [ ] 신호등-도로표지판 TAR 해제 + 시퀀스 분할 → `scripts/extract_frames.py`
    - `--sample_rate 6` (30fps→5fps 서브샘플링, 시각적 중복 제거)
    - **시퀀스 단위 분할**: 9개 TAR → train 6개 / val 1~2개 / test 1~2개
    - test 시퀀스는 연속 프레임 보존 → ByteTrack 추적 평가 및 웹 시연에 사용
  - [ ] GTSDB → YOLO 포맷 변환 → `src/detect/prepare_dataset.py --source gtsdb`
  - [ ] 신호등-도로표지판 프레임 → YOLO 포맷 변환 → `--source aihub_traffic`
    - JSON xyxy bbox → YOLO 정규화 cx cy w h (구현 완료)
  - [ ] 야외 한글 이미지 → YOLO 포맷 변환 → `--source aihub_signboard`
    - JSON COCO-style xywh bbox → YOLO 정규화 cx cy w h (구현 완료)
- [ ] YOLOv8n 학습
  - [ ] GTSDB only 기준선 학습 (AI Hub 도착 전 선행) → `src/detect/yolo_train.py`
  - [ ] GTSDB + AI Hub 합산 학습 → `--source all`
  - [ ] FP16 기준선 mAP 측정 및 기록 → `docs/EXPERIMENTS.md` E0 행
- [ ] ONNX 내보내기
  - [ ] PyTorch → ONNX 변환 → `src/detect/export_yolo_onnx.py`
  - [ ] ONNX 모델 검증 (추론 결과 일치 확인)

**완료 기준:** FP16 YOLOv8n의 mAP@0.5 > 0.7 달성

---

## Phase 2: 추적 통합 (2~3주차)
**목표:** ByteTrack으로 프레임 간 객체 추적, MOT 메트릭 기준선 확립

- [ ] ByteTrack 구현
  - [ ] Kalman Filter + IoU 매칭 구현 → `src/track/bytetrack.py`
  - [ ] AI Hub test 시퀀스 (연속 프레임) 에서 동작 확인
- [ ] BoT-SORT 통합 (ablation용)
  - [ ] 경량 ReID 백본 선택 (OSNet-x0.25 ~0.5M params)
  - [ ] BoT-SORT 구현 → `src/track/botsort.py`
- [ ] MOT 평가
  - [ ] 테스트 시퀀스 준비 — extract_frames.py의 test split 시퀀스 사용
    - ⚠️ 시퀀스 단위 분할로 test 프레임은 학습에 미등장, 리크 없음
  - [ ] MOTA/IDF1/HOTA 평가 코드 → `src/track/eval_tracking.py`
  - [ ] FP16 기준선 추적 메트릭 기록 → `docs/EXPERIMENTS.md`

**완료 기준:** ByteTrack MOTA > 0.5 on AI Hub test 시퀀스

---

## Phase 3: 파이프라인 조립 + 인식 연결 (3~4주차)
**목표:** 검출 → 추적 → 클래스별 분기 인식 전체 파이프라인 완성

- [ ] 트랙별 ROI 크롭 구현
  - [ ] 검출 bbox → 추적된 ID별 이미지 크롭
  - [ ] 시간 버퍼 (최근 T=8 프레임) 관리
- [ ] 인식기 분기 연결
  - [ ] signboard → KoreanOCRNet (기존 `src/korean_ocr_model.py`)
  - [ ] traffic_sign → TrafficSignNet (기존 `src/model.py`)
- [ ] E2E 파이프라인 → `src/pipeline/e2e_pipeline.py`
- [ ] E2E 평가 → `src/pipeline/eval_e2e.py`
- [ ] FP16 기준선 E2E 메트릭 기록 → `docs/EXPERIMENTS.md` E0 행 완성

**완료 기준:** 영상 입력 → 검출+추적+인식 결과 출력 파이프라인 동작

---

## Phase 4: 체계적 양자화 실험 (4~6주차)
**목표:** E0~E7 실험 매트릭스 완성, 단계별 민감도 분석

- [ ] YOLOv8n 양자화 포팅
  - [ ] W8A8 PTQ 적용 (`base_W8A8.py` 로직 포팅) → `src/quant/quantize_yolo.py`
  - [ ] W4A16 QAT 적용 (`base_train_w4a16_qat.py` 래퍼 적용)
  - [ ] SmoothQuant 적용 (`multimodal_w8a8_smoothquant.py` 캘리브레이션 적용)
- [ ] 인식기 양자화 (기존 코드 재활용)
  - [ ] KoreanOCRNet W8A8/W4A16/SmoothQuant/1-Bit
  - [ ] TrafficSignNet W8A8/W4A16/SmoothQuant/1-Bit
- [ ] ReID 백본 양자화 (BoT-SORT 실험용)
  - [ ] OSNet-x0.25 W8A8 → `src/quant/quantize_reid.py`
- [ ] 실험 매트릭스 실행
  - [ ] E1: 검출기만 W8A8
  - [ ] E2: 인식기만 W8A8
  - [ ] E3: 전체 W8A8
  - [ ] E4: 전체 W4A16
  - [ ] E5: 전체 SmoothQuant
  - [ ] E6: BoT-SORT + W8A8 ReID
  - [ ] E7: 극한 (W4A16 검출 + 1-Bit 인식)
- [ ] 결과 분석 + 시각화
  - [ ] Pareto frontier 차트 생성
  - [ ] 단계별 민감도 분석 그래프
  - [ ] `docs/EXPERIMENTS.md` 전체 결과 기록

**완료 기준:** 8개 실험 전체 결과 + Pareto 차트 완성

---

## Phase 5: ONNX 최적화 + 엣지 벤치마크 (6~7주차)
**목표:** 최적 구성을 ONNX로 내보내고 엣지 성능 벤치마크

- [ ] 최적 구성 선정 (Final Score 기준)
- [ ] 전체 파이프라인 ONNX 내보내기 → `src/export/export_pipeline_onnx.py`
- [ ] ONNX Runtime CPU 벤치마크
- [ ] ONNX Runtime Web (WASM) 벤치마크
- [ ] 벤치마크 결과 기록 → `docs/EXPERIMENTS.md`

**완료 기준:** ONNX 파이프라인 30+ FPS on CPU 또는 명확한 병목 분석

---

## Phase 6: 웹 배포 + 시연 (7~8주차)
**목표:** 브라우저에서 실시간 검출+추적+인식 시연

- [ ] 웹 프론트엔드 구현
  - [ ] `web/detection/index.html` — UI 레이아웃
  - [ ] `web/detection/app.js` — ONNX Runtime Web 추론
  - [ ] `web/detection/bytetrack.js` — JS ByteTrack
  - [ ] `web/detection/styles.css` — 스타일링
- [ ] 실시간 기능
  - [ ] 웹캠/영상 입력 → 검출 → 추적 → 인식 오버레이
  - [ ] **AI Hub 프레임 시퀀스 재생 모드** — test 시퀀스 프레임을 브라우저에서 재생하며 파이프라인 시연
  - [ ] 양자화 모델 전환 토글 (FP16/W8A8/W4A16 비교)
  - [ ] FPS + 모델 크기 실시간 표시
- [ ] 서버 fallback 모드
  - [ ] WebSocket 서버 → `scripts/detection_server.py`
- [ ] 최종 시연 준비
  - [ ] AI Hub 도로 영상 클립 (val 분할에서 선별) + 실시간 웹캠 데모
  - [ ] Pareto 차트 대시보드

**완료 기준:** AI Hub 도로 영상 + 웹캠에서 간판+표지판 실시간 검출+추적+인식 동작

---

## 최종 산출물 체크리스트
- [ ] 연구 보고서 (실험 결과 + 분석)
- [ ] 시연 시스템 (웹 앱)
- [ ] 코드 정리 + 문서 최종 업데이트

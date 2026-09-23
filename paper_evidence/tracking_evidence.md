# Tracking evidence audit

2026-09-23 정적 소스 조사 및 기존 annotation 메타데이터 계수만 수행. 추론/평가를 새로 실행하지 않았다.

## 문서 결과와 증거 상태

| 구성 | MOTA | IDF1 명칭 | HOTA 명칭 | IDSW 평균 | FP 평균 | FN 평균 | 근거/상태 |
|---|---:|---:|---:|---:|---:|---:|---|
| v2 E0 FP32 ByteTrack | .295 | .495 | .570 | 28 | 210 | 2378 | README.md:437; VERIFIED_DOC_ONLY |
| v2 E1 W8A8 ByteTrack | .291 | .491 | .565 | 44 | 41 | 2647 | README.md:438; VERIFIED_DOC_ONLY |
| v2 E4 W4A16 ByteTrack | .176 | .309 | .424 | 21 | 41 | 2647 | README.md:439; VERIFIED_DOC_ONLY |
| v2 E5 SmoothQuant ByteTrack | .280 | .479 | .558 | 28 | 207 | 2381 | README.md:440; VERIFIED_DOC_ONLY |
| E6 W8A8 BoT-SORT | .068 | .330 | .444 | 2 | 781 | 2551 | README.md:441; VERIFIED_DOC_ONLY; 별도 v1 FPS 주의 |

src/pipeline/eval_e2e.py:62-102 및 scripts/archive/plot_v2_extras.py:34는 위 결과를 하드코딩한다. 검출/추적 출력을 저장한 원본 로그나 frame별 prediction 파일은 미발견이다. _finalize_v2_results.py:4-13,63-80이 기대하는 logs/track_E0_FP32.log 등은 현재 없다. 따라서 .295의 산출을 독립적으로 재계산할 수 없다.

v1 수치는 별도 문서 기록이다: docs/EXPERIMENTS.md:256-267 E0 .219/.384/.487/IDSW0/FPS21.6; E1 .221/.384/.487/FPS24.8; E4 .105/.192/.322/FPS25.7; E5 .225/.387/.490/FPS20.8. src/track/run_tracking_ablation.py:173-185는 v1 E0 값을 기본 삽입하며 eval_botsort.py:149는 v1 E1 값을 고정한다. v2 재실행 결과와 섞일 위험이 있다. README.md:500은 E6 FPS20.4가 v1 시점이라고 명시하므로 v2 CPU E0~E7 동등 비교 표로 사용하면 안 된다.

## 실제 지표 구현: 표준 MOT 평가와 분리해야 함

1. GT loading: src/track/eval_tracking.py:131-154는 traffic_sign/traffic_light만 읽고 둘 다 class0으로 합친다. 주석과 달리 signboard는 추가하지 않는다. v3 신호등 분리/3-class ontology를 그대로 평가하는 코드가 아니다.
2. GT ID는 실제 identity annotation을 읽지 않고 연속 두 frame box 간 greedy IoU≥.4로 생성한다(157-202). 샘플 간 이동/몽타주 컷은 identity가 끊어질 수 있다. 따라서 IDSW는 pseudo-GT에 대한 추정이다.
3. 평가 matching은 IoU≥.5 greedy이며 class 일치 조건이 없다(252-290). Hungarian/global identity matching이 아니다.
4. `IDTP`는 모든 framewise bbox 매칭에서 1씩 증가하고, ID switch가 있어도 그대로 TP로 센다(281-300). 따라서 코드의 IDF1은 **framewise detection F1 proxy**이며 identity 일관성을 평가하는 표준 IDF1로 표기해서는 안 된다.
5. HOTA는 단일 IoU에서 sqrt(DetA × TP/(TP+IDSW))이다(302-305). threshold sweep 및 track-pair association 기반 표준 HOTA가 아니다. **HOTA-like proxy**로 명시하거나 표준 구현으로 재평가해야 한다.
6. MOTA 산식 자체는 1−(FP+FN+IDSW)/GT이지만 pseudo identity/matching의 제약을 함께 밝혀야 한다(299).
7. 전체 결과는 sequence macro average이며 GT/FP/FN/IDSW 카운트도 평균한다(401-405). 표의 평균 카운트를 MOTA 식에 대입하는 micro 재계산과 같지 않다. 평균3386 객체는 고유 객체 수가 아닌 annotation object-instance 평균이다.

이 구현 사실은 VERIFIED_RAW(소스 정적 확인). 표준 지표로 호명한 주장과는 **CONFLICT**이며, 숫자 자체가 조작되었다는 뜻은 아니다.

## 평가 분모의 현행 메타데이터 확인

data/aihub_traffic/test/images/<sequence>/*.jpg와 같은 이름의 labels JSON을 직접 세고, eval_tracking.py:150의 traffic_sign/traffic_light 필터만 적용했다. 이미지 디코딩/모델 실행/annotation 재가공은 하지 않았다.

| sequence | 주야 | JPG 프레임 | 평가 대상 object-instances | JPG 대응 JSON 누락 |
|---|---|---:|---:|---:|
| c_validation_1920_1200_night_1 | 야간 | 16 | 33 | 0 |
| d_validation_1920_1080_daylight_2 | 주간 | 2401 | 6739 | 0 |
| 합 | 2 sequences | **2417** | **6772** | 0 |
| sequence 평균 | | 1208.5 | **3386** | |

현재 metadata는 README.md:429의 GT3386 평균을 지지한다. 과거 실행 manifest가 없으므로 .295가 정확히 이 2417 frame을 평가한 결과라는 연결은 **미확정**이다. 16 frame 야간과 2401 frame 주간에 macro average가 각각50%를 부여하므로 “주야간 균등”은 sequence 수의 균등일 뿐 frame/object 균형이 아니다.

설정(소스 기본값, 과거 실행 인자 확인 아님): YOLOv8s ONNX, imgsz640, conf.25, NMS IoU.45, CPU EP(eval_tracking.py:30-47); ByteTrack track_thresh.5 / match_thresh.8 / track_buffer30 / frame_rate5(378-383). BoT-SORT는 W8A8 detector + reid_net_w8a8.onnx, CMC 및 ReID 옵션, lam.5, alpha.95(eval_botsort.py:28-33,107-116). missing ReID는 None 경로로 처리되므로 파일만으로 과거 ReID 활성 여부를 확정할 수 없다.

추적 FPS는 decode 이후 detect+tracker 구간만 합산, warmup 없음, 전체 이미지 수/시간(324-355). 이미지 로드 실패가 있으면 분자에는 남고 시간에는 빠진다. 평균 FPS도 sequence별 FPS의 산술평균이다. 서버/브라우저 end-to-end FPS와 직접 비교 불가.

## 논문 기여 판단

Tracking은 현재 후보 C1/C2/C3의 보조 파이프라인 요소로 두는 것이 타당하다. 표준 MOT 성능 우위, BoT-SORT 일반적 열세, 시간적 identity 보존의 강한 주장을 뒷받침하지 못한다. MUST(추적 수치 사용 시): 실제 identity GT 또는 pseudo-GT 한계 명시, 표준 IDF1/HOTA, sequence별 raw counts/predictions/평가 frame manifest. SHOULD: 더 많은 독립 주야간 시퀀스 및 aggregate 규칙 명시. 새로운 평가는 이번 범위에서 실행하지 않았다.

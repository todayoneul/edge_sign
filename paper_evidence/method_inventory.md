# 구현 방법 목록 (문헌 대조 준비용)

감사일 2026-09-23. 웹 문헌 검색 및 신규 실험 없이 현 소스와 저장 ONNX만 조사했다. 코드에서 확인한 구현은 VERIFIED_RAW(정적 코드 사실)이며 해당 옵션으로 과거 실험이 실행되었다는 증거는 아니다. 소스의 이름/주석과 실제 연산을 분리한다. 실측 크기 전체는 quantization_evidence.md 및 artifact_manifest.csv를 참조한다.

## 검출·인식·추적 구조

| 구성 | 확인한 방법 | 근거 |
|---|---|---|
| 초기/v2/v3 검출 | Ultralytics YOLOv8s 계열. 현재 fp32 ONNX는 (1,3,640,640) 입력, (1,6,8400) 출력, model.22 Detect 및 DFL 16-bin 적분 경로. v3도 graph의 class channel 수는 2. 문서상 세대별 class 의미는 분리 필요 | model_space/yolov8s_signs{,_v2,_v3}_fp32.onnx graph; src/quant/quantize_yolo.py:33–73 |
| v4 검출 | YOLO26-n 명명, one2one head model.23, 최종 box 4 channels, cls 2 channels, (1,300,6) 출력. 보존 inference graph에는 DFL 적분 없음. Softmax 두 개는 attention. dfl_loss 로그 열로 반박 불가 | model_space/yolo_v4_signs_fp32.onnx graph; scripts/export_v4_variants.py:33–55 |
| KoreanOCRNet | gray 1×64×64 단일 글자 2350-class 분류기. 32채널 일반 conv → depthwise-separable 64/128/256/256 → GAP → dropout .3 → 1×1 conv classifier. 문장 OCR/CTC recognizer 아님 | src/korean_ocr_model.py:5–77 |
| TrafficSignNet | RGB 3×32×32, conv 16/32/64 + GAP + 1×1 conv 64→class. 43-class GTSDB와 14-class 한국 분류는 별개 모델/평가 | src/model.py:14–65; model_space/traffic_sign_net_fp32.onnx / korean_sign_net_fp32.onnx output shapes |
| ByteTrack | 8-dim constant velocity Kalman(x,y,a,h와 속도), IoU 기반 두 단계 high/low confidence 연관. scipy Hungarian 사용 가능 시 사용, fallback greedy. DeepSORT Kalman 파생 출처 명시 | src/track/bytetrack.py:1–17,42–164,368–426,502–618 |
| BoT-SORT 변형 | 프로젝트 재구현, CMC(ORB 특징/호모그래피), IoU+cosine appearance, SimpleReIDNet 128차원. 공식 pretrained ReID와 동등하다고 간주 금지 | src/track/botsort.py:1–15,74–121,165–210,264 이후 |
| ReID export | 새로운 SimpleReIDNet을 생성하고 별도 checkpoint load 없이 weight fake-quant 후 ONNX export. 저장 ReID가 학습된 모델이라는 증거 없음 | src/quant/quantize_reid.py:30–49 |

## 양자화 방식과 명명 차이

| 범위/명명 | 실제 구현 | weight / activation / 저장 | 근거 |
|---|---|---|---|
| Phase1 W8A8 PTQ | Conv/Linear per-output-channel absolute max/127, round/clamp 후 dequantized weight 대입 | W8 grid simulation; activation A8 양자화 없음 | src/base_W8A8.py:28–58 |
| Phase1 W4A16 QAT | RoundSTE forward round, backward identity. per-channel max/7, clamp [-8,7]. Conv/Linear forward에서 fake quant weight 사용 | 학습 QAT 구현 존재; 실제 packed INT4 배포와 다름 | src/base_train_w4a16_qat.py:29–67; logs/training_log_w4a16.csv |
| Phase1 1-bit KD | BinarySTE sign (0→+1), backward에서 abs(weight)>1 gradient=0; channel mean(abs(W)) scale. KD 학습 경로 존재 | 학습 중 이진 weight simulation. 파일 .pth/.safetensors 길이는 1-bit 이론 크기와 구별 | src/base_train_1bit_kd.py:46–84; logs/training_log_1bit.csv |
| 검출기 W8A8 | per-output-channel absmax/127, round[-128,127], float weight 대입 | W8 only fake-quant; FP32 activation/export, Q/DQ 없음 | src/quant/quantize_yolo.py:137–159; saved *_w8a8.onnx |
| 검출기 W4A16 | absmax/7, round[-8,7], float weight 대입 | W4 simulation; A32 실행 그래프, packed W4/A16 아님 | src/quant/quantize_yolo.py:185–209; export half=False :67 |
| 검출기 SmoothQuant 명명 | activation hook으로 input channel max, alpha=.5 scaling s=a_max^alpha/w_max^(1-alpha), W에 s 흡수 + output-channel W8 rounding, wrapper에서 x/s | smoothing adaptation + weight fake quant; A8 quantizer 없음. 그룹/채널 불일치시 truncation/padding 구현 포함 | src/quant/quantize_yolo.py:242–383; src/multimodal_w8a8_smoothquant.py:34–149 |
| OCR/TrafficSign W8/W4 | Conv/Linear 전부 per-output-channel minmax fake-quant | FP32 initializer, A32, 전체 FP32와 동일 파일 길이 | src/quant/quantize_recognizers.py:51–95; model_space/*_net_w8a8.onnx / *_w4a16.onnx |
| OCR/TrafficSign 1-bit | sign(W)×per-output-channel mean(abs(W)); zero sign→+1 | **PTB/PTQ**, QAT나 KD 아님. FP32 저장/연산 | src/quant/quantize_recognizers.py:98–115 |
| real static INT8 | ORT quant_pre_process + quantize_static QDQ; per_channel=True, weight QInt8, activation QUInt8, reduce_range=False | int8 initializer 및 Q/DQ 실제 존재; kernel/runtime 성능은 별도 증거 필요 | scripts/archive/quantize_onnx_real.py:175–238; scripts/quantize_v3_detector.py:114–146 |
| v3 head exclusion | /model.22 prefix 전체 제외 기본, --quant_head면 exclude=[] | FLOAT head + QDQ backbone/neck. v3 saved graph head Q/DQ=0 | scripts/quantize_v3_detector.py:37,83–107; saved v3 QDQ graph |
| v4 head exclusion | preprocessed graph 이름에서 최대 model index 선택(model.23), 포함 노드 전체 제외 옵션 | fullhead: head Q83/DQ133, INT8 initializer48; excluded: 모두0 | scripts/export_v4_variants.py:80–105; saved full/excluded graph |
| v3 FP16 | post-hoc 변환 경로. 실제 saved FLOAT16 initializer131 + FLOAT1 | 22.382483 MB, DFL 경로 유지 | scripts/export_fp16_detector.py; model_space/yolov8s_signs_v3_fp16.onnx |
| v4 FP16 | native YOLO export half=True device=0. 코드 주석상 post-hoc 변환 mixed Concat 오류 회피 | FLOAT16 initializer208+FLOAT1; 4.971834 MB | scripts/export_v4_variants.py:40–55; saved FP16 graph |
| Dynamic INT8 | saved *_int8_dyn는 ConvInteger 존재. Korean14 *_w8a8도 실제 dynamic ConvInteger 5개로 다른 W8 명명과 다름 | 이름만으로 quantization family 선택 금지 | model_space/* graph; src/quantize_int8.py:18–24 |

_is_quantizable()는 “Detection Head 제외”라고 쓰지만 실제로는 이름에 dfl 또는 detect가 있는 경우만 제외한다(src/quant/quantize_yolo.py:52–58). model.22.cv2/cv3 또는 model.23.one2one_cv2/cv3 이름 자체는 이 조건에 걸리지 않는다. 따라서 fake-quant 실험을 head-excluded로 일괄 표기하면 안 된다. 이와 별도로 static QDQ head exclusion은 저장 graph에서 확인된다.

## Calibration/평가 조건 (기본값과 실행 증거 분리)

| 경로 | 소스상 기본 선택·전처리 | 과거 실행 검증 수준 |
|---|---|---|
| 초기 real QDQ detector | YoloCalibReader n=80, TEST_DIR 아래 정렬 frame, resize640 RGB [0,1]. 없으면 random uint8 dummy | manifest/seed별 실행로그 미확보. 테스트 데이터를 calibration에 사용 가능한 코드로 evaluation 독립성 점검 필요 |
| 초기 real QDQ OCR | OcrCalibReader n=200, 정렬 class의 jpg 최대10개씩, gray64 [0,1], 없으면 random | OCR evaluator는 Normalize(.5,.5)로 [-1,1] 사용. calibration와 평가 전처리 차이 존재; 실제 실행 경위 미확보 |
| 초기 real QDQ TS | GTSDB gt.txt shuffle seed42, 최대200 crop, RGB32 [-1,1] | calibration/eval crop ID overlap manifest 미확보 |
| v3/v4 static | data/yolo_signs_v2/images/val 앞 정렬 JPG150장(default), 직접 resize640 RGB [0,1], per-channel QInt8 weights/QUInt8 activation | 실제 calibration frame IDs/hash 없음. v4 parity도 같은 디렉터리 앞 n장이라 기본 실행 시 calibration/eval 중복 가능 |
| SmoothQuant detector | data/yolo_signs/images/val 재귀 정렬, 10 batches×4(default), shuffle=False, RGB640 [0,1], alpha=.5 | 실제 당시 argv/inputs 미확보 |
| OCR quant 평가 | data/korean_ocr/val, numeric class sort, 앞5000(default), batch256, gray64 Normalize(.5,.5), CPU ORT | raw counters/predictions 없음. doc-only 정확도 |
| GTSDB quant 평가 | 전체 crop 20% random_split seed42, CPU ORT | FP32 best checkpoint scalar만 raw, 양자화 raw 결과 없음 |
| v4 parity | val 정렬 JPG 앞20(default; 문서25), conf>.25, CPU ORT, per-frame 평균 conf를 zero-frame 포함해 평균 | 실행 출력 파일 미발견; 문서1.7/0/1.7 task metric 확정 불가 |

출처: scripts/archive/quantize_onnx_real.py:32–36,53–161; scripts/quantize_v3_detector.py:41–65,91–94; scripts/export_v4_variants.py:58–105; src/quant/quantize_yolo.py:227–251; src/quant/quantize_recognizers.py:268–381; scripts/eval_v4_parity.py:28–81.

## Export·평가 지표

model_space 전체 ONNX의 표준 ai.onnx opset은 14로 확인했다(ORT 전처리 파일은 추가 domain opset 포함). 인식기 export는 TorchScript dynamo=False(src/quant/quantize_recognizers.py:142–153), SmoothQuant 직접 export도 dynamo=False(src/quant/quantize_yolo.py:96–110). Ultralytics export 호출부는 opset14를 지정하나 API 내부 exporter 버전은 별도 환경 의존이다.

Phase1 Final Score는 logs/final_score_report.txt에 saved table이 있으며 SmoothQuant .8068은 해당 문서성 집계 파일의 실제 값이다. 그러나 saved score table의 Memory(MB)를 모델 disk bytes 또는 peak runtime memory로 자동 동일시할 수 없다. Phase2 eval_e2e.py:105–138은 OCR/TS accuracy와 이론 크기를 상수로 정의하고, :257–276은 이 상수를 이용해 Final Score를 산출한다. 그러므로 그 표는 독립 recognition 평가 또는 실제 모델 총 byte 측정이 아니다. 서로 다른 세대의 normalization/모델/태스크를 합산 비교하지 않는다.

## 원인 연구와 재현성 한계

analyze_quant_collapse.py는 모델 weight tensor 분석 + 난수 합성 DFL/CosSim 사례를 생성하는 data-free 진단이다. 실제 activation distribution을 수집한 분석으로 명명하면 안 된다. head exclusion 관찰만으로 weight 가설을 완전히 기각하거나 activation 단일 원인을 확증할 수 없다. 최소 통제: weight-only/activation-only, box/cls head별 QDQ, 동일 calibration/holdout manifest와 raw output 보존. 이 감사에서는 수행하지 않았다.

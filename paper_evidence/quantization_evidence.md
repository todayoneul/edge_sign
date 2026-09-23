# 양자화·인식 근거 감사

감사일: 2026-09-23. 신규 학습·양자화·추론·벤치마크 없이 소스/문서/저장 아티팩트만 읽었다. MB=10^6 bytes, MiB=2^20 bytes. ONNX 구조 검사는 convnext_env Python의 onnx.load(load_external_data=False)로 수행했다. VERIFIED_RAW는 아래 구조·파일 길이의 직접 관찰이며 정확도 실행을 뜻하지 않는다. 체크포인트는 임의 pickle 실행 없이 weights_only=True 읽기만 허용했다.

## 판정 요약

- YOLOv8 v3 CosSim=0.9995·검출 0·헤드 제외 후 11/12 일치: 문서 기록만 발견. 원 프레임 ID별 출력/conf, full-head v3 모델, 실행 로그 미확보. 현재 v3 파일은 헤드 제외된 QDQ임을 직접 확인했다.
- YOLO26 full/head-excluded 두 파일과 실제 QDQ 차이는 확인된다. 그러나 평균 검출수 1.7→0.0→1.7은 저장 raw 패리티 출력이 없어 VERIFIED_DOC_ONLY.
- **구조 충돌:** YOLO26가 dfl_loss 이름 때문에 DFL을 유지한다는 README:722 / docs/EXPERIMENTS.md:399 / scripts/export_v4_variants.py:68 주장은 실제 ONNX와 맞지 않는다. /model.23 최종 box conv 출력은 4채널이고, DFL 적분 경로가 없다. Softmax 2개는 /model.10 및 /model.22의 attention이며 검출 헤드 밖이다.
- OCR 98.4/54.6/0.3%는 문서와 eval_e2e.py의 상수이다. 평가 구현 존재가 실행 결과를 입증하지 않는다.
- W8A8/W4A16/1-bit 명명 fake-quant 파일은 FLOAT 가중치/실행 그래프이다. 진짜 A8/A16 활성화 또는 packed INT4/1-bit 배포로 간주할 수 없다.

## A. YOLOv8 v3 헤드 붕괴

| 항목 | 기록/직접 관찰 | 상태 | 출처 |
|---|---|---|---|
| 전체 tensor CosSim | 0.9995 | VERIFIED_DOC_ONLY | docs/EXPERIMENTS.md:366–370 |
| full-head INT8 검출 | 0 | VERIFIED_DOC_ONLY | docs/EXPERIMENTS.md:369–370; README.md:48 |
| baseline / 회복 | 헤드 제외 11/12 박스 일치, conf 약 0.40; 원 프레임·matching 규칙 없음 | VERIFIED_DOC_ONLY | docs/EXPERIMENTS.md:366 |
| saved v3 QDQ | /model.22 명칭 Q/DQ 0개; DFL FLOAT 경로 유지 | VERIFIED_RAW | model_space/yolov8s_signs_v3_int8_static.onnx graph |
| full-head v3 artifact/log | 별도 보존 파일 미발견. 현재 quantize_v3_detector.py:98–99는 같은 출력에 덮어쓴다 | MISSING | model_space/; scripts/quantize_v3_detector.py |

현재 saved generic yolov8s_signs_int8_static.onnx에는 model.22 QDQ가 존재하지만 **v3 풀헤드 반례로 대체할 수 없다**. 모델 세대/가중치가 다른 파일이다. CosSim 계산 구현은 scripts/archive/quantize_onnx_real.py:284 이후, v3의 verify 호출은 scripts/quantize_v3_detector.py:150 부근에 있다. 코드만으로 0.9995의 실측 경위/입력을 확정하지 못한다.

## B. YOLO26/v4 구조와 패리티

| variant | doc avg_det | doc avg_conf | 실제 bytes | 헤드 Q/DQ | 헤드 INT8 initializer 수 |
|---|---:|---:|---:|---|---:|
| yolo_v4_signs_fp32.onnx | 1.7 | .564 | 9805975 | 0/0 | 0 |
| yolo_v4_signs_int8_static.onnx | 0.0 | .000 | 3098518 | 83/133 | 48 |
| yolo_v4_signs_int8_head_excluded.onnx | 1.7 | .525 | 3416372 | 0/0 | 0 |
| yolo_v4_signs_w4a16.onnx | .6 | .165 | 9805975 | 0/0 | 0 |

패리티 숫자 출처: docs/EXPERIMENTS.md:390–397 (25 frames, conf>0.25). **숫자는 VERIFIED_DOC_ONLY**, bytes/구조만 VERIFIED_RAW. scripts/eval_v4_parity.py:52–81은 정렬된 val JPG 앞 n장을 640×640 직접 resize하고 CPUExecutionProvider에서 conf>0.25를 센다. avg_conf는 각 프레임의 통과 박스 평균을 구한 뒤 빈 프레임을 0으로 포함해 다시 평균한다. 따라서 W4의 0.165가 임계값보다 낮아도 코드와 모순이 아니다. FP16의 raw task metric도 없다.

scripts/export_v4_variants.py:25,58–109는 제외 옵션을 줘도 INT8 출력명을 _int8_static.onnx로 고정한다. 현재 _int8_head_excluded.onnx 보존 파일의 별도 rename/copy 경위와 실행 config/hash manifest는 미확보. 파일 구조는 의도한 제외 차이를 확인하지만 실행 이력은 복원되지 않는다.

YOLO26 FP32 graph: /model.23/one2one_cv2.{0,1,2}/one2one_cv2.{0,1,2}.2/Conv의 initializer shape=[4,16,1,1]. YOLOv8의 16-bin×4=64 box channels 및 /model.22/dfl/Softmax→dfl/conv/Conv와 구별된다. 로그 열 dfl_loss는 구조 검증을 대신하지 못한다. 이 감사는 **보존된 inference graph에서 DFL 적분이 없음을 확인**한 것이며 학습 loss 구현 전체를 분석한 결과는 아니다.

## C. 원인 해석의 한계

scripts/analyze_quant_collapse.py:73–121의 weight SQNR/첨도는 가중치 정적 분석이고 :125–176의 DFL 잡음 및 CosSim 예시는 난수 합성 실험이다(seed=0, :35). 저장된 assets/v3/quant_collapse_analysis.png는 그림이지 실제 모델의 프레임별 activation dump가 아니다. 본 감사에서 이 스크립트는 실행하지 않았다. 독립적인 weight-only/activation-only/head-only 통제가 없으므로 “활성화가 원인으로 확증”, “가중치 원인 기각”, “DFL가 진짜 벽”, “모델 교체로 해결 불가”는 과도하다. 안전한 표현: “문서에 기록된 헤드 INT8 민감도 관찰을 설명하는 활성화/임계값 가설이며, 원인 규명을 위한 통제 실험과 원 출력 보존이 필요하다.”

## D. 인식 정확도

| model / metric | FP32 | W8A8 명명 | W4A16 명명 | 1-bit 명명 | 상태 |
|---|---:|---:|---:|---:|---|
| KoreanOCRNet Top-1 (%) | 98.5 | 98.4 | 54.6 | 0.3 | VERIFIED_DOC_ONLY |
| TrafficSignNet 43-class Top-1 (%) | 62.8 | 63.2 | 49.2 | 12.8 | 양자화 값 VERIFIED_DOC_ONLY; FP32 checkpoint val_acc=0.628099173553719 직접 확인 |
| Korean TrafficSignNet 14-class | 80.3 문서 | raw 값 없음 | variant 없음 | variant 없음 | VERIFIED_DOC_ONLY |

근거: docs/EXPERIMENTS.md:96–104,209; src/pipeline/eval_e2e.py:105–125 하드코딩. model_space/traffic_sign_net_best.pth의 안전한 weights_only 읽기에서 epoch=49, num_classes=43, val_acc=0.628099173553719 확인(저장 scalar이며 per-sample raw prediction 없음). korean_sign_net_best.pth는 weights_only=True에서 허용되지 않는 pickle 객체가 있어 내용을 강제 로드하지 않았다. 80.3은 이 감사 범위에서 raw로 승격하지 않는다.

평가 구현: src/quant/quantize_recognizers.py:268–325는 OCR val NumericalImageFolder의 앞 5000 샘플(default)을 순서대로 평가한다. 전체 2350-class를 균형 표집한 평가가 아님에 유의. :333–381은 GTSDB 전체 crop의 20%를 seed=42 random_split한다. 코드의 평가 조건과 문서 숫자가 같은 실행인지 확인할 로그/ID 목록은 없다. 43-class GTSDB 62.8%와 한국 14-class 80.3%는 NOT_COMPARABLE.

## E. 실제 저장 크기와 그래프

모든 model_space ONNX를 직접 확인했다. 아래 FLOAT/INT 타입 수는 initializer 개수이며 파라미터 수가 아니다. Q/DQ는 전체 노드 개수. dynamic ConvInteger의 지원/속도는 이 구조 검사로 판단하지 않는다.

| file | bytes | MB | MiB | initializer dtypes(count) | Q / DQ / ConvInteger |
|---|---:|---:|---:|---|---|
| model_space/_spike/yolo26n.onnx | 9941944 | 9.941944 | 9.481377 | {'FLOAT': 209, 'INT64': 19} | 0/0/0 |
| model_space/korean_ocr_net_1bit.onnx | 2880185 | 2.880185 | 2.746758 | {'FLOAT': 20} | 0/0/0 |
| model_space/korean_ocr_net_fp32.onnx | 2880185 | 2.880185 | 2.746758 | {'FLOAT': 20} | 0/0/0 |
| model_space/korean_ocr_net_int8_dyn.onnx | 749365 | 0.749365 | 0.714650 | {'FLOAT': 20, 'INT8': 20, 'INT64': 10} | 0/0/10 |
| model_space/korean_ocr_net_int8_static.onnx | 796892 | 0.796892 | 0.759975 | {'UINT8': 12, 'FLOAT': 32, 'INT8': 20, 'INT32': 20} | 16/36/0 |
| model_space/korean_ocr_net_w4a16.onnx | 2880185 | 2.880185 | 2.746758 | {'FLOAT': 20} | 0/0/0 |
| model_space/korean_ocr_net_w8a8.onnx | 2880185 | 2.880185 | 2.746758 | {'FLOAT': 20} | 0/0/0 |
| model_space/korean_sign_net_fp16.onnx | 59743 | 0.059743 | 0.056975 | {'FLOAT16': 10} | 0/0/0 |
| model_space/korean_sign_net_fp32.onnx | 116860 | 0.116860 | 0.111446 | {'FLOAT': 10} | 0/0/0 |
| model_space/korean_sign_net_w8a8.onnx | 37789 | 0.037789 | 0.036038 | {'FLOAT': 10, 'INT8': 10, 'INT64': 5} | 0/0/5 |
| model_space/reid_net_int8_dyn.onnx | 75736 | 0.075736 | 0.072227 | {'FLOAT': 12, 'INT8': 14, 'INT64': 4} | 0/0/6 |
| model_space/reid_net_w8a8.onnx | 254465 | 0.254465 | 0.242677 | {'FLOAT': 12} | 0/0/0 |
| model_space/traffic_sign_net_1bit.onnx | 125814 | 0.125814 | 0.119986 | {'FLOAT': 10} | 0/0/0 |
| model_space/traffic_sign_net_fp32.onnx | 125814 | 0.125814 | 0.119986 | {'FLOAT': 10} | 0/0/0 |
| model_space/traffic_sign_net_int8_dyn.onnx | 40485 | 0.040485 | 0.038610 | {'FLOAT': 10, 'INT8': 10, 'INT64': 5} | 0/0/5 |
| model_space/traffic_sign_net_int8_static.onnx | 44807 | 0.044807 | 0.042731 | {'UINT8': 7, 'FLOAT': 17, 'INT8': 10, 'INT32': 10} | 10/20/0 |
| model_space/traffic_sign_net_w4a16.onnx | 125814 | 0.125814 | 0.119986 | {'FLOAT': 10} | 0/0/0 |
| model_space/traffic_sign_net_w8a8.onnx | 125814 | 0.125814 | 0.119986 | {'FLOAT': 10} | 0/0/0 |
| model_space/yolo_v4_signs_fp16.onnx | 4971834 | 4.971834 | 4.741510 | {'FLOAT16': 208, 'INT64': 19, 'FLOAT': 1} | 0/0/0 |
| model_space/yolo_v4_signs_fp32.onnx | 9805975 | 9.805975 | 9.351707 | {'FLOAT': 209, 'INT64': 19} | 0/0/0 |
| model_space/yolo_v4_signs_int8_head_excluded.onnx | 3416372 | 3.416372 | 3.258106 | {'INT64': 19, 'FLOAT': 473, 'UINT8': 266, 'INT8': 156, 'INT32': 156} | 303/460/0 |
| model_space/yolo_v4_signs_int8_static.onnx | 3098518 | 3.098518 | 2.954977 | {'INT64': 19, 'FLOAT': 544, 'UINT8': 341, 'INT8': 204, 'INT32': 204} | 387/594/0 |
| model_space/yolo_v4_signs_w4a16.onnx | 9805975 | 9.805975 | 9.351707 | {'FLOAT': 209, 'INT64': 19} | 0/0/0 |
| model_space/yolov8s_signs_fp32.onnx | 44747973 | 44.747973 | 42.674993 | {'FLOAT': 132, 'INT64': 12} | 0/0/0 |
| model_space/yolov8s_signs_int8_dyn.onnx | 11491070 | 11.491070 | 10.958738 | {'FLOAT': 132, 'INT64': 75, 'INT8': 128} | 0/0/64 |
| model_space/yolov8s_signs_int8_static.onnx | 11657516 | 11.657516 | 11.117474 | {'INT64': 12, 'FLOAT': 344, 'UINT8': 216, 'INT8': 128, 'INT32': 126} | 243/372/0 |
| model_space/yolov8s_signs_smoothquant.onnx | 44761494 | 44.761494 | 42.687887 | {'FLOAT': 194, 'INT64': 13} | 0/0/0 |
| model_space/yolov8s_signs_v2_fp32.onnx | 44747973 | 44.747973 | 42.674993 | {'FLOAT': 132, 'INT64': 12} | 0/0/0 |
| model_space/yolov8s_signs_v3_fp16.onnx | 22382483 | 22.382483 | 21.345599 | {'FLOAT16': 131, 'INT64': 12, 'FLOAT': 1} | 0/0/0 |
| model_space/yolov8s_signs_v3_fp32.onnx | 44747495 | 44.747495 | 42.674537 | {'FLOAT': 132, 'INT64': 12} | 0/0/0 |
| model_space/yolov8s_signs_v3_int8_static.onnx | 17997724 | 17.997724 | 17.163967 | {'INT64': 12, 'FLOAT': 285, 'UINT8': 153, 'INT8': 90, 'INT32': 90} | 174/264/0 |
| model_space/yolov8s_signs_w4a16.onnx | 44747973 | 44.747973 | 42.674993 | {'FLOAT': 132, 'INT64': 12} | 0/0/0 |
| model_space/yolov8s_signs_w8a8.onnx | 44747973 | 44.747973 | 42.674993 | {'FLOAT': 132, 'INT64': 12} | 0/0/0 |

### 크기 주장 충돌

- README.md:54,80–91 및 :466의 “FP32 총 22.3 MB → INT8 총 11.7 MB”는 현재 E0/E3 파일 조합의 실제 크기와 맞지 않는다. E0 generic detector+OCR+TS = **47,753,972 bytes (47.753972 MB)**. all-static 조합=**12,499,215 bytes (12.499215 MB)**이며 11.657516 MB는 검출기 단독이다. static 파일의 task metric이 fake-quant accuracy와 같은지도 입증되지 않았다.
- v3 head-excluded detector 단독=17.997724 MB로 이미 15 MB 초과. OCR static+TS static 포함=18.839423 MB. v3 한국 14-class FP32 recognizer 조합의 별도 총량은 18.114584 MB (OCR 제외). 따라서 초기 파일을 근거로 현 v3 전체 15 MB 목표 달성을 주장할 수 없다.
- README.md:400의 fake-quant “42.7 MB”는 실제 44.747973 MB ≈42.675 MiB와 단위 혼용. README.md:91의 ~44.7 MB는 decimal 기준에 맞는다.
- src/pipeline/eval_e2e.py:128–138의 21.5/5.4/2.7 및 OCR .69/.17/.09 MB는 실제 저장 bytes를 읽지 않는 상수. 이론 포맷 환산이라 해도 FP32 저장 bytes와 큰 차이가 있어 현재 아티팩트와 일대일 대응하지 않는다.
- README.md:466의 21.5+2.7+0.12≈22.3은 문서 내부 산술도 불일치한다(합 24.32).

### models/ 및 checkpoints/ 전체 파일 길이

아래에는 가중치뿐 아니라 config JSON과 외부 tensor 파일을 포함하여 실제 존재 파일을 열거한다. 체크포인트에는 optimizer/state 등이 포함될 수 있으므로 배포 모델 크기로 해석하지 않는다. 외부 데이터 ONNX는 .onnx와 .onnx.data를 합쳐야 한다.

| file | bytes | decimal MB |
|---|---:|---:|
| models/hf_1bit_model/config.json | 87 | 0.000087 |
| models/hf_1bit_model/model.safetensors | 2095724 | 2.095724 |
| models/hf_mm_1bit_model/config.json | 288 | 0.000288 |
| models/hf_mm_1bit_model/model.safetensors | 30637680 | 30.637680 |
| models/hf_w4a16_model/config.json | 180 | 0.000180 |
| models/hf_w4a16_model/model.safetensors | 15641016 | 15.641016 |
| models/hf_w8a8_model_convnextv2_nano.fcmae_ft_in1k/config.json | 102 | 0.000102 |
| models/hf_w8a8_model_convnextv2_nano.fcmae_ft_in1k/model.safetensors | 15641016 | 15.641016 |
| models/hf_w8a8_smoothquant/smoothquant_w8a8.pth | 30755607 | 30.755607 |
| models/korean_ocr.onnx | 33300 | 0.033300 |
| models/korean_ocr.onnx.data | 2930688 | 2.930688 |
| models/korean_ocr_best.pth | 2901044 | 2.901044 |
| models/korean_ocr_quant.onnx | 758419 | 0.758419 |
| models/yolov8n/yolov8n.pt | 6549796 | 6.549796 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_1.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_10.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_11.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_12.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_13.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_14.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_15.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_16.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_17.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_18.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_19.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_2.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_20.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_21.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_22.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_23.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_24.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_25.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_26.pth | 93946875 | 93.946875 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_27.pth | 93946875 | 93.946875 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_28.pth | 93946875 | 93.946875 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_29.pth | 93946875 | 93.946875 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_3.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_30.pth | 93946875 | 93.946875 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_4.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_5.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_6.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_7.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_8.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_1bit/qat_1bit_epoch_9.pth | 93944501 | 93.944501 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_1.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_10.pth | 92067701 | 92.067701 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_11.pth | 92069429 | 92.069429 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_12.pth | 92069429 | 92.069429 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_13.pth | 92069429 | 92.069429 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_14.pth | 92069429 | 92.069429 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_15.pth | 92069429 | 92.069429 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_2.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_3.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_4.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_5.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_6.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_7.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_8.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_9.pth | 92067055 | 92.067055 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_1.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_10.pth | 31510103 | 31.510103 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_11.pth | 31510103 | 31.510103 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_12.pth | 31510103 | 31.510103 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_13.pth | 31510103 | 31.510103 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_14.pth | 31510103 | 31.510103 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_15.pth | 31510103 | 31.510103 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_2.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_3.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_4.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_5.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_6.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_7.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_8.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_9.pth | 31509933 | 31.509933 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_1.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_10.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_11.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_12.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_13.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_14.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_15.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_2.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_3.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_4.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_5.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_6.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_7.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_8.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_9.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_1.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_10.pth | 30684475 | 30.684475 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_11.pth | 30684475 | 30.684475 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_12.pth | 30684475 | 30.684475 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_13.pth | 30684475 | 30.684475 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_14.pth | 30684475 | 30.684475 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_15.pth | 30684475 | 30.684475 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_2.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_3.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_4.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_5.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_6.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_7.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_8.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_9.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_1.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_10.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_11.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_12.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_13.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_14.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_15.pth | 30684309 | 30.684309 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_2.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_3.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_4.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_5.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_6.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_7.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_8.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_9.pth | 30684143 | 30.684143 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_1.pth | 31310011 | 31.310011 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_10.pth | 93945793 | 93.945793 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_11.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_12.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_13.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_14.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_15.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_16.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_17.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_18.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_19.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_2.pth | 31310011 | 31.310011 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_20.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_21.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_22.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_23.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_24.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_25.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_26.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_27.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_28.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_29.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_3.pth | 31310011 | 31.310011 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_30.pth | 93947521 | 93.947521 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_4.pth | 31310011 | 31.310011 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_5.pth | 31310011 | 31.310011 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_6.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_7.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_8.pth | 93945147 | 93.945147 |
| checkpoints/checkpoints_w4a16/qat_w4a16_epoch_9.pth | 93945147 | 93.945147 |
| checkpoints/vlm_projection/projection_head_epoch_1.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_10.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_11.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_12.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_13.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_14.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_15.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_16.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_17.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_18.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_19.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_2.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_20.pth | 1314925 | 1.314925 |
| checkpoints/vlm_projection/projection_head_epoch_3.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_4.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_5.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_6.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_7.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_8.pth | 1314917 | 1.314917 |
| checkpoints/vlm_projection/projection_head_epoch_9.pth | 1314917 | 1.314917 |

## F. 논문 사용 조건

구조·저장크기 차이는 현 아티팩트로 입증 가능하다. 정확도 민감도 및 head collapse는 현재 문서 수치를 provisional로만 유지한다. MUST: 동일 holdout frame IDs에서 FP32/full-head/head-excluded 출력·confidence·정답 비교·calibration manifest·모델 hash를 보존, OCR 4variant 동일 샘플 평가 raw counters 및 예측 저장. SHOULD: activation-only/weight-only와 cv2/cv3 분리 ablation, activation range/outlier 통계, 독립 calibration/evaluation 분리. 이 감사에서는 추가 실험을 실행하지 않았다.

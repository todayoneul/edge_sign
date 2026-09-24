# Edge-Sign TIIS evidence 재검증 보고서

기준 문서: 첨부 `Edge-Sign_개정원고_v2_검토본.pdf` 9쪽 전체. 평가일: 2026-09-24. 작업 브랜치: `paper/tiis-evidence-revalidation` (base `095f3cd`). 아래 수치는 이 브랜치의 원시 artifact에서 다시 읽은 값이다. 별도 표시가 없는 mAP는 **export된 YOLO26 ONNX의 독립 test** 결과이다.

## 1. Executive Summary

- AI Hub 영상의 sequence가 겹치지 않도록 train 12,375장, calibration 150장, test 2,417장을 확정했다. Test는 6,772개 객체(표지판 3,366, 신호등 3,406)를 포함한다. 밤은 **16장뿐**이다.
- 동일 test/640×640 stretch 전처리/ORT CPU/후처리에서 FP32 mAP@0.5 **0.499865**, full static INT8 QDQ **0**, head-excluded QDQ **0.482600**이다. Full QDQ는 2,417장 전체에서 confidence ≥0.001 검출도 0건이었다. Head 제외는 검출을 되살리지만 FP32보다 mAP@0.5가 0.017266 낮다.
- CPU detector-only 평균은 FP32 **16.115 ms**, head-excluded QDQ **21.553 ms**였다. 이 환경에서는 QDQ가 CPU 가속을 주지 않았다. 브라우저 WASM head-excluded QDQ는 **9.285 FPS**, WebGPU QDQ는 커널 오류로 `unsupported`였다.
- 한국 도로용 YOLO26 head-excluded QDQ + KoreanSignNet FP32의 실제 파일 크기는 **3,533,232 B**다. 2,417장 순차 파이프라인은 메모리 프레임 기준 **44.121 FPS**, 로컬 JPEG 디코딩 포함 **33.269 FPS**다. 카메라/영상 디코딩, 전송, 화면 렌더까지 포함한 배포 30 FPS는 검증하지 않았다.
- 독립 test의 수동 GT 박스에서 추출한 KoreanSignNet 14-class ROI 6,771개에 대한 FP32 Top-1은 **0.808152**다. 검출과 추적을 통과한 파이프라인의 fine-class 정답은 전체 매핑 가능 GT의 **0.353419**였다. 두 지표는 평가 단위가 다르다.
- Test JSON에는 프레임 간 identity ID가 없어 기존 pseudo-GT MOTA/IDF1/HOTA를 본 논문의 주 정량 결과에서 제외한다. 15 MB/30 FPS/accuracy retention **동시 달성은 아직 확정할 수 없다**.

## 2. Experiment Environment

| 항목 | 기록 |
|---|---|
| OS | Windows build 26200, AMD64 |
| CPU | AMD Family 26 Model 68, 12 logical CPUs; ORT intra-op 4 threads |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB; driver 591.86 |
| Python / ORT / ONNX | 3.10.19 / 1.23.2 / 1.21.0 |
| Browser | HeadlessChrome 153.0.0.0, ORT Web 1.22.0; WebGPU adapter detail returned `{}` |
| Browser WASM | single thread, `crossOriginIsolated=false` |
| CUDA EP | `unsupported`: CUDA 세션이 CPU로 폴백. 현재 환경에서 `cublasLt64_12.dll` 로드 실패. GPU 결과로 표시하지 않음 |

원문은 [system_info.txt](../environment/system_info.txt), [python_packages.txt](../environment/python_packages.txt), 브라우저 JSON의 `user_agent`/`webgpu_adapter`에 있다. 모든 모델의 실제 bytes, SHA-256, opset, I/O shape, QDQ 노드 수는 [model_manifest.csv](../models/model_manifest.csv)에서 확인한다. 모델명과 export 스크립트는 계보를 시사하지만, **checkpoint→ONNX의 해시 연결을 저장한 당시 export log는 없어 정확한 계보를 암호학적으로 입증하지 못한다**.

## 3. Dataset Split

| Split | 이미지 | 객체 | 표지판 | 신호등 | sequence | 주간 / 야간 |
|---|---:|---:|---:|---:|---:|---:|
| train | 12,375 | 43,677 | 21,637 | 22,040 | 5 | 12,250 / 125 |
| calibration | 150 | 621 | 319 | 302 | 1 | 150 / 0 |
| test | 2,417 | 6,772 | 3,366 | 3,406 | 2 | 2,401 / 16 |

Train은 `data/yolo_signs_v2/images/train`, calibration은 sorted `val` 첫 150장(기존 v4 양자화 스크립트의 선택 방식), test는 `data/aihub_traffic/test`의 두 전체 sequence다. Train/calibration/test 간 sequence 이름과 JPEG SHA-256이 모두 달랐다. Random seed를 사용하지 않았다. 원본 이미지 복사 없이 [manifest](../splits/)에 경로, 해시, 클래스, 박스, sequence, 주야간을 저장했다. Test JSON 누락/손상으로 제외된 프레임은 0장이다.

이는 같은 AI Hub 수집원의 **sequence 독립 분할**이다. 촬영 위치·도로·영상 원천까지 외부 독립 데이터라는 증거는 없다. Calibration이 주간 한 sequence에만 치우쳐 있고 야간 test가 16장이라 야간 일반화 결론은 약하다.

## 4. Detection Quantization

YOLO26-n 세 변형은 모두 input `[1,3,640,640]`, output `[1,300,6]`(xyxy/conf/class), 동일한 RGB/255 stretch 전처리와 내장 top-300 후처리를 사용했다. 추가 NMS는 없다. AP에는 confidence ≥0.001, 아래 precision/recall/count에는 confidence ≥0.25와 IoU ≥0.5를 적용했다. AP는 Ultralytics 호환 101-point precision-envelope trapezoid 방식이며, 저장된 예측 60개 class×IoU 조합을 Ultralytics 계산과 대조해 최대 절대 오차 **0**을 확인했다([AP_CROSSCHECK.json](AP_CROSSCHECK.json)). 최초 평균 보간 방식의 수치는 `metrics_initial_101mean.json`에 보존했다.

| 모델 | 실제 파일 B | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 검출 수 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 9,805,975 | 0.499865 | 0.242181 | 0.754608 | 0.423213 | 3,798 |
| full static INT8 QDQ | 3,098,518 | 0 | 0 | 0 | 0 | 0 |
| head-excluded static INT8 QDQ | 3,416,372 | 0.482600 | 0.235028 | 0.760033 | 0.402688 | 3,588 |

Head-excluded의 mAP@0.5 차이는 **-0.017266**, mAP@0.5:0.95 차이는 **-0.007153**, recall 차이는 **-0.020526**이다. ONNX graph의 `model.23` Q/DQ 노드는 full QDQ에 **216개**, head-excluded QDQ에 **0개**라 저장된 파일의 head 제외 여부도 구조적으로 확인했다. QDQ 모델의 검출 박스가 같은 클래스 FP32 박스와 이루는 최고 IoU 평균은 **0.903653**(confidence ≥0.25)이다. 이는 one-to-one match가 아닌 보조 패리티 지표다. 각 프레임의 GT, 박스, 클래스, confidence, FP32 대비 최고 IoU, raw output cosine/SQNR을 [predictions.jsonl](../detection/yolo26_head_excluded_qdq/predictions.jsonl)에 보존했다. FP32와 full QDQ에도 같은 형식의 파일이 있다.

전체 raw output cosine의 평균은 full QDQ **0.627166**, head-excluded QDQ **0.848464**였다([tensor_similarity_summary.csv](../tables/tensor_similarity_summary.csv)). PDF/README의 과거 **0.9995** 관찰은 다른 모델 세대와 tensor 기준일 수 있어 이번 v4 결과로 재현되었다고 말할 수 없다. Task-level에서는 full QDQ가 confidence 0으로 붕괴했다. 정확한 원인이 head 전체인지 특정 branch인지는 이번 실험으로 특정하지 못한다.

[detection_by_lighting.csv](../tables/detection_by_lighting.csv)에서 head-excluded의 주간 mAP@0.5는 0.483088, 야간은 0.389330이다. 야간 표본은 16장뿐이라 차이를 확증하지 않는다. Fig. 2는 [deterministic selection rule](../tables/failure_case_selection.json)로 고른 동일 프레임의 세 출력을 보여준다.

## 5. Runtime Benchmark

Server ORT CPU는 test 첫 프레임의 이미 디코딩한 BGR을 사용해 warm-up 10회 후 100회를 측정했다. Detector-only 범위는 **전처리 + ONNX 추론 + 박스 후처리**이며, 디스크 디코딩/다운로드/렌더는 제외된다. 브라우저는 같은 test 첫 프레임을 canvas로 전처리하며, HeadlessChrome 새 profile별 cold model download, session init, first inference, warm-up 10회, 측정 50회를 분리했다. 아래 p50/p95는 detector-only다. 모든 반복 trace는 `runtime/ort/*_trace.csv`, `runtime/browser_optimized/*_trace.csv`에 있다.

| 환경 / 모델 | p50 ms | p95 ms | 평균 ms | FPS | init ms |
|---|---:|---:|---:|---:|---:|
| ORT CPU FP32 | 15.317 | 20.191 | 16.115 | 62.054 | 46 |
| ORT CPU FP16 | 23.506 | 26.546 | 22.694 | 44.065 | 71 |
| ORT CPU full QDQ | 19.911 | 21.772 | 19.804 | 50.496 | 94 |
| ORT CPU head-excluded QDQ | 21.643 | 23.380 | 21.553 | 46.398 | 85 |
| Chrome WASM FP32 | 141.400 | 156.925 | 144.150 | 6.937 | 1,690 |
| Chrome WASM FP16 | 152.250 | 164.055 | 154.534 | 6.471 | 1,586 |
| Chrome WASM full QDQ | 106.300 | 121.225 | 108.764 | 9.194 | 1,779 |
| Chrome WASM head-excluded QDQ | 105.150 | 117.955 | 107.698 | 9.285 | 1,666 |
| Chrome WebGPU FP32 | 77.400 | 79.000 | 77.462 | 12.910 | 1,507 |
| Chrome WebGPU FP16 | 100.450 | 106.340 | 100.766 | 9.924 | 1,456 |
| Chrome WebGPU QDQ 두 변형 | — | — | — | `unsupported` | — |
| ORT CUDA 네 변형 | — | — | — | `unsupported` | — |

WebGPU QDQ는 `DequantizeLinear`의 int32 zero-point 커널 오류로 첫 run에서 실패했다. ORT CUDA는 누락된 cuBLAS DLL 때문에 CPU로 폴백해 GPU 수치로 취급하지 않았다. 브라우저 WebGPU adapter 이름은 API에서 노출되지 않아 실제 GPU 할당은 추가 확인이 필요하다. 초기에 FP16 JS 변환 비용이 과다했던 `runtime/browser/` 기록은 삭제하지 않고 pilot으로 보존했다. 위 표는 변환 코드를 고친 **`browser_optimized/`** 기록만 사용한다. CPU와 브라우저의 전처리/실행 환경이 달라 속도 우월성의 직접 비교로 사용하지 않는다.

### 공개 Hugging Face Space 재측정 (2026-09-24; v3 배포 감사)

공개 [Space](https://huggingface.co/spaces/gyann/edge-sign)는 처음 `RUNTIME_ERROR`/`Scheduling failure: unable to schedule`였고, 인증된 계정으로 재시작한 뒤 `RUNNING`, `/api/status` HTTP 200을 확인했다. Space 소스 commit은 `9fc2593358a678a5b1597e978a63778bc909fc31`, 하드웨어는 `cpu-basic`이다. Space repo의 Dockerfile은 Python 3.11을 사용하고 요구사항은 ORT CPU 1.23.2로 고정돼 있으나, 실제 컨테이너의 CPU 모델·thread 수·패키지 버전은 API로 확인할 수 없었다([환경 기록](../runtime/hf_space_v3/ENVIRONMENT.txt)). 두 검출기 ONNX의 실제 입력은 1×3×640×640, 출력은 1×6×8400, opset 14다. **이 Space의 검출기는 YOLOv8s v3**(FP32 또는 head-excluded static QDQ INT8)다. 위의 새 논문 근거인 **YOLO26-n v4와 모델 세대가 다르므로** 이 배포 수치를 v4 속도나 정확도로 옮겨 적지 않는다.

공개 샘플 `seoul_daylight.mp4`(15프레임, 원본 5.28 FPS)를 클라이언트에서 반복하고, 실제 배포 `/ws/stream`에 10 FPS로 제공했다. 각 모델은 tracker reset → warm-up 10프레임 → 측정 50프레임 순서다. 로컬 비디오 디코드/JPEG·base64·JSON 생성, WebSocket 왕복, Space의 JPEG 디코드·검출·ByteTrack·인식·JSON 응답이 포함된다. **브라우저 canvas 렌더링과 웹캠 캡처는 제외**한다. 아래의 수신 FPS는 입력 제공 10 FPS와 다른 값이며, 큐가 쌓이는 상황에서 받은 결과의 간격을 뜻한다.

| Space v3 구성 | 클라이언트 제공 간격 평균 | Space 파이프라인 평균 / p95 | 결과 수신 FPS | 첫 측정 송신→마지막 수신 FPS | 왕복 p50 / p95 |
|---|---:|---:|---:|---:|---:|
| head-excluded INT8 | 99.420 ms | 349.238 / 486.895 ms | 2.500 | 2.056 | 12,311 / 19,205 ms |
| FP32 | 99.249 ms | 456.192 / 580.050 ms | 1.991 | 1.643 | 16,314 / 25,218 ms |

왕복 시간이 초 단위인 이유는 10 FPS 제공 속도가 Space 처리 용량을 넘어 프레임이 전송/처리 큐에 쌓이기 때문이다. Space 내부 파이프라인만 평균 349/456 ms이므로, **현재 공개 v3 Space의 서버 경로는 30 FPS 목표를 충족하지 못한다.** 이는 공유 CPU 환경의 해당 시점 단일 실행이며 다른 하드웨어에 일반화하지 않는다. INT8 선택 구성의 실제 ONNX 크기는 검출기 17,997,724 B + OCR 2,880,185 B + KoreanSignNet 116,860 B = **20,994,769 B**로 15 MB도 초과한다. Space 리포의 모든 배포 ONNX 파일 합계는 88,124,747 B이며, 선택 구성의 크기와 구분한다. 이 실험으로 v3의 검출 정확도를 새로 평가하지 않았다.

원시 근거: INT8 [config](../runtime/hf_space_v3/20260924_int8_final/config.json)·[trace](../runtime/hf_space_v3/20260924_int8_final/trace.jsonl)·[metrics](../runtime/hf_space_v3/20260924_int8_final/metrics.json), FP32 [config](../runtime/hf_space_v3/20260924_fp32_final/config.json)·[trace](../runtime/hf_space_v3/20260924_fp32_final/trace.jsonl)·[metrics](../runtime/hf_space_v3/20260924_fp32_final/metrics.json), [pilot 선택 기록](../runtime/hf_space_v3/PILOT_NOTES.md). 각 최종 trace는 60개 연속 프레임 ID(1–60), 정확히 10개 warm-up/50개 측정 결과와 모델 variant를 보존한다. 다른 에이전트가 두 trace에서 모든 요약 통계와 전송 간격을 독립 재계산해 일치함을 확인했다.

실제 웹 화면에서는 57초 반복 샘플이 재생됐으나 서버 모드의 박스/단계 지표는 표시되지 않았다. Space 원격 `Viewport.tsx`와 이 브랜치의 파일 SHA256이 모두 `fc06d930c2c7a54ee68c17e3db75701048bee657a757e315fa9bfacd53ad033b`다. 해당 코드의 첫 영상 로드 경로는 `isPlaying=false`를 캡처한 `getFrame`을 타이머에 전달해 프레임을 보내지 않는 문제가 있고, 서버 경로는 표시용 FPS도 갱신하지 않는다. 따라서 **브라우저 렌더까지 포함한 FPS는 여전히 NOT VERIFIED**다. 서버 경로가 이미 30 FPS보다 느리다는 판정과 구분한다. 이전 실험의 오류·예비 결과도 삭제하지 않고 `PILOT_NOTES.md`에 제외 이유를 기록했다.

## 6. Recognition

독립 test JSON의 `type/text/attribute`를 기존 `scripts/prepare_korean_traffic.py`와 같은 14-class 매핑으로 해석했다. 6,772 객체 중 1개는 세부 클래스에 매핑되지 않아 제외했다. GT 박스에 학습 데이터 생성과 같은 8% margin을 주고 32×32 ROI를 만들어 FP32 ONNX를 평가했다. 이는 **oracle-box 분류 정확도**다.

| 모델 | ROI 수 | Top-1 | 상태 |
|---|---:|---:|---|
| KoreanSignNet FP32 | 6,771 | 0.808152 | 완료; class별 accuracy와 confusion matrix 저장 |
| KoreanSignNet W8A8 | — | — | `unsupported`: CPU ORT에서 `ConvInteger` 구현 부재 |

원시 logits와 클래스 예측은 [fp32_predictions.jsonl](../recognition/fp32_predictions.jsonl), class별 값과 confusion matrix는 [fp32_metrics.json](../recognition/fp32_metrics.json), Fig. 5와 [recognition_per_class.csv](../tables/recognition_per_class.csv)에 있다. KoreanOCRNet의 2,350-class 단일 문자 결과는 간판 문자열 정확도로 재해석하지 않았다.

## 7. Tracking

[annotation_audit.json](../tracking/annotation_audit.json)은 test 2,417장/6,772 객체의 수동 프레임 간 identity 필드가 **0개**임을 기록한다. 이번 작업은 ByteTrack을 실제 파이프라인 처리시간에 포함했지만 MOTA/IDF1/HOTA는 측정하지 않았다. 기존 pseudo-GT 수치는 본 논문의 수동 GT 추적 정확도처럼 제시할 수 없다. 수동 ID 라벨링 예산이 없으므로 tracking 정량 주장은 exploratory analysis로 낮춘다.

## 8. End-to-End Target

두 서버 파이프라인은 같은 2,417장 test를 sequence 순서로 한 번 처리했다. Sequence 경계에서 tracker를 재설정했다. `track_thresh=0.25`; FP32 KoreanSignNet이 **예측 track ROI**를 분류하고 원본 fine-class GT와 IoU ≥0.5, 동일 coarse class로 매칭한다. 프레임별 detector/track/recognition/decode 지연과 최종 track 출력을 보존했다. 이 설정은 기존 v3 production class와 v4의 출력 형태가 달라 별도 paper wrapper로 측정했다.

| 구성 | detector B | recognizer B | 합계 B | 메모리 프레임 pipeline FPS | JPEG decode 포함 FPS | 전체 mapped GT 중 fine-class 정답 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26 FP32 + ByteTrack + KoreanSignNet FP32 | 9,805,975 | 116,860 | 9,922,835 | 45.943 | 34.056 | 0.375277 |
| YOLO26 head-excluded QDQ + ByteTrack + KoreanSignNet FP32 | 3,416,372 | 116,860 | 3,533,232 | 44.121 | 33.269 | 0.353419 |

QDQ 파이프라인의 조건부 Top-1(매칭된 track 중 fine-class 정답)은 **0.861721**이고, 전체 매핑 GT 기준은 **0.353419**다. 선택/미검출 객체가 빠지는 조건부 수치를 전체 정확도로 쓰지 않는다. 모델은 작아졌지만 FP32 대비 end-to-end correct/GT가 0.021858 낮아졌다. Detector-only benchmark와 전체 파이프라인의 FPS도 서로 다르다.

현재 공개 HF Space는 위 표의 YOLO26 모델을 배포하지 않는다. 해당 v3 Space는 15 MB와 서버 30 FPS를 모두 충족하지 못했지만, 그것을 아래 **v4 모델 조합**의 실패 판정으로 대체하지 않는다.

| 목표 | 판정 | 근거와 한계 |
|---|---|---|
| 실제 모델 파일 합계 ≤15 MB | **PASS** | head-excluded QDQ + FP32 recognizer 3,533,232 B, tracking 추가 모델 없음 |
| 배포 시스템 ≥30 FPS | **NOT VERIFIED** | CPU 메모리 프레임 44.121 FPS/JPEG 파일 33.269 FPS; 카메라·전송·렌더 제외. 브라우저 QDQ WASM detector만 9.285 FPS, WebGPU QDQ unsupported |
| accuracy retention | **NOT VERIFIED** | 허용 열화 기준이 사전 정의되지 않음. mAP@0.5 0.499865→0.482600, 전체 GT fine-class 정답 0.375277→0.353419 |
| 세 조건 동시 달성 | **NOT VERIFIED** | 단일 실제 브라우저/배포 환경에서 성능·정확도를 함께 확인하지 못함 |

## 9. Paper Claims

| 주장 | 판정 | 필요한 논문 표현 |
|---|---|---|
| YOLO26 full-head static QDQ에서 검출 소실 | **SUPPORTED** | 이 artifact/독립 test 2,417장에 한정해 0건이라고 기술 |
| head 제외로 검출 품질 회복 | **PARTIALLY SUPPORTED** | FP32 mAP@0.5 대비 -0.017266, recall -0.020526을 같이 기술; 무손실 표현 금지 |
| 높은 cosine(0.9995)에도 v4 검출 붕괴 | **NOT SUPPORTED** | 이번 full-QDQ raw output cosine 평균 0.627166; 과거 다른 세대 수치는 재검증 전 historical observation |
| INT8 QDQ가 서버 CPU 검출을 가속 | **NOT SUPPORTED** | 이 환경 FP32 16.115 ms, head-excluded 21.553 ms |
| 작은 YOLO26 모델 묶음 | **SUPPORTED** | 실제 파일 합계 3,533,232 B; 모델 외 자산은 별도 |
| 브라우저 WebGPU에서 이 INT8 QDQ 동작 | **NOT SUPPORTED** | ORT Web 1.22.0 커널 오류; WASM에서는 실행 |
| KoreanSignNet 독립 14-class 분류 정확도 | **PARTIALLY SUPPORTED** | GT ROI Top-1 0.808152; end-to-end fine-class correct/GT 0.353419과 구분 |
| 수동 GT 기반 추적 MOTA/IDF1/HOTA | **NOT SUPPORTED** | manual identity GT 부재 |
| 15 MB/30 FPS/accuracy retention 동시 달성 | **NOT SUPPORTED** | 현재의 배포 범위와 accuracy 기준으로 입증되지 않음 |

**PDF/저장소 수치 불일치:** v4 학습 `runs/detect/edge_sign_v4/results.csv`의 최고 validation mAP@0.5는 epoch 28 **0.75611**로 PDF의 0.756과 맞는다. README/일부 docs의 **0.748**과는 다르다. 이 CSV의 학습 validation 값은 새 640×640 ONNX test mAP@0.5 **0.499865**와 같은 지표/분할로 비교할 수 없다. v2 weight-only 수치, v3의 12/0/11 frame 사례, 과거 23.3→56.3 FPS, pseudo-GT MOTA는 원시 조건을 이번에 재구성하지 않아 historical summary로 남긴다.

## 10. Remaining Experiments

### P0: 투고 전 우선

1. 목표 accuracy retention의 허용 기준과 실제 배포 workload를 사전 고정한다. YOLO26 v4를 동일 전처리/후처리의 별도 배포 환경에 올려 카메라/영상 디코드·전송·렌더를 포함한 지연을 측정한다. 현재 공개 v3 Space의 서버 영상 경로는 먼저 프레임 전송과 FPS 표시 문제를 수정·검증해야 한다. 서버 CPU에서만 30 FPS를 주장하려면 그 범위를 제목/초록에 명시한다.
2. Headless 브라우저의 WebGPU adapter 실체를 확인하고 실제 사용자 브라우저에서 반복한다. QDQ WebGPU 지원이 필요하면 모델/ORT Web 버전을 바꾼 **새 구성**을 검증하되 실패 기록을 유지한다.
3. 야간과 별도 촬영 장소의 독립 test를 확장한다. 현재 night 16장, calibration night 0장이다. Scene/위치 중복도 메타데이터로 점검한다.
4. 수동 identity GT를 소량이라도 구축하려면 annotation protocol/검수 후 MOTA/IDF1/HOTA를 다시 산출한다. 구축하지 않으면 tracking 정량 주장을 제외한다.
5. KoreanSignNet 양자화 인식 수치를 논문에 넣으려면 CPU에서 실행되는 실제 QDQ 모델을 새 출력 경로로 만들고 같은 독립 ROI로 평가한다. 기존 `ConvInteger` W8A8 파일을 실행 성공으로 적지 않는다.

### P1: 원인 분석/확장

1. YOLO26 classification/regression/head 전체를 각각 제외한 QDQ ablation을 동일 calibration/test에서 수행한다. DFL 하나를 원인으로 단정하지 않는다.
2. Calibration 표본 수·주야간 구성과 640/1280 입력 해상도를 별도 통제 실험으로 비교한다. 모델 세대를 넘는 우월성 비교는 하지 않는다.
3. CUDA EP 의존성을 복구해 GPU 수치를 새 실행 환경과 raw trace로 기록한다.

## Figure 및 Table 후보

- [Fig. 1 evidence scope](../figures/fig1_evidence_scope.png): historical v2와 새 v4 근거의 범위.
- [Fig. 2 head-QDQ failure](../figures/fig2_head_qdq_failure.png): 동일 test frame의 3모델 박스/confidence. 원시 자료는 각 `detection/*/predictions.jsonl`.
- [Fig. 3 runtime](../figures/fig3_runtime_latency.png): p50/p95. [runtime_summary.csv](../tables/runtime_summary.csv)와 반복 trace 기반.
- [Fig. 4 failure cases](../figures/fig4_failure_cases.png): 밤/소형 객체 미검출. [선택 규칙](../tables/failure_case_selection.json).
- [Fig. 5 confusion matrix](../figures/fig5_recognition_confusion.png): 14-class oracle ROI; [class별 표](../tables/recognition_per_class.csv).
- Table 후보: [detection_summary.csv](../tables/detection_summary.csv), [detection_by_lighting.csv](../tables/detection_by_lighting.csv), [runtime_summary.csv](../tables/runtime_summary.csv), pipeline `metrics.json`.

원시 이미지, checkpoint, ONNX는 Git에 넣지 않았다. 모델 및 manifest 해시가 재현성의 기준이다. 원본 checkout의 미커밋 논문 개정본과 `main`은 수정하지 않았다.

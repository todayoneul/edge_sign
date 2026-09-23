# Runtime evidence audit

2026-09-23 정적 조사. 새 benchmark/추론은 실행하지 않았다. VERIFIED_RAW는 저장된 원본이나 직접 확인 가능한 구현 사실을 뜻하며, 코드 존재를 실측 결과로 간주하지 않는다.

## 핵심 판정

CPU 56.3 FPS, WebGPU FP32 62 FPS, WebGPU FP16 24 FPS, WASM INT8 2.2 FPS는 현재 로컬에서 대응하는 실행 로그/반복 latency 배열/환경 기록을 찾지 못했다. 모두 **VERIFIED_DOC_ONLY**이다. 이는 실험이 없었다는 판정이 아니라 현재 증거의 보존 한계다. C3는 동기가 될 수 있으나, 확정된 runtime 비교 실험으로 제출하기에는 부족하다.

| 주장 | 세대/모델 | 기록값 | 근거 | 판정 |
|---|---|---|---|---|
| CPU FP32 파이프라인 | v2 YOLOv8s + ByteTrack + OCR/TrafficSignNet | 23.3 FPS | README.md:543-548; scripts/archive/plot_v2_extras.py:37-40 | VERIFIED_DOC_ONLY |
| CPU Static INT8 파이프라인 | v2 yolov8s_signs_int8_static.onnx | 56.3 FPS (E0 detector-only / E3 all 표기) | README.md:545-548; docs/EXPERIMENTS.md:110-124 | VERIFIED_DOC_ONLY |
| CPU Static INT8 이전 결과 | v1/Phase5 | 57.7 FPS; detector 32.4→14.6 ms (2.22×) | docs/EXPERIMENTS.md:289-297; README.md:555 | VERIFIED_DOC_ONLY; v2 56.3과 혼용 금지 |
| 브라우저 FP32 WebGPU | v3 3-class detector spike 맥락, 43MB 표기 | 62 FPS | README.md:671-678; web_modern/public/spike/index.html 모델 선택; spike.js:15-18 | VERIFIED_DOC_ONLY |
| 브라우저 FP16 WebGPU | 동일 v3 spike 맥락, 22MB 표기 | 24 FPS | README.md:676 | VERIFIED_DOC_ONLY |
| 브라우저 INT8 WASM | v3 18MB 표기 | 2.2 FPS | README.md:677 | VERIFIED_DOC_ONLY |
| INT8 WebGPU 실패 | int32 DequantizeLinear 미지원 설명 | 실행 불가 | README.md:678; spike.js:146 | VERIFIED_DOC_ONLY; 오류 원본/브라우저 버전 없음 |
| HF/Docker FPS | CPU 배포 구성 | 미발견 | Dockerfile; requirements-hf.txt | MISSING; 패키징은 배포 benchmark가 아님 |

62/24는 같은 EP에서도 다른 precision의 관찰이지만 raw/환경 통제 미확인이다. **FP32/WebGPU vs INT8/WASM is a precision-runtime pair comparison, not an isolated bit-width comparison.** 또한 v2 CPU 전체 pipeline과 v3 브라우저 detector-only 수치를 동일 모델/동일 작업 속도로 비교할 수 없다. README.md:49의 “INT8 가속은 서버 CPU에서만 유효”는 조사된 특정 조합으로 한정해야 한다. FP16 커널 미성숙이라는 인과 설명도 profiler/커널 근거가 없다.

## 벤치마크 구현과 측정 범위

| 코드 | Warmup / runs | 입력/측정 범위 | 통계와 한계 |
|---|---|---|---|
| scripts/archive/benchmark_pipeline.py:36-46,58-102,270-280 | 기본 3 / 50 | 각 모델 고정 random float32 dummy, CPU EP, sess.run | mean/min/max 반환; median/std/원시 배열 저장 없음. seed 없음 |
| 같은 파일:109-218 | 기본 3 / 50 | 정렬된 test 시퀀스에서 처음 프레임을 모음; 없으면 random dummy; preload 후 process_frame 구간 | 총시간/n 평균, n/총시간 FPS. disk decode/네트워크/렌더 미포함. 초기 warmup 프레임을 timed loop에 다시 사용. 검출량에 따른 인식 호출량 미기록 |
| src/pipeline/eval_e2e.py:196-243,369-385 | 기본 3 / 50프레임 | 위와 유사한 preload + pipeline loop | FPS만 측정. task accuracy는 40-120행의 하드코딩 값 |
| scripts/archive/quantize_onnx_real.py:247-259 | 5 / 기본50 | CPU EP, 고정 입력 sess.run | 총시간/n; per-run sample/median/std 없음 |
| scripts/quantize_v3_detector.py:146-148 | compare_latency의 5 / 50 | real input으로 FP32/INT8 비교 | 코드 존재; 출력 로그 미발견 |
| web_modern/public/spike/spike.js:149-154,336-358 | session 생성 시 1회 warm run; 종료 run 수 없음 | canvas/video/image/synthetic 소스; 전처리→session.run→후처리/렌더, RAF 반복 | 추론시간 20개 rolling mean. FPS는 각 frame 시작 간격의 역수 20개 평균으로 N/총시간과 다름. median/std/export 없음 |
| scripts/spike_ortweb_check.mjs:20-28 | 1회 추론 | Node WASM 0 tensor | 로드 성공 검사; browser FPS 실험 아님 |

현행 pipeline detector provider는 CUDA 우선이며 EDGE_SIGN_CPU_ONLY=1일 때만 CPU 강제(src/pipeline/e2e_pipeline.py:49-54,339). 예전 스크립트의 CPU 제목만으로 현재 동일 실행환경을 보장하지 않는다. archive 이동 후 benchmark_pipeline.py:24 및 _finalize_v2_results.py:28의 parent.parent는 repo root가 아닌 scripts를 가리킨다. 재현 시 경로 검토가 필요하며 이번에 수정하지 않았다.

## 보존/환경

- logs는 train_v4.log, Phase1 CSV 및 final_score_report.txt만 확인되며 과거 logs/track_*.log, eval_e2e_v2.log, quant_int8_v2.log, results_v2.json은 미발견. _finalize_v2_results.py:4-13의 입력/출력 설명은 파일 존재나 실행 결과가 아니다.
- scripts/archive/plot_v2_extras.py:34-40은 MOTA/FPS 숫자 배열을 직접 선언한다. assets/v2/fps_comparison.png 같은 재생성 그림은 독립 raw evidence가 아니다. src/pipeline/eval_e2e.py:40-120도 사전 측정값을 코드에 기록한 것뿐이다.
- requirements.txt:79,128은 ORT-GPU 1.23.2 / torch 2.11.0+cu128 설치 명세. requirements-hf.txt:6은 CPU ORT 1.23.2. 현재 요구사항을 과거 benchmark 실행 버전으로 소급하지 않는다.
- spike.js:79-84는 CDN ORT-Web 1.22.0, WASM numThreads=1을 지정. browser benchmark용 ORT는 CDN에서 동적 로드되므로 web_modern/package-lock.json이 실행 라이브러리/브라우저 버전을 확정하지 못한다.
- logs/train_v4.log:3은 실제 학습 환경 Python 3.10.19, torch 2.11.0+cu128, Ultralytics 8.4.56, NVIDIA GeForce RTX 5070 12227MiB를 기록한다. 이는 **v4 학습** 증거이며 v2 CPU/v3 browser 측정 장치 증거가 아니다.
- 과거 benchmark CPU 모델, RAM, OS 빌드, GPU driver, browser 이름/버전, power mode, 온도/동시 workload, 실제 provider trace는 MISSING. 현재 기계 정보로 보충하지 않았다.
- README.md:531의 detector speedup 2.42×는 ~32/~14 반올림 수치로 확인할 수 없고, 56.3/23.3 pipeline 비율과 같다. docs/EXPERIMENTS.md:294의 2.22×와 버전/측정범위를 명확히 분리해야 한다.

## 추가 확보 우선순위

MUST: 동일 checkpoint/split/hash와 실제 hardware/runtime 버전을 묶은 raw latency 반복 기록; warmup/run count/전처리·후처리·전송 포함범위; mean/median/std/분위수; FP32/INT8 task-quality 동시 확인. SHOULD: 독립 반복/신뢰구간, 실제 edge hardware, 고정 video 및 인식 호출수, CPU/GPU threading·power·provider 통제. 이번에는 실행하지 않았다.

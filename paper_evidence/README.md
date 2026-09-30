# Edge-Sign TIIS evidence index

이 폴더는 논문(투고 준비 중)의 수치를 다시 계산할 수 있는 원시 기록을 담습니다. 결과 요약은 [RUNTIME_MATRIX.md](reports/RUNTIME_MATRIX.md)와 [COCO_VALIDATION.md](reports/COCO_VALIDATION.md), 추가 실험은 [extra/README.md](extra/README.md)에 있습니다. 기존 v2/v3 문서의 숫자는 새 결과와 합치지 않았습니다.

| 질문 | 새 원시 evidence |
|---|---|
| 학습/보정/시험 분리 | [split_summary.json](splits/split_summary.json), `*_manifest.jsonl` |
| YOLO26 검출 FP32/full QDQ/head-excluded QDQ | `detection/yolo26_*/{config.yaml,metrics.json,predictions.jsonl}` |
| 모델 계보와 실행 환경 | [model_manifest.csv](models/model_manifest.csv), `environment/` |
| ORT CPU 및 CUDA 시도 | `runtime/ort/*.{json,csv}` |
| Chrome WASM 및 WebGPU | `runtime/browser_optimized/*.{json,csv}`; `runtime/browser/`는 FP16 전처리 개선 전 첫 실행으로 보존 |
| 공개 Hugging Face Space v3 재측정 | `runtime/hf_space_v3/20260924_{int8,fp32}_final/{config.json,metrics.json,trace.jsonl}`; [예비 실행 제외 이유](runtime/hf_space_v3/PILOT_NOTES.md). YOLO26 v4 배포 결과와 구분 |
| YOLO26 v4 배포 준비 | [SPACE_V4_DEPLOYMENT_READINESS.md](reports/SPACE_V4_DEPLOYMENT_READINESS.md). v4 전용 경로 구현, 출력 형식 감사 반영, v3/v4 설정 차이 |
| 공개 Hugging Face Space v4 측정 | `runtime/hf_space_v4/20260925_{head_excluded_qdq,fp32}_{final,open30}/{config.json,metrics.json,trace.jsonl}`; [측정 기록](runtime/hf_space_v4/MEASUREMENT_NOTES.md), [한눈에 보기](reports/HF_SPACE_V4_MEASUREMENT.md). `final`은 10 FPS 입력(미포화), `open30`은 30 FPS 입력(처리 한계) |
| 평가 기준(정확도·속도·크기)과 근거 | [EVALUATION_CRITERIA.md](reports/EVALUATION_CRITERIA.md). 99% 사전 기준 적용 중, 95% 참고 등급은 2026-09-26 추가 |
| Detector→ByteTrack→KoreanSignNet | `runtime/pipeline_{fp32,head_excluded_qdq}/{config.yaml,metrics.json,trace.csv,predictions.jsonl}` |
| 독립 시험 프레임의 14-class 인식 | `recognition/fp32_{metrics.json,predictions.jsonl}` |
| tracking identity GT 감사 | [annotation_audit.json](tracking/annotation_audit.json) |
| 초기 그림 후보와 표 | `figures/fig1_*.png`–`fig5_*.png`(초기 후보, 현재 초안에는 쓰지 않음, `generate_paper_figures.py`), `tables/` |
| 구성요소 × 정밀도 × 실행 환경 매트릭스 | [RUNTIME_MATRIX.md](reports/RUNTIME_MATRIX.md); `runtime/matrix/`(`summary.json`, `placement.json`, `bootstrap_retention.json`, 실행별 JSON·trace, `*/predictions.jsonl.gz`, `failed_attempts/`) |
| 새 QDQ 변형의 생성 기록 | `models/quantize_variants_log.json`(검출기), `models/recognizer_variants_log.json`(인식기), `splits/recognition_calibration_manifest.jsonl` |
| 인식기 변형 정확도 | `recognition/variants/{fp32,fp16,int8_*}_metrics.json`, `predictions.csv` |
| 구성요소 민감도·단일 기기 지연·파이프라인 그림 | `figures/fig6_component_sensitivity.png`(초안 Fig. 2), `fig7_runtime_latency.png`, `fig8_pipeline_assignment.png`(Windows 단일 기기 참고용, 초안에는 쓰지 않음) (`scripts/paper/plot_runtime_matrix.py`) |
| 기기 비교 그림 (초안 Fig. 3–4) | `figures/fig11_runtime_latency_devices.{pdf,png}`, `fig12_pipeline_devices.{pdf,png}` (`plot_runtime_matrix.py --compare runtime/matrix_mac --recheck runtime/matrix_mac_recheck`). Mac 값은 재측정이 반복한 항목(WASM 4T YOLO26-n·YOLOv8s, 파이프라인 4개 배치)만 재측정 중앙값으로 대체하고 나머지는 1차 값이다 |
| 검출 붕괴 원인 (활성값 ablation) | `runtime/matrix/cpu_{v4,v3}_a8sim_{all,head,decode,outconcat,decode_no_outconcat}/`, `cpu_{v4,v3}_int8_decode_excl/`, `runtime/matrix/decode_tensor_scan.json`, `models/activation_ablation_log.json` (`activation_ablation.py`, `decode_tensor_scan.py`) |
| 파이프라인 5회 반복 | `runtime/matrix/pipeline_t4_ort1300_*_r{2..5}.json`·`_trace.csv`, 집계 `pipeline_repeats_summary.json` (`summarize_pipeline_repeats.py`) |
| 검출→추적→인식 종단 정확도 (정밀도 조합) | `runtime/end_to_end/{summary.json,per_frame.csv}` (`evaluate_end_to_end.py`) |
| 개요·정성 비교 그림 | `figures/fig9_study_overview.png`(초안 Fig. 1, `plot_study_overview.py`), `fig10_qualitative.png`(초안 Fig. 5, `plot_qualitative.py`) |
| 표준 워크로드 외부 검증 (YOLO11l + MLPerf COCO safe subset) | 사전 등록 [COCO_VALIDATION_PLAN.md](reports/COCO_VALIDATION_PLAN.md), 결과 [COCO_VALIDATION.md](reports/COCO_VALIDATION.md); `coco/`(부분집합·보정 manifest, `cpu_<variant>/metrics.json`, `detections.npz`), 지연은 `runtime/matrix/*coco_*` (`coco_validation.py`, `runtime_matrix.py`) |
| 두 번째 기기 측정 | [DEVICE_MEASUREMENT_GUIDE.md](reports/DEVICE_MEASUREMENT_GUIDE.md) (절차), [runtime/matrix_mac/](runtime/matrix_mac/) (Mac 1차 결과, 배경 부하 통제 없음), [runtime/matrix_mac_recheck/](runtime/matrix_mac_recheck/) (2026-09-29 조용한 환경 재측정: WASM FP32 대 INT8, 파이프라인 4개 배치 5회, WASM 수치 일치성; `run_mac_wasm_recheck.sh`), RUNTIME_MATRIX §2.6 |
| 추가 실험 (2026-09-27): 크기별 유지율, 좌표 정규화 기준선, 보정 방법, YOLOv8s 종단 정확도 | [extra/README.md](extra/README.md); 그림 `figures/fig13_size_retention.{pdf,png}` (`plot_size_bins.py`; 초안에는 그림 없이 4.2.2절에서 서술만) |
| 그림 스타일 | 모든 논문 그림은 `scripts/paper/paper_style.py`(Arial 8 pt, 6.3 in 폭, Okabe–Ito 팔레트)로 벡터 PDF와 600 dpi PNG를 함께 만든다 |

## 논문 그림 대응 (2026-09-30 초안 기준)

파일 이름의 번호는 만든 순서이고 초안의 그림 번호와 다르다. 초안의 그림 번호는 아래 표를 따른다.

| 초안 | 내용 | 파일 (`figures/`) | 생성 |
|---|---|---|---|
| Fig. 1 | BRIQ 평가 구조 개요 | `fig9_study_overview.{pdf,png,svg}` | `plot_study_overview.py` |
| Fig. 2 | 구성요소별 정확도 유지율과 95% CI | `fig6_component_sensitivity.{pdf,png}` | `plot_runtime_matrix.py` |
| Fig. 3 | 실행 환경·기기별 검출기 지연 (막대 = 평균, 수염 = p90, 로그 축) | `fig11_runtime_latency_devices.{pdf,png}` | `plot_runtime_matrix.py --compare runtime/matrix_mac --recheck runtime/matrix_mac_recheck` |
| Fig. 4 | 두 기기의 구성요소별 배치 단계 지연 (A = Windows, B = Mac) | `fig12_pipeline_devices.{pdf,png}` | 위와 같음 |
| Fig. 5 | YOLO26-n 정밀도별 정성 비교 | `fig10_qualitative.{pdf,png}` | `plot_qualitative.py` |

초안에 쓰지 않는 그림: `fig1`–`fig5`(초기 후보), `fig7`·`fig8`(단일 기기), `fig13`(크기별 유지율).

데이터 이미지와 ONNX/.pt 파일은 포함하지 않습니다. `model_manifest.csv`의 경로는 원본 체크아웃의 ignored artifact를 가리키며, 모든 스크립트는 `--artifact-root`로 이 위치를 받습니다. 모델 해시와 manifest 해시가 다르면 결과를 재사용하지 마세요.

실행 예: `python scripts/paper/evaluate_qdq_detection.py --artifact-root C:\Users\leegy\Desktop\CNN_Quant --output 새_결과_디렉터리`. 스크립트는 기존 evidence 출력에 덮어쓰지 않도록 설계했습니다.

실제 공개 Space 확인: `python scripts/paper/benchmark_hf_space.py --variant int8 --warmup 10 --iterations 50 --send-fps 10 --output 새_결과_디렉터리`. 이 스크립트는 공개 샘플 영상만 사용하며 `/ws/stream`의 서버 파이프라인과 네트워크를 측정한다. 브라우저 렌더 FPS로 해석하지 않는다.

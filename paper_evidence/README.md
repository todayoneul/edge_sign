# Edge-Sign TIIS evidence index

이 폴더는 첨부 `Edge-Sign_개정원고_v2_검토본.pdf`의 주장 중 **이번 브랜치에서 새로 검증한 결과**를 담습니다. 먼저 [TIIS_EVIDENCE_REPORT.md](reports/TIIS_EVIDENCE_REPORT.md)를 읽으면 됩니다. 기존 v2/v3 문서의 숫자는 새 결과와 합치지 않았습니다.

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
| 논문용 그림과 표 | `figures/fig1_*.png`–`fig5_*.png`, `tables/` |
| **논문 초안** | [paper_draft_KSII_TIIS_ko.md](paper_draft_KSII_TIIS_ko.md) — 구성요소·실행 환경별 양자화 실증 분석 |
| 구성요소 × 정밀도 × 실행 환경 매트릭스 | [RUNTIME_MATRIX.md](reports/RUNTIME_MATRIX.md); `runtime/matrix/`(`summary.json`, `placement.json`, `bootstrap_retention.json`, 실행별 JSON·trace, `*/predictions.jsonl.gz`, `failed_attempts/`) |
| 새 QDQ 변형의 생성 기록 | `models/quantize_variants_log.json`(검출기), `models/recognizer_variants_log.json`(인식기), `splits/recognition_calibration_manifest.jsonl` |
| 인식기 변형 정확도 | `recognition/variants/{fp32,fp16,int8_*}_metrics.json`, `predictions.csv` |
| 논문 그림 6–8 | `figures/fig6_component_sensitivity.png`, `fig7_runtime_latency.png`, `fig8_pipeline_assignment.png` (`scripts/paper/plot_runtime_matrix.py`) |
| 검출 붕괴 원인 (활성값 ablation) | `runtime/matrix/cpu_{v4,v3}_a8sim_{all,head,decode,outconcat,decode_no_outconcat}/`, `cpu_{v4,v3}_int8_decode_excl/`, `runtime/matrix/decode_tensor_scan.json`, `models/activation_ablation_log.json` (`activation_ablation.py`, `decode_tensor_scan.py`) |
| 파이프라인 5회 반복 | `runtime/matrix/pipeline_t4_ort1300_*_r{2..5}.json`·`_trace.csv`, 집계 `pipeline_repeats_summary.json` (`summarize_pipeline_repeats.py`) |
| 검출→추적→인식 종단 정확도 (정밀도 조합) | `runtime/end_to_end/{summary.json,per_frame.csv}` (`evaluate_end_to_end.py`) |
| 논문 그림 추가 | `figures/fig9_study_overview.png`(개요, `plot_study_overview.py`), `fig10_qualitative.png`(정성 비교, `plot_qualitative.py`) |
| 두 번째 기기 측정 | [DEVICE_MEASUREMENT_GUIDE.md](reports/DEVICE_MEASUREMENT_GUIDE.md) |

데이터 이미지와 ONNX/.pt 파일은 포함하지 않습니다. `model_manifest.csv`의 경로는 원본 체크아웃의 ignored artifact를 가리키며, 모든 스크립트는 `--artifact-root`로 이 위치를 받습니다. 모델 해시와 manifest 해시가 다르면 결과를 재사용하지 마세요.

실행 예: `python scripts/paper/evaluate_qdq_detection.py --artifact-root C:\Users\leegy\Desktop\CNN_Quant --output 새_결과_디렉터리`. 스크립트는 기존 evidence 출력에 덮어쓰지 않도록 설계했습니다.

실제 공개 Space 확인: `python scripts/paper/benchmark_hf_space.py --variant int8 --warmup 10 --iterations 50 --send-fps 10 --output 새_결과_디렉터리`. 이 스크립트는 공개 샘플 영상만 사용하며 `/ws/stream`의 서버 파이프라인과 네트워크를 측정한다. 브라우저 렌더 FPS로 해석하지 않는다.

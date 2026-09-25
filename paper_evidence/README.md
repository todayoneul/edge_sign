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
| 공개 Hugging Face Space v4 측정 | `runtime/hf_space_v4/20260925_{head_excluded_qdq,fp32}_final/{config.json,metrics.json,trace.jsonl}`; [측정 기록](runtime/hf_space_v4/MEASUREMENT_NOTES.md), [한눈에 보기](reports/HF_SPACE_V4_MEASUREMENT.md). 10 FPS 입력에서 Space 미포화 |
| Detector→ByteTrack→KoreanSignNet | `runtime/pipeline_{fp32,head_excluded_qdq}/{config.yaml,metrics.json,trace.csv,predictions.jsonl}` |
| 독립 시험 프레임의 14-class 인식 | `recognition/fp32_{metrics.json,predictions.jsonl}` |
| tracking identity GT 감사 | [annotation_audit.json](tracking/annotation_audit.json) |
| 논문용 그림과 표 | `figures/fig1_*.png`–`fig5_*.png`, `tables/` |

데이터 이미지와 ONNX/.pt 파일은 포함하지 않습니다. `model_manifest.csv`의 경로는 원본 체크아웃의 ignored artifact를 가리키며, 모든 스크립트는 `--artifact-root`로 이 위치를 받습니다. 모델 해시와 manifest 해시가 다르면 결과를 재사용하지 마세요.

실행 예: `python scripts/paper/evaluate_qdq_detection.py --artifact-root C:\Users\leegy\Desktop\CNN_Quant --output 새_결과_디렉터리`. 스크립트는 기존 evidence 출력에 덮어쓰지 않도록 설계했습니다.

실제 공개 Space 확인: `python scripts/paper/benchmark_hf_space.py --variant int8 --warmup 10 --iterations 50 --send-fps 10 --output 새_결과_디렉터리`. 이 스크립트는 공개 샘플 영상만 사용하며 `/ws/stream`의 서버 파이프라인과 네트워크를 측정한다. 브라우저 렌더 FPS로 해석하지 않는다.

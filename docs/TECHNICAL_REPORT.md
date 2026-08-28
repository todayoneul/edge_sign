# Edge-Sign — Technical Report Guide

루트 `README.md`는 프로젝트를 빠르게 이해할 수 있도록 portfolio-style로 정리되어 있습니다.

기존의 상세 README에는 아래 내용이 포함되어 있었습니다.

- W8A8 / W4A16 / SmoothQuant / 1-Bit 실험
- Detection / Tracking / Recognition 단계별 민감도
- YOLO detection head INT8 collapse 분석
- Browser WebGPU와 Server CPU runtime 비교
- Phase 1~3 데이터셋·실험 설정·정량 결과
- E0~E7 Pareto analysis
- 재현 명령 및 라이선스 상세

## Full pre-redesign README

전체 상세 기록은 Git history에 그대로 보존되어 있습니다.

**[README before portfolio redesign](https://github.com/todayoneul/edge_sign/blob/94f5882e6f44a21c77084a55f6886894fdcd6b1a/README.md)**

## Current detailed sources

- [Experiment details](EXPERIMENTS.md)
- [`assets/`](../assets/) — experiment plots and qualitative samples
- [`src/quant/`](../src/quant/) — quantization implementation
- [`src/pipeline/`](../src/pipeline/) — end-to-end deployment pipeline
- [Third-party licenses](../THIRD_PARTY_NOTICES.md)

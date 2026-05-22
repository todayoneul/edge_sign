---
language: ko
license: mit
tags:
- sign-language-recognition
- sign-language
- keypoint
- mediapipe
- onnx
- onnxruntime
- edge-sign
- ksl
---

# KSL Sequence Recognition Model - Landmark

이 모델은 한국수어(KSL)를 실시간으로 인식하기 위한 Sequence Classifier ONNX 모델입니다.
브라우저 환경(ONNX Runtime Web)에서 서버 없이도 동작할 수 있도록 INT8로 경량화하여 설계 및 추출되었습니다.

## 모델 정보 (Model Specifications)
- **종류**: Landmark
- **역할**: 50개의 핵심 수어 단어를 처리하는 AIHub 랜드마크 기반 한국수어 단어 인식 모델
- **분류 클래스 수**: 50 클래스
- **입력 텐서 구조 (Input Shape)**: `[1, 40, 959]` (Batch Size, Sequence Length, Feature Dimensions)
- **입력 특징 차원 (Feature Dimension)**: 959차원 (Pose 25점, Face 70점, Left Hand 21점, Right Hand 21점의 2D/3D 좌표 및 가시성/신뢰도 정보를 매핑)

## 파일 구성 (Files)
- `landmark_best.onnx`: 모델 네트워크 구조 및 연산 그래프 정의 파일
- `landmark_best.onnx.data`: 가중치 바이너리 데이터 (External Data)
- `config.json`: 입력 형태 및 아키텍처 하이퍼파라미터 정의
- `landmark_labels.json`: 수어 단어 사본 매핑 (정답 라벨)

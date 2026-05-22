import os
import shutil
import json

def create_readme(model_name, description, classes_count, input_shape, save_dir):
    readme_content = f"""---
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

# KSL Sequence Recognition Model - {model_name}

이 모델은 한국수어(KSL)를 실시간으로 인식하기 위한 Sequence Classifier ONNX 모델입니다.
브라우저 환경(ONNX Runtime Web)에서 서버 없이도 동작할 수 있도록 INT8로 경량화하여 설계 및 추출되었습니다.

## 모델 정보 (Model Specifications)
- **종류**: {model_name}
- **역할**: {description}
- **분류 클래스 수**: {classes_count} 클래스
- **입력 텐서 구조 (Input Shape)**: `{input_shape}` (Batch Size, Sequence Length, Feature Dimensions)
- **입력 특징 차원 (Feature Dimension)**: 959차원 (Pose 25점, Face 70점, Left Hand 21점, Right Hand 21점의 2D/3D 좌표 및 가시성/신뢰도 정보를 매핑)

## 파일 구성 (Files)
- `{model_name.lower()}_best.onnx`: 모델 네트워크 구조 및 연산 그래프 정의 파일
- `{model_name.lower()}_best.onnx.data`: 가중치 바이너리 데이터 (External Data)
- `config.json`: 입력 형태 및 아키텍처 하이퍼파라미터 정의
- `{model_name.lower()}_labels.json`: 수어 단어 사본 매핑 (정답 라벨)
"""
    if "mediapipe" in model_name.lower():
        readme_content += "- `mediapipe_stats.json`: 특징 데이터 Z-score 정규화를 위한 평균(mean) 및 표준편차(std) 값\n"
    
    with open(os.path.join(save_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Created README.md in {save_dir}")

def main():
    web_model_dir = "./web/model"
    output_base_dir = "./models"
    
    # 1. MediaPipe Model Packaging
    mp_hf_dir = os.path.join(output_base_dir, "hf_mediapipe_ksl")
    os.makedirs(mp_hf_dir, exist_ok=True)
    
    mp_files = {
        "mediapipe_best.onnx": "mediapipe_best.onnx",
        "mediapipe_best.onnx.data": "mediapipe_best.onnx.data",
        "mediapipe_labels.json": "mediapipe_labels.json",
        "mediapipe_stats.json": "mediapipe_stats.json"
    }
    
    print("\n--- Packaging MediaPipe KSL Model ---")
    all_files_exist = True
    for src_name, dest_name in mp_files.items():
        src_path = os.path.join(web_model_dir, src_name)
        dest_path = os.path.join(mp_hf_dir, dest_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_name} -> {dest_path}")
        else:
            print(f"WARNING: {src_path} not found!")
            all_files_exist = False
            
    if all_files_exist:
        # Create config.json
        mp_config = {
            "model_type": "ksl_sequence_classifier",
            "input_shape": [1, 30, 959],
            "hidden_dim": 192,
            "num_layers": 2,
            "num_classes": 2771,
            "dropout": 0.1,
            "normalization": "z_score",
            "vocabulary_file": "mediapipe_labels.json",
            "normalization_stats_file": "mediapipe_stats.json"
        }
        with open(os.path.join(mp_hf_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(mp_config, f, indent=2)
        print(f"Created config.json in {mp_hf_dir}")
        
        # Create README.md
        create_readme(
            model_name="MediaPipe",
            description="2,771개의 대규모 어휘 클래스를 처리하는 MediaPipe 기반 한국수어 단어 인식 모델",
            classes_count=2771,
            input_shape="[1, 30, 959]",
            save_dir=mp_hf_dir
        )
        
    # 2. AIHub Landmark Model Packaging
    ah_hf_dir = os.path.join(output_base_dir, "hf_landmark_ksl")
    os.makedirs(ah_hf_dir, exist_ok=True)
    
    ah_files = {
        "landmark_best.onnx": "landmark_best.onnx",
        "landmark_best.onnx.data": "landmark_best.onnx.data",
        "landmark_labels.json": "landmark_labels.json"
    }
    
    print("\n--- Packaging AIHub Landmark Model ---")
    all_files_exist = True
    for src_name, dest_name in ah_files.items():
        src_path = os.path.join(web_model_dir, src_name)
        dest_path = os.path.join(ah_hf_dir, dest_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_name} -> {dest_path}")
        else:
            print(f"WARNING: {src_path} not found!")
            all_files_exist = False
            
    if all_files_exist:
        # Create config.json
        ah_config = {
            "model_type": "ksl_sequence_classifier",
            "input_shape": [1, 40, 959],
            "hidden_dim": 128,
            "num_layers": 2,
            "num_classes": 50,
            "dropout": 0.1,
            "normalization": "none",
            "vocabulary_file": "landmark_labels.json"
        }
        with open(os.path.join(ah_hf_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(ah_config, f, indent=2)
        print(f"Created config.json in {ah_hf_dir}")
        
        # Create README.md
        create_readme(
            model_name="Landmark",
            description="50개의 핵심 수어 단어를 처리하는 AIHub 랜드마크 기반 한국수어 단어 인식 모델",
            classes_count=50,
            input_shape="[1, 40, 959]",
            save_dir=ah_hf_dir
        )
        
    print("\nPackaging completed successfully!")

if __name__ == "__main__":
    main()

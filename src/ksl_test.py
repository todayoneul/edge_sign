import cv2
import numpy as np
import onnxruntime as ort
import os

# 설정
# 15.61MB 크기의 양자화된 INT8 모델 경로
ONNX_MODEL_PATH = "./model_space/convnextv2_ksl_int8.onnx"

# 테스트 이미지 경로 설정
TEST_IMAGE_PATH = "./dataset/train/가능/NIA_SL_WORD2493_REAL01_D_1.jpg"

# 클래스 이름(한글) 로딩
# dataset/train 하위 폴더 이름들을 읽어서 정답지 리스트를 생성합니다.
DATA_DIR = "./dataset/train"
CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def preprocess_image(image_path):
    """이미지를 모델이 요구하는 224x224 규격 및 정규화 수치로 변환합니다."""
    # 한글 경로 지원을 위해 numpy로 읽고 cv2로 디코딩합니다.
    img_array = np.fromfile(image_path, np.uint8)
    if img_array is None or len(img_array) == 0:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 디코딩할 수 없습니다: {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    
    # 0~1 사이로 스케일링
    img = img.astype(np.float32) / 255.0
    
    # ImageNet 정규화 공식 (학습 시와 동일한 기준으로 적용)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # [H, W, C] 차원을 [C, H, W] 로 변경
    img = np.transpose(img, (2, 0, 1))
    # 배치 차원 추가 [1, C, H, W]
    img = np.expand_dims(img, axis=0)
    
    # ONNX Runtime이 요구하는 float32 타입으로 최종 변환 보장
    return img.astype(np.float32)

def softmax(x):
    """결과값을 0~100% 사이의 확률로 변환합니다."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

if __name__ == '__main__':
    print("ONNX Runtime 엔진을 가동합니다.")
    # CPU 환경에서 최고 속도를 내도록 설정
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(ONNX_MODEL_PATH, session_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    print(f"테스트 이미지를 로드합니다. (경로: {TEST_IMAGE_PATH})")
    input_data = preprocess_image(TEST_IMAGE_PATH)
    
    print("INT8 모델 추론을 시작합니다.")
    # 순수 ONNX 추론 수행
    outputs = session.run(None, {input_name: input_data})
    
    # 확률 계산
    probabilities = softmax(outputs[0])[0]
    
    # 상위 3개 예측 결과 추출
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    
    print("\n[예측 결과 TOP 3]")
    print("-" * 30)
    for i, idx in enumerate(top3_indices):
        word = CLASS_NAMES[idx]
        confidence = probabilities[idx] * 100
        if i == 0:
            print(f"1위: {word} ({confidence:.2f}%)")
        elif i == 1:
            print(f"2위: {word} ({confidence:.2f}%)")
        else:
            print(f"3위: {word} ({confidence:.2f}%)")
    print("-" * 30)
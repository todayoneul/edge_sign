import os
import torch
import timm
import torch.nn as nn

# 경로 및 환경 설정 (Phase 1 분류 모델 기준 — 필요 시 경로 수정)
MODEL_PATH = "./checkpoints/w8a8_best.pth"
OUTPUT_ONNX_PATH = "./model_space/convnextv2_fp32.onnx"
NUM_CLASSES = 1000  # ImageNet 기본값; 파인튜닝 체크포인트 사용 시 수정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    os.makedirs("./model_space", exist_ok=True)
    print("1. FP32 원본 모델 구조를 생성합니다.")
    # [수정] Fake Quantization을 적용하지 않고 기본 모델을 생성합니다.
    model = timm.create_model('convnextv2_nano.fcmae_ft_in1k', pretrained=False)
    model.head.fc = nn.Linear(model.head.fc.in_features, NUM_CLASSES)
    
    print("2. QAT 학습된 가중치를 로드합니다.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    print("3. ONNX 포맷으로 모델을 추출합니다.")
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    torch.onnx.export(
        model, 
        dummy_input, 
        OUTPUT_ONNX_PATH,
        dynamo=False,              # 레거시(TorchScript) 익스포터 강제 사용 (양자화 호환성 보장)
        export_params=True,
        opset_version=14, # 안정적인 지원 버전으로 변경
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("FP32 ONNX 파일 생성이 완료되었습니다.")
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# 경로 세팅 (export_onnx.py 실행 후 생성된 FP32 파일 경로와 맞춰서 사용)
FP32_ONNX_PATH = "./model_space/convnextv2_fp32.onnx"
INT8_ONNX_PATH = "./model_space/convnextv2_int8.onnx"

if __name__ == '__main__':
    print("Real INT8 Quantization 프로세스")
    
    if not os.path.exists(FP32_ONNX_PATH):
        raise FileNotFoundError("FP32 ONNX 파일이 없습니다. Step 1을 먼저 실행해 주세요.")

    # 동적 양자화(Dynamic Quantization) 수행
    # - 가중치(Weight)는 INT8로 물리적 변환되어 저장됨
    # - 활성화 함수(Activation)는 추론 시점에 동적으로 INT8 변환되어 연산 가속
    quantize_dynamic(
        model_input=FP32_ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        weight_type=QuantType.QUInt8 # 양자화 타입 지정 (Unsigned / Signed INT8)
    )
    
    fp32_size = os.path.getsize(FP32_ONNX_PATH) / (1024 * 1024)
    int8_size = os.path.getsize(INT8_ONNX_PATH) / (1024 * 1024)
    
    print("\n양자화 압축 결과 리포트")
    print(f"   - 원본 FP32 모델 용량: {fp32_size:.2f} MB")
    print(f"   - 압축 INT8 모델 용량: {int8_size:.2f} MB")
    print(f"   - 압축률: 약 {fp32_size / int8_size:.1f}배")
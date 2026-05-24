import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
ONNX_PATH = MODEL_DIR / "korean_ocr.onnx"
QUANT_ONNX_PATH = MODEL_DIR / "korean_ocr_quant.onnx"

def main():
    if not ONNX_PATH.exists():
        print(f"Error: ONNX model not found at {ONNX_PATH}. Run training first.")
        return

    print("Checking original ONNX model...")
    original_size = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"Original ONNX Model Size: {original_size:.2f} MB")

    print("\nApplying W8A8 Dynamic Quantization...")
    # Apply dynamic quantization (weights to uint8, activations to float32 dynamically)
    # This is highly effective for reducing model size for deployment on CPUs/WebGL.
    quantize_dynamic(
        model_input=str(ONNX_PATH),
        model_output=str(QUANT_ONNX_PATH),
        weight_type=QuantType.QUInt8
    )

    print("\nQuantization complete.")
    quantized_size = QUANT_ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"Quantized ONNX Model Size: {quantized_size:.2f} MB")
    
    compression_ratio = original_size / quantized_size
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Quantized model saved to: {QUANT_ONNX_PATH}")

if __name__ == "__main__":
    main()

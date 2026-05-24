import os
from pathlib import Path
import onnx
from onnxconverter_common import convert_float_to_float16

DATA_DIR = Path(__file__).parent.parent / "data"
FP32_PATH = DATA_DIR / "mobilevit.onnx"
FP16_PATH = DATA_DIR / "mobilevit_quant.onnx" # Keep the _quant name for frontend loading compatibility

def main():
    if not FP32_PATH.exists():
        print(f"Error: FP32 MobileViT ONNX model not found at {FP32_PATH}. Please train and export first.")
        return

    print(f"Converting model {FP32_PATH} to FP16 half-precision for WebGPU acceleration...")
    
    try:
        # Load the ONNX model
        model = onnx.load(str(FP32_PATH))
        
        # Convert model to Float16
        # This reduces file size by 50% and dramatically speeds up WebGPU shader executions.
        model_fp16 = convert_float_to_float16(model)
        
        # Save the FP16 model
        onnx.save(model_fp16, str(FP16_PATH))
        
        # Verify sizes
        fp32_size = FP32_PATH.stat().st_size / 1024
        fp16_size = FP16_PATH.stat().st_size / 1024
        
        print(f"FP16 Conversion complete!")
        print(f"Original FP32 model size: {fp32_size:.2f} KB (~{fp32_size/1024:.2f} MB)")
        print(f"Converted FP16 model size: {fp16_size:.2f} KB (~{fp16_size/1024:.2f} MB) (Reduction: {(1 - fp16_size/fp32_size)*100:.1f}%)")
        
        # Simple structural integrity check
        onnx.checker.check_model(onnx.load(str(FP16_PATH)))
        print("ONNX FP16 model structure check passed.")
        
    except Exception as e:
        print(f"FP16 Conversion failed with error: {e}")

if __name__ == "__main__":
    main()

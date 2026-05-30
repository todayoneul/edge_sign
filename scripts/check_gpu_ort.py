"""onnxruntime-gpu가 CUDAExecutionProvider를 사용 가능한지 검증."""
import os
from pathlib import Path

# torch cu128가 동봉한 CUDA/cuDNN DLL을 onnxruntime-gpu가 찾도록 등록
import torch
_torch_lib = Path(torch.__file__).parent / "lib"
if _torch_lib.exists():
    os.add_dll_directory(str(_torch_lib))

import onnxruntime as ort
print("providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP 미가용"

import numpy as np
ROOT = Path(__file__).parent.parent
yolo = ROOT / "model_space" / "yolov8s_signs_w8a8.onnx"
sess = ort.InferenceSession(str(yolo), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("session providers:", sess.get_providers())
assert sess.get_providers()[0] == "CUDAExecutionProvider", "GPU 세션 생성 실패"
x = np.zeros((1, 3, 640, 640), np.float32)
sess.run(None, {sess.get_inputs()[0].name: x})
print("OK: GPU 추론 1프레임 성공")

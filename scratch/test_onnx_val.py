import os
import glob
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

def main():
    model_path = r"C:\Users\leegy\Desktop\CNN_Quant\web\korean_ocr_quant.onnx"
    idx_to_char_path = r"C:\Users\leegy\Desktop\CNN_Quant\web\idx_to_char.json"
    val_dir = r"C:\Users\leegy\Desktop\CNN_Quant\data\korean_ocr\val"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    if not os.path.exists(idx_to_char_path):
        print(f"idx_to_char.json not found at {idx_to_char_path}")
        return
        
    with open(idx_to_char_path, "r", encoding="utf-8") as f:
        idx_to_char = json.load(f)
        
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    paths = glob.glob(os.path.join(val_dir, "*", "*.jpg"))
    if not paths:
        print("No validation images found!")
        return
        
    print(f"Loaded model. Running evaluation on 100 random validation images...")
    np.random.seed(42)
    selected_paths = np.random.choice(paths, 100, replace=False)
    
    correct = 0
    for p in selected_paths:
        # Load image, convert to grayscale, resize to 64x64
        img = Image.open(p).convert("L")
        img_resized = img.resize((64, 64), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32)
        
        # Normalize: (val / 255.0 - 0.5) / 0.5
        arr = (arr / 255.0 - 0.5) / 0.5
        arr = np.expand_dims(arr, axis=(0, 1)) # shape: [1, 1, 64, 64]
        
        # Run inference
        outputs = session.run(["output"], {"input": arr})
        logits = outputs[0][0]
        
        pred_idx = np.argmax(logits)
        pred_char = idx_to_char[str(pred_idx)]
        
        # Ground truth class idx is the directory name
        gt_idx = int(os.path.basename(os.path.dirname(p)))
        gt_char = idx_to_char[str(gt_idx)]
        
        if pred_char == gt_char:
            correct += 1
        else:
            print(f"Mismatch: File {os.path.basename(p)} -> GT: {gt_char}, Pred: {pred_char}")
            
    print(f"Accuracy on 100 sample images: {correct}%")

if __name__ == "__main__":
    main()

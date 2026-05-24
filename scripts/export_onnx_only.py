import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from korean_ocr_model import KoreanOCRNet

# Reconfigure encoding to avoid Windows UnicodeEncodeError with emojis
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = KoreanOCRNet(num_classes=2350).to(device)
    model_path = Path(__file__).parent.parent / "models" / "korean_ocr_best.pth"
    onnx_path = Path(__file__).parent.parent / "models" / "korean_ocr.onnx"
    
    print(f"Loading best weights from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 64, 64, device=device)
    
    print(f"Exporting model to ONNX format at: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX export complete successfully!")

if __name__ == "__main__":
    main()

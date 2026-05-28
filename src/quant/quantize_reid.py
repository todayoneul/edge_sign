"""
SimpleReIDNet W8A8 양자화 + ONNX 내보내기 (E6 실험용).

사용법:
  python src/quant/quantize_reid.py            # W8A8 ONNX 내보내기
  python src/quant/quantize_reid.py --mode fp32 # FP32 ONNX 내보내기
"""
import argparse
import sys
import io
from pathlib import Path

import torch
import torch.nn as nn

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

MODEL_SPACE = ROOT / "model_space"
MODEL_SPACE.mkdir(parents=True, exist_ok=True)

from src.quant.quantize_recognizers import apply_w8a8_ptq, export_to_onnx


def build_reid(embed_dim: int = 128) -> nn.Module:
    from src.track.botsort import SimpleReIDNet
    return SimpleReIDNet(embed_dim=embed_dim).eval()


def quantize_reid(mode: str = "w8a8", embed_dim: int = 128) -> Path:
    """SimpleReIDNet → ONNX (fp32 or w8a8)."""
    model = build_reid(embed_dim)
    dummy = torch.zeros(1, 3, 64, 64)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SimpleReIDNet: {n_params:,} params, embed_dim={embed_dim}")

    if mode == "w8a8":
        n = apply_w8a8_ptq(model)
        print(f"  W8A8 fake-quant: {n} 레이어")
        out = MODEL_SPACE / "reid_net_w8a8.onnx"
    else:
        out = MODEL_SPACE / "reid_net_fp32.onnx"

    export_to_onnx(model, dummy, out,
                   input_names=["image"], output_names=["embedding"])

    size_kb = out.stat().st_size / 1024
    print(f"  -> {out.name}  ({size_kb:.1f} KB)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp32", "w8a8"], default="w8a8")
    parser.add_argument("--embed_dim", type=int, default=128)
    args = parser.parse_args()

    print(f"\n[ReID {args.mode.upper()}] SimpleReIDNet ONNX 내보내기")
    path = quantize_reid(args.mode, args.embed_dim)
    print(f"완료: {path}")


if __name__ == "__main__":
    main()

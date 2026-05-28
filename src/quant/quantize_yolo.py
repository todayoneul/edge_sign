"""
YOLOv8s 검출기 양자화 스크립트.

Phase 1의 base_W8A8.py / multimodal_w8a8_smoothquant.py 패턴을 YOLOv8에 포팅.
PyTorch 레벨 fake-quantization 후 ONNX 내보내기.

지원 모드:
  w8a8     — W8A8 PTQ (per-channel MinMax, Phase 1 base_W8A8 동일 방식)
  w4a16    — W4A16 PTQ (4-bit 가중치 / 16-bit 활성화)
  smoothquant — SmoothQuant + W8A8 (Phase 1 multimodal_smoothquant 동일 방식)

사용법:
  python src/quant/quantize_yolo.py --mode w8a8
  python src/quant/quantize_yolo.py --mode w4a16
  python src/quant/quantize_yolo.py --mode smoothquant --calib_batches 10
"""
import argparse
import sys
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

ROOT = Path(__file__).parent.parent.parent
MODEL_SPACE = ROOT / "model_space"
WEIGHTS = ROOT / "runs" / "detect" / "edge_sign_v2_e0_full3" / "weights" / "best.pt"
DATA_DIR = ROOT / "data" / "yolo_signs"
YOLO_DATASET = DATA_DIR / "dataset.yaml"

MODEL_SPACE.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────
# 공통 유틸
# ────────────────────────────────────────────

def load_yolo_model(weights=WEIGHTS):
    from ultralytics import YOLO
    return YOLO(str(weights))


def _is_quantizable(name: str, module: nn.Module) -> bool:
    """Conv2d / Linear 중 Detection Head 제외."""
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return False
    # YOLO detection head (model.22.*) — 마지막 출력 보호
    skip_keywords = ["dfl", "detect"]
    return not any(k in name.lower() for k in skip_keywords)


def export_to_onnx(yolo_model, out_name: str, opset: int = 14) -> Path:
    """수정된 PyTorch 모델을 ONNX로 내보내기 (ultralytics .export() 사용)."""
    result = yolo_model.export(
        format="onnx",
        imgsz=640,
        half=False,
        simplify=True,
        opset=opset,
        dynamic=False,
    )
    src = Path(result)
    dst = MODEL_SPACE / out_name
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  → 저장: {dst} ({size_mb:.2f} MB)")
    return dst


def export_nn_to_onnx(nn_model: nn.Module, out_name: str, opset: int = 14) -> Path:
    """
    이미 수정된 nn.Module을 torch.onnx.export로 직접 내보내기.
    SmoothQuant처럼 wrapper가 포함된 경우 사용 (ultralytics .export()의 fuse() 충돌 회피).
    """
    import onnx
    import onnxslim

    MODEL_SPACE.mkdir(parents=True, exist_ok=True)
    dst = MODEL_SPACE / out_name
    tmp = MODEL_SPACE / ("_tmp_" + out_name)

    nn_model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        torch.onnx.export(
            nn_model,
            dummy,
            str(tmp),
            opset_version=opset,
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
            dynamo=False,          # TorchScript 기반 exporter 사용 (PyTorch 2.x 호환)
        )

    # onnxslim으로 최적화
    try:
        slimmed = onnxslim.slim(str(tmp))
        onnx.save(slimmed, str(dst))
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.rename(dst)

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  → 저장: {dst} ({size_mb:.2f} MB)")
    return dst


def verify_onnx(path: Path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})
    print(f"  검증 OK: output shape = {out[0].shape}")


# ────────────────────────────────────────────
# W8A8 PTQ (Phase 1 base_W8A8.py 동일 방식)
# ────────────────────────────────────────────

def apply_w8a8_ptq(model_nn: nn.Module) -> int:
    """
    Conv2d / Linear 레이어에 per-channel MinMax W8A8 fake-quantization 적용.
    Phase 1의 apply_w8a8_ptq() 와 동일한 로직.
    """
    quantized = 0
    for name, module in model_nn.named_modules():
        if not _is_quantizable(name, module):
            continue
        with torch.no_grad():
            w = module.weight.data
            # Per-output-channel MinMax scale
            if w.dim() == 4:                          # Conv2d: [out, in, kH, kW]
                max_val = w.view(w.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
            else:                                      # Linear: [out, in]
                max_val = w.abs().max(dim=1)[0].view(-1, 1)
            scale = (max_val / 127.0).clamp(min=1e-8)
            q_w = torch.round(w / scale).clamp(-128, 127)
            module.weight.data = q_w * scale          # fake-dequant
        quantized += 1
    return quantized


def run_w8a8(weights=WEIGHTS):
    print("\n[W8A8 PTQ] 시작")
    yolo = load_yolo_model(weights)
    nn_model = yolo.model

    n = apply_w8a8_ptq(nn_model)
    print(f"  양자화 레이어: {n}개 (Detection Head 제외)")

    out = export_to_onnx(yolo, "yolov8s_signs_w8a8.onnx")
    verify_onnx(out)
    return out


# ────────────────────────────────────────────
# W4A16 PTQ (4-bit 가중치 / FP16 활성화)
# ────────────────────────────────────────────

def apply_w4a16_ptq(model_nn: nn.Module) -> int:
    """
    4-bit 가중치 양자화 시뮬레이션 (활성화는 FP32 유지).
    Phase 1의 W4A16 QAT 가중치 표현과 동일한 INT4 범위(-8 ~ 7).
    """
    quantized = 0
    for name, module in model_nn.named_modules():
        if not _is_quantizable(name, module):
            continue
        with torch.no_grad():
            w = module.weight.data
            if w.dim() == 4:
                max_val = w.view(w.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
            else:
                max_val = w.abs().max(dim=1)[0].view(-1, 1)
            scale = (max_val / 7.0).clamp(min=1e-8)   # INT4: [-8, 7]
            q_w = torch.round(w / scale).clamp(-8, 7)
            module.weight.data = q_w * scale
        quantized += 1
    return quantized


def run_w4a16(weights=WEIGHTS):
    print("\n[W4A16 PTQ] 시작")
    yolo = load_yolo_model(weights)
    nn_model = yolo.model

    n = apply_w4a16_ptq(nn_model)
    print(f"  양자화 레이어: {n}개 (4-bit 가중치)")

    out = export_to_onnx(yolo, "yolov8s_signs_w4a16.onnx")
    verify_onnx(out)
    return out


# ────────────────────────────────────────────
# SmoothQuant + W8A8 (Phase 1 동일 방식)
# ────────────────────────────────────────────

def _build_calib_loader(num_batches=10, batch_size=4):
    """val 이미지를 캘리브레이션 데이터로 사용."""
    import cv2
    from torch.utils.data import DataLoader, Dataset

    img_dir = DATA_DIR / "images" / "val"
    img_paths = sorted(img_dir.rglob("*.jpg"))[:num_batches * batch_size]

    class YOLOImageDataset(Dataset):
        def __init__(self, paths, imgsz=640):
            self.paths = paths
            self.imgsz = imgsz

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = cv2.imread(str(self.paths[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.imgsz, self.imgsz))
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return tensor

    ds = YOLOImageDataset(img_paths)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


class _SmoothWrapper(nn.Module):
    """
    SmoothQuant Wrapper: forward에서 입력을 1/s로 스케일링 후 가중치(s 흡수+W8) 레이어 실행.
    ONNX export 시 스케일 나눗셈이 그래프에 포함됨 (Phase 1 SmoothQuantWrapper 동일 방식).
    """
    def __init__(self, module: nn.Module, smooth_scale: torch.Tensor):
        super().__init__()
        self.module = module
        self.register_buffer("smooth_scale", smooth_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x / self.smooth_scale.view(1, -1, 1, 1)
        else:
            x = x / self.smooth_scale
        return self.module(x)


def apply_smoothquant(model_nn: nn.Module, calib_loader, alpha: float = 0.5) -> int:
    """
    SmoothQuant: 활성화 캘리브레이션 → wrapper로 입력 스케일링 + 가중치 흡수 + W8A8.
    wrapper.forward()에 x/s 연산이 포함돼 ONNX export 시 그래프에 반영됨.
    """
    device = next(model_nn.parameters()).device
    model_nn.eval()

    # 1. 활성화 최대값 수집 (per-input-channel)
    act_max: dict = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().abs()
            if x.dim() == 4:
                ch_max = x.amax(dim=(0, 2, 3))
            else:
                ch_max = x.amax(dim=0) if x.dim() >= 2 else x
            act_max[name] = torch.max(act_max[name], ch_max) if name in act_max else ch_max
        return hook

    target_names = [
        name for name, m in model_nn.named_modules() if _is_quantizable(name, m)
    ]
    target_mods = dict(model_nn.named_modules())

    for name in target_names:
        hooks.append(target_mods[name].register_forward_hook(make_hook(name)))

    print(f"  캘리브레이션 중 ({len(calib_loader)} 배치)...")
    with torch.no_grad():
        for batch in calib_loader:
            model_nn(batch.to(device))
    for h in hooks:
        h.remove()

    # 2. Wrapper 교체 + 가중치 W8A8 적용
    def _set_module(root, dotted_name, new_module):
        parts = dotted_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    quantized = 0
    for name in target_names:
        if name not in act_max:
            continue
        module = target_mods[name]
        w = module.weight.data
        a_max = act_max[name].to(device).clamp(min=1e-8)

        if w.dim() == 4:
            in_ch = w.size(1)
        else:
            in_ch = w.size(1)

        # a_max 채널 수 맞추기
        if a_max.shape[0] != in_ch:
            if a_max.shape[0] > in_ch:
                a_max = a_max[:in_ch]
            else:
                pad = a_max.mean().expand(in_ch - a_max.shape[0])
                a_max = torch.cat([a_max, pad])

        # per-input-channel weight max
        if w.dim() == 4:
            w_max = w.abs().amax(dim=(0, 2, 3)).clamp(min=1e-8)
        else:
            w_max = w.abs().amax(dim=0).clamp(min=1e-8)

        smooth_s = (a_max ** alpha) / (w_max ** (1 - alpha) + 1e-8)
        smooth_s = smooth_s.clamp(1e-3, 1e3)

        # 가중치에 smooth_s 흡수 + W8 fake-quant
        with torch.no_grad():
            if w.dim() == 4:
                w_scaled = w * smooth_s.view(1, in_ch, 1, 1)
                out_max = w_scaled.view(w.size(0), -1).abs().max(dim=1)[0].view(-1,1,1,1)
            else:
                w_scaled = w * smooth_s.view(1, in_ch)
                out_max = w_scaled.abs().max(dim=1)[0].view(-1, 1)
            q_scale = (out_max / 127.0).clamp(min=1e-8)
            q_w = torch.round(w_scaled / q_scale).clamp(-128, 127)
            module.weight.data = q_w * q_scale

        # Wrapper 교체 (ONNX export 시 x/smooth_s 연산 포함)
        wrapper = _SmoothWrapper(module, smooth_s)
        _set_module(model_nn, name, wrapper)
        quantized += 1

    return quantized


def run_smoothquant(weights=WEIGHTS, calib_batches=10, alpha=0.5):
    print("\n[SmoothQuant + W8A8] 시작")
    yolo = load_yolo_model(weights)
    nn_model = yolo.model

    # ★ fuse() 먼저: Conv+BN 융합 후 SmoothWrapper 교체
    #   그래야 ultralytics fuse() 재호출 없이 torch.onnx.export 가능
    nn_model = nn_model.fuse()
    nn_model.eval()

    calib_loader = _build_calib_loader(num_batches=calib_batches)
    n = apply_smoothquant(nn_model, calib_loader, alpha=alpha)
    print(f"  SmoothQuant 적용 레이어: {n}개 (alpha={alpha})")

    # torch.onnx.export 직접 사용 (ultralytics .export()의 fuse() 재호출 회피)
    out = export_nn_to_onnx(nn_model, "yolov8s_signs_smoothquant.onnx")
    verify_onnx(out)
    return out


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv8s 검출기 양자화")
    parser.add_argument("--mode", choices=["w8a8", "w4a16", "smoothquant", "all"],
                        default="all", help="양자화 모드")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS),
                        help="학습된 best.pt 경로")
    parser.add_argument("--calib_batches", type=int, default=10,
                        help="SmoothQuant 캘리브레이션 배치 수")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant alpha (0=weight만, 1=activation만)")
    args = parser.parse_args()

    modes = ["w8a8", "w4a16", "smoothquant"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "w8a8":
            run_w8a8(args.weights)
        elif mode == "w4a16":
            run_w4a16(args.weights)
        elif mode == "smoothquant":
            run_smoothquant(args.weights, args.calib_batches, args.alpha)

    print("\n모든 양자화 완료. model_space/ 디렉토리 확인:")
    for f in sorted(MODEL_SPACE.glob("yolov8s_signs_*.onnx")):
        print(f"  {f.name}: {f.stat().st_size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()

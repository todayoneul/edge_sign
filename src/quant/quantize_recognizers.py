"""
KoreanOCRNet + TrafficSignNet 양자화 실험 스크립트.

Phase 1 base_W8A8.py / base_train_1bit_kd.py 패턴 재활용.
fake-quant PTQ (W8A8 / W4A16 / 1-Bit) + ONNX 내보내기 + val 평가.

실험 매핑:
  E2: FP16 검출기 + W8A8 인식기 → 본 스크립트로 w8a8 내보내기 후 정확도 측정
  E3: W8A8 전체 → 검출기(E1) + 인식기(E2)
  E4-recog: W4A16 인식기
  E7-recog: 1-Bit 인식기

사용법:
  python src/quant/quantize_recognizers.py              # 전체 (w8a8/w4a16/1bit)
  python src/quant/quantize_recognizers.py --mode w8a8  # 특정 모드
  python src/quant/quantize_recognizers.py --eval_only  # 기존 ONNX 평가만
"""
import argparse
import sys
import io
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Windows 터미널 인코딩 문제 해결
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

MODEL_SPACE  = ROOT / "model_space"
MODEL_DIR    = ROOT / "models"
OCR_CKPT     = MODEL_DIR / "korean_ocr_best.pth"
TSIGN_CKPT   = MODEL_SPACE / "traffic_sign_net_best.pth"
OCR_DATA     = ROOT / "data" / "korean_ocr"
GTSDB_DIR    = ROOT / "data" / "GTSDB" / "FullIJCNN2013"

MODEL_SPACE.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 공통 양자화 함수 (Phase 1 패턴 재활용)
# ─────────────────────────────────────────────

def _quantizable(name: str, module: nn.Module) -> bool:
    """Conv2d / Linear 레이어만 대상 (BN, Activation 제외)."""
    return isinstance(module, (nn.Conv2d, nn.Linear))


def apply_w8a8_ptq(model: nn.Module) -> int:
    """Per-output-channel MinMax W8A8 fake-quantization (base_W8A8.py 동일 방식)."""
    count = 0
    for name, module in model.named_modules():
        if not _quantizable(name, module):
            continue
        with torch.no_grad():
            w = module.weight.data
            if w.dim() == 4:   # Conv2d
                max_val = w.view(w.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
            else:               # Linear
                max_val = w.abs().max(dim=1)[0].view(-1, 1)
            scale = (max_val / 127.0).clamp(min=1e-8)
            q_w = torch.round(w / scale).clamp(-128, 127)
            module.weight.data = q_w * scale
        count += 1
    return count


def apply_w4a16_ptq(model: nn.Module) -> int:
    """Per-output-channel MinMax W4A16 fake-quantization."""
    count = 0
    for name, module in model.named_modules():
        if not _quantizable(name, module):
            continue
        with torch.no_grad():
            w = module.weight.data
            if w.dim() == 4:
                max_val = w.view(w.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
            else:
                max_val = w.abs().max(dim=1)[0].view(-1, 1)
            scale = (max_val / 7.0).clamp(min=1e-8)
            q_w = torch.round(w / scale).clamp(-8, 7)
            module.weight.data = q_w * scale
        count += 1
    return count


def apply_1bit_ptq(model: nn.Module) -> int:
    """PTB: Post-Training Binarization (sign(W) × ||W||_1 / n, base_train_1bit_kd.py 패턴)."""
    count = 0
    for name, module in model.named_modules():
        if not _quantizable(name, module):
            continue
        with torch.no_grad():
            w = module.weight.data
            if w.dim() == 4:
                scale = w.abs().mean(dim=(1, 2, 3), keepdim=True)
            else:
                scale = w.abs().mean(dim=1, keepdim=True)
            binary_w = torch.sign(w)
            binary_w[binary_w == 0] = 1.0   # 0은 +1로 처리
            module.weight.data = binary_w * scale
        count += 1
    return count


# ─────────────────────────────────────────────
# ONNX 내보내기
# ─────────────────────────────────────────────

def export_to_onnx(model: nn.Module, dummy: torch.Tensor, out_path: Path,
                   input_names: list, output_names: list, opset: int = 14):
    """PyTorch 모델 → ONNX (dynamo=False)."""
    try:
        import onnxslim
        use_slim = True
    except ImportError:
        use_slim = False

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    torch.onnx.export(
        model.cpu().eval(), dummy.cpu(),
        str(tmp_path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch"}, output_names[0]: {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )

    if use_slim:
        onnxslim.slim(str(tmp_path), str(out_path))
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.rename(out_path)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"    -> {out_path.name}  ({size_mb:.3f} MB)")
    return out_path


# ─────────────────────────────────────────────
# KoreanOCRNet 로드 + 양자화 + ONNX
# ─────────────────────────────────────────────

def load_ocr_model() -> nn.Module:
    from src.korean_ocr_model import KoreanOCRNet
    model = KoreanOCRNet(num_classes=2350)
    if OCR_CKPT.exists():
        state = torch.load(str(OCR_CKPT), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  OCR 체크포인트 로드: {OCR_CKPT}")
    else:
        print(f"  [WARN] OCR 체크포인트 없음: {OCR_CKPT}")
    return model.eval()


def quantize_ocr(mode: str) -> Path:
    """KoreanOCRNet 양자화 + ONNX 내보내기."""
    model = load_ocr_model()
    dummy = torch.zeros(1, 1, 64, 64)

    if mode == "fp32":
        out = MODEL_SPACE / "korean_ocr_net_fp32.onnx"
    elif mode == "w8a8":
        n = apply_w8a8_ptq(model)
        print(f"    W8A8 fake-quant: {n} 레이어")
        out = MODEL_SPACE / "korean_ocr_net_w8a8.onnx"
    elif mode == "w4a16":
        n = apply_w4a16_ptq(model)
        print(f"    W4A16 fake-quant: {n} 레이어")
        out = MODEL_SPACE / "korean_ocr_net_w4a16.onnx"
    elif mode == "1bit":
        n = apply_1bit_ptq(model)
        print(f"    1-Bit PTB: {n} 레이어")
        out = MODEL_SPACE / "korean_ocr_net_1bit.onnx"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return export_to_onnx(model, dummy, out, ["image"], ["logits"])


# ─────────────────────────────────────────────
# TrafficSignNet 로드 + 양자화 + ONNX
# ─────────────────────────────────────────────

def load_tsign_model(num_classes: int = 43) -> nn.Module:
    from src.model import TrafficSignNet
    model = TrafficSignNet(num_classes=num_classes)
    if TSIGN_CKPT.exists():
        ckpt = torch.load(str(TSIGN_CKPT), map_location="cpu", weights_only=False)
        nc = ckpt.get("num_classes", num_classes)
        if nc != num_classes:
            model = TrafficSignNet(num_classes=nc)
        model.load_state_dict(ckpt["model_state"])
        print(f"  TrafficSignNet 체크포인트 로드: {TSIGN_CKPT} (num_classes={nc})")
    else:
        print(f"  [WARN] TrafficSignNet 체크포인트 없음: {TSIGN_CKPT}")
    return model.eval()


def quantize_tsign(mode: str) -> Path:
    """TrafficSignNet 양자화 + ONNX 내보내기."""
    model = load_tsign_model()
    dummy = torch.zeros(1, 3, 32, 32)

    if mode == "fp32":
        out = MODEL_SPACE / "traffic_sign_net_fp32.onnx"   # 이미 존재
    elif mode == "w8a8":
        n = apply_w8a8_ptq(model)
        print(f"    W8A8 fake-quant: {n} 레이어")
        out = MODEL_SPACE / "traffic_sign_net_w8a8.onnx"
    elif mode == "w4a16":
        n = apply_w4a16_ptq(model)
        print(f"    W4A16 fake-quant: {n} 레이어")
        out = MODEL_SPACE / "traffic_sign_net_w4a16.onnx"
    elif mode == "1bit":
        n = apply_1bit_ptq(model)
        print(f"    1-Bit PTB: {n} 레이어")
        out = MODEL_SPACE / "traffic_sign_net_1bit.onnx"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mode != "fp32" or not out.exists():
        export_to_onnx(model, dummy, out, ["images"], ["logits"])
    return out


# ─────────────────────────────────────────────
# 평가: KoreanOCRNet (val set)
# ─────────────────────────────────────────────

def eval_ocr_onnx(onnx_path: Path, max_samples: int = 5000) -> dict:
    """KoreanOCRNet ONNX 평가 (data/korean_ocr/val/)."""
    import onnxruntime as ort
    from torchvision import transforms, datasets

    val_dir = OCR_DATA / "val"
    if not val_dir.exists():
        print(f"  [SKIP] OCR val 데이터 없음: {val_dir}")
        return {}

    # NumericalImageFolder: 클래스 이름을 숫자로 정렬
    class NumericalImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            import os
            classes = sorted(
                [d.name for d in os.scandir(directory) if d.is_dir()], key=int
            )
            class_to_idx = {cls: int(cls) for cls in classes}
            return classes, class_to_idx

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_ds = NumericalImageFolder(root=str(val_dir), transform=transform)
    # 평가 속도를 위해 최대 max_samples개만 사용
    indices = list(range(min(max_samples, len(val_ds))))
    from torch.utils.data import Subset, DataLoader
    subset = Subset(val_ds, indices)
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    correct1 = correct5 = total = 0
    t0 = time.time()

    for imgs, labels in loader:
        imgs_np = imgs.numpy()
        out = sess.run([output_name], {input_name: imgs_np})[0]  # [B, 2350]
        top5 = np.argsort(out, axis=1)[:, -5:][:, ::-1]         # [B, 5] 내림차순
        labels_np = labels.numpy()
        correct1 += (top5[:, 0] == labels_np).sum()
        correct5 += sum(labels_np[i] in top5[i] for i in range(len(labels_np)))
        total += len(labels_np)

    elapsed = time.time() - t0
    top1 = correct1 / total * 100
    top5 = correct5 / total * 100
    fps  = total / elapsed

    return {"top1": round(top1, 2), "top5": round(top5, 2),
            "samples": total, "fps": round(fps, 1)}


# ─────────────────────────────────────────────
# 평가: TrafficSignNet (GTSDB val 크롭)
# ─────────────────────────────────────────────

def eval_tsign_onnx(onnx_path: Path) -> dict:
    """TrafficSignNet ONNX 평가 (GTSDB val 크롭)."""
    import onnxruntime as ort
    import cv2
    from src.detect.train_traffic_sign_net import GTSDBCropDataset
    from torch.utils.data import DataLoader, random_split

    full_ds = GTSDBCropDataset(GTSDB_DIR, img_size=32, augment=False)
    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    _, val_ds = random_split(full_ds, [n_train, n_val],
                             generator=torch.Generator().manual_seed(42))

    # DataLoader 대신 직접 배치 처리 (PIL → numpy 변환 포함)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    correct1 = correct5 = 0
    t0 = time.time()

    for img_tensor, label in val_ds:
        # img_tensor: [3, 32, 32] float32
        img_np = img_tensor.numpy()[np.newaxis]  # [1, 3, 32, 32]
        out = sess.run([output_name], {input_name: img_np})[0]  # [1, 43]
        top5 = np.argsort(out[0])[::-1][:5]
        if top5[0] == label:
            correct1 += 1
        if label in top5:
            correct5 += 1

    total = len(val_ds)
    elapsed = time.time() - t0
    top1 = correct1 / total * 100
    top5 = correct5 / total * 100

    return {"top1": round(top1, 2), "top5": round(top5, 2),
            "samples": total, "fps": round(total / elapsed, 1)}


# ─────────────────────────────────────────────
# 메인: 양자화 + 평가 + 결과 출력
# ─────────────────────────────────────────────

MODES = ["fp32", "w8a8", "w4a16", "1bit"]

EXP_MAP = {
    "fp32":  "E0 기준선",
    "w8a8":  "E2/E3 W8A8",
    "w4a16": "E4 W4A16",
    "1bit":  "E7 1-Bit PTB",
}


def run_all(modes: list[str], eval_ocr: bool = True, eval_tsign: bool = True,
            ocr_samples: int = 5000):
    results = []

    for mode in modes:
        print(f"\n{'='*55}")
        print(f"[{mode.upper()}] KoreanOCRNet + TrafficSignNet")
        print(f"{'='*55}")

        # ── KoreanOCRNet
        print(f"  [OCR] 양자화 내보내기...")
        try:
            ocr_path = quantize_ocr(mode)
        except Exception as e:
            print(f"  [OCR] 내보내기 실패: {e}")
            ocr_path = None

        ocr_metrics = {}
        if eval_ocr and ocr_path and ocr_path.exists():
            print(f"  [OCR] 평가 중 (최대 {ocr_samples}개)...")
            ocr_metrics = eval_ocr_onnx(ocr_path, max_samples=ocr_samples)
            if ocr_metrics:
                print(f"    Top-1={ocr_metrics['top1']:.2f}%  "
                      f"Top-5={ocr_metrics['top5']:.2f}%  "
                      f"({ocr_metrics['samples']}개, {ocr_metrics['fps']:.0f} FPS)")

        # ── TrafficSignNet
        print(f"  [TrafficSign] 양자화 내보내기...")
        try:
            ts_path = quantize_tsign(mode)
        except Exception as e:
            print(f"  [TrafficSign] 내보내기 실패: {e}")
            ts_path = None

        ts_metrics = {}
        if eval_tsign and ts_path and ts_path.exists():
            print(f"  [TrafficSign] 평가 중...")
            ts_metrics = eval_tsign_onnx(ts_path)
            if ts_metrics:
                print(f"    Top-1={ts_metrics['top1']:.2f}%  "
                      f"Top-5={ts_metrics['top5']:.2f}%  "
                      f"({ts_metrics['samples']}개, {ts_metrics['fps']:.0f} FPS)")

        # OCR ONNX 크기
        ocr_size = ocr_path.stat().st_size / 1024 / 1024 if ocr_path and ocr_path.exists() else 0
        ts_size  = ts_path.stat().st_size / 1024 / 1024  if ts_path  and ts_path.exists()  else 0

        results.append({
            "mode": mode,
            "exp": EXP_MAP.get(mode, mode),
            "ocr_top1": ocr_metrics.get("top1", "—"),
            "ocr_top5": ocr_metrics.get("top5", "—"),
            "ts_top1":  ts_metrics.get("top1", "—"),
            "ts_top5":  ts_metrics.get("top5", "—"),
            "ocr_size": round(ocr_size, 3),
            "ts_size":  round(ts_size, 3),
        })

    # ── 결과 요약표
    print(f"\n{'='*70}")
    print("인식기 양자화 결과 요약")
    print(f"{'='*70}")
    hdr = (f"{'모드':<8} {'실험':<12} "
           f"{'OCR Top1':>9} {'OCR Top5':>9} {'TS Top1':>8} {'TS Top5':>8} "
           f"{'OCR MB':>7} {'TS MB':>6}")
    print(hdr)
    print("-" * 70)

    e0 = next((r for r in results if r["mode"] == "fp32"), None)

    for r in results:
        ocr1 = r['ocr_top1']
        ts1  = r['ts_top1']
        # 변화량
        if e0 and r["mode"] != "fp32":
            if isinstance(ocr1, float) and isinstance(e0["ocr_top1"], float):
                ocr1_str = f"{ocr1:.1f} ({ocr1 - e0['ocr_top1']:+.1f})"
            else:
                ocr1_str = str(ocr1)
            if isinstance(ts1, float) and isinstance(e0["ts_top1"], float):
                ts1_str = f"{ts1:.1f} ({ts1 - e0['ts_top1']:+.1f})"
            else:
                ts1_str = str(ts1)
        else:
            ocr1_str = f"{ocr1:.1f}" if isinstance(ocr1, float) else str(ocr1)
            ts1_str  = f"{ts1:.1f}"  if isinstance(ts1, float) else str(ts1)

        print(f"{r['mode']:<8} {r['exp']:<12} "
              f"{ocr1_str:>9} {str(r['ocr_top5']):>9} "
              f"{ts1_str:>8} {str(r['ts_top5']):>8} "
              f"{r['ocr_size']:>7.3f} {r['ts_size']:>6.3f}")

    print("\n[민감도 분석]")
    if e0:
        for r in results:
            if r["mode"] == "fp32":
                continue
            for name, base_key, key in [
                ("OCR Top-1", "ocr_top1", "ocr_top1"),
                ("TrafficSign Top-1", "ts_top1", "ts_top1"),
            ]:
                bv = e0[base_key]
                v  = r[key]
                if isinstance(v, float) and isinstance(bv, float):
                    d = v - bv
                    p = d / bv * 100
                    print(f"  {r['mode']} {name}: {bv:.1f}% -> {v:.1f}% ({d:+.1f}pp, {p:+.1f}%)")

    print("\n위 결과를 docs/EXPERIMENTS.md 인식 결과 표에 기입하세요.")
    return results


def main():
    parser = argparse.ArgumentParser(description="인식기 양자화 + 평가")
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    parser.add_argument("--no_ocr",   action="store_true", help="OCR 평가 건너뛰기")
    parser.add_argument("--no_tsign", action="store_true", help="TrafficSign 평가 건너뛰기")
    parser.add_argument("--ocr_samples", type=int, default=5000,
                        help="OCR val 최대 샘플 수 (기본 5000)")
    args = parser.parse_args()

    modes = MODES if args.mode == "all" else [args.mode]
    run_all(modes,
            eval_ocr=not args.no_ocr,
            eval_tsign=not args.no_tsign,
            ocr_samples=args.ocr_samples)


if __name__ == "__main__":
    main()

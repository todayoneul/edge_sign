"""
[Phase 8] 한국 표지판/신호등 ROI 분류기 학습 (14클래스)

독일 GTSDB 분류기(TrafficSignNet 43class)를 대체.
data/roi_cls (prepare_korean_traffic.py 출력)에서 학습.

전처리는 추론(e2e_pipeline._run_tsign)과 동일: (rgb/255 - 0.5) / 0.5
클래스 불균형은 CrossEntropy class weight로 보정.

사용법:
  python scripts/train_korean_classifier.py --epochs 40
  python scripts/train_korean_classifier.py --export_only   # ONNX만
출력:
  model_space/korean_sign_net_fp32.onnx
  model_space/korean_sign_net_best.pth
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.model import TrafficSignNet

ROI_DIR   = ROOT / "data" / "roi_cls"
CKPT_OUT  = ROOT / "model_space" / "korean_sign_net_best.pth"
ONNX_OUT  = ROOT / "model_space" / "korean_sign_net_fp32.onnx"
IMG_SIZE  = 32

CLASSES = json.loads((ROI_DIR / "classes.json").read_text(encoding="utf-8"))["names"]
NUM_CLASSES = len(CLASSES)


class ROIDataset(Dataset):
    def __init__(self, split: str, augment: bool):
        self.augment = augment
        self.samples = []  # (path, cls_idx)
        # ROI 폴더는 클래스 인덱스(ASCII) — 한글 폴더는 OpenCV read 실패
        for ci in range(NUM_CLASSES):
            for p in (ROI_DIR / split / f"{ci:02d}").glob("*.jpg"):
                self.samples.append((p, ci))
        self.counts = np.bincount([c for _, c in self.samples], minlength=NUM_CLASSES)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5  # 추론과 동일 정규화
        if self.augment:
            if np.random.rand() < 0.5:
                rgb = rgb + np.random.uniform(-0.15, 0.15)  # 밝기 jitter
            # 좌우 반전은 사용 안 함: 속도숫자·좌회전 화살표·방향 지시표지는
            # 반전 시 의미가 바뀌어(좌→우) 라벨 노이즈가 됨.
        t = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        return t, cls


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_cls_c = np.zeros(NUM_CLASSES); per_cls_t = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            correct += (pred == y).sum(); total += len(y)
            for yi, pi in zip(y, pred):
                per_cls_t[yi] += 1
                if yi == pi: per_cls_c[yi] += 1
    acc = correct / max(1, total)
    per = per_cls_c / np.maximum(1, per_cls_t)
    return acc, per


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ROIDataset("train", augment=True)
    val_ds   = ROIDataset("val", augment=False)
    print(f"클래스 {NUM_CLASSES}, Train {len(train_ds):,}  Val {len(val_ds):,}  device={device}")
    print("train 클래스 분포:", dict(zip(CLASSES, train_ds.counts.tolist())))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = TrafficSignNet(num_classes=NUM_CLASSES).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"파라미터: {nparam:,}")

    # 클래스 가중치 (불균형 보정): inverse freq
    freq = np.maximum(1, train_ds.counts)
    w = (freq.sum() / (NUM_CLASSES * freq)).astype(np.float32)
    w = np.clip(w, 0.3, 5.0)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
        sched.step()
        acc, per = evaluate(model, val_loader, device)
        if acc > best:
            best = acc
            CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(),
                        "num_classes": NUM_CLASSES, "classes": CLASSES,
                        "val_acc": acc}, str(CKPT_OUT))
        if ep % 2 == 0 or ep == 1:
            worst = sorted(zip(CLASSES, per), key=lambda t: t[1])[:3]
            print(f"ep{ep:2d} loss={tot_loss/len(train_ds):.3f} "
                  f"val_acc={acc:.4f} best={best:.4f}  약점:{[(c,round(float(p),2)) for c,p in worst]}")
    print(f"\n[완료] best_val_acc={best:.4f}  저장: {CKPT_OUT}")
    return best


def export_onnx():
    ckpt = torch.load(str(CKPT_OUT), map_location="cpu", weights_only=False)
    model = TrafficSignNet(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"체크포인트 로드 val_acc={ckpt.get('val_acc'):.4f}")
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)

    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".onnx"))
    torch.onnx.export(model, dummy, str(tmp), opset_version=14,
                      input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                      do_constant_folding=True, dynamo=False)
    # onnxslim 있으면 슬림, 없으면 그대로 사용 (선택적 최적화)
    try:
        import onnxslim
        onnxslim.slim(str(tmp), str(ONNX_OUT))
        tmp.unlink(missing_ok=True)
    except ImportError:
        import shutil
        shutil.move(str(tmp), str(ONNX_OUT))
        print("  (onnxslim 미설치 — 슬림 생략)")
    print(f"[OK] ONNX: {ONNX_OUT} ({ONNX_OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--export_only", action="store_true")
    args = ap.parse_args()
    if not args.export_only:
        train(args)
    print("\n[ONNX 내보내기]")
    export_onnx()

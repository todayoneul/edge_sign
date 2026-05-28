"""
TrafficSignNet 학습 스크립트 (GTSDB 43-class).

GTSDB gt.txt에서 교통표지판 크롭 → 분류 학습 → ONNX 내보내기.

사용법:
  python src/detect/train_traffic_sign_net.py           # 학습 + ONNX 내보내기
  python src/detect/train_traffic_sign_net.py --epochs 30
  python src/detect/train_traffic_sign_net.py --export_only  # 학습 건너뛰고 ONNX만
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

GTSDB_DIR  = ROOT / "data" / "GTSDB" / "FullIJCNN2013"
MODEL_OUT  = ROOT / "model_space" / "traffic_sign_net_fp32.onnx"
CKPT_OUT   = ROOT / "model_space" / "traffic_sign_net_best.pth"
NUM_CLASSES = 43   # GTSDB 전체 클래스 사용
IMG_SIZE    = 32   # TrafficSignNet 입력 크기

# ──────────────────────────────────────────────
# GTSDB 43-class 이름 (요약)
# ──────────────────────────────────────────────
CLASS_NAMES = {
    0: "Speed limit (20km/h)",   1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",   3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",   5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",  9: "No passing",
    10: "No passing for heavy vehicles", 11: "Right of way at next intersection",
    12: "Priority road",         13: "Yield",
    14: "Stop",                  15: "No vehicles",
    16: "No heavy vehicles",     17: "No entry",
    18: "General caution",       19: "Dangerous curve (left)",
    20: "Dangerous curve (right)", 21: "Double curve",
    22: "Bumpy road",            23: "Slippery road",
    24: "Road narrows (right)",  25: "Road work",
    26: "Traffic signals",       27: "Pedestrians",
    28: "Children crossing",     29: "Bicycles crossing",
    30: "Beware of ice/snow",    31: "Wild animals crossing",
    32: "End of all restrictions", 33: "Turn right ahead",
    34: "Turn left ahead",       35: "Ahead only",
    36: "Go straight or right",  37: "Go straight or left",
    38: "Keep right",            39: "Keep left",
    40: "Roundabout mandatory",  41: "End of no passing",
    42: "End of no passing (heavy vehicles)",
}


# ──────────────────────────────────────────────
# 1. 데이터셋
# ──────────────────────────────────────────────

class GTSDBCropDataset(Dataset):
    """GTSDB gt.txt에서 교통표지판 크롭 데이터셋."""

    def __init__(self, gtsdb_dir: Path, img_size: int = 32, augment: bool = True):
        self.img_size = img_size
        self.augment  = augment
        self.samples: list[tuple[np.ndarray, int]] = []
        self._load(gtsdb_dir)

    def _load(self, gtsdb_dir: Path):
        gt_file = gtsdb_dir / "gt.txt"
        if not gt_file.exists():
            raise FileNotFoundError(f"gt.txt 없음: {gt_file}")

        img_cache: dict[str, np.ndarray] = {}
        lines_ok = 0

        with open(gt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) != 6:
                    continue
                fname, x1, y1, x2, y2, cls = parts
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(cls)

                img_path = gtsdb_dir / fname
                if not img_path.exists():
                    continue

                if fname not in img_cache:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img_cache[fname] = img

                img = img_cache[fname]
                # 패딩 추가 후 크롭 (sign 주변 5% 마진)
                h, w = img.shape[:2]
                margin_x = max(2, int((x2 - x1) * 0.05))
                margin_y = max(2, int((y2 - y1) * 0.05))
                cx1 = max(0, x1 - margin_x)
                cy1 = max(0, y1 - margin_y)
                cx2 = min(w, x2 + margin_x)
                cy2 = min(h, y2 + margin_y)

                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                self.samples.append((crop, cls))
                lines_ok += 1

        print(f"  GTSDB 크롭 로드: {lines_ok}개 (유니크 이미지 {len(img_cache)})")

    def __len__(self):
        return len(self.samples)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """BGR→RGB, [0,1] 정규화, augment."""
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std

        if self.augment:
            # 수평 뒤집기 (교통 표지판은 대칭 아닌 것도 있으므로 50% 확률)
            if np.random.rand() < 0.3:
                img = img[:, ::-1, :]
            # 밝기 jitter
            img = img + np.random.uniform(-0.1, 0.1)

        tensor = torch.from_numpy(img.transpose(2, 0, 1).copy())
        return tensor

    def __getitem__(self, idx):
        crop, cls = self.samples[idx]
        return self._preprocess(crop), cls


# ──────────────────────────────────────────────
# 2. 모델 (num_classes=43 버전)
# ──────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """TrafficSignNet (src/model.py 기반, num_classes 가변)."""
    from src.model import TrafficSignNet
    model = TrafficSignNet(num_classes=num_classes)
    return model


# ──────────────────────────────────────────────
# 3. 학습
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"TrafficSignNet train: {NUM_CLASSES}class, device={device}")
    print(f"{'='*55}")

    # 데이터셋
    full_ds = GTSDBCropDataset(GTSDB_DIR, img_size=IMG_SIZE, augment=True)
    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False   # val은 augment 없음

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0)

    print(f"  Train {n_train}  Val {n_val}")

    model = build_model(NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  파라미터: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
            train_correct += (out.argmax(1) == labels).sum().item()
        scheduler.step()

        train_loss /= n_train
        train_acc   = train_correct / n_train

        # ── Val
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
        val_acc = val_correct / n_val

        marker = " ← best" if val_acc > best_acc else ""
        print(f"  Ep {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}{marker}")

        if val_acc > best_acc:
            best_acc = val_acc
            CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES,
            }, str(CKPT_OUT))

    print(f"\n  [Done] train complete  best_val_acc={best_acc:.4f}")
    print(f"  저장: {CKPT_OUT}")
    return best_acc


# ──────────────────────────────────────────────
# 4. ONNX 내보내기
# ──────────────────────────────────────────────

def export_onnx(ckpt_path: Path | None = None):
    """best.pth → ONNX (opset 14)."""
    import onnxslim

    model = build_model(NUM_CLASSES)
    if ckpt_path and ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        nc = ckpt.get("num_classes", NUM_CLASSES)
        if nc != NUM_CLASSES:
            model = build_model(nc)
        model.load_state_dict(ckpt["model_state"])
        print(f"  체크포인트 로드: {ckpt_path}  (val_acc={ckpt.get('val_acc', '?'):.4f})")
    else:
        print("  ⚠️  체크포인트 없음 — 랜덤 가중치로 ONNX 내보내기 (구조 확인용)")

    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    import tempfile, shutil
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    torch.onnx.export(
        model, dummy, str(tmp_path),
        opset_version=14,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    onnxslim.slim(str(tmp_path), str(MODEL_OUT))
    tmp_path.unlink(missing_ok=True)

    size_mb = MODEL_OUT.stat().st_size / 1024 / 1024
    print(f"  [OK] ONNX export done: {MODEL_OUT}  ({size_mb:.2f} MB)")
    return MODEL_OUT


# ──────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrafficSignNet 학습 + ONNX 내보내기")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch",       type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--export_only", action="store_true",
                        help="학습 건너뛰고 기존 체크포인트로 ONNX만 내보내기")
    args = parser.parse_args()

    if not args.export_only:
        train(args)

    print("\n[ONNX 내보내기]")
    export_onnx(CKPT_OUT if CKPT_OUT.exists() else None)


if __name__ == "__main__":
    main()

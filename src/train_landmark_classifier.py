import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class LandmarkDataset(Dataset):
    def __init__(self, manifest_path, augment=False, noise_std=0.01, scale_range=0.1):
        self.items = []
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range

        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row["path"], int(row["label_id"])))

        if not self.items:
            raise ValueError("Manifest is empty.")

        sample = np.load(self.items[0][0])
        self.seq_len = sample["data"].shape[0]
        self.feature_dim = sample["data"].shape[1]

    def __len__(self):
        return len(self.items)

    def _augment(self, data):
        if not self.augment:
            return data
            
        # 1. 스케일링 증강
        scale = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
        
        # 2. 이동(Translation) 증강
        shift = np.random.uniform(-self.scale_range, self.scale_range, size=(1, data.shape[1]))
        
        # 3. 노이즈(Jitter) 증강
        noise = np.random.normal(0.0, self.noise_std, size=data.shape)
        
        data = data * scale + shift + noise
        
        # 4. 시간적 마스킹 (Temporal Dropout) - 일부 프레임을 무작위로 0으로 처리하여 강건성 확보
        if np.random.rand() < 0.5:
            num_mask = max(1, int(data.shape[0] * 0.1)) # 최대 10%의 프레임 마스킹
            mask_indices = np.random.choice(data.shape[0], num_mask, replace=False)
            data[mask_indices] = 0.0
            
        # 5. [AIhub 논문(KSL-Guide) 참조] Random Frame Skip Sampling
        # 수어 동작의 속도 차이를 모사하기 위한 프레임 샘플링(크롭 및 리샘플링) 기법
        if np.random.rand() < 0.5:
            seq_len = data.shape[0]
            start_idx = np.random.randint(0, max(2, seq_len // 4))
            end_idx = np.random.randint(seq_len - max(1, seq_len // 4), seq_len)
            
            if end_idx > start_idx + 1:
                sampled_indices = np.linspace(start_idx, end_idx - 1, num=seq_len, dtype=int)
                # 미세한 시간 지터(Temporal Jitter) 추가
                jitter = np.random.randint(-1, 2, size=seq_len)
                sampled_indices = np.clip(sampled_indices + jitter, 0, seq_len - 1)
                data = data[sampled_indices]
            
        return data

    def __getitem__(self, idx):
        path, label = self.items[idx]
        sample = np.load(path)
        data = sample["data"].astype(np.float32)
        data = self._augment(data)
        return torch.from_numpy(data).float(), label, data.shape[0]


def collate_landmarks(batch):
    data_list, label_list, len_list = zip(*batch)
    lengths = torch.tensor(len_list, dtype=torch.long)
    labels = torch.tensor(label_list, dtype=torch.long)
    padded = pad_sequence(data_list, batch_first=True)
    return padded, labels, lengths


class LandmarkModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        
        # 입력 차원 매핑 및 특징 정규화
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # [AIhub 공식 모델(SignModel) 영감]
        # 공식 코드(lib/model/sign_model.py)의 시간적 특징 추출(Temporal Encode) 방식을 차용하여
        # 단순 GRU에 의존하지 않고, 1D-CNN (Temporal Convolution)과 MaxPooling을 선행 적용합니다.
        # 이를 통해 지역적(Local) 시공간적 패턴을 더욱 효과적으로 포착할 수 있습니다.
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU()
        )
        
        # 기존 양방향 GRU를 1D-CNN 이후에 결합하여 장기(Long-term) 문맥 포착 유지
        self.gru = nn.GRU(
            hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # MLP 헤드 고도화 (단순 Linear -> MLP + LayerNorm)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths):
        # x: (B, T, C)
        x = self.proj(x)
        
        # 1D-CNN 입력 조건 (B, C, T)로 차원 변경
        x = x.transpose(1, 2)
        
        # AIhub 모델 구조에서 차용한 Temporal Encoding 적용
        x = self.temporal_encoder(x)
        
        # 다시 GRU 입력을 위해 (B, T', C') 차원 복원
        x = x.transpose(1, 2)

        # Conv/Pool 이후 길이 보정 (MaxPool1d 2회)
        conv_lengths = torch.div(lengths, 4, rounding_mode='floor').clamp(min=1)
        packed = pack_padded_sequence(
            x, conv_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)

        # 마지막 레이어의 양방향 hidden state 결합
        h_n = h_n.view(self.gru.num_layers, 2, x.size(0), self.gru.hidden_size)
        last_layer = h_n[-1]
        pooled = torch.cat([last_layer[0], last_layer[1]], dim=1)

        return self.head(pooled)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, label, lengths in loader:
        data = data.to(device)
        label = label.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(data, lengths)
        loss = criterion(logits, label)
        loss.backward()
        
        # 그래디언트 폭발(Exploding Gradients) 방지를 위한 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == label).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label, lengths in loader:
            data = data.to(device)
            label = label.to(device)
            lengths = lengths.to(device)
            logits = model(data, lengths)
            loss = criterion(logits, label)

            total_loss += loss.item() * data.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += data.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a landmark sequence classifier.")
    parser.add_argument("--manifest", default="./dataset/landmarks/manifest.csv", help="Manifest CSV path.")
    parser.add_argument("--labels", default="./dataset/landmarks/labels.json", help="Labels JSON path.")
    # 수렴을 위해 기본 학습 에포크 상향
    parser.add_argument("--epochs", type=int, default=100, help="Epochs.") 
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    # 기본 모델 크기 확장
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size.")
    parser.add_argument("--layers", type=int, default=2, help="GRU layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--augment", action="store_true", help="Enable landmark augmentation.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)
    num_classes = len(labels)

    dataset = LandmarkDataset(args.manifest, augment=args.augment)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # 데이터 로더 병목 방지
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
        collate_fn=collate_landmarks,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
        collate_fn=collate_landmarks,
    )

    model = LandmarkModel(
        input_dim=dataset.feature_dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    # 1400여 개의 클래스에 대한 과적합 방지를 위해 Label Smoothing 추가 -> 데이터셋이 작아 오버피팅이 필요하므로 제거
    criterion = nn.CrossEntropyLoss()
    
    # AdamW 옵티마이저 (Weight Decay 강화 -> 다시 완화)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 코사인 어닐링 (Cosine Annealing) 학습률 스케줄러 적용
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=1e-5)

    best_val_acc = 0.0
    os.makedirs("./checkpoints", exist_ok=True)
    
    print(f"[{device}] 모델 학습 시작! (Total Classes: {num_classes}, Epochs: {args.epochs})")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"LR={current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./checkpoints/landmark_best.pth")
            print(f" -> Best model saved! (Val Acc: {best_val_acc:.3f})")

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
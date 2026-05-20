import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class MediapipeDataset(Dataset):
    def __init__(
        self,
        items,
        *,
        augment=False,
        noise_std=0.01,
        scale_range=0.1,
        normalize="none",
        stats=None,
    ):
        self.items = items
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.normalize = normalize
        self.stats = stats

        if not self.items:
            raise ValueError("Dataset is empty.")

        sample = np.load(self.items[0][0])["data"]
        self.feature_dim = sample.shape[1]

    def __len__(self):
        return len(self.items)

    def _augment(self, data):
        if not self.augment:
            return data

        scale = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)
        shift = np.random.uniform(-self.scale_range, self.scale_range, size=(1, data.shape[1]))
        noise = np.random.normal(0.0, self.noise_std, size=data.shape)

        data = data * scale + shift + noise

        if np.random.rand() < 0.5:
            num_mask = max(1, int(data.shape[0] * 0.1))
            mask_indices = np.random.choice(data.shape[0], num_mask, replace=False)
            data[mask_indices] = 0.0

        if np.random.rand() < 0.5:
            seq_len = data.shape[0]
            start_idx = np.random.randint(0, max(2, seq_len // 4))
            end_idx = np.random.randint(seq_len - max(1, seq_len // 4), seq_len)

            if end_idx > start_idx + 1:
                sampled_indices = np.linspace(start_idx, end_idx - 1, num=seq_len, dtype=int)
                jitter = np.random.randint(-1, 2, size=seq_len)
                sampled_indices = np.clip(sampled_indices + jitter, 0, seq_len - 1)
                data = data[sampled_indices]

        return data

    def _normalize(self, data):
        if self.normalize == "none":
            return data
        if self.normalize == "meanstd" and self.stats is not None:
            mean, std = self.stats
            return (data - mean) / std
        if self.normalize == "per_sample":
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-6
            return (data - mean) / std
        return data

    def __getitem__(self, idx):
        path, label = self.items[idx]
        sample = np.load(path)["data"].astype(np.float32)
        sample = self._augment(sample)
        sample = self._normalize(sample)
        return torch.from_numpy(sample).float(), label, sample.shape[0]


def collate_sequences(batch):
    data_list, label_list, len_list = zip(*batch)
    lengths = torch.tensor(len_list, dtype=torch.long)
    labels = torch.tensor(label_list, dtype=torch.long)
    padded = pad_sequence(data_list, batch_first=True)
    return padded, labels, lengths


class LandmarkModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths):
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)

        conv_lengths = torch.div(lengths, 4, rounding_mode="floor").clamp(min=1)
        packed = pack_padded_sequence(
            x, conv_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h_n = h_n.view(self.gru.num_layers, 2, x.size(0), self.gru.hidden_size)
        last_layer = h_n[-1]
        pooled = torch.cat([last_layer[0], last_layer[1]], dim=1)
        return self.head(pooled)


def stratified_split(items, val_split, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for idx, (_, label) in enumerate(items):
        by_label[label].append(idx)

    train_idx, val_idx = [], []
    for indices in by_label.values():
        rng.shuffle(indices)
        if val_split <= 0 or len(indices) <= 1:
            train_idx.extend(indices)
            continue
        n_val = max(1, int(len(indices) * val_split))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def compute_mean_std(items, indices, feature_dim):
    total = 0
    sum_x = np.zeros(feature_dim, dtype=np.float64)
    sum_x2 = np.zeros(feature_dim, dtype=np.float64)

    for idx in indices:
        data = np.load(items[idx][0])["data"].astype(np.float32)
        total += data.shape[0]
        sum_x += data.sum(axis=0)
        sum_x2 += np.square(data).sum(axis=0)

    if total == 0:
        raise ValueError("No frames to compute stats.")

    mean = (sum_x / total).astype(np.float32)
    var = (sum_x2 / total) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)
    return mean, std


def load_items(manifest_path):
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["path"], int(row["label_id"])))
    return items


def topk_correct(logits, labels, k=5):
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(labels.unsqueeze(1))
        return correct.any(dim=1).float().sum().item()


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
    top5_correct = 0
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
            top5_correct += topk_correct(logits, label, k=5)
            total += data.size(0)

    return total_loss / total, correct / total, top5_correct / total


def main():
    parser = argparse.ArgumentParser(description="Train MediaPipe keypoint classifier.")
    parser.add_argument("--manifest", default="./dataset/mediapipe_from_videos/manifest.csv", help="Manifest CSV path.")
    parser.add_argument("--labels", default="./dataset/mediapipe_from_videos/labels.json", help="Labels JSON path.")
    parser.add_argument("--epochs", type=int, default=120, help="Epochs.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=192, help="Hidden size.")
    parser.add_argument("--layers", type=int, default=2, help="GRU layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--normalize", choices=["none", "meanstd", "per_sample"], default="meanstd")
    parser.add_argument("--stats", default="", help="Path to save/load mean/std stats.")
    parser.add_argument("--no-class-weight", action="store_true", help="Disable class-balanced weights.")
    parser.add_argument("--log-csv", default="./logs/mediapipe_training.csv", help="CSV log path.")
    parser.add_argument("--out", default="./checkpoints/mediapipe_best.pth", help="Best model output.")
    parser.add_argument("--out-last", default="./checkpoints/mediapipe_last.pth", help="Last model output.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest}. Run scripts/collect_mediapipe_dataset.py first."
        )
    if not os.path.exists(args.labels):
        raise FileNotFoundError(
            f"Labels not found: {args.labels}. Run scripts/collect_mediapipe_dataset.py first."
        )

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)
    num_classes = len(labels)

    items = load_items(args.manifest)
    train_idx, val_idx = stratified_split(items, args.val_split, args.seed)

    sample = np.load(items[0][0])["data"]
    feature_dim = sample.shape[1]

    stats = None
    if args.normalize == "meanstd":
        stats_path = args.stats or str(Path(args.out).with_suffix(".stats.npz"))
        if os.path.exists(stats_path):
            cached = np.load(stats_path)
            stats = (cached["mean"], cached["std"])
        else:
            mean, std = compute_mean_std(items, train_idx, feature_dim)
            np.savez(stats_path, mean=mean, std=std)
            stats = (mean, std)

    train_set = MediapipeDataset(
        items,
        augment=args.augment,
        normalize=args.normalize,
        stats=stats,
    )
    val_set = MediapipeDataset(
        items,
        augment=False,
        normalize=args.normalize,
        stats=stats,
    )

    train_loader = DataLoader(
        Subset(train_set, train_idx),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        Subset(val_set, val_idx),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sequences,
    )

    model = LandmarkModel(
        input_dim=feature_dim,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    use_class_weight = not args.no_class_weight
    if use_class_weight:
        counts = np.zeros(num_classes, dtype=np.float32)
        for _, label in items:
            counts[label] += 1
        weights = 1.0 / np.clip(counts, 1.0, None)
        weights = weights / weights.sum() * num_classes
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=1e-5)

    best_val_acc = 0.0
    os.makedirs(Path(args.out).parent, exist_ok=True)

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if log_file.tell() == 0:
        log_writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_top5",
        ])

    print(f"[{device}] MediaPipe training start (Classes: {num_classes}, Epochs: {args.epochs})")

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_top5 = eval_one_epoch(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            print(
                f"Epoch {epoch:03d}/{args.epochs}: "
                f"LR={current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Top5: {val_top5:.3f}"
            )

            log_writer.writerow([
                epoch,
                f"{current_lr:.6f}",
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{val_top5:.6f}",
            ])
            log_file.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.out)
                print(f" -> Best model saved: {args.out} (Val Acc: {best_val_acc:.3f})")

            torch.save(model.state_dict(), args.out_last)
    finally:
        log_file.close()

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()

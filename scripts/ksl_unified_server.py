import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# self-contained LandmarkModel definition for robustness
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
        
        self.gru = nn.GRU(
            hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
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


def apply_w8a8_ptq(model):
    quantized_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if "head" in name or "classifier" in name:
                continue
            with torch.no_grad():
                weight = module.weight.data
                max_val = weight.abs().max()
                scale = max_val / 127.0
                if scale > 0:
                    q_weight = torch.round(weight / scale).clamp(-128, 127)
                    module.weight.data = q_weight * scale
                    quantized_layers += 1
    return quantized_layers


def load_model_helper(weights_path, labels_path, hidden_dim, num_classes, device, use_w8a8=False):
    print(f"Loading weights from {weights_path} with hidden_dim={hidden_dim}, classes={num_classes}...")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    if len(labels) != num_classes:
        print(f"WARNING: labels length ({len(labels)}) does not match num_classes ({num_classes}). Using first {num_classes} labels or padding.")
        if len(labels) > num_classes:
            labels = labels[:num_classes]
        else:
            labels = labels + [f"Class {i}" for i in range(len(labels), num_classes)]

    model = LandmarkModel(
        input_dim=959,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.1,
    )
    
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    if use_w8a8:
        num_q = apply_w8a8_ptq(model)
        print(f"Applied W8A8 PTQ to {num_q} layers.")

    model = model.to(device)
    model.eval()
    return model, labels


def predict_sequence(model, sequence, device):
    seq = torch.from_numpy(sequence).unsqueeze(0).to(device)
    lengths = torch.tensor([sequence.shape[0]], device=device)
    with torch.no_grad():
        logits = model(seq, lengths)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
    return int(idx.item()), float(conf.item())


# Setup FastAPI App
app = FastAPI(title="KSL Unified Real-Time Interpreter Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_websocket_inference(
    websocket: WebSocket,
    model: LandmarkModel,
    labels: list,
    stats_mean: np.ndarray = None,
    stats_std: np.ndarray = None,
    model_name: str = "Model"
):
    await websocket.accept()
    print(f"Client connected to WebSocket endpoint: {model_name}")

    # Read configuration parameters dynamically from query parameters if present
    query_params = websocket.query_params
    window_size = int(query_params.get("window_size", 30))
    min_frames = int(query_params.get("min_frames", 10))
    vote_size = int(query_params.get("vote_size", 10))
    min_votes = int(query_params.get("min_votes", 6))
    min_conf = float(query_params.get("min_conf", 0.3))
    min_gap = float(query_params.get("min_gap", 1.0))
    infer_interval = float(query_params.get("infer_interval", 0.1))

    print(f"[{model_name}] Connection config: window_size={window_size}, min_frames={min_frames}, "
          f"vote_size={vote_size}, min_votes={min_votes}, min_conf={min_conf}, "
          f"min_gap={min_gap}, infer_interval={infer_interval}")

    seq_buffer = deque(maxlen=window_size)
    vote_buffer = deque(maxlen=vote_size)
    
    last_emit = 0.0
    last_infer = 0.0
    last_frame_time = time.time()
    fps_history = deque(maxlen=30)

    device = app.state.device

    try:
        while True:
            # Receive 959-dimensional Float32 landmark features (959 * 4 = 3836 bytes)
            data = await websocket.receive_bytes()
            if len(data) != 959 * 4:
                continue

            features = np.frombuffer(data, dtype=np.float32).copy()

            # Normalization (Mean-Std Z-score) if stats are provided (MediaPipe Model)
            if stats_mean is not None and stats_std is not None:
                features = (features - stats_mean) / (stats_std + 1e-8)

            seq_buffer.append(features)

            now = time.time()
            # Control inference frequency
            if now - last_infer < infer_interval:
                continue
            last_infer = now

            if len(seq_buffer) < min_frames:
                await websocket.send_text(json.dumps({
                    "label": "-",
                    "confidence": 0.0,
                    "stable": "-",
                    "fps": 0.0,
                }))
                continue

            sequence = np.stack(seq_buffer, axis=0)
            pred_idx, conf = predict_sequence(model, sequence, device)
            label = labels[pred_idx]

            vote_buffer.append(pred_idx)
            counts = Counter(vote_buffer)
            top_idx, top_count = counts.most_common(1)[0]

            stable_label = None
            if conf >= min_conf and top_count >= min_votes:
                if now - last_emit >= min_gap:
                    stable_label = labels[top_idx]
                    last_emit = now

            frame_time = now - last_frame_time
            last_frame_time = now
            if frame_time > 0:
                fps_history.append(1.0 / frame_time)
            fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            await websocket.send_text(json.dumps({
                "label": label,
                "confidence": conf,
                "stable": stable_label or "",
                "fps": fps,
            }))
    except WebSocketDisconnect:
        print(f"Client disconnected from WebSocket endpoint: {model_name}")
    except Exception as e:
        print(f"Error in {model_name} inference loop: {e}")


@app.websocket("/ws/mediapipe")
async def websocket_mediapipe(websocket: WebSocket):
    await run_websocket_inference(
        websocket=websocket,
        model=app.state.mediapipe_model,
        labels=app.state.mediapipe_labels,
        stats_mean=app.state.mediapipe_mean,
        stats_std=app.state.mediapipe_std,
        model_name="MediaPipe Model"
    )


@app.websocket("/ws/landmark")
async def websocket_landmark(websocket: WebSocket):
    await run_websocket_inference(
        websocket=websocket,
        model=app.state.landmark_model,
        labels=app.state.landmark_labels,
        stats_mean=None,
        stats_std=None,
        model_name="AIHub Landmark Model"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Unified KSL Interpreter WebSocket Server.")
    
    # MediaPipe Model Configs
    parser.add_argument("--mp-weights", default="./checkpoints/mediapipe_best.pth", help="MediaPipe model weights path.")
    parser.add_argument("--mp-labels", default="./dataset/mediapipe_from_videos/labels.json", help="MediaPipe labels path.")
    parser.add_argument("--mp-stats", default="./checkpoints/mediapipe_best.stats.npz", help="MediaPipe normalisation stats path.")
    
    # AIHub Model Configs
    parser.add_argument("--ah-weights", default="./checkpoints/landmark_best.pth", help="AIHub landmark model weights path.")
    parser.add_argument("--ah-labels", default="./dataset/landmarks_top50/labels.json", help="AIHub landmark labels path.")
    
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda.")
    parser.add_argument("--w8a8", action="store_true", help="Apply W8A8 PTQ post-training quantization to models.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address.")
    parser.add_argument("--port", type=int, default=8000, help="Port.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    app.state.device = device

    print(f"Device: {device}")
    print(f"Quantization enabled (W8A8 PTQ): {args.w8a8}")

    # 1. Load MediaPipe Model
    if os.path.exists(args.mp_weights):
        # Hidden size is 192, Classes is 2771
        mp_w = args.mp_weights
        mp_l = args.mp_labels
        mp_s = args.mp_stats
        
        # Load stats
        if os.path.exists(mp_s):
            stats = np.load(mp_s)
            app.state.mediapipe_mean = stats["mean"]
            app.state.mediapipe_std = stats["std"]
            print(f"Loaded MediaPipe normalisation statistics from {mp_s}.")
        else:
            app.state.mediapipe_mean = None
            app.state.mediapipe_std = None
            print(f"WARNING: MediaPipe normalisation stats not found at {mp_s}. Normalisation will be skipped.")

        app.state.mediapipe_model, app.state.mediapipe_labels = load_model_helper(
            weights_path=mp_w,
            labels_path=mp_l,
            hidden_dim=192,
            num_classes=2771,
            device=device,
            use_w8a8=args.w8a8
        )
    else:
        sys.exit(f"CRITICAL: MediaPipe best weights not found at {args.mp_weights}")

    # 2. Load AIHub Model
    if os.path.exists(args.ah_weights):
        # Hidden size is 128, Classes is 50
        ah_w = args.ah_weights
        ah_l = args.ah_labels

        app.state.landmark_model, app.state.landmark_labels = load_model_helper(
            weights_path=ah_w,
            labels_path=ah_l,
            hidden_dim=128,
            num_classes=50,
            device=device,
            use_w8a8=args.w8a8
        )
    else:
        sys.exit(f"CRITICAL: AIHub best weights not found at {args.ah_weights}")

    # 3. Mount Static Files Directory for Frontend
    web_dir = Path(__file__).resolve().parents[1] / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")
        print(f"Mounted static frontend from: {web_dir}")
    else:
        print(f"WARNING: Static web directory not found at: {web_dir}")

    # Start Server
    import uvicorn
    print(f"Starting unified FastAPI server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

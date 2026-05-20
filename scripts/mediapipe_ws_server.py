import argparse
import json
import os
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from train_landmark_classifier import LandmarkModel

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

def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_model(weights_path, labels_path, device, use_w8a8):
    labels = load_labels(labels_path)
    num_classes = len(labels)

    model = LandmarkModel(
        input_dim=959,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.1,
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    if use_w8a8:
        apply_w8a8_ptq(model)

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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    args = app.state.args
    model = app.state.model
    labels = app.state.labels

    seq_buffer = deque(maxlen=args.window_size)
    vote_buffer = deque(maxlen=args.vote_size)
    last_emit = 0.0
    last_infer = 0.0
    last_frame_time = time.time()
    fps_history = deque(maxlen=30)

    try:
        while True:
            # 클라이언트로부터 959개의 Float32 데이터 수신
            data = await websocket.receive_bytes()
            if len(data) != 959 * 4: # Float32는 4바이트
                continue
                
            features = np.frombuffer(data, dtype=np.float32)
            seq_buffer.append(features)

            now = time.time()
            if now - last_infer < args.infer_interval:
                continue
            last_infer = now

            if len(seq_buffer) < args.min_frames:
                await websocket.send_text(json.dumps({
                    "label": "-",
                    "confidence": 0.0,
                    "stable": "-",
                    "fps": 0.0,
                }))
                continue

            sequence = np.stack(seq_buffer, axis=0)
            pred_idx, conf = predict_sequence(model, sequence, app.state.device)
            label = labels[pred_idx]

            vote_buffer.append(pred_idx)
            counts = Counter(vote_buffer)
            top_idx, top_count = counts.most_common(1)[0]

            stable_label = None
            if conf >= args.min_conf and top_count >= args.min_votes:
                if now - last_emit >= args.min_gap:
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
        return

def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe Keypoint web server.")
    parser.add_argument("--weights", default="./checkpoints/landmark_best.pth", help="Path to model weights.")
    parser.add_argument("--labels", default="./dataset/landmarks/labels.json", help="Path to labels JSON.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda.")
    parser.add_argument("--window-size", type=int, default=30, help="Frames per inference window.")
    parser.add_argument("--min-frames", type=int, default=10, help="Minimum frames before inference.")
    parser.add_argument("--vote-size", type=int, default=10, help="Prediction vote buffer size.")
    parser.add_argument("--min-votes", type=int, default=6, help="Votes required for stable output.")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--min-gap", type=float, default=1.0, help="Debounce gap in seconds.")
    parser.add_argument("--infer-interval", type=float, default=0.1, help="Seconds between inferences.")
    parser.add_argument("--no-w8a8", action="store_true", help="Disable W8A8 PTQ.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address.")
    parser.add_argument("--port", type=int, default=8001, help="Port.")
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device(args.device)
    model, labels = create_model(
        weights_path=args.weights,
        labels_path=args.labels,
        device=device,
        use_w8a8=not args.no_w8a8,
    )

    app.state.args = args
    app.state.model = model
    app.state.labels = labels
    app.state.device = device

    print(f"서버가 ws://{args.host}:{args.port}/ws 에서 시작되었습니다.")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
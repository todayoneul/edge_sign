import argparse
import json
import os
import subprocess
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from train_landmark_classifier import LandmarkModel

EXPECTED_LEN = {
    "pose_keypoints_2d": 75,
    "face_keypoints_2d": 210,
    "hand_left_keypoints_2d": 63,
    "hand_right_keypoints_2d": 63,
    "pose_keypoints_3d": 100,
    "face_keypoints_3d": 280,
    "hand_left_keypoints_3d": 84,
    "hand_right_keypoints_3d": 84,
}
KEY_ORDER = list(EXPECTED_LEN.keys())


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


def extract_features_from_json(data):
    people = data.get("people", [])
    if isinstance(people, list) and people:
        person = people[0]
    elif isinstance(people, dict):
        person = people
    else:
        person = {}

    features = []
    for key in KEY_ORDER:
        values = person.get(key, None)
        if values:
            features.extend(values)
        else:
            features.extend([0.0] * EXPECTED_LEN[key])
    return np.array(features, dtype=np.float32)


class OpenPoseRunner:
    def __init__(self, openpose_bin, model_dir, work_dir, net_resolution="320x176"):
        self.openpose_bin = Path(openpose_bin) if openpose_bin else None
        self.model_dir = Path(model_dir) if model_dir else None
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.net_resolution = net_resolution

    def extract(self, bgr_frame):
        if self.openpose_bin is None or self.model_dir is None:
            return None

        frame_path = self.work_dir / f"frame_{self.counter:06d}.jpg"
        json_path = self.work_dir / f"frame_{self.counter:06d}_keypoints.json"
        self.counter += 1

        cv2.imwrite(str(frame_path), bgr_frame)

        cmd = [
            str(self.openpose_bin),
            "--image_path", str(frame_path),
            "--write_json", str(self.work_dir),
            "--model_folder", str(self.model_dir),
            "--display", "0",
            "--render_pose", "0",
            "--hand",
            "--face",
            "--number_people_max", "1",
            "--net_resolution", self.net_resolution,
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

        if not json_path.exists():
            frame_path.unlink(missing_ok=True)
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = extract_features_from_json(data)
        except Exception:
            features = None
        finally:
            frame_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)

        return features


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
    model.load_state_dict(state, strict=True)

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
    extractor = OpenPoseRunner(
        openpose_bin=args.openpose_bin,
        model_dir=args.openpose_model_dir,
        work_dir=args.work_dir,
        net_resolution=args.net_resolution,
    )

    seq_buffer = deque(maxlen=args.window_size)
    vote_buffer = deque(maxlen=args.vote_size)
    last_emit = 0.0
    last_infer = 0.0
    last_frame_time = time.time()
    fps_history = deque(maxlen=30)

    try:
        while True:
            data = await websocket.receive_bytes()
            np_buf = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            features = extractor.extract(frame)
            if features is None:
                continue

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
    parser = argparse.ArgumentParser(description="AIhub OpenPose keypoint web server.")
    parser.add_argument("--weights", default="./checkpoints/landmark_best.pth", help="Path to model weights.")
    parser.add_argument("--labels", default="./dataset/landmarks_top50/labels.json", help="Path to labels JSON.")
    parser.add_argument("--openpose-bin", default=os.environ.get("OPENPOSE_BIN"), help="OpenPose executable.")
    parser.add_argument("--openpose-model-dir", default=os.environ.get("OPENPOSE_MODEL_DIR"), help="OpenPose model folder.")
    parser.add_argument("--work-dir", default="./data/openpose_tmp", help="Working directory for OpenPose outputs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda.")
    parser.add_argument("--window-size", type=int, default=40, help="Frames per inference window.")
    parser.add_argument("--min-frames", type=int, default=16, help="Minimum frames before inference.")
    parser.add_argument("--vote-size", type=int, default=15, help="Prediction vote buffer size.")
    parser.add_argument("--min-votes", type=int, default=9, help="Votes required for stable output.")
    parser.add_argument("--min-conf", type=float, default=0.45, help="Confidence threshold.")
    parser.add_argument("--min-gap", type=float, default=1.5, help="Debounce gap in seconds.")
    parser.add_argument("--infer-interval", type=float, default=0.15, help="Seconds between inferences.")
    parser.add_argument("--net-resolution", default="320x176", help="OpenPose net resolution.")
    parser.add_argument("--no-w8a8", action="store_true", help="Disable W8A8 PTQ.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address.")
    parser.add_argument("--port", type=int, default=8001, help="Port.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.openpose_bin or not args.openpose_model_dir:
        raise RuntimeError("OPENPOSE_BIN and OPENPOSE_MODEL_DIR must be set.")

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

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

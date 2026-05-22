import os
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from train_landmark_classifier import LandmarkModel

class LandmarkModelONNXWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.proj = original_model.proj
        self.temporal_encoder = original_model.temporal_encoder
        self.gru = original_model.gru
        self.head = original_model.head

    def forward(self, x):
        # x: (B, T, C)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)
        
        # Bypass packing
        _, h_n = self.gru(x)
        h_n = h_n.view(self.gru.num_layers, 2, x.size(0), self.gru.hidden_size)
        last_layer = h_n[-1]
        pooled = torch.cat([last_layer[0], last_layer[1]], dim=1)
        return self.head(pooled)

def main():
    os.makedirs("./web/model", exist_ok=True)
    
    device = torch.device("cpu")
    
    # 1. Export Landmark Model (50 classes, hidden=128)
    print("Exporting landmark_best.pth (50 classes) to ONNX...")
    model_landmark = LandmarkModel(
        input_dim=959,
        hidden_dim=128,
        num_layers=2,
        num_classes=50,
        dropout=0.1
    )
    model_landmark.load_state_dict(torch.load("./checkpoints/landmark_best.pth", map_location=device), strict=True)
    model_landmark.eval()
    
    wrapper_landmark = LandmarkModelONNXWrapper(model_landmark)
    wrapper_landmark.eval()
    
    dummy_input_landmark = torch.randn(1, 40, 959) # Default window size for AIHub landmark is 40
    
    torch.onnx.export(
        wrapper_landmark,
        dummy_input_landmark,
        "./web/model/landmark_best.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("landmark_best.onnx exported successfully.")
    
    # 2. Export MediaPipe Model (2771 classes, hidden=192)
    print("Exporting mediapipe_best.pth (2771 classes) to ONNX...")
    model_mediapipe = LandmarkModel(
        input_dim=959,
        hidden_dim=192,
        num_layers=2,
        num_classes=2771,
        dropout=0.1
    )
    model_mediapipe.load_state_dict(torch.load("./checkpoints/mediapipe_best.pth", map_location=device), strict=True)
    model_mediapipe.eval()
    
    wrapper_mediapipe = LandmarkModelONNXWrapper(model_mediapipe)
    wrapper_mediapipe.eval()
    
    dummy_input_mediapipe = torch.randn(1, 30, 959) # Default window size for MediaPipe is 30
    
    torch.onnx.export(
        wrapper_mediapipe,
        dummy_input_mediapipe,
        "./web/model/mediapipe_best.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("mediapipe_best.onnx exported successfully.")

    # 3. Export normalisation stats and labels to web/model/
    stats_path = "./checkpoints/mediapipe_best.stats.npz"
    if os.path.exists(stats_path):
        import numpy as np
        import json
        print(f"Exporting normalisation stats from {stats_path} to web/model/mediapipe_stats.json...")
        stats = np.load(stats_path)
        mean = stats["mean"].tolist()
        std = stats["std"].tolist()
        with open("./web/model/mediapipe_stats.json", "w", encoding="utf-8") as f:
            json.dump({"mean": mean, "std": std}, f)
        print("mediapipe_stats.json exported successfully.")
    
    import shutil
    
    mp_labels_src = "./dataset/mediapipe_from_videos/labels.json"
    if os.path.exists(mp_labels_src):
        print("Copying MediaPipe labels to web/model/mediapipe_labels.json...")
        shutil.copy(mp_labels_src, "./web/model/mediapipe_labels.json")
        
    ah_labels_src = "./dataset/landmarks_top50/labels.json"
    if os.path.exists(ah_labels_src):
        print("Copying AIHub labels to web/model/landmark_labels.json...")
        shutil.copy(ah_labels_src, "./web/model/landmark_labels.json")

if __name__ == "__main__":
    main()

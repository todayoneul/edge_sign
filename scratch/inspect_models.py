import torch
from pathlib import Path

checkpoints_dir = Path("checkpoints")
for p in ["landmark_best.pth", "mediapipe_best.pth"]:
    path = checkpoints_dir / p
    if path.exists():
        state = torch.load(path, map_location="cpu")
        print(f"File: {p}")
        for k in ["proj.0.weight", "head.4.weight"]:
            if k in state:
                print(f"  {k}: {state[k].shape}")
            else:
                # print all keys matching head or proj
                matching = [key for key in state.keys() if "head" in key or "proj" in key]
                print(f"  matching keys for {k}: {matching}")
                break

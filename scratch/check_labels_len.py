import json

for path in ["./dataset/landmarks_top50/labels.json", "./dataset/mediapipe_from_videos/labels.json"]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Path: {path}, Length: {len(data)}")

import os
import glob
import pandas as pd
from pathlib import Path
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS_DIR = Path(__file__).parent.parent / "web" / "assets"

# Target classes mapping for offline verification gallery
SAMPLES_TO_GENERATE = {
    1: "sample_speed30.jpg",
    2: "sample_speed50.jpg",
    5: "sample_speed80.jpg",
    12: "sample_priority.jpg",
    13: "sample_yield.jpg",
    14: "sample_stop.jpg",
    17: "sample_no_entry.jpg",
    18: "sample_caution.jpg",
    33: "sample_turn_right.jpg",
    34: "sample_turn_left.jpg",
    38: "sample_keep_right.jpg",
    40: "sample_roundabout.jpg"
}

def find_gtsrb_images():
    paths_to_check = [
        DATA_DIR / "gtsrb" / "GTSRB" / "Training",
        DATA_DIR / "gtsrb" / "GTSRB" / "Final_Training" / "Images"
    ]
    for path in paths_to_check:
        if path.exists():
            return path
    raise FileNotFoundError("GTSRB path not found.")

def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        train_dir = find_gtsrb_images()
    except FileNotFoundError as e:
        print(e)
        return
        
    for orig_id, filename in SAMPLES_TO_GENERATE.items():
        class_folder = train_dir / f"{orig_id:05d}"
        if not class_folder.exists():
            print(f"Class folder {class_folder} missing.")
            continue
            
        csv_files = list(class_folder.glob("*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0], sep=";")
            # Take the 5th image to avoid the first very dark/blurry ones
            idx = min(5, len(df) - 1)
            row = df.iloc[idx]
            img_path = class_folder / row["Filename"]
            
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                # Crop to ROI so the web UI receives a clean cropped image
                x1, y1, x2, y2 = int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"])
                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))
                
                # Save to web/assets/
                dest_path = ASSETS_DIR / filename
                img.save(dest_path, "JPEG")
                print(f"Generated sample asset: {dest_path.name}")
        else:
            # Fallback direct glob
            img_files = list(class_folder.glob("*.ppm")) + list(class_folder.glob("*.jpg"))
            if img_files:
                img = Image.open(img_files[0]).convert("RGB")
                dest_path = ASSETS_DIR / filename
                img.save(dest_path, "JPEG")
                print(f"Generated sample asset (fallback): {dest_path.name}")

if __name__ == "__main__":
    main()

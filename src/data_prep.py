import os
import json
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data"
LABEL_FILE = Path(__file__).parent.parent / "web" / "labels.json"

# Define our selected 12 classes mapping: original_class_id -> (new_class_idx, label_name, korean_name)
SELECTED_CLASSES = {
    1: (0, "Speed Limit 30", "속도제한 30km/h"),
    2: (1, "Speed Limit 50", "속도제한 50km/h"),
    5: (2, "Speed Limit 80", "속도제한 80km/h"),
    12: (3, "Priority Road", "우선도로"),
    13: (4, "Yield", "양보"),
    14: (5, "Stop", "정지"),
    17: (6, "No Entry", "진입금지"),
    18: (7, "General Caution", "주의"),
    33: (8, "Turn Right Ahead", "우회전"),
    34: (9, "Turn Left Ahead", "좌회전"),
    38: (10, "Keep Right", "우측통행"),
    40: (11, "Roundabout Mandatory", "회전교차로")
}

def find_gtsrb_images():
    # Check common paths for torchvision or manual extraction
    paths_to_check = [
        DATA_DIR / "gtsrb" / "GTSRB" / "Training",
        DATA_DIR / "gtsrb" / "GTSRB" / "Final_Training" / "Images",
        DATA_DIR / "GTSRB" / "Final_Training" / "Images",
        DATA_DIR / "GTSRB_Final_Training_Images" / "GTSRB" / "Final_Training" / "Images",
        DATA_DIR / "Final_Training" / "Images"
    ]
    
    for path in paths_to_check:
        if path.exists():
            print(f"Found GTSRB training images at: {path}")
            return path
            
    # Recursive search as last resort
    found = list(DATA_DIR.glob("**/Final_Training/Images"))
    if found:
        print(f"Found GTSRB training images via search at: {found[0]}")
        return found[0]
        
    raise FileNotFoundError("Could not find GTSRB training images. Ensure download script finished successfully.")

def main():
    try:
        train_img_dir = find_gtsrb_images()
    except FileNotFoundError as e:
        print(e)
        return

    # Export label mapping to JSON for Web UI
    labels_mapping = {}
    for orig_id, (new_idx, name_en, name_ko) in SELECTED_CLASSES.items():
        labels_mapping[str(new_idx)] = {
            "english": name_en,
            "korean": name_ko,
            "original_id": orig_id
        }
        
    # Create web folder if it doesn't exist
    LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        json.dump(labels_mapping, f, indent=4, ensure_ascii=False)
    print(f"Exported class mapping to {LABEL_FILE}")

    # Gather data paths
    data_records = []
    
    # GTSRB contains folders like 00000, 00001, etc.
    for orig_id, (new_idx, name_en, name_ko) in SELECTED_CLASSES.items():
        class_folder = train_img_dir / f"{orig_id:05d}"
        if not class_folder.exists():
            print(f"Warning: Class folder {class_folder} does not exist.")
            continue
            
        # Read the CSV inside the class folder for metadata (bounding boxes, etc.)
        csv_files = list(class_folder.glob("*.csv"))
        if not csv_files:
            # If no CSV, just grab images directly
            img_files = list(class_folder.glob("*.ppm")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
            for img_path in img_files:
                data_records.append({
                    "path": str(img_path),
                    "class_id": new_idx,
                    "roi_x1": 0, "roi_y1": 0, "roi_x2": 0, "roi_y2": 0
                })
        else:
            # Parse CSV metadata
            # Columns: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
            df = pd.read_csv(csv_files[0], sep=";")
            for _, row in df.iterrows():
                img_path = class_folder / row["Filename"]
                if img_path.exists():
                    data_records.append({
                        "path": str(img_path),
                        "class_id": new_idx,
                        "roi_x1": int(row["Roi.X1"]),
                        "roi_y1": int(row["Roi.Y1"]),
                        "roi_x2": int(row["Roi.X2"]),
                        "roi_y2": int(row["Roi.Y2"])
                    })

    df_dataset = pd.DataFrame(data_records)
    print(f"Total filtered samples collected: {len(df_dataset)}")
    
    if len(df_dataset) == 0:
        print("No samples found. Please wait for the download task to complete and try again.")
        return

    # Print distribution
    print("\nClass Distribution:")
    for new_idx in sorted(df_dataset["class_id"].unique()):
        count = len(df_dataset[df_dataset["class_id"] == new_idx])
        label = SELECTED_CLASSES[list(SELECTED_CLASSES.keys())[list(SELECTED_CLASSES.values()).index(next(v for v in SELECTED_CLASSES.values() if v[0] == new_idx))]][1]
        print(f"Class {new_idx} ({label}): {count} samples")

    # Split into train/validation sets (80/20 stratified split)
    train_df, val_df = train_test_split(df_dataset, test_size=0.2, random_state=42, stratify=df_dataset["class_id"])
    
    # Save split info to data directory
    data_meta_dir = DATA_DIR / "processed"
    data_meta_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(data_meta_dir / "train_split.csv", index=False)
    val_df.to_csv(data_meta_dir / "val_split.csv", index=False)
    print(f"\nSaved splits to {data_meta_dir} (Train: {len(train_df)}, Val: {len(val_df)})")

if __name__ == "__main__":
    main()

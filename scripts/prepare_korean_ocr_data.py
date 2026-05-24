import zipfile
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_ROOT = Path(r"C:\Users\leegy\Desktop\CNN_Quant\AIhub\다양한 형태의 한글 문자 OCR")
OUTPUT_DIR = BASE_DIR / "data" / "korean_ocr"

TRAIN_LABEL_ZIP = DATASET_ROOT / "Training" / "[라벨]Training_필기체.zip"
TRAIN_IMAGE_ZIP = DATASET_ROOT / "Training" / "[원천]Training_필기체.zip"
VAL_LABEL_ZIP = DATASET_ROOT / "Validation" / "[라벨]validation_필기체.zip"
VAL_IMAGE_ZIP = DATASET_ROOT / "Validation" / "[원천]validation_필기체.zip"

def build_label_map(label_zip_path):
    """
    Read the label ZIP file and build a map of filename_base -> character
    """
    print(f"Building label map from {label_zip_path.name}...")
    label_map = {}
    with zipfile.ZipFile(label_zip_path, 'r') as z:
        # Filter for JSON files under 1.글자/
        json_files = [n for n in z.namelist() if '1.글자/' in n and n.endswith('.json')]
        total = len(json_files)
        print(f"Found {total} character-level JSON files.")
        
        # Read JSON files
        for idx, name in enumerate(json_files):
            if idx > 0 and idx % 50000 == 0:
                print(f"  Processed {idx}/{total} JSON files...")
            try:
                base = os.path.basename(name).replace('.json', '')
                with z.open(name) as f:
                    data = json.loads(f.read().decode('utf-8'))
                    char = data["text"]["letter"]["value"]
                    label_map[base] = char
            except Exception as e:
                print(f"Error reading {name}: {e}")
    print(f"Finished building label map. Total mapped files: {len(label_map)}")
    return label_map

def extract_images(image_zip_path, label_map, char_to_idx, split_dir, max_samples_per_class=None):
    """
    Extract JPG files from image ZIP, mapping them to class folders based on label_map
    """
    print(f"\nExtracting images from {image_zip_path.name} to {split_dir}...")
    split_dir.mkdir(parents=True, exist_ok=True)
    
    class_counts = {i: 0 for i in range(len(char_to_idx))}
    
    with zipfile.ZipFile(image_zip_path, 'r') as z:
        # Filter for JPG files under 1.글자/
        jpg_files = [n for n in z.namelist() if '1.글자/' in n and n.endswith('.jpg')]
        total = len(jpg_files)
        print(f"Found {total} character-level JPG files in source zip.")
        
        extracted_count = 0
        skipped_count = 0
        missing_label_count = 0
        
        # Helper to extract a single file
        def extract_single(name):
            nonlocal extracted_count, skipped_count, missing_label_count
            base = os.path.basename(name).replace('.jpg', '')
            char = label_map.get(base)
            if not char:
                missing_label_count += 1
                return
            
            class_idx = char_to_idx.get(char)
            if class_idx is None:
                missing_label_count += 1
                return
            
            # Check limit
            if max_samples_per_class is not None and class_counts[class_idx] >= max_samples_per_class:
                skipped_count += 1
                return
            
            class_counts[class_idx] += 1
            
            # Target output path
            class_dir = split_dir / str(class_idx)
            class_dir.mkdir(exist_ok=True)
            out_path = class_dir / f"{base}.jpg"
            
            try:
                # Read from zip and write to output
                with z.open(name) as f_in:
                    data = f_in.read()
                with open(out_path, 'wb') as f_out:
                    f_out.write(data)
                extracted_count += 1
            except Exception as e:
                print(f"Error extracting {name}: {e}")

        # Process all files
        for idx, name in enumerate(jpg_files):
            if idx > 0 and idx % 50000 == 0:
                print(f"  Processed {idx}/{total} JPG files... Extracted: {extracted_count}, Skipped: {skipped_count}")
            extract_single(name)
            
    print(f"Extraction complete for {split_dir.name}:")
    print(f"  Extracted: {extracted_count}")
    print(f"  Skipped (exceeded limit): {skipped_count}")
    print(f"  Missing labels: {missing_label_count}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract and structure Korean OCR dataset")
    parser.add_argument("--max-samples-train", type=int, default=None, help="Max training samples per class")
    parser.add_argument("--max-samples-val", type=int, default=None, help="Max validation samples per class")
    args = parser.parse_args()

    # 1. Build Label Maps
    val_label_map = build_label_map(VAL_LABEL_ZIP)
    train_label_map = build_label_map(TRAIN_LABEL_ZIP)
    
    # 2. Collect all unique characters across both splits
    all_chars = sorted(list(set(val_label_map.values()) | set(train_label_map.values())))
    print(f"\nTotal unique characters identified: {len(all_chars)}")
    
    if len(all_chars) != 2350:
        print(f"Warning: Expected 2350 characters, but found {len(all_chars)}!")
    
    # Save char mapping files
    char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
    
    (BASE_DIR / "data").mkdir(exist_ok=True)
    with open(BASE_DIR / "data" / "char_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(char_to_idx, f, ensure_ascii=False, indent=2)
    with open(BASE_DIR / "data" / "idx_to_char.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_char, f, ensure_ascii=False, indent=2)
        
    print(f"Saved character mappings to data/char_to_idx.json and data/idx_to_char.json")
    
    # 3. Extract Validation Images
    extract_images(VAL_IMAGE_ZIP, val_label_map, char_to_idx, OUTPUT_DIR / "val", max_samples_per_class=args.max_samples_val)
    
    # 4. Extract Training Images
    extract_images(TRAIN_IMAGE_ZIP, train_label_map, char_to_idx, OUTPUT_DIR / "train", max_samples_per_class=args.max_samples_train)
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main()

import os
import glob
import numpy as np
from PIL import Image

def main():
    val_dir = r"C:\Users\leegy\Desktop\CNN_Quant\data\korean_ocr\val"
    paths = glob.glob(os.path.join(val_dir, "*", "*.jpg"))
    if not paths:
        print("No images found!")
        return
        
    print(f"Found {len(paths)} validation images. Analyzing first 15:")
    for p in paths[:15]:
        img = Image.open(p).convert("L")
        arr = np.array(img)
        h, w = arr.shape
        # Ink is black/dark, let's threshold it
        ink = arr < 220
        rows = np.any(ink, axis=1)
        cols = np.any(ink, axis=0)
        if np.any(rows) and np.any(cols):
            r_indices = np.where(rows)[0]
            c_indices = np.where(cols)[0]
            min_r, max_r = r_indices[0], r_indices[-1]
            min_c, max_c = c_indices[0], c_indices[-1]
            stroke_w = max_c - min_c + 1
            stroke_h = max_r - min_r + 1
            margin_left = min_c
            margin_right = w - 1 - max_c
            margin_top = min_r
            margin_bottom = h - 1 - max_r
            print(f"Img: {os.path.basename(p)}, Size: {w}x{h}, Ink: {stroke_w}x{stroke_h}, "
                  f"Margins (L/R/T/B): {margin_left}/{margin_right}/{margin_top}/{margin_bottom}")
        else:
            print(f"Img: {os.path.basename(p)}, Size: {w}x{h}, (No ink found < 220!)")

if __name__ == "__main__":
    main()

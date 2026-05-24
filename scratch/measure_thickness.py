import os
import glob
import numpy as np
from PIL import Image
import cv2

def main():
    val_dir = r"C:\Users\leegy\Desktop\CNN_Quant\data\korean_ocr\val"
    paths = glob.glob(os.path.join(val_dir, "*", "*.jpg"))
    if not paths:
        print("No images found!")
        return
        
    print(f"Analyzing stroke widths on 100 validation images...")
    widths = []
    
    np.random.seed(42)
    selected_paths = np.random.choice(paths, 100, replace=False)
    
    for p in selected_paths:
        img = Image.open(p).convert("L")
        # Resize to 64x64 as done in validation pipeline
        img_resized = img.resize((64, 64), Image.BILINEAR)
        arr = np.array(img_resized)
        
        # Threshold to get binary mask of ink (white = background, black = ink)
        # Ink is typically < 200
        binary = (arr < 200).astype(np.uint8)
        
        # Calculate thickness: we can use distance transform or sum of pixels / contour length
        # Let's count number of ink pixels
        ink_pixels = np.sum(binary)
        
        # Find contours to estimate stroke length
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        total_perimeter = 0
        for c in contours:
            total_perimeter += cv2.arcLength(c, True)
            
        if total_perimeter > 0:
            # Stroke thickness is roughly: 2 * (ink_pixels) / total_perimeter
            # since a stroke has two sides (outer and inner)
            thickness = 2.0 * ink_pixels / total_perimeter
            widths.append(thickness)
            
    print(f"Average stroke thickness in 64x64 resized images: {np.mean(widths):.2f} pixels")
    print(f"Min stroke thickness: {np.min(widths):.2f} pixels")
    print(f"Max stroke thickness: {np.max(widths):.2f} pixels")

if __name__ == "__main__":
    main()

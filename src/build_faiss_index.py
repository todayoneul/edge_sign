import os
import torch
import torch.nn.functional as F
import faiss
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

# Configuration
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INDEX_SIZE = 1000 # 구축할 갤러리 이미지 수
SAVE_DIR = "./data/faiss_db"
GALLERY_DIR = os.path.join(SAVE_DIR, "gallery_images")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)

print(f"🚀 Initializing CLIP Vision Encoder for FAISS indexing...")

# 1. Load CLIP Model
vision_encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

# 2. Build Index
print(f"📥 Downloading General Conceptual Captions dataset for FAISS index...")
ds = load_dataset("conceptual_captions", split="train", streaming=True)
ds = ds.shuffle(seed=42, buffer_size=5000)

d = vision_encoder.config.projection_dim # 512 for CLIP ViT-B/32
index = faiss.IndexFlatIP(d) # Inner product for cosine similarity
metadata = []

count = 0
with torch.no_grad():
    for item in ds:
        if count >= INDEX_SIZE:
            break
            
        try:
            response = requests.get(item['image_url'], timeout=3)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            continue
            
        # Save image locally
        img_path = os.path.join(GALLERY_DIR, f"general_{count}.jpg")
        img.resize((224,224)).save(img_path, format="JPEG", quality=85)
        
        # Extract features using CLIP
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        feat = F.normalize(vision_encoder(**inputs).image_embeds, p=2, dim=-1).cpu().float().numpy()
        
        # Add to index
        index.add(feat)
        metadata.append(img_path)
        count += 1
        
        if count % 100 == 0:
            print(f"Indexed {count} images...")

# Save FAISS index and metadata
faiss.write_index(index, os.path.join(SAVE_DIR, "general_clip.index"))
with open(os.path.join(SAVE_DIR, "general_metadata.txt"), "w") as f:
    for p in metadata:
        f.write(f"{p}\n")

print(f"✅ FAISS Index successfully built with {index.ntotal} images!")
print(f"💾 Saved to {SAVE_DIR}")

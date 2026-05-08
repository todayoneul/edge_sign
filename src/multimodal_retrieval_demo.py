import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import argparse

# 1. Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "./checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_15.pth"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
ASSETS_DIR = "./assets/demo"
os.makedirs(ASSETS_DIR, exist_ok=True)

# --- 1-Bit Layers (Reused) ---
class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.where(weight == 0, torch.ones_like(weight), torch.sign(weight))
    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[weight.abs() > 1.0] = 0
        return grad_input

def binarize_weight(weight):
    if weight.dim() == 4: scale = weight.abs().mean(dim=(1, 2, 3), keepdim=True)
    elif weight.dim() == 2: scale = weight.abs().mean(dim=1, keepdim=True)
    else: scale = weight.abs().mean()
    return BinarySTE.apply(weight) * scale

class BinaryConv2d(nn.Conv2d):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        return F.conv2d(input, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)

class BinaryLinear(nn.Linear):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        return F.linear(input, bw, self.bias)

def replace_layers_with_1bit(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and "stem" not in name and "head" not in name:
            bin_conv = BinaryConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                                    module.stride, module.padding, module.dilation, module.groups, module.bias is not None)
            bin_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None: bin_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, bin_conv)
        elif isinstance(module, nn.Linear) and "head" not in name and "classifier" not in name:
            bin_linear = BinaryLinear(module.in_features, module.out_features, module.bias is not None)
            bin_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None: bin_linear.bias.data.copy_(module.bias.data)
            setattr(model, name, bin_linear)
        else: replace_layers_with_1bit(module)

# --- Demo Engine ---
class Multimodal1BitDemo:
    def __init__(self):
        print("🎨 Initializing Term Project Demo Engine...")
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_ID).bfloat16().to(DEVICE).eval()
        
        self.student = timm.create_model(MODEL_NAME, pretrained=False)
        replace_layers_with_1bit(self.student)
        self.student.head.fc = nn.Linear(self.student.head.fc.in_features, 512)
        
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        self.student.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        self.student = self.student.bfloat16().to(DEVICE).eval()
        
        data_config = timm.data.resolve_model_data_config(self.student)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        self.gallery_images = []
        self.gallery_features = None

    def build_gallery(self, dataset_name="ILSVRC/imagenet-1k", size=1000):
        print(f"📂 Building Image Gallery from {dataset_name} ({size} images)...")
        ds = load_dataset(dataset_name, split="validation", streaming=True)
        
        feats = []
        for i, item in enumerate(tqdm(ds, total=size)):
            if i >= size: break
            img = item['image'].convert("RGB")
            self.gallery_images.append(img)
            
            img_tensor = self.transform(img).unsqueeze(0).to(DEVICE).bfloat16()
            with torch.no_grad():
                feat = F.normalize(self.student(img_tensor), p=2, dim=-1)
                feats.append(feat.cpu())
        
        self.gallery_features = torch.cat(feats, dim=0)
        print("✅ Gallery built successfully.")

    def search(self, query_text, top_k=5):
        print(f"🔍 Searching for: '{query_text}'")
        inputs = self.tokenizer(query_text, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_feat = F.normalize(self.text_model(**inputs).text_embeds, p=2, dim=-1)
        
        sims = F.cosine_similarity(text_feat.cpu(), self.gallery_features)
        scores, indices = torch.topk(sims, top_k)
        
        # Visualization
        fig = plt.figure(figsize=(18, 6), facecolor='#f0f0f0')
        plt.suptitle(f"Term Project: 1-Bit Zero-Shot Retrieval\nQuery: \"{query_text}\"", fontsize=20, fontweight='bold', y=1.05)
        
        for i in range(top_k):
            ax = plt.subplot(1, top_k, i + 1)
            plt.imshow(self.gallery_images[indices[i]])
            plt.title(f"Rank {i+1}\nSim: {scores[i]:.4f}", fontsize=14, pad=10)
            plt.axis('off')
            # Add a border based on score
            rect = plt.Rectangle((0,0), 1, 1, fill=False, color='green', lw=4, transform=ax.transAxes)
            ax.add_patch(rect)

        plt.tight_layout()
        save_name = f"demo_{query_text.replace(' ', '_')}.png"
        plt.savefig(os.path.join(ASSETS_DIR, save_name), bbox_inches='tight', dpi=300)
        print(f"💾 Professional result saved to {ASSETS_DIR}/{save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="a photo of a cute puppy")
    args = parser.parse_args()
    
    demo = Multimodal1BitDemo()
    demo.build_gallery(size=500) # Quick build for demo
    demo.search(args.query)
    demo.search("an old lighthouse on a rocky cliff")
    demo.search("a high-speed racing car")
    demo.search("a bowl of delicious ramen")

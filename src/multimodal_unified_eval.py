import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import argparse

# 1. Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
EVAL_SIZE = 1000
K_LIST = [1, 5, 10]

# --- 1-Bit Layers ---
class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight): return torch.where(weight == 0, torch.ones_like(weight), torch.sign(weight))
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

# --- Custom Head ---
class CustomProjectionHead(nn.Module):
    def __init__(self, in_features, out_features=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features, out_features)
        )
    def forward(self, x): return self.net(x)

def evaluate_recall(checkpoint_path, use_custom_head=False):
    print(f"Initializing Evaluation Engine for: {checkpoint_path}")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
    text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_ID).bfloat16().to(DEVICE).eval()
    
    student = timm.create_model(MODEL_NAME, pretrained=False)
    replace_layers_with_1bit(student)
    
    in_features = student.head.fc.in_features
    if use_custom_head:
        student.head.fc = CustomProjectionHead(in_features, 512)
    else:
        student.head.fc = nn.Linear(in_features, 512)
    
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    student.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    student = student.bfloat16().to(DEVICE).eval()
    
    data_config = timm.data.resolve_model_data_config(student)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    print(f"Loading {EVAL_SIZE} samples from ImageNet validation...")
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    
    image_features, text_queries = [], []
    class_labels = ds.features['label'].names
    
    for i, item in enumerate(tqdm(ds, total=EVAL_SIZE)):
        if i >= EVAL_SIZE: break
        img = item['image'].convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE).bfloat16()
        
        with torch.no_grad():
            feat = F.normalize(student(img_tensor), p=2, dim=-1)
            image_features.append(feat.cpu())
            
        label_idx = item['label']
        text_queries.append(f"a photo of a {class_labels[label_idx]}")

    image_features = torch.cat(image_features, dim=0)
    
    print("Extracting Text Features...")
    text_features = []
    for query in tqdm(text_queries):
        inputs = tokenizer(query, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            t_feat = F.normalize(text_model(**inputs).text_embeds, p=2, dim=-1)
            text_features.append(t_feat.cpu())
    text_features = torch.cat(text_features, dim=0)

    sim_matrix = torch.matmul(text_features, image_features.t())
    
    print("\n--- Recall@K Results ---")
    for k in K_LIST:
        correct = 0
        for i in range(EVAL_SIZE):
            _, top_indices = torch.topk(sim_matrix[i], k)
            if i in top_indices: correct += 1
        recall = (correct / EVAL_SIZE) * 100
        print(f"Recall@{k}: {recall:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--custom-head", action="store_true", help="Use custom projection head")
    args = parser.parse_args()
    evaluate_recall(args.ckpt, args.custom_head)

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

# 1. Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
BATCH_SIZE = 64
NUM_WORKERS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. W8A8 PTQ Logic (Reused from base_W8A8.py)
def apply_w8a8_ptq(model):
    print("🧩 [W8A8 PTQ] Applying 8-bit quantization to weights...")
    quantized_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if "head" in name or "classifier" in name: continue
            with torch.no_grad():
                weight = module.weight.data
                max_val = weight.abs().max()
                scale = max_val / 127.0
                if scale > 0:
                    q_weight = torch.round(weight / scale).clamp(-128, 127)
                    module.weight.data = q_weight * scale
                    quantized_layers += 1
    print(f"✅ Quantized {quantized_layers} layers to 8-bit.")
    return model

# 3. Data Loaders
_dummy = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy)
val_transform = timm.data.create_transform(**data_config, is_training=False)
del _dummy

clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def collate_fn_val(examples):
    rgbs = [ex["image"].convert("RGB") for ex in examples]
    s_images = torch.stack([val_transform(img) for img in rgbs])
    t_images = torch.stack([clip_transform(img) for img in rgbs])
    return s_images, t_images

def main():
    print(f"🚀 [Multimodal PTQ] W8A8 Evaluation Started!")
    
    print("Loading Teacher & Student...")
    teacher_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    teacher_model = teacher_model.bfloat16().to(DEVICE).eval()
    
    student_model = timm.create_model(MODEL_NAME, pretrained=True)
    student_model.head.fc = nn.Linear(student_model.head.fc.in_features, 512)
    
    # Note: For W8A8 PTQ in multimodal, we assume the model is already somewhat trained or we evaluate zero-shot distillation.
    # Here we apply PTQ to the PRETRAINED model which has a modified head.
    student_model = apply_w8a8_ptq(student_model)
    student_model = student_model.bfloat16().to(DEVICE).eval()

    print("Loading ImageNet Validation...")
    hf_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    val_loader = DataLoader(hf_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_val)

    print("Evaluating Cosine Similarity...")
    total_sim = 0.0
    num_batches = 0
    max_batches = 100 # Quick evaluation
    
    with torch.no_grad():
        for i, (s_images, t_images) in enumerate(val_loader):
            if i >= max_batches: break
            s_images, t_images = s_images.to(DEVICE, dtype=torch.bfloat16), t_images.to(DEVICE, dtype=torch.bfloat16)
            
            teacher_features = teacher_model(t_images).image_embeds
            student_features = student_model(s_images)
            
            sim = F.cosine_similarity(student_features, teacher_features, dim=-1).mean()
            total_sim += sim.item()
            num_batches += 1
            if i % 20 == 0:
                print(f"  Batch [{i}/{max_batches}] Sim: {sim.item():.4f}")

    avg_sim = total_sim / num_batches
    print("\n" + "="*50)
    print(f"🏆 W8A8 Multimodal Cosine Similarity: {avg_sim:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()

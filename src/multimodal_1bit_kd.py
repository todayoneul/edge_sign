import os
import time
import sys
import csv
import glob
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger("PIL").setLevel(logging.ERROR)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
EPOCHS = 15
LEARNING_RATE = 5e-4
SAVE_DIR = "./checkpoints/checkpoints_mm_1bit"
LOG_DIR = "./logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 2. 1-Bit Binarization Layers (Same as previous project)
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
    if weight.dim() == 4:
        scale = weight.abs().mean(dim=(1, 2, 3), keepdim=True)
    elif weight.dim() == 2:
        scale = weight.abs().mean(dim=1, keepdim=True)
    else:
        scale = weight.abs().mean()
    return BinarySTE.apply(weight) * scale

class BinaryConv2d(nn.Conv2d):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.conv2d(input, bw, bias, self.stride, self.padding, self.dilation, self.groups)

class BinaryLinear(nn.Linear):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, bw, bias)

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
        else:
            replace_layers_with_1bit(module)

# 3. Data Loaders & Transforms
_dummy = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy)
student_transform = timm.data.create_transform(**data_config, is_training=True)
student_val_transform = timm.data.create_transform(**data_config, is_training=False)
del _dummy

# CLIP Transform
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def collate_fn_train(examples):
    rgbs = [ex["image"].convert("RGB") for ex in examples]
    s_images = torch.stack([student_transform(img) for img in rgbs])
    t_images = torch.stack([clip_transform(img) for img in rgbs])
    return s_images, t_images

def collate_fn_val(examples):
    rgbs = [ex["image"].convert("RGB") for ex in examples]
    s_images = torch.stack([student_val_transform(img) for img in rgbs])
    t_images = torch.stack([clip_transform(img) for img in rgbs])
    return s_images, t_images

# 4. Main Training Loop
def main():
    print(f"[Term Project] Multimodal 1-Bit KD Started!")
    csv_file_path = os.path.join(LOG_DIR, "training_log_mm_1bit.csv")
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Cosine_Loss", "Val_Cosine_Sim", "Learning_Rate", "Time_sec"])

    print("Loading CLIP Vision Teacher Model...")
    teacher_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    teacher_model = teacher_model.bfloat16().to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print("Loading 1-Bit ConvNeXt Student Model...")
    student_model = timm.create_model(MODEL_NAME, pretrained=True)
    replace_layers_with_1bit(student_model)
    # CLIP Vit-Base projection dim is 512. Adjust student head.
    student_model.head.fc = nn.Linear(student_model.head.fc.in_features, 512)
    student_model = student_model.bfloat16().to(DEVICE)

    print("Loading ImageNet Dataset...")
    hf_dataset = load_dataset("ILSVRC/imagenet-1k")
    # For speed in 3-week project, we limit max steps or just run normally.
    train_loader = DataLoader(hf_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(hf_dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_val)

    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CosineEmbeddingLoss()

    start_epoch = 1
    checkpoints = glob.glob(os.path.join(SAVE_DIR, "mm_1bit_epoch_*.pth"))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        epoch_str = latest_ckpt.split('_epoch_')[-1].split('.pth')[0]
        start_epoch = int(epoch_str) + 1
        print(f"Auto-Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        student_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    target = torch.ones(BATCH_SIZE).to(DEVICE) # Target for Cosine Embedding Loss

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start_time = time.time()
        print(f"\n[Epoch {epoch}/{EPOCHS}] Distilling CLIP into 1-Bit CNN...")
        student_model.train()
        train_loss = 0.0
        
        for i, (s_images, t_images) in enumerate(train_loader):
            s_images = s_images.to(DEVICE, dtype=torch.bfloat16)
            t_images = t_images.to(DEVICE, dtype=torch.bfloat16)
            current_batch_size = s_images.size(0)
            target_batch = target[:current_batch_size]
            
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_features = teacher_model(t_images).image_embeds
            
            student_features = student_model(s_images)
            
            loss = criterion(student_features, teacher_features, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
            if i % 500 == 0:
                print(f"  Step [{i}/{len(train_loader)}] Cosine Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        student_model.eval()
        total_cos_sim = 0.0
        print(f"[Epoch {epoch}] Validation (Cosine Similarity)...")
        with torch.no_grad():
            for i, (s_images, t_images) in enumerate(val_loader):
                s_images = s_images.to(DEVICE, dtype=torch.bfloat16)
                t_images = t_images.to(DEVICE, dtype=torch.bfloat16)
                
                t_feats = teacher_model(t_images).image_embeds
                s_feats = student_model(s_images)
                
                cos_sim = F.cosine_similarity(s_feats, t_feats, dim=1).mean().item()
                total_cos_sim += cos_sim
        
        avg_val_sim = total_cos_sim / len(val_loader)
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} Val Avg Cosine Similarity: {avg_val_sim:.4f} (Time: {epoch_time:.1f}s)")

        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_cos_sim': avg_val_sim
        }, os.path.join(SAVE_DIR, f"mm_1bit_epoch_{epoch}.pth"))
        
        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{avg_val_sim:.4f}", f"{current_lr:.6f}", f"{epoch_time:.1f}"])

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

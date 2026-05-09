import os
import time
import csv
import glob
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
LEARNING_RATE = 5e-5  # Slightly lower for QAT
SAVE_DIR = "./checkpoints/checkpoints_mm_w4a16"
LOG_DIR = "./logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 2. W4A16 QAT Layers (Reused from base_train_w4a16_qat.py)
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output

def fake_quantize_4bit(weight):
    if weight.dim() == 4:
        max_val = weight.view(weight.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
    elif weight.dim() == 2:
        max_val = weight.abs().max(dim=1)[0].view(-1, 1)
    else:
        max_val = weight.abs().max()
    scale = torch.clamp(max_val / 7.0, min=1e-8) 
    q_weight = torch.clamp(RoundSTE.apply(weight / scale), -8, 7)
    return q_weight * scale

class W4A16Conv2d(nn.Conv2d):
    def forward(self, input):
        w_q = fake_quantize_4bit(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.conv2d(input, w_q, bias, self.stride, self.padding, self.dilation, self.groups)

class W4A16Linear(nn.Linear):
    def forward(self, input):
        w_q = fake_quantize_4bit(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, w_q, bias)

def replace_layers_with_w4a16(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and "head" not in name:
            qat_conv = W4A16Conv2d(module.in_channels, module.out_channels, module.kernel_size, 
                                   module.stride, module.padding, module.dilation, module.groups, module.bias is not None)
            qat_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None: qat_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, qat_conv)
        elif isinstance(module, nn.Linear) and "head" not in name and "classifier" not in name:
            qat_linear = W4A16Linear(module.in_features, module.out_features, module.bias is not None)
            qat_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None: qat_linear.bias.data.copy_(module.bias.data)
            setattr(model, name, qat_linear)
        else:
            replace_layers_with_w4a16(module)

# 3. Data Loaders & Transforms (Same as multimodal_1bit_kd.py)
_dummy = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy)
student_transform = timm.data.create_transform(**data_config, is_training=True)
student_val_transform = timm.data.create_transform(**data_config, is_training=False)
del _dummy

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
    print(f"🚀 [Multimodal QAT] W4A16 Multimodal KD Started!")
    csv_file_path = os.path.join(LOG_DIR, "training_log_mm_w4a16.csv")
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Cosine_Loss", "Val_Cosine_Sim", "Learning_Rate", "Time_sec"])

    print("Loading CLIP Vision Teacher...")
    teacher_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    teacher_model = teacher_model.bfloat16().to(DEVICE).eval()
    for param in teacher_model.parameters(): param.requires_grad = False

    print("Loading W4A16 Student Model...")
    student_model = timm.create_model(MODEL_NAME, pretrained=True)
    replace_layers_with_w4a16(student_model)
    student_model.head.fc = nn.Linear(student_model.head.fc.in_features, 512)
    student_model = student_model.bfloat16().to(DEVICE)

    print("Loading ImageNet...")
    hf_dataset = load_dataset("ILSVRC/imagenet-1k")
    train_loader = DataLoader(hf_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(hf_dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_val)

    optimizer = optim.AdamW(student_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CosineEmbeddingLoss()

    start_epoch = 1
    # Check for latest checkpoint to resume
    latest_epoch = 0
    for ckpt_file in glob.glob(os.path.join(SAVE_DIR, "mm_w4a16_epoch_*.pth")):
        try:
            ep = int(ckpt_file.split("_epoch_")[-1].split(".")[0])
            if ep > latest_epoch:
                latest_epoch = ep
        except ValueError:
            pass
            
    if latest_epoch > 0:
        resume_checkpoint = os.path.join(SAVE_DIR, f"mm_w4a16_epoch_{latest_epoch}.pth")
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Fast-forwarded scheduler to epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start_time = time.time()
        student_model.train()
        train_loss = 0.0
        
        for i, (s_images, t_images) in enumerate(train_loader):
            s_images, t_images = s_images.to(DEVICE, dtype=torch.bfloat16), t_images.to(DEVICE, dtype=torch.bfloat16)
            
            with torch.no_grad():
                teacher_features = teacher_model(t_images).image_embeds
                teacher_features = F.normalize(teacher_features, p=2, dim=-1)
            
            optimizer.zero_grad()
            student_features = student_model(s_images)
            student_features = F.normalize(student_features, p=2, dim=-1)
            
            loss = criterion(student_features, teacher_features, torch.ones(s_images.size(0), device=DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"  Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        student_model.eval()
        total_sim = 0.0
        with torch.no_grad():
            for s_images, t_images in val_loader:
                s_images, t_images = s_images.to(DEVICE, dtype=torch.bfloat16), t_images.to(DEVICE, dtype=torch.bfloat16)
                teacher_features = teacher_model(t_images).image_embeds
                student_features = student_model(s_images)
                
                sim = F.cosine_similarity(student_features, teacher_features, dim=-1).mean()
                total_sim += sim.item()
        
        avg_val_sim = total_sim / len(val_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"🏆 Epoch {epoch} - Val Cosine Sim: {avg_val_sim:.4f}")

        # Save Checkpoint
        torch.save({
            'model_state_dict': student_model.state_dict(),
            'epoch': epoch,
            'val_cos_sim': avg_val_sim
        }, os.path.join(SAVE_DIR, f"mm_w4a16_epoch_{epoch}.pth"))

        # Log
        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{avg_val_sim:.4f}", f"{scheduler.get_last_lr()[0]:.6f}", f"{epoch_time:.1f}"])

if __name__ == '__main__':
    main()

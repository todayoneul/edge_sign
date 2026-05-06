import os
import time
import sys
import csv
import glob
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
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
import numpy as np # 💡 비트 패킹을 위해 numpy가 필요합니다!
from safetensors.torch import save_file # 💡 safetensors 추가!

# 1. 환경 및 설정
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k' 
BATCH_SIZE = 64
NUM_WORKERS = 8 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 30        
LEARNING_RATE = 5e-4 
SAVE_DIR = "./checkpoints_1bit"
LOG_DIR = "./logs"

# KD (지식 증류) 하이퍼파라미터
TEMPERATURE = 4.0 
ALPHA = 0.9       


# 2. 1-Bit Binarization & 커스텀 레이어

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
    binary_w = BinarySTE.apply(weight)
    return binary_w * scale

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


# 3. 지식 증류 (KD) 손실 함수
def kd_loss_fn(student_logits, teacher_logits, labels, T=TEMPERATURE, alpha=ALPHA):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (T * T)
    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def export_huggingface_1bit(model, save_dir="./models/hf_1bit_model"):
    print("\n [1-Bit] 극한의 비트 패킹(Bit-packing) 추출을 시작합니다...")
    os.makedirs(save_dir, exist_ok=True)
    export_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                # 1. 원래 가중치(FP16)를 가져옵니다.
                weight = module.weight.data
                
                # 2. 채널별 스케일(Scale) 계산 (복원을 위해 따로 저장해야 함)
                if weight.dim() == 4: scale = weight.abs().mean(dim=(1, 2, 3), keepdim=True)
                elif weight.dim() == 2: scale = weight.abs().mean(dim=1, keepdim=True)
                else: scale = weight.abs().mean()
                
                export_state_dict[f"{name}.scale"] = scale.to(torch.float16)
                
                # 3. 이진화 (+1 / -1) 및 Boolean 마스크(0과 1)로 변환
                binary_mask = (weight > 0).cpu().numpy() 
                
                # 4. 🔥 비트 패킹 (8개의 불리언 값을 1개의 uint8 정수로 압축!)
                packed_bits = np.packbits(binary_mask)
                export_state_dict[f"{name}.weight_packed"] = torch.from_numpy(packed_bits)
                
            if hasattr(module, 'bias') and module.bias is not None:
                export_state_dict[f"{name}.bias"] = module.bias.data.to(torch.float16)
                
        # Normalization 레이어는 그대로 보존
        elif "norm" in name.lower() or isinstance(module, torch.nn.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                export_state_dict[f"{name}.weight"] = module.weight.to(torch.float16)
            if hasattr(module, 'bias') and module.bias is not None:
                export_state_dict[f"{name}.bias"] = module.bias.to(torch.float16)

    config = {"architectures": ["ConvNeXtV2ForImageClassification"], "quantization": "1-Bit_Packed"}
    with open(os.path.join(save_dir, "config.json"), "w") as f: json.dump(config, f)
    
    safetensors_path = os.path.join(save_dir, "model.safetensors")
    save_file(export_state_dict, safetensors_path)
    
    # 1500만 파라미터가 1.8MB로 압축되는 기적을 눈으로 확인하세요!
    print("="*50)
    print(f"1-Bit 비트 패킹 포맷 저장 완료! (위치: {save_dir})")
    print(f"실제 디스크 차지 용량: {os.path.getsize(safetensors_path) / (1024**2):.2f} MB")
    print("="*50)

# 4. 전처리 및 데이터로더 설정 (위치 정상화)
_dummy_model = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy_model)
transform_val = timm.data.create_transform(**data_config, is_training=False)
transform_train = timm.data.create_transform(**data_config, is_training=True)
del _dummy_model

def collate_fn_train(examples):
    return torch.stack([transform_train(ex["image"].convert("RGB")) for ex in examples]), torch.tensor([ex["label"] for ex in examples])
def collate_fn_val(examples):
    return torch.stack([transform_val(ex["image"].convert("RGB")) for ex in examples]), torch.tensor([ex["label"] for ex in examples])


# 5. 메인 학습 루프
def main():
    print(f"[Phase 3] 1-Bit Binary CNN + 지식 증류(KD) 학습 시작!")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_file_path = os.path.join(LOG_DIR, "training_log_1bit.csv")

    # CSV 헤더 생성
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_KD_Loss", "Val_Accuracy", "Learning_Rate", "Time_sec"])

    # 1. 16비트 선생님 모델 로드 (학습 X)
    print("FP16 선생님 모델 준비 중...")
    teacher_model = timm.create_model(MODEL_NAME, pretrained=True)
    teacher_model = teacher_model.bfloat16().to(DEVICE)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 2. 1비트 학생 모델 로드 (학습 O)
    print("1-Bit 학생 모델 준비 중...")
    student_model = timm.create_model(MODEL_NAME, pretrained=True)
    replace_layers_with_1bit(student_model)
    student_model = student_model.bfloat16().to(DEVICE)

    # 데이터셋 로드 
    print("ImageNet 데이터셋 로드 중...")
    hf_dataset = load_dataset("ILSVRC/imagenet-1k")
    train_loader = DataLoader(hf_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, collate_fn=collate_fn_train)
    val_loader = DataLoader(hf_dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, collate_fn=collate_fn_val)

    # 최적화 도구 
    optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 오토 리줌 (Auto-Resume) 로직
    start_epoch = 1
    checkpoints = glob.glob(os.path.join(SAVE_DIR, "qat_1bit_epoch_*.pth"))
    
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime) 
        epoch_str = latest_ckpt.split('_epoch_')[-1].split('.pth')[0]
        start_epoch = int(epoch_str) + 1
        
        print(f"\n[Auto-Resume] 체크포인트 발견! 기존 학생 모델을 불러옵니다: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            student_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("1-Bit 학생 가중치, 옵티마이저, 스케줄러 복구 완료!")
        else:
            student_model.load_state_dict(checkpoint)

    # 4. KD 에폭 루프
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start_time = time.time()
            print(f"\n[Epoch {epoch}/{EPOCHS}] 1-Bit 학습 + 선생님 지도 중...")
            student_model.train()
            train_loss = 0.0
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE, dtype=torch.bfloat16)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                # 선생님의 가르침 받기
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                
                # 학생의 예측 및 KD 손실 계산
                student_logits = student_model(images)
                loss = kd_loss_fn(student_logits, teacher_logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                if i % 500 == 0:
                    print(f"  Step [{i}/{len(train_loader)}] KD Loss: {loss.item():.4f}")
            
            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)

            # 5. 학생 혼자 평가 (Validation)
            student_model.eval()
            correct, total = 0, 0
            print(f" [Epoch {epoch}] 1-Bit 학생 모델 정확도 평가 중...")
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE, dtype=torch.bfloat16)
                    labels = labels.to(DEVICE)
                    
                    outputs = student_model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f" Epoch {epoch} 1-Bit Top-1 Accuracy: {acc:.2f} % (Time: {epoch_time:.1f}s)")

            # 6. 체크포인트 저장
            save_path = os.path.join(SAVE_DIR, f"qat_1bit_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'acc': acc
            }, save_path)
            
            # CSV 로깅
            with open(csv_file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{acc:.2f}", f"{current_lr:.6f}", f"{epoch_time:.1f}"])
        print("\n 30 에폭 학습이 모두 종료되었습니다. 최종 모델 추출을 시작합니다.")
        export_huggingface_1bit(student_model)
    except KeyboardInterrupt:
        print("\n학습 강제 중단! 진행 상황은 안전하게 저장되었습니다.")
        sys.exit(0)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
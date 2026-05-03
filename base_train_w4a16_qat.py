import os
import time
import csv
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import json
from torch.utils.data import DataLoader
from datasets import load_dataset
from safetensors.torch import save_file

# 1. 환경 및 설정
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k' 
BATCH_SIZE = 64
NUM_WORKERS = 8 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 30        
LEARNING_RATE = 1e-5 #
SAVE_DIR = "./checkpoints_w4a16"
LOG_DIR = "./logs"   # 데이터를 저장 폴더


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

def replace_layers_with_qat(model):
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
        else: replace_layers_with_qat(module)

def export_huggingface_w4a16(model, save_dir="./hf_w4a16_model"):
    print(f"\n📦 허깅페이스 표준 포맷(Safetensors)으로 추출을 시작합니다...")
    os.makedirs(save_dir, exist_ok=True)
    
    export_state_dict = {}
    
    # 1. 모델의 모든 레이어를 순회하며 가중치 추출
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # QAT 레이어인 경우: 4비트로 시뮬레이션 된 가중치를 진짜 추출
            if hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():
                    # 우리의 fake_quantize_4bit 로직을 거친 최종 가중치 획득
                    w_q = fake_quantize_4bit(module.weight)
                    
                    # float16을 진짜 정수형(int8)으로 강제 캐스팅
                    # 4비트(-8 ~ 7) 값이기 때문에 int8 그릇에 완벽하게 들어갑니다.
                    w_q_int = w_q.to(torch.int8) 
                    export_state_dict[f"{name}.weight"] = w_q_int
            
            # 편향(Bias)은 원래 16비트를 유지해야 하므로 그대로 저장
            if hasattr(module, 'bias') and module.bias is not None:
                export_state_dict[f"{name}.bias"] = module.bias.to(torch.float16)
        
        # 정규화(LayerNorm 등) 레이어는 양자화하지 않으므로 그대로 저장
        elif "norm" in name.lower() or isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                export_state_dict[f"{name}.weight"] = module.weight.to(torch.float16)
            if hasattr(module, 'bias') and module.bias is not None:
                export_state_dict[f"{name}.bias"] = module.bias.to(torch.float16)

    # 2. 허깅페이스 config.json 생성 (나중에 모델 구조를 알기 위함)
    config = {
        "architectures": ["ConvNeXtV2ForImageClassification"],
        "model_type": "convnextv2",
        "quantization": "W4A16",
        "num_classes": 1000,
        "torch_dtype": "int8"
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 3. 허깅페이스 표준 safetensors 형식으로 저장
    safetensors_path = os.path.join(save_dir, "model.safetensors")
    save_file(export_state_dict, safetensors_path)
    
    # 4. 실제 디스크 저장 용량 측정
    file_size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print("="*50)
    print(f"허깅페이스 포맷 저장 완료! (위치: {save_dir})")
    print(f"실제 디스크 차지 용량: {file_size_mb:.2f} MB")
    print("="*50)

# 전처리 설정
_dummy_model = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy_model)
transform_val = timm.data.create_transform(**data_config, is_training=False)
transform_train = timm.data.create_transform(**data_config, is_training=True)
del _dummy_model

def collate_fn_train(examples):
    return torch.stack([transform_train(ex["image"].convert("RGB")) for ex in examples]), torch.tensor([ex["label"] for ex in examples])
def collate_fn_val(examples):
    return torch.stack([transform_val(ex["image"].convert("RGB")) for ex in examples]), torch.tensor([ex["label"] for ex in examples])


# 3. 메인 학습 및 평가 루프 (이어하기 & 로깅 추가)
def main():
    print(f"🚀 [Phase 2] W4A16 QAT 학습 시작: {MODEL_NAME} (BFloat16 Engine)")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    csv_file_path = os.path.join(LOG_DIR, "training_log_w4a16.csv")

    model = timm.create_model(MODEL_NAME, pretrained=True)
    replace_layers_with_qat(model)
    model = model.bfloat16().to(DEVICE)

    # 최적화 도구 세팅
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # [핵심 기술 1] 오토 리줌 (Auto-Resume) 로직
    start_epoch = 1
    checkpoints = glob.glob(os.path.join(SAVE_DIR, "qat_w4a16_epoch_*.pth"))
    
    if checkpoints:
        # 가장 높은 에폭 숫자 찾기
        latest_ckpt = max(checkpoints, key=os.path.getctime) 
        epoch_str = latest_ckpt.split('_epoch_')[-1].split('.pth')[0]
        start_epoch = int(epoch_str) + 1
        
        print(f"\n🔄 [Auto-Resume] 체크포인트 발견! 기존 모델을 불러옵니다: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        
        '''
        # 이전 1~3 에폭은 가중치만 저장했으므로, 예외 처리
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ 가중치, 옵티마이저, 스케줄러 상태 복구 완료!")
        else:
            model.load_state_dict(checkpoint)
            print("⚠️ 구버전 체크포인트(가중치만 존재)를 불러왔습니다. 옵티마이저는 초기화됩니다.")
            
        if start_epoch > EPOCHS:
            print("🎉 이미 목표 에폭(EPOCHS)까지 학습이 완료되었습니다!")
            return
        '''
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print("✅ 모델 가중치 로드 완료! 허깅페이스 포맷으로 추출을 시작합니다.")
        
        # 💡 2. 진짜 가중치가 들어간 모델을 추출합니다.
        export_huggingface_w4a16(model)
        
        # 💡 3. 추출이 목적이므로 여기서 스크립트를 깔끔하게 종료합니다. (선택 사항)
        import sys
        print("🛑 추출이 완료되어 프로그램을 종료합니다.")
        sys.exit(0)

    # 데이터셋 로드
    print("📁 ImageNet 데이터셋 로드 중...")
    hf_dataset = load_dataset("ILSVRC/imagenet-1k")
    train_loader = DataLoader(hf_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, prefetch_factor=4,
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(hf_dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, prefetch_factor=4,
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_val)

    # [핵심 기술 2] CSV 로거 초기화 (파일이 없으면 헤더 생성)
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Accuracy", "Learning_Rate", "Time_sec"])

    # 4. QAT 에폭 루프 (start_epoch 부터 시작)
    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start_time = time.time()
            print(f"\n[Epoch {epoch}/{EPOCHS}] W4A16 QAT 진행 중...")
            model.train()
            train_loss = 0.0
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE, dtype=torch.bfloat16)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                
                if i % 500 == 0:
                    print(f"  Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
                    
            scheduler.step()
            avg_train_loss = train_loss / len(train_loader)

            # 5. 평가 (Validation)
            model.eval()
            correct, total = 0, 0
            print(f"🔍 [Epoch {epoch}] 정확도 평가 중...")
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE, dtype=torch.bfloat16)
                    labels = labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"🏆 Epoch {epoch} W4A16 Top-1 Accuracy: {acc:.2f} % (Time: {epoch_time:.1f}s)")

            # [핵심 기술 3] 스마트 체크포인트 저장 (옵티마이저 통째로 저장)
            save_path = os.path.join(SAVE_DIR, f"qat_w4a16_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'acc': acc
            }, save_path)
            
            # [핵심 기술 4] CSV 기록 쓰기
            with open(csv_file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{acc:.2f}", f"{current_lr:.6f}", f"{epoch_time:.1f}"])

    except KeyboardInterrupt:
        # Ctrl+C가 눌렸을 때 이 코드가 실행됩니다.
        print("\n" + "="*50)
        print("🛑 [긴급 정지] 사용자가 학습을 중단했습니다 (Ctrl+C).")
        print("💾 이전 에폭까지의 진행 상황은 안전하게 저장되어 있습니다.")
        print("🧟 좀비 프로세스 생성을 막기 위해 시스템을 강제 종료합니다...")
        print("="*50)
        
        # 데드락에 빠지기 전에 파이썬 프로세스를 강제로 셧다운!
        sys.exit(0)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# 1. 환경 및 설정

MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k' 
BATCH_SIZE = 64
NUM_WORKERS = 8 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 10        # 4비트는 10~30 에폭이면 충분히 복구됩니다.
LEARNING_RATE = 1e-5 # 미세 조정(Fine-tuning)이므로 아주 작은 학습률 사용
SAVE_DIR = "./checkpoints_w4a16"

# 💡 [핵심 기술 1] STE (Straight-Through Estimator)
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 역전파 시 반올림 연산을 무시하고 그라디언트를 그대로 통과시킴
        return grad_output


def fake_quantize_4bit(weight):
    # 💡 [수정됨] Weight Collapse 방지를 위한 Per-Channel(채널별) 스케일 계산
    if weight.dim() == 4:
        # Conv2d 레이어 [출력채널, 입력채널, K, K] -> 출력 채널별 독립 스케일
        max_val = weight.view(weight.size(0), -1).abs().max(dim=1)[0]
        max_val = max_val.view(-1, 1, 1, 1) # 원래 모양으로 브로드캐스팅 준비
    elif weight.dim() == 2:
        # Linear 레이어 [출력특성, 입력특성] -> 출력 특성별 독립 스케일
        max_val = weight.abs().max(dim=1)[0]
        max_val = max_val.view(-1, 1)
    else:
        max_val = weight.abs().max()

    scale = max_val / 7.0 
    scale = torch.clamp(scale, min=1e-8) 
    
    q_weight = RoundSTE.apply(weight / scale)
    q_weight = torch.clamp(q_weight, -8, 7)
    
    return q_weight * scale
            
# ==========================================
# 💡 [핵심 기술 2] 커스텀 QAT 레이어 정의 및 스왑
# ==========================================
class W4A16Conv2d(nn.Conv2d):
    def forward(self, input):
        # 학습 중(Forward)에 실시간으로 4비트 양자화 시뮬레이션
        w_q = fake_quantize_4bit(self.weight)
        w_q = w_q.to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.conv2d(input, w_q, bias, self.stride, self.padding, self.dilation, self.groups)

class W4A16Linear(nn.Linear):
    def forward(self, input):
        w_q = fake_quantize_4bit(self.weight)
        w_q = w_q.to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, w_q, bias)

def replace_layers_with_qat(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and "head" not in name:
            qat_conv = W4A16Conv2d(module.in_channels, module.out_channels, module.kernel_size, 
                                   module.stride, module.padding, module.dilation, module.groups, 
                                   module.bias is not None)
            qat_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None: qat_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, qat_conv)
            
        elif isinstance(module, nn.Linear) and "head" not in name and "classifier" not in name:
            qat_linear = W4A16Linear(module.in_features, module.out_features, module.bias is not None)
            qat_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None: qat_linear.bias.data.copy_(module.bias.data)
            setattr(model, name, qat_linear)
        else:
            # 하위 모듈로 재귀 탐색
            replace_layers_with_qat(module)

# ==========================================
# 3. 메인 학습 및 평가 루프
# ==========================================
_dummy_model = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy_model)
transform_val = timm.data.create_transform(**data_config, is_training=False)
transform_train = timm.data.create_transform(**data_config, is_training=True) # Data Augmentation
del _dummy_model

def collate_fn_train(examples):
    images = [transform_train(example["image"].convert("RGB")) for example in examples]
    labels = [example["label"] for example in examples]
    return torch.stack(images), torch.tensor(labels)

def collate_fn_val(examples):
    images = [transform_val(example["image"].convert("RGB")) for example in examples]
    labels = [example["label"] for example in examples]
    return torch.stack(images), torch.tensor(labels)

def main():
    print(f"🚀 [Phase 2] W4A16 QAT 학습 시작: {MODEL_NAME} (BFloat16 Engine)")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================
    # 💡 1. 모델 준비 (궁극의 해결책: bfloat16 통일)
    # ==========================================
    model = timm.create_model(MODEL_NAME, pretrained=True)
    replace_layers_with_qat(model)
    
    # 모델 전체를 bfloat16으로 덮어씌웁니다. (NaN 방어 + 타입 충돌 완벽 해결)
    model = model.bfloat16().to(DEVICE)

    # 2. 데이터셋 로드
    print("📁 ImageNet 데이터셋 로드 중...")
    hf_dataset = load_dataset("ILSVRC/imagenet-1k")
    
    train_loader = DataLoader(hf_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(hf_dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_val)

    # 3. 최적화 도구 세팅
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ==========================================
    # 4. QAT 에폭 루프 (가장 깔끔해진 형태)
    # ==========================================
    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}] W4A16 QAT 진행 중...")
        model.train()
        train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # 💡 입력 이미지도 bfloat16으로 맞춰줍니다.
            images = images.to(DEVICE, dtype=torch.bfloat16)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Scaler나 autocast 없이 순수하게 연산!
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 혹시 모를 그라디언트 스파이크만 방어
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if i % 500 == 0:
                print(f"  Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        scheduler.step()

        # 5. 평가 (Validation)
        model.eval()
        correct = 0
        total = 0
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
        print(f"🏆 Epoch {epoch} W4A16 Top-1 Accuracy: {acc:.2f} %")

        # 체크포인트 저장
        save_path = os.path.join(SAVE_DIR, f"qat_w4a16_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
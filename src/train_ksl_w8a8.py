import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

# 1. 하이퍼파라미터 및 경로 설정
# 전처리된 데이터셋 경로 (절대 경로 권장)
DATA_DIR = os.path.abspath(os.path.join(".", "dataset", "train"))
BATCH_SIZE = 64        # GPU 메모리(12GB)에 맞춘 배치 사이즈
EPOCHS = 15            # 파인튜닝 에포크 수
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. 커스텀 W8A8 양자화 연산자 (STE)
class FakeQuantize8BitSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        scale = weight.abs().max() / 127.0
        scale = torch.clamp(scale, min=1e-8) 
        ctx.save_for_backward(scale)
        quantized_weight = torch.clamp(torch.round(weight / scale), -128, 127)
        return quantized_weight * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def apply_w8a8_fake_quantization(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.weight.data = FakeQuantize8BitSTE.apply(module.weight)
    return model

# 3. 메인 실행부
if __name__ == '__main__':
    print(f"[{DEVICE}] 데이터를 로드합니다. (경로: {DATA_DIR})")

    # 데이터 증강 및 정규화
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 전체 데이터셋 로드
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    NUM_CLASSES = len(full_dataset.classes)
    
    # Train / Validation 9:1 분할
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"총 {total_size}장 중, 학습용: {train_size}장 / 검증용: {val_size}장 (클래스: {NUM_CLASSES}개)")

    # 멀티프로세싱 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, prefetch_factor=2)

    # 4. 모델 구축 (ImageNet Base + W8A8 QAT)
    print("\nConvNeXtV2-Nano 모델을 설정합니다. (ImageNet Base)")
    
    # ImageNet 사전학습 모델 로드
    model = timm.create_model('convnextv2_nano.fcmae_ft_in1k', pretrained=True)
    
    # W8A8 Fake Quantization 적용
    model = apply_w8a8_fake_quantization(model)

    # 분류기 헤드를 수어 클래스 개수에 맞게 교체
    model.head.fc = nn.Linear(model.head.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)


    # 5. 차등 학습률(Differential LR) 및 학습/검증 루프
    criterion = nn.CrossEntropyLoss()
    
    # 백본 모델과 분류기 헤드에 차등 학습률 적용
    head_params = list(model.head.fc.parameters())
    base_params = [p for n, p in model.named_parameters() if not n.startswith('head.fc')]
    
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("./checkpoints", exist_ok=True)
    print("\n수어 도메인 특화 W8A8 파인튜닝을 시작합니다.")

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        # [1단계: 학습]
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*train_correct/train_total:.2f}%"})
            
        scheduler.step()
        
        # [2단계: 검증]
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_time = time.time() - start_time
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # 결과 출력
        print(f"[Epoch {epoch}] 소요 시간: {epoch_time:.1f}초")
        print(f"   - 학습 손실: {train_loss/len(train_loader):.4f} | 학습 정확도: {train_acc:.2f}%")
        print(f"   - 검증 손실: {val_loss/len(val_loader):.4f} | 검증 정확도: {val_acc:.2f}%")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"./checkpoints/w8a8_ksl_best.pth")
            print(f"   최고 성능 모델이 저장되었습니다. (검증 정확도: {best_val_acc:.2f}%)")
            
        # 에포크마다 정기 저장
        torch.save(model.state_dict(), f"./checkpoints/w8a8_ksl_epoch_{epoch}.pth")

    print("\n모든 학습이 완료되었습니다.")
import os
import time
import csv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
import logging
logging.getLogger("PIL").setLevel(logging.ERROR) 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from datasets import load_dataset

MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k' 
BATCH_SIZE = 64
NUM_WORKERS = 8 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = "./logs"

import json
from safetensors.torch import save_file 
def export_huggingface_w8a8(model, save_dir="./hf_w8a8_model"):
    print("\n[W8A8] 허깅페이스 표준 포맷(Safetensors) 추출 시작...")
    os.makedirs(save_dir, exist_ok=True)
    export_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                # W8A8 양자화된 가중치를 int8로 강제 캐스팅
                export_state_dict[f"{name}.weight"] = module.weight.data.to(torch.int8)
            if hasattr(module, 'bias') and module.bias is not None:
                export_state_dict[f"{name}.bias"] = module.bias.data.to(torch.float16)
        elif "norm" in name.lower() or isinstance(module, torch.nn.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                export_state_dict[f"{name}.weight"] = module.weight.to(torch.float16)
            if hasattr(module, 'bias') and module.bias is not None:
                export_state_dict[f"{name}.bias"] = module.bias.to(torch.float16)

    config = {"architectures": ["ConvNeXtV2ForImageClassification"], "quantization": "W8A8", "torch_dtype": "int8"}
    with open(os.path.join(save_dir, "config.json"), "w") as f: json.dump(config, f)
    
    safetensors_path = os.path.join(save_dir, "model.safetensors")
    save_file(export_state_dict, safetensors_path)
    print(f"W8A8 저장 완료! 용량: {os.path.getsize(safetensors_path) / (1024**2):.2f} MB")

def apply_w8a8_ptq(model):
    print("\n[커스텀 W8A8 엔진] 모델 가중치 8비트 압축 시작...")
    quantized_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if "head" in name or "classifier" in name:
                continue
                
            with torch.no_grad():
                weight = module.weight.data
                max_val = weight.abs().max()
                scale = max_val / 127.0 
                
                if scale > 0:
                    q_weight = torch.round(weight / scale).clamp(-128, 127)
                    module.weight.data = q_weight * scale
                    quantized_layers += 1
                    
    print(f" 총 {quantized_layers}개의 핵심 레이어가 8비트 해상도로 변환되었습니다!")
    return model

# 전처리 설정
_dummy_model = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy_model)
transform = timm.data.create_transform(**data_config, is_training=False)
del _dummy_model 

def collate_fn(examples):
    images = [transform(example["image"].convert("RGB")) for example in examples]
    labels = [example["label"] for example in examples]
    return torch.stack(images), torch.tensor(labels)


def main():
    print(f" [Phase 1.2] W8A8 PTQ 전체 데이터셋 정밀 평가 시작: {MODEL_NAME}")
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_file_path = os.path.join(LOG_DIR, "evaluation_w8a8_ptq.csv")

    # 1. 원본 모델 로드 및 W8A8 적용
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model = model.half().to(DEVICE) # W8A8은 추론만 하므로 FP16 사용 (bfloat16도 가능)
    model.eval()
    model = apply_w8a8_ptq(model)

    # 2. 데이터셋 로드
    print(" ImageNet Validation 데이터셋 로드 중 (50,000장)...")
    try:
        hf_val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")
        
        val_loader = DataLoader(
            hf_val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, collate_fn=collate_fn
        )
    except Exception as e:
        print(f" 데이터셋 에러: {e}")
        return

    # GPU 웜업 (정확한 FPS 측정을 위해 필수)
    print(" GPU 웜업 진행 중...")
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, dtype=torch.float16, device=DEVICE)
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
    torch.cuda.synchronize()

    # 3. 전체 데이터셋 순회 및 평가
    correct_top1 = 0
    total_samples = 0
    
    print("8비트 모델 추론 시작 (전체 Validation 데이터셋)...")
    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(DEVICE, dtype=torch.float16)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total_samples += labels.size(0)
            correct_top1 += predicted.eq(labels).sum().item()
            
            # 진행 상황 알림 (터미널이 살아있는지 확인용)
            if i % 100 == 0:
                print(f"  ... 진행 중: [{total_samples}/50000] 장 처리 완료")

    torch.cuda.synchronize()
    end_time = time.time()

    # 4. 최종 결과 계산
    total_time = end_time - start_time
    fps = total_samples / total_time
    top1_acc = (correct_top1 / total_samples) * 100

    param_size = sum(p.nelement() for p in model.parameters())
    theoretical_w8a8_mb = (param_size * 1) / (1024**2) # 1 Byte (8-bit) per param

    print("\n" + "="*50)
    print("[W8A8 PTQ 최종 결과 리포트]")
    print("="*50)
    print(f"성능(P) - Top-1 Accuracy: {top1_acc:.2f} %")
    print(f"속도(S) - Throughput: {fps:.2f} FPS")
    print(f"메모리(M) - Theoretical Size: {theoretical_w8a8_mb:.2f} MB")
    print(f"소요 시간: {total_time:.1f} 초")
    print("="*50)

    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Quantization", "Top-1_Accuracy(%)", "Throughput(FPS)", "Memory(MB)", "Time(sec)"])
        writer.writerow([MODEL_NAME, "W8A8 (PTQ)", f"{top1_acc:.2f}", f"{fps:.2f}", f"{theoretical_w8a8_mb:.2f}", f"{total_time:.1f}"])
        
    print(f"결과가 성공적으로 저장되었습니다: {csv_file_path}")
    export_huggingface_w8a8(model, save_dir=f"./hf_w8a8_model_{MODEL_NAME}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        import sys
        print("\n사용자에 의해 평가가 중단되었습니다.")
        sys.exit(0)
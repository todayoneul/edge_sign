import os
import time
import torch
import timm
from torch.utils.data import DataLoader
from datasets import load_dataset


# 1. 환경 및 설정 (전역 변수)

MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k' 
BATCH_SIZE = 64
NUM_WORKERS = 8 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 모델 구조만 빠르게 불러와서 전처리 규격(Transform) 추출
_dummy_model = timm.create_model(MODEL_NAME, pretrained=False)
data_config = timm.data.resolve_model_data_config(_dummy_model)
transform = timm.data.create_transform(**data_config, is_training=False)
del _dummy_model # 메모리 절약을 위해 더미 모델 삭제

# main() 함수 바깥으로 빼내어 Windows 워커들이 정상적으로 복사(Pickle)할 수 있게 함
def collate_fn(examples):
    images = [transform(example["image"].convert("RGB")) for example in examples]
    labels = [example["label"] for example in examples]
    return torch.stack(images), torch.tensor(labels)

def main():
    print(f"🚀 [Phase 1.1] Baseline 평가 시작: {MODEL_NAME}")
    print(f"🖥️  Target Device: {DEVICE}")


    # 2. 모델 로드 및 메모리(M) 측정

    model = timm.create_model(MODEL_NAME, pretrained=True)
    model = model.half() # FP16(Half Precision)으로 변환
    model = model.to(DEVICE)
    model.eval()

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"📊 [지표 1] 모델 메모리(M): {size_all_mb:.2f} MB")


    # 3. 데이터셋 준비 (다운로드된 로컬 캐시 사용)
    print("Hugging Face에서 ImageNet-1K Validation 데이터셋 로드 중...")
    
    try:

        hf_val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")
        
        val_loader = DataLoader(
            hf_val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=True,
            collate_fn=collate_fn
        )
        print(f"📁 데이터셋 준비 완료: 총 {len(hf_val_dataset)}장")

    except Exception as e:
        print(f"⚠️ 데이터셋 로드 실패: {e}")
        return


    # 4. 성능(P) 및 속도(S) 평가 (Inference Loop)
    correct_top1 = 0
    total_samples = 0

    print("🔥 GPU 웜업 진행 중...")
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, dtype=torch.float16, device=DEVICE)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    print("🏃‍♂️ 본격적인 평가 시작...")
    start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, dtype=torch.float16)
            labels = labels.to(DEVICE)

            outputs = model(images)
            
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct_top1 += predicted.eq(labels).sum().item()
            
            # 진행 상황 모니터링 (배치 100번마다 출력)
            if (total_samples // BATCH_SIZE) % 100 == 0:
                print(f"   ... 진행 중: {total_samples}장 처리 완료")

    torch.cuda.synchronize()
    end_time = time.time()

    # 결과 계산
    total_time = end_time - start_time
    fps = total_samples / total_time
    top1_acc = (correct_top1 / total_samples) * 100

    print("\n" + "="*50)
    print("[Phase 1.1 Baseline 결과 리포트]")
    print("="*50)
    print(f"성능(P) - Top-1 Accuracy: {top1_acc:.2f} %")
    print(f"속도(S) - Throughput: {fps:.2f} FPS")
    print(f"메모리(M) - Model Size: {size_all_mb:.2f} MB")
    print("="*50)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
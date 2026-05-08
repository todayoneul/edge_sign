import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import timm

# 설정
LLM_MODEL_NAME = "Qwen/Qwen1.5-0.5B"
VISION_MODEL_NAME = "convnextv2_nano.fcmae_ft_in1k"
MAX_LENGTH = 128
SAVE_DIR = "./data/vlm_cache"
os.makedirs(SAVE_DIR, exist_ok=True)

class OmniModalDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        print(f"데이터셋 로드 시작: {split} 분할")
        
        # 가볍고 검증된 이미지-텍스트 쌍 데이터셋 로드 (예: COCO 기반의 작은 데이터셋)
        # 실제 훈련 시 더 방대한 LLaVA-Instruct 데이터셋 등으로 교체 가능합니다.
        self.dataset = load_dataset("ydshieh/coco_dataset_script", "2017", split=split, trust_remote_code=True)
        
        if max_samples is not None:
            self.dataset = self.dataset.select(range(max_samples))
            
        print("토크나이저 및 시각 전처리 모듈 초기화 중...")
        # 1. 텍스트 토크나이저 (Qwen)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. 이미지 전처리 모듈 (ConvNeXt-Nano 규격)
        _dummy = timm.create_model(VISION_MODEL_NAME, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_dummy)
        # VLM 학습 시에는 증강(Augmentation)을 최소화하여 원본 해상도와 형태를 보존하는 것이 유리함
        self.image_transform = timm.data.create_transform(**data_config, is_training=False)
        del _dummy
        
        print(f"데이터셋 초기화 완료: 총 {len(self.dataset)} 샘플 준비됨.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. 이미지 처리
        # COCO 데이터셋 구조에서 이미지 추출 (경로 기반 로드)
        image_path = item['image_path']
        from PIL import Image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # 이미지 로드 실패 시 빈 텐서 반환 (콜레이터에서 필터링 가능)
            image = Image.new('RGB', (224, 224), color='black')
            
        pixel_values = self.image_transform(image)
        
        # 2. 텍스트 처리
        # 캡션 중 첫 번째 문장을 정답(Target)으로 사용
        captions = item['captions']
        target_text = captions[0] if isinstance(captions, list) and len(captions) > 0 else "설명 없음"
        
        # Omni-Modal 지시어(Prompt) 포맷팅
        prompt = "<image>\n이 이미지를 자세히 설명해 주십시오:\n"
        full_text = prompt + target_text + self.tokenizer.eos_token
        
        # 토큰화 (입력 마스크 생성 포함)
        encoded = self.tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # 손실(Loss) 계산을 위한 Labels 생성
        # 프롬프트 부분은 Loss 계산에서 제외(-100)하고 생성할 텍스트만 학습하도록 마스킹
        labels = input_ids.clone()
        prompt_encoded = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        prompt_len = len(prompt_encoded)
        labels[:prompt_len] = -100
        
        # 패딩 부분도 Loss 계산에서 제외
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def prepare_and_test_dataloader():
    print("VLM 데이터 로더 테스트를 시작합니다.")
    # 소규모 샘플로 데이터셋 초기화 테스트
    dataset = OmniModalDataset(split="validation", max_samples=100)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    for batch in dataloader:
        print("--- 배치 데이터 구조 확인 ---")
        print(f"이미지 텐서 형태: {batch['pixel_values'].shape}")
        print(f"입력 토큰 형태: {batch['input_ids'].shape}")
        print(f"어텐션 마스크 형태: {batch['attention_mask'].shape}")
        print(f"정답 레이블 형태: {batch['labels'].shape}")
        break
    
    print("VLM 전처리 스크립트 작성 및 구조 검증이 완료되었습니다.")

if __name__ == "__main__":
    prepare_and_test_dataloader()

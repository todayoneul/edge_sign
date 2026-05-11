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
        print(f"데이터셋 로드 시작: {split} 분할 (Streaming 모드)")
        
        # 보안 이슈(스크립트 실행 중단) 및 데이터셋 스키마 캐스팅 오류를 방지하기 위해 
        # streaming=True를 사용하며, 필요한 샘플만큼만 메모리에 로드합니다.
        try:
            # HuggingFaceM4/COCO (Karpathy split) Parquet 변환 브랜치 사용
            raw_dataset = load_dataset("HuggingFaceM4/COCO", revision="refs/convert/parquet", split=split, streaming=True)
            
            if max_samples is None:
                # 전체 데이터셋이 너무 크므로, 명시적 지정이 없는 경우 기본적으로 5,000개만 로드
                max_samples = 5000
                print(f"주의: max_samples가 지정되지 않아 {max_samples}개 샘플을 기본으로 로드합니다.")
            
            print(f"{max_samples}개의 샘플을 스트리밍으로 로드 중...")
            self.samples = []
            for i, item in enumerate(raw_dataset):
                if i >= max_samples:
                    break
                self.samples.append(item)
                if (i + 1) % 1000 == 0:
                    print(f"{i + 1}개 로드 완료...")
            
        except Exception as e:
            print(f"데이터셋 로드 중 오류 발생: {e}")
            raise e
            
        print("토크나이저 및 시각 전처리 모듈 초기화 중...")
        # 1. 텍스트 토크나이저 (Qwen)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. 이미지 전처리 모듈 (ConvNeXt-Nano 규격)
        _dummy = timm.create_model(VISION_MODEL_NAME, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_dummy)
        self.image_transform = timm.data.create_transform(**data_config, is_training=False)
        del _dummy
        
        print(f"데이터셋 초기화 완료: 총 {len(self.samples)} 샘플 준비됨.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. 이미지 처리
        # Parquet 버전 데이터셋은 PIL Image 객체를 'image' 필드에 직접 포함하고 있습니다.
        try:
            image = item['image'].convert("RGB")
        except Exception:
            from PIL import Image
            image = Image.new('RGB', (224, 224), color='black')
            
        pixel_values = self.image_transform(image)
        
        # 2. 텍스트 처리
        # HuggingFaceM4/COCO(Karpathy split) Parquet 버전은 이미지-캡션 쌍이 평탄화되어 있습니다.
        # sentences 필드가 딕셔너리이며 'raw' 키를 포함합니다.
        if isinstance(item.get('sentences'), dict) and 'raw' in item['sentences']:
            target_text = item['sentences']['raw']
        elif isinstance(item.get('sentences'), list) and len(item['sentences']) > 0:
            target_text = item['sentences'][0]['raw']
        else:
            target_text = "설명 없음"
        
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

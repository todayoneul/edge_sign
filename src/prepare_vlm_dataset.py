import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import timm
import requests
from PIL import Image
from io import BytesIO

# 설정
LLM_MODEL_NAME = "Qwen/Qwen1.5-0.5B"
VISION_MODEL_NAME = "convnextv2_nano.fcmae_ft_in1k"
MAX_LENGTH = 128
SAVE_DIR = "./data/vlm_cache"
os.makedirs(SAVE_DIR, exist_ok=True)

class OmniModalIterableDataset(IterableDataset):
    def __init__(self, split="train", max_samples=None):
        print(f"데이터셋 로드 시작: Conceptual Captions 모드 (Streaming)")
        self.max_samples = max_samples
        
        try:
            # 대규모 일반 이미지-텍스트 데이터셋 (인터넷의 임의의 사진 대응을 위해)
            self.raw_dataset = load_dataset("conceptual_captions", split=split, streaming=True)
            self.raw_dataset = self.raw_dataset.shuffle(seed=42, buffer_size=1000)
        except Exception as e:
            print(f"데이터셋 로드 중 오류 발생: {e}")
            raise e
            
        print("토크나이저 및 시각 전처리 모듈 초기화 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        _dummy = timm.create_model(VISION_MODEL_NAME, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_dummy)
        self.image_transform = timm.data.create_transform(**data_config, is_training=False)
        del _dummy

    def __iter__(self):
        count = 0
        for item in self.raw_dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break
                
            try:
                # 인터넷에서 이미지 실시간 다운로드
                response = requests.get(item['image_url'], timeout=3)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception:
                # 다운로드 실패 시 해당 샘플은 무시하고 다음으로 넘어감
                continue
                
            pixel_values = self.image_transform(image)
            
            # 일반 캡션 가져오기
            target_text = item.get('caption', "설명 없음")
            
            prompt = "<image>\n이 이미지를 자세히 설명해 주십시오:\n"
            full_text = prompt + target_text + self.tokenizer.eos_token
            
            encoded = self.tokenizer(
                full_text,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            
            labels = input_ids.clone()
            prompt_encoded = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
            prompt_len = len(prompt_encoded)
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100

            count += 1
            yield {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

def prepare_and_test_dataloader():
    print("VLM 데이터 로더 테스트를 시작합니다.")
    dataset = OmniModalIterableDataset(split="train", max_samples=5)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    
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

import os
import random
import hashlib
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

DEFAULT_SOURCES = [
    "conceptual_captions",
    "coco_captions",
    "flickr30k"
]
DEFAULT_SOURCE_WEIGHTS = [0.6, 0.25, 0.15]

class OmniModalIterableDataset(IterableDataset):
    def __init__(
        self,
        split="train",
        max_samples=None,
        sources=None,
        source_weights=None,
        cache_images=True,
        cache_dir=SAVE_DIR,
        seed=42,
        caption_prompt="<image>\n이 이미지를 자세히 설명해 주십시오:\n",
        vqa_prompt_template="<image>\n질문: {question}\n답변:"
    ):
        print("데이터셋 로드 시작: 멀티 소스 스트리밍 모드")
        self.max_samples = max_samples
        self.cache_images = cache_images
        self.cache_dir = cache_dir
        self.seed = seed
        self.caption_prompt = caption_prompt
        self.vqa_prompt_template = vqa_prompt_template
        os.makedirs(self.cache_dir, exist_ok=True)

        if sources is None:
            sources = DEFAULT_SOURCES

        if source_weights is None:
            if sources == DEFAULT_SOURCES and len(DEFAULT_SOURCE_WEIGHTS) == len(sources):
                source_weights = DEFAULT_SOURCE_WEIGHTS
            else:
                source_weights = [1.0] * len(sources)

        if len(sources) != len(source_weights):
            raise ValueError("sources와 source_weights의 길이가 일치해야 합니다.")

        resolved_sources = []
        resolved_weights = []
        for src, weight in zip(sources, source_weights):
            ds = self._load_streaming_dataset(src, split)
            if ds is None:
                continue
            ds = ds.shuffle(seed=self.seed, buffer_size=1000)
            resolved_sources.append((src, ds))
            resolved_weights.append(weight)

        if len(resolved_sources) == 0:
            raise RuntimeError("사용 가능한 데이터셋이 없습니다. sources 설정을 확인해 주십시오.")

        self.sources = resolved_sources
        self.source_weights = resolved_weights

        print("토크나이저 및 시각 전처리 모듈 초기화 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _dummy = timm.create_model(VISION_MODEL_NAME, pretrained=False)
        data_config = timm.data.resolve_model_data_config(_dummy)
        self.image_transform = timm.data.create_transform(**data_config, is_training=False)
        del _dummy

    def _load_streaming_dataset(self, source_name, split):
        try:
            return load_dataset(source_name, split=split, streaming=True)
        except Exception as e:
            print(f"데이터셋 로드 실패: {source_name} | 오류: {e}")
            return None

    def _load_image_from_url(self, url):
        if url is None:
            return None

        cache_path = None
        if self.cache_images:
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{url_hash}.jpg")
            if os.path.exists(cache_path):
                try:
                    return Image.open(cache_path).convert("RGB")
                except Exception:
                    pass

        try:
            response = requests.get(url, timeout=3)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            return None

        if self.cache_images and cache_path is not None:
            try:
                image.save(cache_path, format="JPEG", quality=90)
            except Exception:
                pass

        return image

    def _extract_image(self, item):
        if "image" in item and item["image"] is not None:
            try:
                return item["image"].convert("RGB")
            except Exception:
                return None

        if "image_url" in item:
            return self._load_image_from_url(item.get("image_url"))

        return None

    def _extract_caption(self, item):
        if "caption" in item and item["caption"]:
            return item["caption"]
        if "text" in item and item["text"]:
            return item["text"]
        if "sentence" in item and item["sentence"]:
            return item["sentence"]
        if "sentences" in item and item["sentences"]:
            sentences = item["sentences"]
            if isinstance(sentences, list) and len(sentences) > 0:
                first = sentences[0]
                if isinstance(first, dict) and "raw" in first:
                    return first["raw"]
                if isinstance(first, str):
                    return first
        if "annotations" in item and item["annotations"]:
            annotations = item["annotations"]
            if isinstance(annotations, list) and len(annotations) > 0:
                first = annotations[0]
                if isinstance(first, dict) and "caption" in first:
                    return first["caption"]
        return "설명 없음"

    def _extract_vqa(self, item):
        if "question" not in item:
            return None, None

        question = item.get("question")
        answer = None
        if "answer" in item:
            answer = item.get("answer")
        elif "answers" in item:
            answers = item.get("answers")
            if isinstance(answers, list) and len(answers) > 0:
                first = answers[0]
                if isinstance(first, dict) and "answer" in first:
                    answer = first["answer"]
                elif isinstance(first, str):
                    answer = first

        if question is None or answer is None:
            return None, None

        prompt = self.vqa_prompt_template.format(question=question)
        return prompt, answer

    def __iter__(self):
        count = 0
        rng = random.Random(self.seed)
        streams = [iter(ds) for _, ds in self.sources]
        source_weights = list(self.source_weights)

        while True:
            if self.max_samples is not None and count >= self.max_samples:
                break

            src_idx = rng.choices(range(len(streams)), weights=source_weights, k=1)[0]
            try:
                item = next(streams[src_idx])
            except StopIteration:
                streams[src_idx] = iter(self.sources[src_idx][1])
                continue

            image = self._extract_image(item)
            if image is None:
                continue

            pixel_values = self.image_transform(image)

            prompt, target_text = self._extract_vqa(item)
            if prompt is None:
                prompt = self.caption_prompt
                target_text = self._extract_caption(item)

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

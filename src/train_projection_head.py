import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import argparse

from prepare_vlm_dataset import OmniModalDataset
from omni_modal_vlm import OmniModal1BitVLM

def train_projection_head(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스 설정: {device}")

    # 1. 모델 초기화
    print("Omni-Modal VLM 모델을 초기화합니다.")
    model = OmniModal1BitVLM()
    
    # Vision Encoder에 1-Bit 체크포인트 로드
    if args.vision_ckpt and os.path.exists(args.vision_ckpt):
        print(f"1-Bit Vision Encoder 체크포인트 로드: {args.vision_ckpt}")
        ckpt = torch.load(args.vision_ckpt, map_location='cpu')
        
        # 'model_state_dict' 또는 모델 자체의 state_dict에서 'head' 부분 제외하고 로드
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # 기존 1-Bit 체크포인트에는 'head' 파라미터가 있을 수 있으므로 필터링
        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        
        model.vision_encoder.load_state_dict(filtered_dict, strict=False)
        print("Vision Encoder 로드 완료.")
    else:
        print("경고: 1-Bit Vision Encoder 체크포인트가 지정되지 않았거나 파일을 찾을 수 없습니다. 랜덤 초기화로 진행합니다.")

    # 2. 파라미터 동결 (Vision Encoder & LLM 동결, Projection Head만 학습)
    print("파라미터 동결 설정: Vision Encoder 및 LLM 동결. Projection Head만 학습합니다.")
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
        
    for param in model.llm.parameters():
        param.requires_grad = False
        
    for param in model.projection_head.parameters():
        param.requires_grad = True

    # 모델 전체를 bfloat16으로 변환하여 연산 일관성 유지 (LLM이 이미 bf16이므로 전체를 맞춤)
    model = model.to(device, dtype=torch.bfloat16)

    # 3. 데이터셋 및 데이터로더 준비
    print("데이터셋을 로드합니다.")
    train_dataset = OmniModalDataset(split="train", max_samples=args.max_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 4. 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 5. 학습 루프
    print(f"학습 시작: 총 {args.epochs} 에포크")
    model.train()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # 입력 데이터 준비
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 순전파
            outputs = model(
                images=pixel_values,
                text_input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 역전파
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 평균 Loss: {avg_loss:.4f}")
        
        # 모델 저장 (Projection Head 가중치만 저장)
        save_path = os.path.join(args.save_dir, f"projection_head_epoch_{epoch+1}.pth")
        torch.save(model.projection_head.state_dict(), save_path)
        print(f"에포크 {epoch+1} 프로젝션 헤드 저장 완료: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Projection Head for Omni-Modal VLM")
    parser.add_argument("--vision_ckpt", type=str, default="./checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_15.pth", help="Path to 1-Bit Vision Encoder checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/vlm_projection", help="Directory to save checkpoints")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for training (for quick testing)")
    
    args = parser.parse_args()
    train_projection_head(args)

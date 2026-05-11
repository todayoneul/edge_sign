import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import argparse

from prepare_vlm_dataset import OmniModalIterableDataset
from omni_modal_vlm import OmniModalW8A8VLM

def train_projection_head(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스 설정: {device}")

    # 1. 모델 초기화
    print("Omni-Modal VLM 모델을 초기화합니다.")
    model = OmniModalW8A8VLM()
    
    # Vision Encoder에 체크포인트 로드
    if args.vision_ckpt and os.path.exists(args.vision_ckpt):
        print(f"Vision Encoder 체크포인트 로드: {args.vision_ckpt}")
        ckpt = torch.load(args.vision_ckpt, map_location='cpu')
        
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        
        model.vision_encoder.load_state_dict(filtered_dict, strict=False)
        print("Vision Encoder 로드 완료.")
    else:
        print("경고: Vision Encoder 체크포인트가 지정되지 않았거나 파일을 찾을 수 없습니다. 랜덤 초기화로 진행합니다.")

    # 2. 파라미터 동결
    print("파라미터 동결 설정: Vision Encoder 및 LLM 동결. Projection Head만 학습합니다.")
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.llm.parameters():
        param.requires_grad = False
    for param in model.projection_head.parameters():
        param.requires_grad = True

    model = model.to(device, dtype=torch.bfloat16)

    # 3. 데이터셋 및 데이터로더 준비
    print("데이터셋을 로드합니다 (무제한 스트리밍).")
    train_dataset = OmniModalIterableDataset(split="train", max_samples=args.max_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)

    # 4. 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    estimated_total_samples = args.max_samples if args.max_samples is not None else 833
    total_steps = (estimated_total_samples // args.batch_size) * args.epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 5. 학습 루프
    print(f"학습 시작: 총 {args.epochs} 에포크 (에포크당 최대 {estimated_total_samples} 샘플 추정)")
    model.train()
    os.makedirs(args.save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=estimated_total_samples // args.batch_size)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                images=pixel_values,
                text_input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            steps += 1
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 중간 저장 (옵션)
            if steps % 5000 == 0:
                save_path = os.path.join(args.save_dir, f"projection_head_epoch_{epoch+1}_step_{steps}.pth")
                torch.save(model.projection_head.state_dict(), save_path)
            
        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch+1} 평균 Loss: {avg_loss:.4f}")
        
        save_path = os.path.join(args.save_dir, f"projection_head_epoch_{epoch+1}.pth")
        torch.save(model.projection_head.state_dict(), save_path)
        print(f"에포크 {epoch+1} 프로젝션 헤드 저장 완료: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Projection Head for Omni-Modal VLM")
    parser.add_argument("--vision_ckpt", type=str, default="./models/hf_w8a8_smoothquant/smoothquant_w8a8.pth", help="Path to Vision Encoder checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/vlm_projection", help="Directory to save checkpoints")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for training (for quick testing)")
    
    args = parser.parse_args()
    train_projection_head(args)

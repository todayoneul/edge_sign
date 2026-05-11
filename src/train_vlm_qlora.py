import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from prepare_vlm_dataset import OmniModalIterableDataset
from omni_modal_vlm import OmniModalW8A8VLM


def parse_csv_list(value):
    if value is None or value.strip() == "":
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_csv_floats(value):
    if value is None or value.strip() == "":
        return None
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def sanitize_name(name):
    return name.replace("/", "__").replace(":", "_")


def load_vision_checkpoint(model, vision_ckpt):
    if vision_ckpt and os.path.exists(vision_ckpt):
        print(f"Vision Encoder 체크포인트 로드: {vision_ckpt}")
        ckpt = torch.load(vision_ckpt, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
        model.vision_encoder.load_state_dict(filtered_dict, strict=False)
        print("Vision Encoder 로드 완료.")
    else:
        print("경고: Vision Encoder 체크포인트를 찾지 못했습니다. 랜덤 초기화로 진행합니다.")


def load_projection_head(model, proj_ckpt):
    if proj_ckpt and os.path.exists(proj_ckpt):
        try:
            state = torch.load(proj_ckpt, map_location="cpu")
            weight = state.get("weight") if isinstance(state, dict) else None
            if weight is not None and weight.shape != model.projection_head.weight.shape:
                print("경고: 프로젝션 헤드 차원이 일치하지 않아 로드를 생략합니다.")
                return
            model.projection_head.load_state_dict(state)
            print(f"프로젝션 헤드 로드 완료: {proj_ckpt}")
        except Exception as e:
            print(f"프로젝션 헤드 로드 실패: {e}")


def build_dataloader(
    split,
    batch_size,
    max_samples,
    sources,
    source_weights,
    cache_images,
    cache_dir,
    seed
):
    try:
        dataset = OmniModalIterableDataset(
            split=split,
            max_samples=max_samples,
            sources=sources,
            source_weights=source_weights,
            cache_images=cache_images,
            cache_dir=cache_dir,
            seed=seed
        )
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        return None

    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def evaluate_loss(model, dataloader, device, max_steps):
    if dataloader is None:
        return None

    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_steps is not None and steps >= max_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                images=pixel_values,
                text_input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1

    model.train()
    if steps == 0:
        return None
    return total_loss / steps


def train_single_llm(args, llm_name, report_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스 설정: {device}")
    print(f"QLoRA 파인튜닝 시작: {llm_name}")

    model = OmniModalW8A8VLM(
        llm_name=llm_name,
        use_qlora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=parse_csv_list(args.lora_target_modules)
    )

    load_vision_checkpoint(model, args.vision_ckpt)
    load_projection_head(model, args.proj_ckpt)

    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.projection_head.parameters():
        param.requires_grad = True

    model.vision_encoder = model.vision_encoder.to(device, dtype=torch.bfloat16)
    model.projection_head = model.projection_head.to(device, dtype=torch.bfloat16)

    sources = parse_csv_list(args.sources)
    source_weights = parse_csv_floats(args.source_weights)

    train_loader = build_dataloader(
        split="train",
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        sources=sources,
        source_weights=source_weights,
        cache_images=not args.disable_cache,
        cache_dir=args.cache_dir,
        seed=args.seed
    )

    if train_loader is None:
        print("학습용 데이터셋을 로드하지 못해 종료합니다.")
        return

    eval_loader = None
    if args.eval_samples > 0:
        eval_loader = build_dataloader(
            split=args.eval_split,
            batch_size=args.batch_size,
            max_samples=args.eval_samples,
            sources=sources,
            source_weights=source_weights,
            cache_images=not args.disable_cache,
            cache_dir=args.cache_dir,
            seed=args.seed
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        if args.max_samples is None:
            steps_per_epoch = 200
        else:
            steps_per_epoch = max(1, args.max_samples // args.batch_size)

    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    llm_tag = sanitize_name(llm_name)
    output_dir = os.path.join(args.output_dir, llm_tag)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    train_loss_last = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        progress_bar = tqdm(range(steps_per_epoch), desc=f"에포크 {epoch+1}/{args.epochs}")
        data_iter = iter(train_loader)

        for _ in progress_bar:
            if args.max_steps is not None and steps >= args.max_steps:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

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
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if steps > 0:
            train_loss_last = total_loss / steps
            print(f"에포크 {epoch+1} 평균 Loss: {train_loss_last:.4f}")

        proj_path = os.path.join(output_dir, f"projection_head_epoch_{epoch+1}.pth")
        torch.save(model.projection_head.state_dict(), proj_path)

        adapter_path = os.path.join(output_dir, f"lora_adapter_epoch_{epoch+1}")
        model.llm.save_pretrained(adapter_path)

    eval_loss = None
    if eval_loader is not None:
        eval_loss = evaluate_loss(model, eval_loader, device, args.eval_max_steps)
        if eval_loss is not None:
            print(f"평가 Loss: {eval_loss:.4f}")

    elapsed = time.time() - start_time
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        is_new = not os.path.exists(report_path)
        with open(report_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("llm_name,train_loss,eval_loss,epochs,steps_per_epoch,elapsed_sec\n")
            f.write(
                f"{llm_name},{train_loss_last},{eval_loss},{args.epochs},{steps_per_epoch},{elapsed:.2f}\n"
            )

    print("QLoRA 파인튜닝이 완료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="Train QLoRA for Omni-Modal VLM")
    parser.add_argument("--vision_ckpt", type=str, default="./models/hf_w8a8_smoothquant/smoothquant_w8a8.pth")
    parser.add_argument("--proj_ckpt", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--llm_list", type=str, default=None, help="쉼표로 구분된 LLM 후보 목록")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--eval_max_steps", type=int, default=30)
    parser.add_argument("--sources", type=str, default=None)
    parser.add_argument("--source_weights", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./data/vlm_cache")
    parser.add_argument("--disable_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/vlm_qlora")
    parser.add_argument("--report_path", type=str, default="./logs/vlm_qlora_compare.csv")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    args = parser.parse_args()

    llm_list = parse_csv_list(args.llm_list)
    if llm_list is None:
        llm_list = [args.llm_name]

    for name in llm_list:
        train_single_llm(args, name, args.report_path)


if __name__ == "__main__":
    main()

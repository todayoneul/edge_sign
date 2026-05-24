import os
import sys
import shutil

# Reconfigure stdout/stderr to utf-8 to avoid encoding issues with print/emojis on Windows
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
from korean_ocr_model import KoreanOCRNet
from onnxruntime.quantization import quantize_dynamic, QuantType

class NumericalImageFolder(datasets.ImageFolder):
    """
    Subclass of ImageFolder that sorts class folders numerically (0, 1, 2...)
    rather than alphabetically (0, 1, 10, 100...) to align with idx_to_char mapping.
    """
    def find_classes(self, directory):
        classes = sorted([d.name for d in os.scandir(directory) if d.is_dir()], key=int)
        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        return classes, class_to_idx

# Configurations
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "korean_ocr"
MODEL_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODEL_DIR / "korean_ocr_best.pth"
ONNX_SAVE_PATH = MODEL_DIR / "korean_ocr.onnx"

BATCH_SIZE = 256  # Large batch size since RTX 5070 has lots of VRAM and training set is large
EPOCHS = 15       # 15 epochs is usually enough to get >95% accuracy on this clean handwriting dataset
LEARNING_RATE = 0.001

def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (Grayscale 1x64x64, normalized)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: Dataset directories not found at {DATA_DIR}. Run prepare_korean_ocr_data.py first.")
        return

    # Datasets and Loaders
    print("Loading datasets...")
    train_dataset = NumericalImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = NumericalImageFolder(root=str(val_dir), transform=val_transform)
    
    # We use num_workers=4 for fast loading, and pin_memory=True for GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.classes)
    print(f"Dataset loaded. Classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Model, Loss, Optimizer, Scheduler, Scaler for Mixed Precision
    model = KoreanOCRNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    best_acc1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train_1 = 0
        correct_train_5 = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
            total_train += labels.size(0)
            correct_train_1 += (acc1 / 100.0) * labels.size(0)
            correct_train_5 += (acc5 / 100.0) * labels.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc1 = (correct_train_1 / total_train) * 100
        train_acc5 = (correct_train_5 / total_train) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val_1 = 0
        correct_val_5 = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * images.size(0)
                acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))
                total_val += labels.size(0)
                correct_val_1 += (acc1 / 100.0) * labels.size(0)
                correct_val_5 += (acc5 / 100.0) * labels.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc1 = (correct_val_1 / total_val) * 100
        val_acc5 = (correct_val_5 / total_val) * 100
        
        print(f"Epoch [{epoch}/{EPOCHS}]")
        print(f"  Train Loss: {epoch_loss:.4f} | Top-1 Acc: {train_acc1:.2f}%, Top-5 Acc: {train_acc5:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Top-1 Acc: {val_acc1:.2f}%, Top-5 Acc: {val_acc5:.2f}%")
        
        # Save best model
        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Saved new best model weight with Val Top-1 Acc: {val_acc1:.2f}%")
            
        scheduler.step()

    print(f"\nTraining completed! Best Validation Top-1 Accuracy: {best_acc1:.2f}%")

    # Load best checkpoint and Export to ONNX
    print("\nLoading best weights for ONNX export...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # Dummy input representing batch size 1, Grayscale, 64x64
    dummy_input = torch.randn(1, 1, 64, 64, device=device)
    
    print(f"Exporting model to ONNX format at: {ONNX_SAVE_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_SAVE_PATH),
        export_params=True,
        opset_version=14, # Robust compatibility for ONNX runtime web
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX export complete.")

    # 4. Quantize to W8A8
    QUANT_ONNX_PATH = MODEL_DIR / "korean_ocr_quant.onnx"
    print("\n[Auto-Pipeline] Applying W8A8 Dynamic Quantization...")
    quantize_dynamic(
        model_input=str(ONNX_SAVE_PATH),
        model_output=str(QUANT_ONNX_PATH),
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized model saved to: {QUANT_ONNX_PATH}")
    print(f"Original size: {ONNX_SAVE_PATH.stat().st_size/1024/1024:.2f} MB")
    print(f"Quantized size: {QUANT_ONNX_PATH.stat().st_size/1024/1024:.2f} MB")

    # 5. Copy to Web Folder
    WEB_DIR = BASE_DIR / "web"
    print("\n[Auto-Pipeline] Copying assets to web folder...")
    shutil.copy(BASE_DIR / "data" / "idx_to_char.json", WEB_DIR / "idx_to_char.json")
    shutil.copy(QUANT_ONNX_PATH, WEB_DIR / "korean_ocr_quant.onnx")
    print("Assets successfully copied to web folder!")

if __name__ == "__main__":
    main()

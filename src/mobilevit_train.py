import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from pathlib import Path
import timm

# Configurations
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_SAVE_PATH = DATA_DIR / "mobilevit_best.pth"
ONNX_SAVE_PATH = DATA_DIR / "mobilevit.onnx"
BATCH_SIZE = 32 # Transformers require slightly smaller batches to fit memory
EPOCHS = 10     # Transfer learning converges very quickly (within 5-10 epochs)
LEARNING_RATE = 3e-4

class GTSRBCropDataset(Dataset):
    """Custom Dataset to read GTSRB crops based on coordinates and convert to tensor"""
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        class_id = int(row["class_id"])
        
        # Load image (RGB)
        image = Image.open(img_path).convert("RGB")
        
        # Crop to ROI if coordinates exist
        x1, y1, x2, y2 = row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"]
        if x2 > x1 and y2 > y1:
            image = image.crop((x1, y1, x2, y2))
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, class_id

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard ViT resolution (224x224)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_csv = PROCESSED_DIR / "train_split.csv"
    val_csv = PROCESSED_DIR / "val_split.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        print(f"Error: Split files not found at {PROCESSED_DIR}. Run data_prep.py first.")
        return

    # Datasets and Loaders
    train_dataset = GTSRBCropDataset(train_csv, transform=train_transform)
    val_dataset = GTSRBCropDataset(val_csv, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Load MobileViT from timm (transfer learning)
    print("Loading pre-trained MobileViT-XXS model...")
    # 'mobilevit_xxs' is a highly efficient light-weight Vision Transformer
    model = timm.create_model('mobilevit_xxs', pretrained=True, num_classes=12)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # AdamW optimizer is widely recommended for Vision Transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = (correct_train / total_train) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved new best transformer weights with Val Acc: {val_acc:.2f}%")
            
        scheduler.step()

    print(f"\nTraining completed! Best Validation Accuracy: {best_acc:.2f}%")

    # Load best checkpoint and Export to ONNX
    print("\nLoading best weights for ONNX export...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # Dummy input representing batch size 1, RGB, 224x224
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    print(f"Exporting MobileViT model to ONNX format at: {ONNX_SAVE_PATH}")
    # Export using opset 14
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_SAVE_PATH),
        export_params=True,
        opset_version=14, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("MobileViT ONNX export complete successfully.")

if __name__ == "__main__":
    main()

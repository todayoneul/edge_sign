import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms

# 1. Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "./checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_15.pth" # Needs to be trained first
SAVE_DIR = "./models/hf_w8a8_smoothquant"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. SmoothQuant Core Logic
class SmoothQuantWrapper(nn.Module):
    def __init__(self, module, smooth_scale):
        super().__init__()
        self.module = module
        self.register_buffer('smooth_scale', smooth_scale)
        
        # 1. Scale the weights (W * diag(s))
        with torch.no_grad():
            if isinstance(module, nn.Linear):
                module.weight.data.mul_(smooth_scale)
            elif isinstance(module, nn.Conv2d):
                if module.groups == module.in_channels and module.in_channels == module.out_channels:
                    module.weight.data.mul_(smooth_scale.view(-1, 1, 1, 1))
                else:
                    module.weight.data.mul_(smooth_scale.view(1, -1, 1, 1))
                
        # 2. Quantize the smoothed weights to W8
        self._quantize_weights()

    def _quantize_weights(self):
        weight = self.module.weight.data
        if weight.dim() == 4:
            max_val = weight.view(weight.size(0), -1).abs().max(dim=1)[0].view(-1, 1, 1, 1)
        else:
            max_val = weight.abs().max(dim=1)[0].view(-1, 1)
            
        scale = torch.clamp(max_val / 127.0, min=1e-8) 
        q_weight = torch.clamp(torch.round(weight / scale), -128, 127)
        self.module.weight.data = q_weight * scale # Dequantize for simulation

    def forward(self, x):
        # 3. Inverse scale the activations (X * diag(s^-1))
        if isinstance(self.module, nn.Linear):
            x_smoothed = x / self.smooth_scale
        elif isinstance(self.module, nn.Conv2d):
            x_smoothed = x / self.smooth_scale.view(1, -1, 1, 1)
            
        # Optional: Apply A8 quantization to x_smoothed here for full W8A8
        a_max = x_smoothed.abs().max()
        a_scale = torch.clamp(a_max / 127.0, min=1e-8)
        x_q = torch.clamp(torch.round(x_smoothed / a_scale), -128, 127) * a_scale
        
        return self.module(x_q)

def calibrate_and_apply_smoothquant(model, dataloader, alpha=0.5, num_calib_batches=5):
    print("🔍 [SmoothQuant] Starting Activation Calibration...")
    model.eval()
    
    act_dict = {}
    def get_act_hook(name):
        def hook(module, input, output):
            x = input[0].detach().abs()
            if isinstance(module, nn.Linear):
                # max across batch and sequence
                act_max = x.max(dim=0)[0]
                if x.dim() == 3: act_max = act_max.max(dim=0)[0]
            elif isinstance(module, nn.Conv2d):
                # max across batch, height, width -> per channel
                act_max = x.max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]
                
            if name not in act_dict: act_dict[name] = act_max
            else: act_dict[name] = torch.max(act_dict[name], act_max)
        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and "head" not in name:
            hooks.append(module.register_forward_hook(get_act_hook(name)))

    # Run calibration batch
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_calib_batches: break # Calibrate on configurable batches
            images = images.to(DEVICE, dtype=torch.bfloat16)
            model(images)
            
    for h in hooks: h.remove()
    print("✅ Calibration complete. Applying SmoothQuant transformations...")

    # Apply SmoothQuant
    for name, module in dict(model.named_modules()).items():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and "head" not in name:
            act_max = act_dict[name].clamp(min=1e-5)
            weight_max = module.weight.detach().abs().max(dim=0)[0]
            if weight_max.dim() > 1:
                weight_max = weight_max.view(weight_max.size(0), -1).max(dim=1)[0]
            weight_max = weight_max.clamp(min=1e-5)
            
            # SmoothQuant Math: s = max(|X|)^alpha / max(|W|)^(1-alpha)
            smooth_scale = (act_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)
            
            # Replace layer
            sq_layer = SmoothQuantWrapper(module, smooth_scale)
            
            # Navigate parent module to setattr
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], sq_layer)
            
    print("🚀 W8A8 SmoothQuant Model generated successfully.")
    return model

# 3. Main
def main():
    parser = argparse.ArgumentParser(description="W8A8 SmoothQuant PTQ")
    parser.add_argument("--calib-batches", type=int, default=5, help="Number of calibration batches (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha ratio (default: 0.5)")
    args = parser.parse_args()

    print("Starting W8A8 SmoothQuant PTQ Process...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Error: FP16 baseline checkpoint not found at {CHECKPOINT_PATH}.")
        print("Please train the FP16 model using `src/multimodal_fp16_baseline.py` first!")
        return

    model = timm.create_model(MODEL_NAME, pretrained=False)
    model.head.fc = nn.Linear(model.head.fc.in_features, 512)
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model = model.bfloat16().to(DEVICE).eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    hf_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    def collate_fn(examples):
        return torch.stack([transform(ex["image"].convert("RGB")) for ex in examples]), None
    
    calib_loader = DataLoader(hf_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model_sq = calibrate_and_apply_smoothquant(model, calib_loader, alpha=args.alpha, num_calib_batches=args.calib_batches)
    
    save_path = os.path.join(SAVE_DIR, "smoothquant_w8a8.pth")
    torch.save(model_sq.state_dict(), save_path)
    print(f"💾 SmoothQuant model saved to {save_path}")

if __name__ == '__main__':
    main()

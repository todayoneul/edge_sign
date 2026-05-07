import os
import torch
import torch.nn as nn
import timm
import json
from safetensors.torch import save_file

# 1. Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "./checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_15.pth"
SAVE_DIR = "./models/hf_mm_1bit_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 1-Bit Binarization Layers (Reused)
class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.where(weight == 0, torch.ones_like(weight), torch.sign(weight))
    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[weight.abs() > 1.0] = 0
        return grad_input

def binarize_weight(weight):
    if weight.dim() == 4:
        scale = weight.abs().mean(dim=(1, 2, 3), keepdim=True)
    elif weight.dim() == 2:
        scale = weight.abs().mean(dim=1, keepdim=True)
    else:
        scale = weight.abs().mean()
    return BinarySTE.apply(weight) * scale

class BinaryConv2d(nn.Conv2d):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return torch.nn.functional.conv2d(input, bw, bias, self.stride, self.padding, self.dilation, self.groups)

class BinaryLinear(nn.Linear):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(input, bw, bias)

def replace_layers_with_1bit(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and "stem" not in name and "head" not in name:
            bin_conv = BinaryConv2d(module.in_channels, module.out_channels, module.kernel_size, 
                                    module.stride, module.padding, module.dilation, module.groups, module.bias is not None)
            bin_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None: bin_conv.bias.data.copy_(module.bias.data)
            setattr(model, name, bin_conv)
        elif isinstance(module, nn.Linear) and "head" not in name and "classifier" not in name:
            bin_linear = BinaryLinear(module.in_features, module.out_features, module.bias is not None)
            bin_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None: bin_linear.bias.data.copy_(module.bias.data)
            setattr(model, name, bin_linear)
        else:
            replace_layers_with_1bit(module)

def export_huggingface_mm_1bit():
    print(f"📦 [Export] 1-Bit Multimodal Model to Hugging Face format...")
    
    # Load model structure
    model = timm.create_model(MODEL_NAME, pretrained=False)
    replace_layers_with_1bit(model)
    model.head.fc = nn.Linear(model.head.fc.in_features, 512)
    
    # Load weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Convert to Safetensors format
    # For 1-bit, we store binarized weights as sign (+1/-1) but in float16 for compatibility,
    # or we can pack them. For HF standard, we keep the simulated float weights.
    export_state_dict = {}
    for name, param in model.named_parameters():
        # Store as float16 for distribution
        export_state_dict[name] = param.data.to(torch.float16)

    # Save Safetensors
    safetensors_path = os.path.join(SAVE_DIR, "model.safetensors")
    save_file(export_state_dict, safetensors_path)
    
    # Create config.json
    config = {
        "architectures": ["ConvNeXtV2ForImageClassification"],
        "model_type": "convnextv2",
        "quantization": "1-Bit_Simulated",
        "embedding_dim": 512,
        "base_model": MODEL_NAME,
        "task": "Multimodal_ZeroShot_Retrieval",
        "torch_dtype": "float16"
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    file_size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print("="*50)
    print(f"✅ Export Complete! (Location: {SAVE_DIR})")
    print(f"📊 Disk Size: {file_size_mb:.2f} MB")
    print("="*50)

if __name__ == "__main__":
    export_huggingface_mm_1bit()

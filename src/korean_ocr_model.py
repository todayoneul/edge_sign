import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class KoreanOCRNet(nn.Module):
    """
    Ultra-lightweight CNN optimized for edge devices (WASM/ONNX Runtime).
    Input: (batch_size, 1, 64, 64) -> Grayscale handwriting image
    Output: (batch_size, 2350) -> Class logits for 2,350 Korean characters
    Total Parameters: ~700k
    """
    def __init__(self, num_classes=2350):
        super(KoreanOCRNet, self).__init__()
        
        # Block 1: 1x64x64 -> 32x32x32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 32x32x32 -> 64x16x16
        self.ds_conv2 = DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 64x16x16 -> 128x8x8
        self.ds_conv3 = DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4: 128x8x8 -> 256x4x4
        self.ds_conv4 = DepthwiseSeparableConv(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Block 5: 256x4x4 -> 256x1x1
        self.ds_conv5 = DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification layers (using 1x1 conv to avoid ONNX export GEMM bugs)
        self.dropout = nn.Dropout(0.3)
        self.fc_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.ds_conv2(x)
        x = self.pool2(x)
        
        x = self.ds_conv3(x)
        x = self.pool3(x)
        
        x = self.ds_conv4(x)
        x = self.pool4(x)
        
        x = self.ds_conv5(x)
        x = self.global_pool(x)
        
        x = self.dropout(x)
        x = self.fc_conv(x) # shape: (batch_size, num_classes, 1, 1)
        x = torch.flatten(x, 1) # shape: (batch_size, num_classes)
        return x

if __name__ == "__main__":
    # Test shape and print parameters
    model = KoreanOCRNet(num_classes=2350)
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

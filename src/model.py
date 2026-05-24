import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficSignNet(nn.Module):
    """
    Ultra-lightweight CNN model optimized for Edge / Web CPU inference.
    Input size: (batch_size, 3, 32, 32)
    Output size: (batch_size, 12)
    Number of parameters: ~65k
    """
    def __init__(self, num_classes=12):
        super(TrafficSignNet, self).__init__()
        
        # Block 1: Input 3x32x32 -> Output 16x16x16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 2: Input 16x16x16 -> Output 32x8x8
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Block 3: Input 32x8x8 -> Output 64x4x4
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global Average Pooling to collapse spatial dimensions (4x4) to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 1x1 Convolution layers replacing Fully Connected layers 
        # to bypass PyTorch 2.x ONNX Gemm exporter conversion bugs.
        self.fc1_conv = nn.Conv2d(64, 64, kernel_size=1)
        self.dropout = nn.Dropout(0.3)
        self.fc2_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # Downsample to 16x16
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # Downsample to 8x8
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2) # Downsample to 4x4
        
        # Global Pooling -> Shape: (batch_size, 64, 1, 1)
        x = self.global_pool(x)
        
        # 1x1 Conv classification (acting as FC layers)
        x = self.fc1_conv(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2_conv(x) # Shape: (batch_size, 12, 1, 1)
        
        # Flatten at the very end
        x = torch.flatten(x, 1) # Shape: (batch_size, 12)
        return x

if __name__ == "__main__":
    # Test model shape and parameters
    model = TrafficSignNet(num_classes=12)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

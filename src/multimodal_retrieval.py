import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from torchvision import transforms

# 1. Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "./checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_15.pth"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
GALLERY_SIZE = 1000  # Number of images to use for the retrieval gallery
TOP_K = 5
ASSETS_DIR = "./assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# 2. 1-Bit Binarization Layers (Reused from training)
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
        return F.conv2d(input, bw, bias, self.stride, self.padding, self.dilation, self.groups)

class BinaryLinear(nn.Linear):
    def forward(self, input):
        bw = binarize_weight(self.weight).to(input.dtype)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, bw, bias)

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

# 3. Model Loading
def load_student_model():
    print(f"Loading 1-Bit Student Model from {CHECKPOINT_PATH}...")
    model = timm.create_model(MODEL_NAME, pretrained=False)
    replace_layers_with_1bit(model)
    model.head.fc = nn.Linear(model.head.fc.in_features, 512)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.bfloat16().to(DEVICE).eval()
    return model

def load_clip_text_model():
    print(f"Loading CLIP Text Model: {CLIP_MODEL_ID}...")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
    text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_ID).bfloat16().to(DEVICE).eval()
    return tokenizer, text_model

# 4. Feature Extraction
def extract_gallery_features(model, dataset, transform):
    print(f"Extracting features for {GALLERY_SIZE} images...")
    features = []
    images = []
    
    for i in tqdm(range(GALLERY_SIZE)):
        item = dataset[i]
        img = item['image'].convert("RGB")
        images.append(img)
        
        img_tensor = transform(img).unsqueeze(0).to(DEVICE).bfloat16()
        with torch.no_grad():
            feat = model(img_tensor)
            feat = F.normalize(feat, p=2, dim=-1)
            features.append(feat.cpu())
            
    return torch.cat(features, dim=0), images

# 5. Retrieval & Visualization
def run_retrieval(query_text, text_model, tokenizer, image_features, images):
    print(f"Querying: '{query_text}'")
    
    # Text embedding
    inputs = tokenizer(query_text, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_feat = text_model(**inputs).text_embeds
        text_feat = F.normalize(text_feat, p=2, dim=-1)
    
    # Cosine Similarity
    similarities = F.cosine_similarity(text_feat.cpu(), image_features)
    top_k_scores, top_k_indices = torch.topk(similarities, TOP_K)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Query: '{query_text}'", fontsize=16)
    
    for i in range(TOP_K):
        plt.subplot(1, TOP_K, i + 1)
        plt.imshow(images[top_k_indices[i]])
        plt.title(f"Score: {top_k_scores[i].item():.4f}")
        plt.axis('off')
        
    save_path = os.path.join(ASSETS_DIR, f"retrieval_{query_text.replace(' ', '_')}.png")
    plt.savefig(save_path)
    print(f"Result saved to {save_path}")
    plt.show()

def main():
    # Transforms (from training script)
    _dummy = timm.create_model(MODEL_NAME, pretrained=False)
    data_config = timm.data.resolve_model_data_config(_dummy)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    
    # Load Dataset
    print("Loading ImageNet-1K Validation Set...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    # Convert streaming dataset to a list for indexing (take GALLERY_SIZE items)
    gallery_list = []
    print(f"Fetching {GALLERY_SIZE} samples for gallery...")
    for i, item in enumerate(dataset):
        gallery_list.append(item)
        if i >= GALLERY_SIZE - 1:
            break
            
    # Load Models
    student_model = load_student_model()
    tokenizer, text_model = load_clip_text_model()
    
    # Extract Features
    image_features, images = extract_gallery_features(student_model, gallery_list, val_transform)
    
    # Sample Queries
    test_queries = [
        "a photo of a golden retriever",
        "a red sports car",
        "a bowl of fresh fruit",
        "a lighthouse by the sea"
    ]
    
    for query in test_queries:
        run_retrieval(query, text_model, tokenizer, image_features, images)

if __name__ == "__main__":
    main()

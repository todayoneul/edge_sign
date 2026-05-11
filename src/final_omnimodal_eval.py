import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

# Imports for different quantization layers
import sys
sys.path.append(os.path.dirname(__file__))
from multimodal_w8a8_qat import replace_layers_with_w8a8
from multimodal_w4a16_qat import replace_layers_with_w4a16
from multimodal_unified_eval import replace_layers_with_1bit, CustomProjectionHead
from multimodal_w8a8_smoothquant import SmoothQuantWrapper

# Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
EVAL_SIZE = 1000
WARMUP_STEPS = 50
TIMING_STEPS = 500

MODELS_TO_EVAL = [
    {
        "name": "FP16 (Baseline)",
        "ckpt": "checkpoints/checkpoints_mm_fp16/mm_fp16_epoch_15.pth",
        "type": "fp16",
        "memory_mb": 125.0
    },
    {
        "name": "W8A8 (QAT)",
        "ckpt": "checkpoints/checkpoints_mm_w8a8/mm_w8a8_epoch_15.pth",
        "type": "w8a8",
        "memory_mb": 14.9
    },
    {
        "name": "W8A8 (SmoothQuant PTQ)",
        "ckpt": "models/hf_w8a8_smoothquant/smoothquant_w8a8.pth",
        "type": "w8a8_sq",
        "memory_mb": 30.7 # Approximately measured earlier
    },
    {
        "name": "W4A16 (QAT)",
        "ckpt": "checkpoints/checkpoints_mm_w4a16/mm_w4a16_epoch_15.pth",
        "type": "w4a16",
        "memory_mb": 14.92
    },
    {
        "name": "1-Bit (Linear Head)",
        "ckpt": "checkpoints/checkpoints_mm_1bit/mm_1bit_epoch_15.pth",
        "type": "1bit_linear",
        "memory_mb": 1.99
    },
    {
        "name": "1-Bit (Custom Head)",
        "ckpt": "checkpoints/checkpoints_mm_1bit_custom/mm_1bit_custom_epoch_15.pth",
        "type": "1bit_custom",
        "memory_mb": 1.99
    }
]

def load_eval_model(model_config):
    student = timm.create_model(MODEL_NAME, pretrained=False)
    in_features = student.head.fc.in_features
    
    if model_config["type"] == "fp16":
        student.head.fc = nn.Linear(in_features, 512)
    elif model_config["type"] == "w8a8":
        replace_layers_with_w8a8(student)
        student.head.fc = nn.Linear(in_features, 512)
    elif model_config["type"] == "w8a8_sq":
        student.head.fc = nn.Linear(in_features, 512)
        # Dummy wrap for smoothquant so state_dict can be loaded
        for name, module in dict(student.named_modules()).items():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and "head" not in name:
                # Provide a dummy smooth_scale of the correct shape. We can guess shape by out_channels or in_channels.
                # Actually, smooth_scale in the original code has length = in_channels.
                dummy_scale = torch.ones(module.in_channels if isinstance(module, nn.Conv2d) else module.in_features)
                sq_layer = SmoothQuantWrapper(module, dummy_scale)
                parts = name.split('.')
                parent = student
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], sq_layer)
    elif model_config["type"] == "w4a16":
        replace_layers_with_w4a16(student)
        student.head.fc = nn.Linear(in_features, 512)
    elif model_config["type"] == "1bit_linear":
        replace_layers_with_1bit(student)
        student.head.fc = nn.Linear(in_features, 512)
    elif model_config["type"] == "1bit_custom":
        replace_layers_with_1bit(student)
        student.head.fc = CustomProjectionHead(in_features, 512)
        
    ckpt = torch.load(model_config["ckpt"], map_location=DEVICE)
    student.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    student = student.bfloat16().to(DEVICE).eval()
    return student

def main():
    print("Initializing Unified Evaluation Engine for Final Score...")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
    text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_ID).bfloat16().to(DEVICE).eval()
    
    # Load dataset
    _dummy = timm.create_model(MODEL_NAME, pretrained=False)
    data_config = timm.data.resolve_model_data_config(_dummy)
    transform = timm.data.create_transform(**data_config, is_training=False)
    del _dummy
    
    print(f"Loading {EVAL_SIZE} samples from ImageNet validation...")
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    
    val_images = []
    text_queries = []
    class_labels = ds.features['label'].names
    
    for i, item in enumerate(ds):
        if i >= EVAL_SIZE: break
        img = item['image'].convert("RGB")
        img_tensor = transform(img).to(DEVICE).bfloat16()
        val_images.append(img_tensor)
        label_idx = item['label']
        text_queries.append(f"a photo of a {class_labels[label_idx]}")
        
    val_images = torch.stack(val_images) # [EVAL_SIZE, C, H, W]
    
    print("Extracting Text Features...")
    text_features = []
    for query in text_queries:
        inputs = tokenizer(query, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            t_feat = F.normalize(text_model(**inputs).text_embeds, p=2, dim=-1)
            text_features.append(t_feat.cpu())
    text_features = torch.cat(text_features, dim=0)
    
    results = []
    
    for config in MODELS_TO_EVAL:
        print(f"\nEvaluating {config['name']}...")
        model = load_eval_model(config)
        
        image_features = []
        
        # 1. Measure Recall@1 (Performance)
        with torch.no_grad():
            for i in range(EVAL_SIZE):
                img_tensor = val_images[i].unsqueeze(0)
                feat = F.normalize(model(img_tensor), p=2, dim=-1)
                image_features.append(feat.cpu())
        
        image_features = torch.cat(image_features, dim=0)
        sim_matrix = torch.matmul(text_features, image_features.t())
        
        correct = 0
        for i in range(EVAL_SIZE):
            _, top_indices = torch.topk(sim_matrix[i], 1)
            if i in top_indices: correct += 1
        recall_1 = (correct / EVAL_SIZE) * 100
        
        # 2. Measure Speed (Throughput -> Latency)
        # Warmup
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=DEVICE)
        with torch.no_grad():
            for _ in range(WARMUP_STEPS):
                _ = model(dummy_input)
                
        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(TIMING_STEPS):
                _ = model(dummy_input)
        end_time = time.time()
        
        latency = (end_time - start_time) / TIMING_STEPS # seconds per image
        throughput = 1.0 / latency
        
        results.append({
            "name": config["name"],
            "recall": recall_1,
            "latency": latency,
            "throughput": throughput,
            "memory": config["memory_mb"]
        })
        
        print(f"Recall@1: {recall_1:.2f}% | Latency: {latency*1000:.2f}ms | Throughput: {throughput:.2f}fps | Memory: {config['memory_mb']}MB")
        
    print("\n--- Calculating Final Scores ---")
    P_vals = [r['recall'] for r in results]
    S_vals = [r['latency'] for r in results] # latency is inverse of throughput
    M_vals = [r['memory'] for r in results]
    
    P_min, P_max = min(P_vals), max(P_vals)
    S_min, S_max = min(S_vals), max(S_vals)
    M_min, M_max = min(M_vals), max(M_vals)
    
    for r in results:
        P = r['recall']
        S = r['latency']
        M = r['memory']
        
        PerfNorm = max(0, min(1, (P - P_min) / (P_max - P_min))) if P_max > P_min else 1.0
        SpeedNorm = max(0, min(1, (S_max - S) / (S_max - S_min))) if S_max > S_min else 1.0
        MemNorm = max(0, min(1, (M_max - M) / (M_max - M_min))) if M_max > M_min else 1.0
        
        FinalScore = 0.6 * PerfNorm + 0.2 * SpeedNorm + 0.2 * MemNorm
        r['final_score'] = FinalScore
        r['perf_norm'] = PerfNorm
        r['speed_norm'] = SpeedNorm
        r['mem_norm'] = MemNorm
        
    # Sort by final score descending
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    print(f"{'Model':<25} | {'Recall@1':<10} | {'Latency':<10} | {'Memory':<10} | {'Final Score':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} | {r['recall']:<8.2f}% | {r['latency']*1000:<8.2f}ms | {r['memory']:<8.2f}MB | {r['final_score']:<10.4f}")

    # Save to a report text file
    with open("final_score_report.txt", "w", encoding="utf-8") as f:
        f.write("Model,Recall@1(%),Latency(ms),Memory(MB),PerfNorm,SpeedNorm,MemNorm,FinalScore\n")
        for r in results:
            f.write(f"{r['name']},{r['recall']:.2f},{r['latency']*1000:.2f},{r['memory']:.2f},{r['perf_norm']:.4f},{r['speed_norm']:.4f},{r['mem_norm']:.4f},{r['final_score']:.4f}\n")
            
if __name__ == "__main__":
    main()

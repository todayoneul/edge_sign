import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_DIR = "./logs"
ASSETS_DIR = "./assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

models = {
    "training_log_mm_1bit.csv": {
        "title": "Multimodal 1-Bit KD Progress",
        "save_name": "mm_1bit_kd_progress.png"
    },
    "training_log_mm_fp16.csv": {
        "title": "Multimodal FP16 Baseline Progress",
        "save_name": "mm_fp16_progress.png"
    },
    "training_log_mm_w8a8.csv": {
        "title": "Multimodal W8A8 QAT Progress",
        "save_name": "mm_w8a8_progress.png"
    },
    "training_log_mm_w4a16.csv": {
        "title": "Multimodal W4A16 QAT Progress",
        "save_name": "mm_w4a16_progress.png"
    },
    "training_log_mm_1bit_custom.csv": {
        "title": "Multimodal 1-Bit Custom Head Progress",
        "save_name": "mm_1bit_custom_progress.png"
    }
}

def plot_progress(csv_filename, config):
    csv_path = os.path.join(LOG_DIR, csv_filename)
    if not os.path.exists(csv_path):
        print(f"No log file found at {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Cosine Loss', color=color)
    ax1.plot(df['Epoch'], df['Train_Cosine_Loss'], color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val Cosine Similarity', color=color)
    ax2.plot(df['Epoch'], df['Val_Cosine_Sim'], color=color, marker='s', label='Val Similarity')
    ax2.tick_params(axis='y', labelcolor=color)

    # Set title BEFORE tight_layout to avoid cropping, add padding
    plt.title(f"{config['title']}: Loss & Cosine Similarity", pad=20)
    fig.tight_layout()
    
    save_path = os.path.join(ASSETS_DIR, config['save_name'])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def main():
    for csv_file, config in models.items():
        plot_progress(csv_file, config)

if __name__ == "__main__":
    main()

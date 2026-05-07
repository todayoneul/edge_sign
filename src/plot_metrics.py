import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_DIR = "./logs"
ASSETS_DIR = "./assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

def plot_multimodal_kd():
    csv_path = os.path.join(LOG_DIR, "training_log_mm_1bit.csv")
    if not os.path.exists(csv_path):
        print(f"No log file found at {csv_path}. Train the model first.")
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

    fig.tight_layout()
    plt.title('Multimodal 1-Bit KD Progress: Loss & Cosine Similarity')
    
    save_path = os.path.join(ASSETS_DIR, "mm_1bit_kd_progress.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_multimodal_kd()
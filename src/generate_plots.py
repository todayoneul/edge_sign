import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('assets', exist_ok=True)

# 1. Multimodal Training Progress (Epoch vs Cosine Sim)
logs = {
    "FP16 Baseline": "logs/training_log_mm_fp16.csv",
    "W8A8 QAT": "logs/training_log_mm_w8a8.csv",
    "W4A16 QAT": "logs/training_log_mm_w4a16.csv",
    "1-Bit Linear": "logs/training_log_mm_1bit.csv",
    "1-Bit Custom Head": "logs/training_log_mm_1bit_custom.csv"
}

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
for label, path in logs.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Check column names
        val_col = [c for c in df.columns if 'val_cosine_sim' in c.lower()]
        if not val_col:
            val_col = [c for c in df.columns if 'val' in c.lower() and 'loss' not in c.lower()]
        if len(val_col) > 0:
            plt.plot(df['Epoch'], df[val_col[0]], marker='o', label=label, linewidth=2)

plt.title('Omni-Modal Alignment Progress (Validation Cosine Similarity)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Cosine Similarity', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('assets/mm_all_progress.png', dpi=300)
print("Saved mm_all_progress.png")

# 2. Pareto Frontier (Memory vs Final Score)
if os.path.exists('final_score_report.txt'):
    df_score = pd.read_csv('final_score_report.txt')
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    sns.scatterplot(data=df_score, x='Memory(MB)', y='FinalScore', hue='Model', s=200, palette='deep')
    
    # Annotate points
    for i, row in df_score.iterrows():
        plt.annotate(f"{row['Model']}\n({row['Memory(MB)']:.2f}MB, {row['FinalScore']:.2f})", 
                     (row['Memory(MB)'], row['FinalScore']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.xscale('log')
    plt.title('Omni-Modal Models: Memory vs Final Score (Pareto Frontier)', fontsize=14, fontweight='bold')
    plt.xlabel('Model Size (MB, Log Scale)', fontsize=12)
    plt.ylabel('Final Score', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('assets/mm_final_pareto.png', dpi=300)
    print("Saved mm_final_pareto.png")

# 3. Bar Chart of Latency
if os.path.exists('final_score_report.txt'):
    plt.figure(figsize=(10, 6))
    df_score = df_score.sort_values(by='Latency(ms)')
    sns.barplot(data=df_score, x='Model', y='Latency(ms)', palette='viridis')
    plt.title('Inference Latency Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('assets/mm_latency_comparison.png', dpi=300)
    print("Saved mm_latency_comparison.png")


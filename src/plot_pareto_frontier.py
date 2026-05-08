import matplotlib.pyplot as plt
import os

# 설정
ASSETS_DIR = "./assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

def plot_pareto_frontier():
    # 향후 실제 학습이 끝난 후 아래 데이터 리스트를 수정해 주십시오.
    # [Model Name, Size(MB), Recall@1(%)]
    data = [
        ("FP16 (Baseline)", 125.0, 65.0), # 가상의 임시 데이터
        ("W8A8 (SmoothQuant)", 14.9, 64.0),
        ("W4A16 (QAT)", 7.5, 45.0),
        ("1-Bit (Custom Head)", 1.99, 11.3),
        ("1-Bit (Linear Head)", 1.99, 14.2)
    ]
    
    names = [item[0] for item in data]
    sizes = [item[1] for item in data]
    recalls = [item[2] for item in data]

    plt.figure(figsize=(10, 6), facecolor='#f8f9fa')
    
    # 산점도 그리기
    plt.scatter(sizes, recalls, s=150, c='blue', alpha=0.7, edgecolors='black')
    
    # 레이블 달기
    for i, name in enumerate(names):
        plt.annotate(name, (sizes[i], recalls[i]), xytext=(10, 5), 
                     textcoords='offset points', fontsize=10, fontweight='bold')

    # 파레토 프론티어 곡선을 위한 정렬 및 연결 (Size 기준 오름차순)
    sorted_indices = sorted(range(len(sizes)), key=lambda k: sizes[k])
    sorted_sizes = [sizes[i] for i in sorted_indices]
    sorted_recalls = [recalls[i] for i in sorted_indices]
    
    plt.plot(sorted_sizes, sorted_recalls, linestyle='--', color='gray', alpha=0.6)

    # 축 설정 (메모리는 로그 스케일로 표시하여 1MB와 125MB 차이를 시각적으로 명확히 함)
    plt.xscale('log')
    plt.xlabel('Model Size (MB, Log Scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
    plt.title('Pareto Frontier: Model Size vs Zero-Shot Retrieval Recall', fontsize=14, fontweight='bold', pad=15)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(ASSETS_DIR, "pareto_frontier.png")
    plt.savefig(save_path, dpi=300)
    print(f"파레토 프론티어 시각화 완료! 결과가 다음 경로에 저장되었습니다: {save_path}")
    print("향후 실제 평가 수치로 스크립트 내의 'data' 배열을 업데이트해 주십시오.")

if __name__ == "__main__":
    plot_pareto_frontier()

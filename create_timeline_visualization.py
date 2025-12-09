"""
연구 배경용 타임라인 시각화
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 6))

# 데이터
nodes = ['Node 1\n(고성능)', 'Node 2\n(중성능)', 'Node 3\n(저성능)']
times = [11.7, 22.4, 50.4]
colors = ['#3498db', '#f39c12', '#e74c3c']

# 바 그래프
y_pos = np.arange(len(nodes))
bars = ax.barh(y_pos, times, color=colors, alpha=0.7, height=0.6)

# 각 바에 시간 표시
for i, (bar, time) in enumerate(zip(bars, times)):
    ax.text(time + 1, i, f'{time}초', va='center', fontsize=12, fontweight='bold')
    
    # 대기 시간 표시
    if i < 2:
        wait_time = 50.4 - time
        ax.text(time/2, i, f'완료 후\n{wait_time:.1f}초 대기', 
                va='center', ha='center', fontsize=10, color='white', fontweight='bold')

# 병목 표시
ax.axvline(x=50.4, color='red', linestyle='--', linewidth=3, alpha=0.7, label='JCT = 50.4초 (병목)')
ax.text(50.4, -0.5, '← 병목!\n모든 노드 대기', ha='right', fontsize=12, 
        color='red', fontweight='bold')

# 데이터 할당 표시
ax.text(5, 2.5, '각 노드: 20,000개 (균등 분배)', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax.set_yticks(y_pos)
ax.set_yticklabels(nodes, fontsize=12)
ax.set_xlabel('학습 시간 (초)', fontsize=13, fontweight='bold')
ax.set_title('Baseline (균등 분배) - 병목 현상', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 60)
ax.grid(axis='x', alpha=0.3)
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('results/figures/final/baseline_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Baseline 타임라인 저장")

# Grid (α=0.7) 타임라인
fig, ax = plt.subplots(figsize=(14, 6))

times_grid = [20.6, 20.7, 21.5]
data_grid = [34027, 16986, 8987]
colors_grid = ['#27ae60', '#2ecc71', '#58d68d']

bars = ax.barh(y_pos, times_grid, color=colors_grid, alpha=0.7, height=0.6)

for i, (bar, time, data) in enumerate(zip(bars, times_grid, data_grid)):
    ax.text(time + 0.5, i, f'{time}초', va='center', fontsize=12, fontweight='bold')
    ax.text(time/2, i, f'{data:,}개', va='center', ha='center', 
            fontsize=10, color='white', fontweight='bold')

ax.axvline(x=21.5, color='green', linestyle='--', linewidth=3, alpha=0.7, label='JCT = 21.5초')
ax.text(21.5, -0.5, '균형! →\n거의 동시 완료', ha='left', fontsize=12, 
        color='green', fontweight='bold')

ax.text(10, 2.5, '성능 기반 차등 분배\n56.7% / 28.3% / 15.0%', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax.set_yticks(y_pos)
ax.set_yticklabels(nodes, fontsize=12)
ax.set_xlabel('학습 시간 (초)', fontsize=13, fontweight='bold')
ax.set_title('Grid Search (α=0.7) - 병목 해소', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 60)
ax.grid(axis='x', alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig('results/figures/final/grid_alpha70_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Grid α=0.7 타임라인 저장")

print("\n" + "="*60)
print("✅ 타임라인 시각화 완료!")
print("="*60)
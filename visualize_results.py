"""
실험 결과 시각화
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 한글 폰트 설정 (맥 기준)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 결과 디렉토리
os.makedirs('results/figures', exist_ok=True)

# 데이터 로드
with open('results/baseline.json', 'r') as f:
    baseline = json.load(f)

with open('results/simple_adaptive.json', 'r') as f:
    simple_adaptive = json.load(f)

with open('results/optimized_results.json', 'r') as f:
    optimized = json.load(f)

with open('results/summary_report.json', 'r') as f:
    summary = json.load(f)

# 1. JCT 비교 막대 그래프
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Baseline\n(균등)', 'Simple\nAdaptive', 'Grid\nSearch', 'Random\nSearch', 'Bayesian\nOpt']
jcts = [
    baseline['jct_simulated'],
    simple_adaptive['jct_simulated'],
    optimized['grid_search']['jct_simulated'],
    optimized['random_search']['jct_simulated'],
    optimized['bayesian_opt']['jct_simulated']
]

colors = ['#e74c3c', '#27ae60', '#3498db', '#9b59b6', '#f39c12']
bars = ax.bar(methods, jcts, color=colors, alpha=0.8, edgecolor='black')

# 값 표시
for i, (bar, jct) in enumerate(zip(bars, jcts)):
    height = bar.get_height()
    improvement = (jcts[0] - jct) / jcts[0] * 100 if i > 0 else 0
    label = f'{jct:.1f}s'
    if i > 0:
        label += f'\n({improvement:.1f}% ↓)'
    ax.text(bar.get_x() + bar.get_width()/2., height,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('JCT (Job Completion Time, 초)', fontsize=12)
ax.set_title('실험 방법별 JCT 비교', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(jcts) * 1.2)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/jct_comparison.png', dpi=300, bbox_inches='tight')
print("✅ JCT 비교 그래프 저장: results/figures/jct_comparison.png")

# 2. 시간 균형도 비교
fig, ax = plt.subplots(figsize=(10, 6))

balances = [
    baseline['balance']['balance_score'],
    simple_adaptive['balance']['balance_score'],
    optimized['grid_search']['balance']['balance_score'],
    optimized['random_search']['balance']['balance_score'],
    optimized['bayesian_opt']['balance']['balance_score']
]

bars = ax.bar(methods, balances, color=colors, alpha=0.8, edgecolor='black')

for bar, balance in zip(bars, balances):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{balance:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('시간 균형도 (0~1)', fontsize=12)
ax.set_title('실험 방법별 시간 균형도 비교', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='목표 (0.9)')
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('results/figures/balance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 시간 균형도 그래프 저장: results/figures/balance_comparison.png")

# 3. 노드별 학습 시간 비교 (Gantt Chart 스타일)
fig, axes = plt.subplots(5, 1, figsize=(12, 10))

experiments = [
    ('Baseline (균등)', baseline),
    ('Simple Adaptive', simple_adaptive),
    ('Grid Search', optimized['grid_search']),
    ('Random Search', optimized['random_search']),
    ('Bayesian Opt', optimized['bayesian_opt'])
]

for idx, (name, data) in enumerate(experiments):
    ax = axes[idx]
    
    times = data['simulated_times']
    node_names = ['Node 1\n(100%)', 'Node 2\n(50%)', 'Node 3\n(25%)']
    
    # 막대 그래프
    bars = ax.barh(node_names, times, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7, edgecolor='black')
    
    # JCT 선
    jct = data['jct_simulated']
    ax.axvline(x=jct, color='red', linestyle='--', linewidth=2, label=f'JCT: {jct:.1f}s')
    
    # 값 표시
    for i, (bar, time) in enumerate(zip(bars, times)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{time:.1f}s', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('학습 시간 (초)', fontsize=10)
    ax.set_title(f'{name} - JCT: {jct:.1f}s, 균형도: {data["balance"]["balance_score"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(times) * 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/node_time_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 노드별 시간 비교 그래프 저장: results/figures/node_time_comparison.png")

# 4. 데이터 분배 비율 비교
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(experiments))
width = 0.25

node1_data = [e[1]['data_distributions'][0]/60000*100 for e in experiments]
node2_data = [e[1]['data_distributions'][1]/60000*100 for e in experiments]
node3_data = [e[1]['data_distributions'][2]/60000*100 for e in experiments]

bars1 = ax.bar(x - width, node1_data, width, label='Node 1 (100%)', color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, node2_data, width, label='Node 2 (50%)', color='#f39c12', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, node3_data, width, label='Node 3 (25%)', color='#e74c3c', alpha=0.8, edgecolor='black')

# 값 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('데이터 할당 비율 (%)', fontsize=12)
ax.set_title('실험 방법별 데이터 분배 비율', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([e[0] for e in experiments], rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/data_distribution.png', dpi=300, bbox_inches='tight')
print("✅ 데이터 분배 비율 그래프 저장: results/figures/data_distribution.png")

# 5. Grid Search 탐색 히트맵
with open('experiments/grid_search/results.json', 'r') as f:
    grid_results = json.load(f)

# α, β 조합별 평균 JCT
alpha_values = sorted(set([r['alpha'] for r in grid_results['all_results']]))
beta_values = sorted(set([r['beta'] for r in grid_results['all_results']]))

heatmap_data = np.zeros((len(beta_values), len(alpha_values)))
for r in grid_results['all_results']:
    i = beta_values.index(r['beta'])
    j = alpha_values.index(r['alpha'])
    heatmap_data[i, j] = r['jct']

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            xticklabels=[f'{a:.2f}' for a in alpha_values],
            yticklabels=[f'{b:.2f}' for b in beta_values],
            cbar_kws={'label': 'JCT (초)'}, ax=ax)
ax.set_xlabel('α (CPU 가중치)', fontsize=12)
ax.set_ylabel('β (Memory 가중치)', fontsize=12)
ax.set_title('Grid Search 탐색 결과 히트맵', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/grid_search_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Grid Search 히트맵 저장: results/figures/grid_search_heatmap.png")

print("\n" + "="*60)
print("✅ 모든 그래프 생성 완료!")
print("저장 위치: results/figures/")
print("="*60)
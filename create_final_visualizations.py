"""
최종 발표용 시각화 생성
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

os.makedirs('results/figures/final', exist_ok=True)

# 데이터 로드
with open('results/baseline.json', 'r') as f:
    baseline = json.load(f)
with open('results/simple_adaptive.json', 'r') as f:
    simple = json.load(f)
with open('experiments/grid_search/results.json', 'r') as f:
    grid_old = json.load(f)
with open('results/alpha_extended.json', 'r') as f:
    grid_new = json.load(f)
with open('results/alpha_70_validation.json', 'r') as f:
    alpha70 = json.load(f)

# ============================================================
# 1. 최종 JCT 비교 (α=0.70 포함)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

methods = ['Baseline\n(균등)', 'Simple\nAdaptive', 
           'Grid Search\n(α=0.5)', 'Grid Search\n(α=0.7)', 
           'Random\nSearch', 'Bayesian\nOpt']
jcts = [50.4, 20.9, 28.1, 21.53, 29.3, 32.2]
colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12', '#9b59b6', '#e67e22']

bars = ax.bar(methods, jcts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 개선율 표시
for i, (bar, jct) in enumerate(zip(bars, jcts)):
    if i == 0:
        label = 'Baseline'
    else:
        improvement = (50.4 - jct) / 50.4 * 100
        label = f'{jct:.1f}초\n({improvement:.1f}% ↓)'
    
    ax.text(bar.get_x() + bar.get_width()/2, jct + 1.5,
            label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# α=0.7에 별표 추가
ax.text(3, jcts[3] + 4, '★ 최적', ha='center', fontsize=14, 
        color='red', fontweight='bold')

ax.set_ylabel('JCT (초)', fontsize=14, fontweight='bold')
ax.set_title('실험 결과: JCT 비교 (낮을수록 좋음 ↓)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 60)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=20.9, color='green', linestyle='--', alpha=0.5, linewidth=2, 
           label='Simple Adaptive 기준')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('results/figures/final/jct_comparison_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ JCT 비교 그래프 (최종) 저장")

# ============================================================
# 2. 시간 균형도 비교 (α=0.70 포함)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

balances = [0.422, 0.972, 0.800, 0.981, 0.798, 0.717]
bars = ax.bar(methods, balances, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, balance in zip(bars, balances):
    ax.text(bar.get_x() + bar.get_width()/2, balance + 0.02,
            f'{balance:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 목표선
ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.7, label='목표: 0.9')

# α=0.7 강조
ax.text(3, balances[3] - 0.05, '★ 최고', ha='center', fontsize=14, 
        color='red', fontweight='bold')

ax.set_ylabel('시간 균형도', fontsize=14, fontweight='bold')
ax.set_title('시간 균형도 비교 (높을수록 좋음 ↑)', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('results/figures/final/balance_comparison_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 시간 균형도 비교 그래프 (최종) 저장")

# ============================================================
# 3. α=0.70 노드별 시간 비교
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Baseline
baseline_times = [11.7, 22.4, 50.4]
baseline_data = [20000, 20000, 20000]

ax1.barh(['Node 3\n(저성능)', 'Node 2\n(중성능)', 'Node 1\n(고성능)'], 
         baseline_times[::-1], color=['#e74c3c', '#f39c12', '#3498db'], alpha=0.7)
ax1.set_xlabel('학습 시간 (초)', fontsize=12, fontweight='bold')
ax1.set_title('Baseline (균등 분배)\nJCT: 50.4초', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 55)
for i, (time, data) in enumerate(zip(baseline_times[::-1], baseline_data[::-1])):
    ax1.text(time + 1, i, f'{time:.1f}초\n({data:,}개)', va='center', fontsize=10)
ax1.axvline(x=50.4, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(50.4, -0.5, '← 병목!', ha='right', fontsize=11, color='red', fontweight='bold')

# α=0.70
alpha70_times = [20.63, 20.74, 21.53]
alpha70_data = [34027, 16986, 8987]

ax2.barh(['Node 3\n(저성능)', 'Node 2\n(중성능)', 'Node 1\n(고성능)'], 
         alpha70_times[::-1], color=['#27ae60', '#2ecc71', '#58d68d'], alpha=0.7)
ax2.set_xlabel('학습 시간 (초)', fontsize=12, fontweight='bold')
ax2.set_title('Grid Search (α=0.70)\nJCT: 21.5초', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 55)
for i, (time, data) in enumerate(zip(alpha70_times[::-1], alpha70_data[::-1])):
    ax2.text(time + 1, i, f'{time:.1f}초\n({data:,}개)', va='center', fontsize=10)
ax2.axvline(x=21.53, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2.text(21.53, -0.5, '균형! →', ha='left', fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/final/node_comparison_alpha70.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ α=0.70 노드별 비교 그래프 저장")

# ============================================================
# 4. α 값에 따른 JCT 변화 (핵심!)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

# α=0.70 실험 결과에서 추출
alpha_values = []
jct_values = []
for result in grid_new['all_results']:
    alpha_values.append(result['alpha'])
    jct_values.append(result['jct'])

# 정렬
sorted_pairs = sorted(zip(alpha_values, jct_values))
alpha_sorted, jct_sorted = zip(*sorted_pairs)

# α별 평균
alpha_unique = sorted(set(alpha_sorted))
jct_avg = []
for a in alpha_unique:
    avg = np.mean([jct for alpha, jct in sorted_pairs if alpha == a])
    jct_avg.append(avg)

ax.plot(alpha_unique, jct_avg, 'o-', linewidth=3, markersize=10, 
        color='#3498db', label='Grid Search 결과')
ax.scatter(alpha_sorted, jct_sorted, alpha=0.3, s=50, color='gray', label='개별 실험')

# 최적점 표시
best_idx = jct_avg.index(min(jct_avg))
ax.scatter(alpha_unique[best_idx], jct_avg[best_idx], s=500, marker='★', 
           color='red', edgecolors='black', linewidths=2, zorder=5,
           label=f'최적: α={alpha_unique[best_idx]:.1f}, JCT={jct_avg[best_idx]:.1f}초')

# Simple Adaptive 표시
ax.axhline(y=20.9, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label='Simple Adaptive (α≈1.0): 20.9초')

ax.set_xlabel('α (CPU 가중치)', fontsize=14, fontweight='bold')
ax.set_ylabel('JCT (초)', fontsize=14, fontweight='bold')
ax.set_title('α 값에 따른 JCT 변화\n(CPU 가중치가 높을수록 성능 향상)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0.45, 1.05)
ax.grid(alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig('results/figures/final/alpha_vs_jct.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ α vs JCT 그래프 저장")

# ============================================================
# 5. 데이터 분배 비교 (Baseline vs α=0.70)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Baseline
nodes = ['Node 1\n(고성능)', 'Node 2\n(중성능)', 'Node 3\n(저성능)']
baseline_dist = [20000, 20000, 20000]
colors_nodes = ['#3498db', '#f39c12', '#e74c3c']

wedges1, texts1, autotexts1 = ax1.pie(baseline_dist, labels=nodes, autopct='%1.1f%%',
                                        colors=colors_nodes, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Baseline (균등 분배)\n각 20,000개', fontsize=14, fontweight='bold')

# α=0.70
alpha70_dist = [34027, 16986, 8987]
wedges2, texts2, autotexts2 = ax2.pie(alpha70_dist, labels=nodes, autopct='%1.1f%%',
                                        colors=colors_nodes, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Grid Search (α=0.70)\n성능 기반 차등 분배', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/final/data_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 데이터 분배 비교 그래프 저장")

# ============================================================
# 6. 종합 결과 테이블 이미지
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

data = [
    ['방법', 'JCT (초)', '개선율', '균형도', '데이터 분배', '특징'],
    ['Baseline', '50.4', '-', '0.422', '20k / 20k / 20k', '균등 분배'],
    ['Simple Adaptive', '20.9', '58.6% ↓', '0.972', '34k / 17k / 9k', 'CPU 비율'],
    ['Grid (α=0.5)', '28.1', '44.2% ↓', '0.800', '30k / 18k / 12k', '보수적'],
    ['Grid (α=0.7) ★', '21.5', '57.3% ↓', '0.981', '34k / 17k / 9k', '최적'],
    ['Random Search', '29.3', '41.8% ↓', '0.798', '-', '불안정'],
    ['Bayesian Opt', '32.2', '36.1% ↓', '0.717', '-', '수렴 느림']
]

table = ax.table(cellText=data, cellLoc='center', loc='center',
                colWidths=[0.18, 0.12, 0.12, 0.12, 0.20, 0.16])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 헤더
for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 데이터 행
colors_row = ['#ffebee', '#e8f5e9', '#fff3e0', '#fff9c4', '#f3e5f5', '#fce4ec']
for i in range(1, 7):
    for j in range(6):
        table[(i, j)].set_facecolor(colors_row[i-1])
        table[(i, j)].set_edgecolor('black')
        if i == 4:  # Grid (α=0.7) 강조
            table[(i, j)].set_facecolor('#fff9c4')
            table[(i, j)].set_text_props(weight='bold')

plt.title('실험 결과 종합', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/final/summary_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 종합 결과 테이블 저장")

print("\n" + "="*60)
print("✅ 모든 최종 시각화 완료!")
print("저장 위치: results/figures/final/")
print("="*60)
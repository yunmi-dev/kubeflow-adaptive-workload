"""
최적화 기법별 상세 시각화
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('results/figures', exist_ok=True)

print("데이터 로딩 중...")

# 데이터 로드
try:
    with open('experiments/grid_search/results.json', 'r') as f:
        grid_data = json.load(f)
    print("✓ Grid Search 데이터 로드 완료")
except Exception as e:
    print(f"✗ Grid Search 로드 실패: {e}")
    exit(1)

try:
    with open('experiments/random_search/results.json', 'r') as f:
        random_data = json.load(f)
    print("✓ Random Search 데이터 로드 완료")
except Exception as e:
    print(f"✗ Random Search 로드 실패: {e}")
    exit(1)

try:
    with open('experiments/bayesian_opt/results.json', 'r') as f:
        bayesian_data = json.load(f)
    print("✓ Bayesian Opt 데이터 로드 완료")
except Exception as e:
    print(f"✗ Bayesian Opt 로드 실패: {e}")
    exit(1)

# ============================================================
# 1. 가중치 비교 막대 그래프
# ============================================================
print("\n[1/5] 가중치 비교 그래프 생성 중...")
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Grid Search', 'Random Search', 'Bayesian Opt']
best_weights = [
    grid_data['best_weights'],
    random_data['best_weights'],
    bayesian_data['best_weights']
]

x = np.arange(len(methods))
width = 0.18

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
labels = ['α (CPU)', 'β (Memory)', 'γ (Network)', 'δ (Load)']
keys = ['alpha', 'beta', 'gamma', 'delta']

for i, (label, key, color) in enumerate(zip(labels, keys, colors)):
    values = [w[key] for w in best_weights]
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('가중치 값', fontsize=12)
ax.set_title('최적화 기법별 최적 가중치 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc='upper right')
ax.set_ylim(0, 0.6)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/optimal_weights_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ optimal_weights_comparison.png 저장 완료")

# ============================================================
# 2. Random Search 산점도
# ============================================================
print("[2/5] Random Search 산점도 생성 중...")
fig, ax = plt.subplots(figsize=(10, 6))

random_results = random_data['all_results']
alphas = [r['alpha'] for r in random_results]
jcts = [r['jct'] for r in random_results]

scatter = ax.scatter(alphas, jcts, c=jcts, cmap='RdYlGn_r', 
                     s=100, alpha=0.6, edgecolors='black')

best = random_data['best_weights']
ax.scatter(best['alpha'], best['jct'], 
          color='red', s=300, marker='*', 
          edgecolors='black', linewidths=2,
          label=f'최적: α={best["alpha"]:.3f}, JCT={best["jct"]:.1f}초', zorder=5)

ax.set_xlabel('α (CPU 가중치)', fontsize=12)
ax.set_ylabel('JCT (초)', fontsize=12)
ax.set_title('Random Search: α와 JCT 관계', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('JCT (초)', fontsize=11)

plt.tight_layout()
plt.savefig('results/figures/random_search_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ random_search_scatter.png 저장 완료")

# ============================================================
# 3. Bayesian Optimization 수렴 곡선
# ============================================================
print("[3/5] Bayesian 수렴 곡선 생성 중...")
fig, ax = plt.subplots(figsize=(10, 6))

bayesian_results = bayesian_data['all_results']
iterations = list(range(1, len(bayesian_results) + 1))
jcts_bayesian = [r['jct'] for r in bayesian_results]

best_so_far = []
current_best = float('inf')
for jct in jcts_bayesian:
    if jct < current_best:
        current_best = jct
    best_so_far.append(current_best)

ax.plot(iterations, jcts_bayesian, 'o-', 
        color='#9b59b6', alpha=0.5, linewidth=2, 
        markersize=8, label='각 시도의 JCT')
ax.plot(iterations, best_so_far, 's-', 
        color='#e74c3c', linewidth=3, 
        markersize=8, label='현재까지 최고 기록')

if len(iterations) >= 5:
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    ax.text(5, max(jcts_bayesian), '초기화 완료 →', 
            ha='right', va='top', fontsize=10, color='gray')

ax.set_xlabel('평가 횟수', fontsize=12)
ax.set_ylabel('JCT (초)', fontsize=12)
ax.set_title('Bayesian Optimization 수렴 곡선', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

final_best = best_so_far[-1]
ax.text(len(iterations), final_best, 
        f'  최종: {final_best:.1f}초', 
        ha='left', va='center', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('results/figures/bayesian_convergence.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ bayesian_convergence.png 저장 완료")

# ============================================================
# 4. 3가지 기법 수렴 비교
# ============================================================
print("[4/5] 3가지 기법 수렴 비교 생성 중...")
fig, ax = plt.subplots(figsize=(12, 6))

# Grid Search
grid_results = grid_data['all_results']
grid_iterations = list(range(1, len(grid_results) + 1))
grid_jcts = [r['jct'] for r in grid_results]
grid_best = []
current_best = float('inf')
for jct in grid_jcts:
    if jct < current_best:
        current_best = jct
    grid_best.append(current_best)

# Random Search
random_best = []
current_best = float('inf')
for jct in jcts:
    if jct < current_best:
        current_best = jct
    random_best.append(current_best)

ax.plot(grid_iterations, grid_best, 'o-', 
        color='#3498db', linewidth=2.5, markersize=6,
        label=f'Grid Search (최종: {grid_best[-1]:.1f}초)')
ax.plot(range(1, len(random_best)+1), random_best, 's-', 
        color='#9b59b6', linewidth=2.5, markersize=6,
        label=f'Random Search (최종: {random_best[-1]:.1f}초)')
ax.plot(iterations, best_so_far, '^-', 
        color='#f39c12', linewidth=2.5, markersize=6,
        label=f'Bayesian Opt (최종: {best_so_far[-1]:.1f}초)')

ax.set_xlabel('평가 횟수', fontsize=12)
ax.set_ylabel('현재까지 최고 JCT (초)', fontsize=12)
ax.set_title('3가지 최적화 기법 수렴 속도 비교', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/convergence_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ convergence_comparison.png 저장 완료")

# ============================================================
# 5. 최적화 기법 특성 요약 테이블
# ============================================================
print("[5/5] 최적화 기법 비교 테이블 생성 중...")
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

data = [
    ['기법', '평가횟수', '최종 JCT', '최적 α', '장점', '단점'],
    ['Grid Search', 
     f"{len(grid_results)}회", 
     f"{grid_data['best_weights']['jct']:.1f}초",
     f"{grid_data['best_weights']['alpha']:.2f}",
     '전역 최적해 보장\n재현 가능',
     '계산 비용 높음'],
    ['Random Search', 
     f"{len(random_results)}회", 
     f"{random_data['best_weights']['jct']:.1f}초",
     f"{random_data['best_weights']['alpha']:.3f}",
     '빠른 탐색\n구현 간단',
     '불안정\n재현성 낮음'],
    ['Bayesian Opt', 
     f"{len(bayesian_results)}회", 
     f"{bayesian_data['best_weights']['jct']:.1f}초",
     f"{bayesian_data['best_weights']['alpha']:.3f}",
     '샘플 효율적\n이론적 우수',
     '초기 수렴 느림\n구현 복잡']
]

table = ax.table(cellText=data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.12, 0.12, 0.12, 0.25, 0.24])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

colors = ['#e8f4f8', '#f0e8f8', '#f8f0e8']
for i in range(1, 4):
    for j in range(6):
        table[(i, j)].set_facecolor(colors[i-1])
        table[(i, j)].set_edgecolor('black')

plt.title('최적화 기법 특성 비교', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/optimization_summary_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ optimization_summary_table.png 저장 완료")

print("\n" + "="*60)
print("✅ 모든 시각화 완료!")
print("저장 위치: results/figures/")
print("="*60)
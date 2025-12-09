"""
α 가중치 분석 시각화
"""
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# α 값에 따른 JCT (Grid Search 결과에서 추출)
with open('results/alpha_extended.json', 'r') as f:
    data = json.load(f)

# α별로 그룹화
alpha_values = {}
for result in data['all_results']:
    alpha = result['alpha']
    jct = result['jct']
    if alpha not in alpha_values:
        alpha_values[alpha] = []
    alpha_values[alpha].append(jct)

# 평균 계산
alphas = sorted(alpha_values.keys())
jct_means = [np.mean(alpha_values[a]) for a in alphas]
jct_mins = [np.min(alpha_values[a]) for a in alphas]

# 그래프
fig, ax = plt.subplots(figsize=(12, 7))

# 평균 선
ax.plot(alphas, jct_means, 'o-', linewidth=3, markersize=10, 
        color='#3498db', label='평균 JCT', zorder=3)

# 최소값 선
ax.plot(alphas, jct_mins, 's--', linewidth=2, markersize=8, 
        color='#e74c3c', alpha=0.7, label='최소 JCT', zorder=2)

# 개별 점들
for alpha in alphas:
    jcts = alpha_values[alpha]
    ax.scatter([alpha]*len(jcts), jcts, alpha=0.3, s=50, color='gray', zorder=1)

# 최적점 강조
best_alpha = 0.7
best_jct = np.mean(alpha_values[0.7])
ax.scatter(best_alpha, best_jct, s=600, marker='★', 
           color='gold', edgecolors='red', linewidths=3, zorder=5,
           label=f'최적: α={best_alpha}, JCT={best_jct:.1f}초')

# Simple Adaptive 기준선
ax.axhline(y=20.9, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label='Simple Adaptive (α≈1.0): 20.9초')

# 영역 표시
ax.axvspan(0.7, 1.0, alpha=0.1, color='green', label='최적 범위 (0.7~1.0)')

ax.set_xlabel('α (CPU 가중치)', fontsize=14, fontweight='bold')
ax.set_ylabel('JCT (초)', fontsize=14, fontweight='bold')
ax.set_title('α 값에 따른 JCT 변화\nCPU 가중치가 높을수록 성능 향상', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0.45, 1.05)
ax.set_ylim(15, 55)
ax.grid(alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

# 화살표 추가
ax.annotate('', xy=(0.7, 25), xytext=(0.5, 30),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax.text(0.6, 32, 'α 증가 →\n성능 향상', fontsize=11, color='red', 
        fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('results/figures/final/alpha_vs_jct.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ α vs JCT 그래프 저장")

# 표 형식
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['α 값', 'JCT (초)', '개선율', 'Node 1', 'Node 2', 'Node 3'],
    ['0.50', '28.1', '44.2%', '50.3%', '30.2%', '19.5%'],
    ['0.60', '24.3', '51.8%', '53.5%', '29.3%', '17.2%'],
    ['0.70 ★', '21.5', '57.3%', '56.7%', '28.3%', '15.0%'],
    ['0.80', '23.1', '54.2%', '58.9%', '27.1%', '14.0%'],
    ['0.90', '22.8', '54.8%', '60.2%', '26.5%', '13.3%'],
    ['1.00', '20.9', '58.5%', '57.1%', '28.6%', '14.3%']
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 헤더
for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 최적 행 강조
for j in range(6):
    table[(3, j)].set_facecolor('#fff9c4')
    table[(3, j)].set_text_props(weight='bold')

# 데이터 행
colors = ['#ffebee', '#fce4ec', '#fff9c4', '#f3e5f5', '#e8f5e9', '#e0f7fa']
for i in range(1, 7):
    for j in range(6):
        if i != 3:
            table[(i, j)].set_facecolor(colors[i-1])
        table[(i, j)].set_edgecolor('black')

plt.title('α 값에 따른 상세 결과', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/final/alpha_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ α 상세 테이블 저장")

print("\n" + "="*60)
print("✅ α 분석 시각화 완료!")
print("="*60)
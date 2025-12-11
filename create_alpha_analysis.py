"""
α 가중치 분석 시각화 (마커 수정 버전)
"""
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("α 분석 시작...")

# 데이터 로드
with open('results/alpha_extended.json', 'r') as f:
    data = json.load(f)

print(f"✓ alpha_extended.json 로드 성공")
print(f"  - all_results 개수: {len(data['all_results'])}")

# ============================================================
# 1. α vs JCT 그래프
# ============================================================
print("\n[1/2] α vs JCT 그래프 생성 중...")

# α별로 그룹화
alpha_values = {}
for result in data['all_results']:
    alpha = round(result['alpha'], 1)
    jct = result['jct']
    if alpha not in alpha_values:
        alpha_values[alpha] = []
    alpha_values[alpha].append(jct)

alphas = sorted(alpha_values.keys())
jct_means = [np.mean(alpha_values[a]) for a in alphas]
jct_mins = [np.min(alpha_values[a]) for a in alphas]

print(f"  - 발견된 α 값들: {alphas}")
print(f"  - 평균 JCT: {dict(zip(alphas, [f'{x:.1f}' for x in jct_means]))}")

# 최적점
best_idx = jct_means.index(min(jct_means))
best_alpha = alphas[best_idx]
best_jct = jct_means[best_idx]

print(f"  - 최적점: α={best_alpha}, JCT={best_jct:.1f}초")

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

# 최적점 강조 (별 대신 원으로)
ax.scatter(best_alpha, best_jct, s=800, marker='o', 
           color='gold', edgecolors='red', linewidths=4, zorder=5,
           label=f'최적: α={best_alpha}, JCT={best_jct:.1f}초')

# 별 텍스트 추가
ax.text(best_alpha, best_jct, '★', fontsize=40, ha='center', va='center',
        color='red', zorder=6, fontweight='bold')

# Simple Adaptive 기준선
ax.axhline(y=20.9, color='green', linestyle='--', linewidth=2, alpha=0.7,
           label='Simple Adaptive (α≈1.0): 20.9초')

# 최적 영역 표시
ax.axvspan(0.7, 1.0, alpha=0.1, color='green', label='최적 범위 (0.7~1.0)')

ax.set_xlabel('α (CPU 가중치)', fontsize=14, fontweight='bold')
ax.set_ylabel('JCT (초)', fontsize=14, fontweight='bold')
ax.set_title('α 값에 따른 JCT 변화\nCPU 가중치가 높을수록 성능 향상', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0.45, 1.05)
ax.set_ylim(min(jct_mins) - 5, max(jct_means) + 5)
ax.grid(alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

# 화살표 추가
if best_alpha > 0.5:
    arrow_x = 0.6
    arrow_y = max(jct_means) - 5
    ax.annotate('', xy=(best_alpha, best_jct + 1), xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(arrow_x, arrow_y + 2, 'α 증가 →\n성능 향상', fontsize=11, 
            color='red', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('results/figures/final/alpha_vs_jct.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ alpha_vs_jct.png 저장 완료")

# ============================================================
# 2. α 상세 테이블
# ============================================================
print("\n[2/2] α 상세 테이블 생성 중...")

fig, ax = plt.subplots(figsize=(11, 6))
ax.axis('tight')
ax.axis('off')

# α별 최고 결과 추출
alpha_best = {}
for result in data['all_results']:
    alpha = round(result['alpha'], 1)
    if alpha not in alpha_best or result['jct'] < alpha_best[alpha]['jct']:
        alpha_best[alpha] = result

# 테이블 데이터
baseline_jct = 50.4
table_data = [
    ['α 값', 'JCT (초)', '개선율', 'β', 'γ', 'δ']
]

for alpha in sorted(alpha_best.keys()):
    result = alpha_best[alpha]
    jct = result['jct']
    improvement = (baseline_jct - jct) / baseline_jct * 100
    beta = result.get('beta', 0)
    gamma = result.get('gamma', 0)
    delta = result.get('delta', 0)
    
    # 최적점에 별 표시
    alpha_str = f'{alpha:.1f} ★' if alpha == best_alpha else f'{alpha:.1f}'
    
    table_data.append([
        alpha_str,
        f'{jct:.1f}',
        f'{improvement:.1f}%',
        f'{beta:.2f}',
        f'{gamma:.2f}',
        f'{delta:.2f}'
    ])

print(f"  - 테이블 행 수: {len(table_data)}")

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.13, 0.15, 0.15, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 헤더 스타일
for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 데이터 행 스타일
best_row = None
for i, row in enumerate(table_data[1:], 1):
    if '★' in row[0]:
        best_row = i
        for j in range(6):
            table[(i, j)].set_facecolor('#fff9c4')
            table[(i, j)].set_text_props(weight='bold')
    else:
        for j in range(6):
            table[(i, j)].set_facecolor('#f5f5f5')
    
    for j in range(6):
        table[(i, j)].set_edgecolor('black')

plt.title('α 값에 따른 상세 결과', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/final/alpha_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ alpha_table.png 저장 완료")

print("\n" + "="*60)
print("✅ 모든 α 분석 시각화 완료!")
print(f"저장 위치: results/figures/final/")
print(f"  - alpha_vs_jct.png")
print(f"  - alpha_table.png")
print("="*60)
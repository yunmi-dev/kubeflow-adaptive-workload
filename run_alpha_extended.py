"""
α 범위 확장 실험 (0.5~1.0)
"""
from src.optimizer import WeightOptimizer
import json

def objective_function(alpha, beta, gamma, delta):
    # 기존 objective_function 복사
    from run_experiments import ExperimentRunner
    runner = ExperimentRunner()
    return runner.objective_function(alpha, beta, gamma, delta)

optimizer = WeightOptimizer(objective_function)

print("α 범위 확장 Grid Search 시작...")

# α를 0.5~1.0으로 확장!
result = optimizer.grid_search(
    alpha_range=[0.50, 0.60, 0.70, 0.80, 0.90, 1.00],  # 확장!
    beta_range=[0.0, 0.1, 0.2],
    gamma_range=[0.0, 0.05, 0.1],
    delta_constraint=(0.0, 0.3)  # δ 범위도 조정
)

with open('results/alpha_extended.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n최적 가중치: α={result['best_weights']['alpha']:.2f}")
print(f"JCT: {result['best_weights']['jct']:.2f}초")
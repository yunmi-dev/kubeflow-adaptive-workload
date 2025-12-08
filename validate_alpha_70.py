"""
α=0.70 최적 가중치로 실제 학습 검증
"""
from src.trainer import DistributedTrainer
from src.performance_scorer import PerformanceScorer
from src.data_distributor import DataDistributor
import json

print("="*60)
print("α=0.70 실제 검증 실험")
print("="*60)

# 설정
total_data = 60000
alpha, beta, gamma, delta = 0.70, 0.10, 0.00, 0.20

# 노드별 메트릭 시뮬레이션
scorer = PerformanceScorer()
nodes = [
    {'cpu_factor': 1.0},
    {'cpu_factor': 0.5},
    {'cpu_factor': 0.25}
]

node_metrics = []
for node in nodes:
    metrics = scorer.simulate_node_metrics(node['cpu_factor'])
    node_metrics.append(metrics)

# 성능 점수 계산
scores = []
for metrics in node_metrics:
    score = scorer.calculate_performance_score(metrics, alpha, beta, gamma, delta)
    scores.append(score)

print(f"\n성능 점수:")
for i, score in enumerate(scores):
    print(f"  Node {i+1}: {score:.4f}")

# 데이터 분배
distributor = DataDistributor(total_data)
distributions = distributor.distribute_data(scores)

print(f"\n데이터 분배:")
for i, dist in enumerate(distributions):
    print(f"  Node {i+1}: {dist:,} ({dist/total_data*100:.1f}%)")

# 실제 학습 실행
model_config = {'batch_size': 32, 'epochs': 3}
trainer = DistributedTrainer(model_config)

result = trainer.run_distributed_training(
    distributions,
    experiment_name="α=0.70 검증"
)

# 결과 저장
with open('results/alpha_70_validation.json', 'w') as f:
    json.dump(result, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else o)

print(f"\n최종 JCT: {result['jct_simulated']:.2f}초")
print("="*60)
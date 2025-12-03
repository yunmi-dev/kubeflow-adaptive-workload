"""
전체 실험 자동화 스크립트
"""
import json
import os
import time
import numpy as np
from datetime import datetime

# 프로젝트 모듈 import
from src.performance_scorer import PerformanceScorer
from src.data_distributor import DataDistributor
from src.trainer import DistributedTrainer, NodeProfile
from src.optimizer import WeightOptimizer

class ExperimentRunner:
    def __init__(self):
        self.total_data = 60000
        self.scorer = PerformanceScorer()
        self.distributor = DataDistributor(self.total_data)
        
        # 학습 설정
        self.model_config = {
            'batch_size': 32,
            'epochs': 3  # 실험용 (빠르게)
        }
        
        self.trainer = DistributedTrainer(self.model_config)
        
        # 노드 프로필
        self.nodes = self.trainer.nodes
        
        # 결과 저장 디렉토리
        os.makedirs('results', exist_ok=True)
        os.makedirs('experiments/grid_search', exist_ok=True)
        os.makedirs('experiments/random_search', exist_ok=True)
        os.makedirs('experiments/bayesian_opt', exist_ok=True)
    
    def objective_function(self, alpha: float, beta: float, gamma: float, delta: float) -> float:
        """
        목적 함수: 주어진 가중치로 JCT 계산
        """
        # 각 노드의 메트릭 시뮬레이션
        node_metrics = []
        for node in self.nodes:
            metrics = self.scorer.simulate_node_metrics(node.cpu_factor)
            node_metrics.append(metrics)
        
        # 성능 점수 계산
        scores = []
        for metrics in node_metrics:
            score = self.scorer.calculate_performance_score(metrics, alpha, beta, gamma, delta)
            scores.append(score)
        
        # 데이터 분배
        distributions = self.distributor.distribute_data(scores)
        
        # JCT 추정 (실제 학습 없이 빠르게)
        # 실제 시간 = 데이터 크기 / 성능 점수
        estimated_times = []
        for i, (dist, node) in enumerate(zip(distributions, self.nodes)):
            # 기준: 1000 samples당 1초 (고성능 노드 기준)
            base_time = dist / 1000.0
            # CPU 성능 반영
            simulated_time = base_time / node.cpu_factor
            estimated_times.append(simulated_time)
        
        jct = max(estimated_times)
        
        return jct
    
    def run_baseline_experiment(self):
        """Baseline 실험: 균등 분배"""
        print("\n" + "="*80)
        print("🎯 Baseline 실험 (균등 분배)")
        print("="*80)
        
        baseline_dist = [20000, 20000, 20000]
        result = self.trainer.run_distributed_training(
            baseline_dist,
            experiment_name="Baseline (균등 분배)"
        )
        
        # 저장
        with open('results/baseline.json', 'w') as f:
            json.dump(result, f, indent=2, default=self._json_converter)
        
        return result
    
    def run_simple_adaptive_experiment(self):
        """Simple Adaptive 실험: 단순 성능 비율"""
        print("\n" + "="*80)
        print("🎯 Simple Adaptive 실험 (단순 성능 비율 4:2:1)")
        print("="*80)
        
        # 성능 비율: 1.0 : 0.5 : 0.25 = 4 : 2 : 1
        adaptive_dist = [
            int(self.total_data * 4 / 7),
            int(self.total_data * 2 / 7),
            int(self.total_data * 1 / 7)
        ]
        adaptive_dist[0] += self.total_data - sum(adaptive_dist)
        
        result = self.trainer.run_distributed_training(
            adaptive_dist,
            experiment_name="Simple Adaptive (성능 비율)"
        )
        
        # 저장
        with open('results/simple_adaptive.json', 'w') as f:
            json.dump(result, f, indent=2, default=self._json_converter)
        
        return result
    
    def run_optimization_experiments(self):
        """3가지 최적화 기법 실험"""
        
        optimizer = WeightOptimizer(self.objective_function)
        
        # 1. Grid Search
        print("\n" + "="*80)
        print("🔍 Grid Search 최적화")
        print("="*80)
        
        grid_result = optimizer.grid_search(
            alpha_range=[0.35, 0.4, 0.45, 0.5],
            beta_range=[0.2, 0.25, 0.3],
            gamma_range=[0.1, 0.15, 0.2],
            delta_constraint=(0.1, 0.2)
        )
        
        with open('experiments/grid_search/results.json', 'w') as f:
            json.dump(grid_result, f, indent=2, default=self._json_converter)
        
        # 2. Random Search
        print("\n" + "="*80)
        print("🎲 Random Search 최적화")
        print("="*80)
        
        random_result = optimizer.random_search(
            n_iterations=50,  # 빠르게 하기 위해 50회
            alpha_range=(0.3, 0.5),
            beta_range=(0.2, 0.3),
            gamma_range=(0.1, 0.2),
            delta_constraint=(0.1, 0.2)
        )
        
        with open('experiments/random_search/results.json', 'w') as f:
            json.dump(random_result, f, indent=2, default=self._json_converter)
        
        # 3. Bayesian Optimization
        print("\n" + "="*80)
        print("🧠 Bayesian Optimization 최적화")
        print("="*80)
        
        bayesian_result = optimizer.bayesian_optimization(
            n_iterations=20,  # 빠르게 하기 위해 20회
            n_initial=5,
            alpha_range=(0.3, 0.5),
            beta_range=(0.2, 0.3),
            gamma_range=(0.1, 0.2),
            delta_constraint=(0.1, 0.2)
        )
        
        with open('experiments/bayesian_opt/results.json', 'w') as f:
            json.dump(bayesian_result, f, indent=2, default=self._json_converter)
        
        return {
            'grid_search': grid_result,
            'random_search': random_result,
            'bayesian_opt': bayesian_result
        }
    
    def run_final_validation(self, optimization_results):
        """최적 가중치로 실제 분산 학습 실행"""
        
        print("\n" + "="*80)
        print("✅ 최종 검증: 최적 가중치로 실제 학습")
        print("="*80)
        
        final_results = {}
        
        for method_name, opt_result in optimization_results.items():
            best_weights = opt_result['best_weights']
            alpha = best_weights['alpha']
            beta = best_weights['beta']
            gamma = best_weights['gamma']
            delta = best_weights['delta']
            
            print(f"\n{'='*60}")
            print(f"방법: {method_name}")
            print(f"최적 가중치: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}")
            print(f"{'='*60}")
            
            # 메트릭 시뮬레이션
            node_metrics = []
            for node in self.nodes:
                metrics = self.scorer.simulate_node_metrics(node.cpu_factor)
                node_metrics.append(metrics)
            
            # 성능 점수 계산
            scores = []
            for metrics in node_metrics:
                score = self.scorer.calculate_performance_score(metrics, alpha, beta, gamma, delta)
                scores.append(score)
            
            # 데이터 분배
            distributions = self.distributor.distribute_data(scores)
            
            print(f"\n데이터 분배: {distributions}")
            
            # 실제 학습
            result = self.trainer.run_distributed_training(
                distributions,
                experiment_name=f"Optimized by {method_name}"
            )
            
            final_results[method_name] = result
        
        # 저장
        with open('results/optimized_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=self._json_converter)
        
        return final_results
    
    def generate_summary_report(self, baseline, simple_adaptive, optimized_results):
        """최종 요약 보고서 생성"""
        
        print("\n" + "="*80)
        print("📊 최종 실험 결과 요약")
        print("="*80)
        
        baseline_jct = baseline['jct_simulated']
        simple_jct = simple_adaptive['jct_simulated']
        
        print(f"\n1. Baseline (균등 분배)")
        print(f"   JCT: {baseline_jct:.2f}s")
        print(f"   시간 균형도: {baseline['balance']['balance_score']:.4f}")
        
        print(f"\n2. Simple Adaptive (성능 비율 4:2:1)")
        print(f"   JCT: {simple_jct:.2f}s")
        print(f"   개선율: {(baseline_jct - simple_jct) / baseline_jct * 100:.2f}%")
        print(f"   시간 균형도: {simple_adaptive['balance']['balance_score']:.4f}")
        
        print(f"\n3. 최적화 기법별 결과:")
        
        summary = {
            'baseline': {
                'jct': baseline_jct,
                'balance': baseline['balance']['balance_score']
            },
            'simple_adaptive': {
                'jct': simple_jct,
                'improvement': (baseline_jct - simple_jct) / baseline_jct * 100,
                'balance': simple_adaptive['balance']['balance_score']
            },
            'optimized_methods': {}
        }
        
        for method_name, result in optimized_results.items():
            jct = result['jct_simulated']
            improvement = (baseline_jct - jct) / baseline_jct * 100
            balance = result['balance']['balance_score']
            
            print(f"\n   [{method_name}]")
            print(f"   JCT: {jct:.2f}s")
            print(f"   개선율: {improvement:.2f}%")
            print(f"   시간 균형도: {balance:.4f}")
            
            summary['optimized_methods'][method_name] = {
                'jct': jct,
                'improvement': improvement,
                'balance': balance
            }
        
        # 최고 성능 방법
        best_method = min(optimized_results.items(), key=lambda x: x[1]['jct_simulated'])
        best_name, best_result = best_method
        best_jct = best_result['jct_simulated']
        best_improvement = (baseline_jct - best_jct) / baseline_jct * 100
        
        print(f"\n" + "="*80)
        print(f"🏆 최고 성능: {best_name}")
        print(f"   JCT: {best_jct:.2f}s")
        print(f"   총 개선율: {best_improvement:.2f}%")
        print(f"   시간 균형도: {best_result['balance']['balance_score']:.4f}")
        print("="*80)
        
        summary['best_method'] = {
            'name': best_name,
            'jct': best_jct,
            'total_improvement': best_improvement
        }
        
        # 저장
        with open('results/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_converter)
        
        return summary
    
    def _json_converter(self, o):
        """JSON 직렬화를 위한 변환기"""
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

def main():
    print("="*80)
    print("🚀 Kubeflow 적응형 워크로드 분배 시스템 - 전체 실험")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    runner = ExperimentRunner()
    
    # 1. Baseline 실험
    baseline = runner.run_baseline_experiment()
    
    # 2. Simple Adaptive 실험
    simple_adaptive = runner.run_simple_adaptive_experiment()
    
    # 3. 최적화 실험 (Grid/Random/Bayesian)
    optimization_results = runner.run_optimization_experiments()
    
    # 4. 최종 검증 (최적 가중치로 실제 학습)
    optimized_results = runner.run_final_validation(optimization_results)
    
    # 5. 요약 보고서 생성
    summary = runner.generate_summary_report(baseline, simple_adaptive, optimized_results)
    
    print(f"\n{'='*80}")
    print(f"✅ 모든 실험 완료!")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과 저장 위치: ./results/")
    print("="*80)
    
    return summary

if __name__ == "__main__":
    summary = main()
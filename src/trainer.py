"""
분산 학습 시뮬레이터 (이기종 환경)
- 실제 학습 수행 후 성능 계수로 시간 스케일링
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple
import json

class NodeProfile:
    """노드 프로필 (이기종 하드웨어 시뮬레이션)"""
    def __init__(self, node_id: int, cpu_factor: float, memory_mb: int, name: str):
        self.node_id = node_id
        self.cpu_factor = cpu_factor  # 1.0 = 고성능, 0.5 = 중성능, 0.25 = 저성능
        self.memory_mb = memory_mb
        self.name = name
    
    def __repr__(self):
        return f"Node{self.node_id}({self.name}, CPU={self.cpu_factor*100}%, MEM={self.memory_mb}MB)"

class DistributedTrainer:
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.batch_size = model_config.get('batch_size', 32)
        self.epochs = model_config.get('epochs', 5)
        
        # 이기종 노드 프로필 정의
        self.nodes = [
            NodeProfile(1, cpu_factor=1.0, memory_mb=8192, name="High-Perf"),
            NodeProfile(2, cpu_factor=0.5, memory_mb=4096, name="Mid-Perf"),
            NodeProfile(3, cpu_factor=0.25, memory_mb=2048, name="Low-Perf")
        ]
    
    def create_model(self) -> keras.Model:
        """
        간단한 CNN 모델 (MNIST)
        """
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_on_node(
        self, 
        node: NodeProfile,
        x_train: np.ndarray,
        y_train: np.ndarray,
        verbose: int = 0
    ) -> Tuple[float, float, Dict]:
        """
        특정 노드에서 학습 수행
        
        Returns:
            actual_time: 실제 걸린 시간
            simulated_time: 성능 계수 적용한 시뮬레이션 시간
            metrics: 학습 메트릭
        """
        print(f"\n{'='*60}")
        print(f"[{node.name}] Node {node.node_id} 학습 시작")
        print(f"  - 할당 데이터: {len(x_train):,} samples")
        print(f"  - CPU 성능: {node.cpu_factor*100:.0f}% (기준 대비)")
        print(f"  - 메모리: {node.memory_mb} MB")
        print(f"{'='*60}")
        
        # 모델 생성
        model = self.create_model()
        
        # 실제 학습 수행
        start_time = time.time()
        
        history = model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
            validation_split=0.1
        )
        
        actual_time = time.time() - start_time
        
        # 성능 계수 적용: 느린 노드는 같은 작업을 더 오래 수행
        # 실제 하드웨어라면 CPU가 느려서 시간이 더 걸림
        simulated_time = actual_time / node.cpu_factor
        
        # 학습 메트릭
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        metrics = {
            'node_id': node.node_id,
            'node_name': node.name,
            'cpu_factor': node.cpu_factor,
            'data_size': len(x_train),
            'actual_time': actual_time,
            'simulated_time': simulated_time,
            'final_loss': final_loss,
            'final_accuracy': final_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'samples_per_sec': len(x_train) / simulated_time
        }
        
        print(f"\n[{node.name}] 학습 완료")
        print(f"  - 실제 시간: {actual_time:.2f}s")
        print(f"  - 시뮬레이션 시간: {simulated_time:.2f}s (CPU {node.cpu_factor*100:.0f}% 반영)")
        print(f"  - 최종 정확도: {final_acc:.4f}")
        print(f"  - 처리 속도: {metrics['samples_per_sec']:.2f} samples/sec")
        
        return actual_time, simulated_time, metrics
    
    def run_distributed_training(
        self,
        data_distributions: List[int],
        experiment_name: str = "experiment"
    ) -> Dict:
        """
        분산 학습 실험 실행
        
        Args:
            data_distributions: 각 노드에 할당할 데이터 개수 [node1, node2, node3]
            experiment_name: 실험 이름
        
        Returns:
            실험 결과 딕셔너리
        """
        print(f"\n{'#'*60}")
        print(f"# 실험 시작: {experiment_name}")
        print(f"# 데이터 분배: {data_distributions}")
        print(f"{'#'*60}")
        
        # MNIST 데이터 로드
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train_full = x_train_full.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
        
        # 데이터 분할
        start_idx = 0
        node_data = []
        for dist in data_distributions:
            end_idx = start_idx + dist
            node_data.append((
                x_train_full[start_idx:end_idx],
                y_train_full[start_idx:end_idx]
            ))
            start_idx = end_idx
        
        # 각 노드에서 학습 (실제로는 병렬이지만 여기서는 순차)
        all_metrics = []
        actual_times = []
        simulated_times = []
        
        for i, node in enumerate(self.nodes):
            x_train, y_train = node_data[i]
            
            actual_t, sim_t, metrics = self.train_on_node(
                node, x_train, y_train, verbose=0
            )
            
            actual_times.append(actual_t)
            simulated_times.append(sim_t)
            all_metrics.append(metrics)
        
        # JCT 계산 (가장 느린 노드가 전체 시간 결정)
        jct_actual = max(actual_times)
        jct_simulated = max(simulated_times)
        
        # 시간 균형도 계산
        balance_metrics = self._calculate_balance(simulated_times)
        
        # 결과 정리
        result = {
            'experiment_name': experiment_name,
            'data_distributions': data_distributions,
            'nodes': [n.__dict__ for n in self.nodes],
            'node_metrics': all_metrics,
            'jct_actual': jct_actual,
            'jct_simulated': jct_simulated,
            'actual_times': actual_times,
            'simulated_times': simulated_times,
            'balance': balance_metrics,
            'total_data': sum(data_distributions)
        }
        
        # 결과 출력
        self._print_summary(result)
        
        return result
    
    def _calculate_balance(self, times: List[float]) -> Dict[str, float]:
        """시간 균형도 계산"""
        times_arr = np.array(times)
        mean_time = np.mean(times_arr)
        std_time = np.std(times_arr)
        
        balance = 1 - (std_time / mean_time) if mean_time > 0 else 0
        
        return {
            'mean_time': float(mean_time),
            'std_time': float(std_time),
            'balance_score': float(balance),
            'min_time': float(np.min(times_arr)),
            'max_time': float(np.max(times_arr))
        }
    
    def _print_summary(self, result: Dict):
        """결과 요약 출력"""
        print(f"\n{'='*60}")
        print(f"실험 결과 요약: {result['experiment_name']}")
        print(f"{'='*60}")
        
        print(f"\n📊 데이터 분배:")
        for i, dist in enumerate(result['data_distributions']):
            ratio = dist / result['total_data'] * 100
            print(f"  Node {i+1}: {dist:,} samples ({ratio:.1f}%)")
        
        print(f"\n⏱️  학습 시간 (시뮬레이션):")
        for i, t in enumerate(result['simulated_times']):
            print(f"  Node {i+1}: {t:.2f}s")
        
        print(f"\n🎯 핵심 지표:")
        print(f"  JCT (Job Completion Time): {result['jct_simulated']:.2f}s")
        print(f"  평균 시간: {result['balance']['mean_time']:.2f}s")
        print(f"  시간 편차 (σ): {result['balance']['std_time']:.2f}s")
        print(f"  시간 균형도: {result['balance']['balance_score']:.4f}")
        
        print(f"\n✅ 모델 정확도:")
        for metric in result['node_metrics']:
            print(f"  Node {metric['node_id']}: {metric['final_accuracy']:.4f}")
        
        print(f"{'='*60}\n")

def compare_baseline_vs_adaptive():
    """Baseline vs Adaptive 비교 실험"""
    
    config = {
        'batch_size': 32,
        'epochs': 3  # 빠른 테스트용
    }
    
    trainer = DistributedTrainer(config)
    
    total_data = 60000
    
    # 1. Baseline: 균등 분배
    baseline_dist = [20000, 20000, 20000]
    baseline_result = trainer.run_distributed_training(
        baseline_dist,
        experiment_name="Baseline (균등 분배)"
    )
    
    # 2. Adaptive: 성능 기반 분배 (성능 비율 1.0:0.5:0.25 = 4:2:1)
    # 총합 = 7, Node1 = 4/7, Node2 = 2/7, Node3 = 1/7
    adaptive_dist = [
        int(total_data * 4 / 7),  # ~34,286
        int(total_data * 2 / 7),  # ~17,143
        int(total_data * 1 / 7)   # ~8,571
    ]
    # 합계 맞추기
    adaptive_dist[0] += total_data - sum(adaptive_dist)
    
    adaptive_result = trainer.run_distributed_training(
        adaptive_dist,
        experiment_name="Adaptive (성능 기반 분배)"
    )
    
    # 3. 비교 분석
    print(f"\n{'#'*60}")
    print("# 최종 비교 분석")
    print(f"{'#'*60}\n")
    
    baseline_jct = baseline_result['jct_simulated']
    adaptive_jct = adaptive_result['jct_simulated']
    improvement = (baseline_jct - adaptive_jct) / baseline_jct * 100
    
    print(f"📈 JCT 비교:")
    print(f"  Baseline:  {baseline_jct:.2f}s")
    print(f"  Adaptive:  {adaptive_jct:.2f}s")
    print(f"  개선율:    {improvement:.2f}%")
    
    print(f"\n⚖️  시간 균형도:")
    print(f"  Baseline:  {baseline_result['balance']['balance_score']:.4f}")
    print(f"  Adaptive:  {adaptive_result['balance']['balance_score']:.4f}")
    
    # 결과 저장
    comparison = {
        'baseline': baseline_result,
        'adaptive': adaptive_result,
        'improvement_percent': improvement
    }
    
    return comparison

if __name__ == "__main__":
    print("🚀 분산 학습 시뮬레이션 시작\n")
    results = compare_baseline_vs_adaptive()
    
    # JSON 저장
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/baseline_vs_adaptive.json', 'w') as f:
        # numpy 타입 변환
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        
        json.dump(results, f, indent=2, default=convert)
    
    print("\n💾 결과 저장됨: results/baseline_vs_adaptive.json")